"""Single GPU/CPU or torchrun DDP causal-LM training."""
import argparse
import hashlib
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .model import GPT, config_for, save_checkpoint
from .tokenizer import ByteBPE


def sample_batch(data, batch_size, seq_len, rng, device):
    if len(data) <= seq_len:
        raise ValueError(f"need > {seq_len} tokens, have {len(data)}")
    starts = rng.integers(0, len(data) - seq_len, size=batch_size)
    offsets = np.arange(seq_len + 1)
    windows = np.asarray(data[starts[:, None] + offsets[None, :]], dtype=np.int64)
    tensor = torch.from_numpy(windows).to(device)
    return tensor[:, :-1], tensor[:, 1:]


def load_split(directory: Path, split: str):
    binary = directory / f"{split}.bin"
    if binary.exists():
        if binary.stat().st_size == 0:
            return np.empty(0, dtype="<u2")
        return np.memmap(binary, dtype="<u2", mode="r")
    # Backward-compatible with early teaching runs.
    return np.load(directory / f"{split}.npy", mmap_mode="r")


@torch.no_grad()
def evaluate(model, val, batch_size, seq_len, seed, device, n_batches=8):
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_batches):
        x, y = sample_batch(val, batch_size, seq_len, rng, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--config", choices=["tiny", "small", "medium", "base100m"], default="tiny")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=0)
    p.add_argument("--global-tokens", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-steps", type=int, nargs="*", default=[],
                   help="exact optimizer steps to evaluate; use --eval-every 0 to disable periodic evaluation")
    p.add_argument("--save-every", type=int, default=0, help="write resumable latest.pt every N steps; 0 = final only")
    p.add_argument("--resume", type=Path, help="latest.pt; --steps remains the total target step")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=10042,
                   help="fixed validation sampling seed, shared across training seeds")
    p.add_argument("--eval-batches", type=int, default=8,
                   help="number of fixed-seed validation batches per evaluation")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.eval_every < 0 or a.eval_batches < 1 or any(step < 1 or step > a.steps for step in a.eval_steps):
        raise ValueError("invalid evaluation schedule")
    eval_steps = set(a.eval_steps)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world > 1:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    tok = ByteBPE.load(a.tokenizer)
    tokenizer_sha256 = hashlib.sha256(a.tokenizer.read_bytes()).hexdigest()
    cfg = config_for(a.config, tok.size)
    seq = a.seq_len or cfg.max_seq_len
    if seq > cfg.max_seq_len:
        raise ValueError("seq_len exceeds model context")
    microtokens = world * a.batch_size * seq
    global_tokens = a.global_tokens or microtokens
    if global_tokens % microtokens:
        raise ValueError("global-tokens must be divisible by world * batch-size * seq-len")
    accum = global_tokens // microtokens
    train = load_split(a.data, "train")
    val = load_split(a.data, "val")
    if len(train) <= seq or len(val) <= seq:
        raise ValueError("train and val both need more than seq_len tokens")
    rng = np.random.default_rng(a.seed + rank)
    net = GPT(cfg).to(device)
    checkpoint = None
    start_step = 0
    if a.resume:
        checkpoint = torch.load(a.resume, map_location="cpu", weights_only=False)
        expected = {"config": asdict(cfg), "global_tokens": global_tokens, "world_size": world,
                    "seq_len": seq, "tokenizer_sha256": tokenizer_sha256}
        if "eval_seed" in checkpoint:
            expected["eval_seed"] = a.eval_seed
        if "eval_batches" in checkpoint:
            expected["eval_batches"] = a.eval_batches
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"resume mismatch for {key}: {checkpoint.get(key)} != {value}")
        net.load_state_dict(checkpoint["model"])
        start_step = int(checkpoint["step"])
        if start_step >= a.steps:
            raise ValueError("--steps must exceed the saved step when resuming")
        rng.bit_generator.state = checkpoint["rank_rng_states"][rank]
    if world > 1:
        net = DDP(net, device_ids=[local_rank] if device.type == "cuda" else None)
    raw_model = net.module if world > 1 else net
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.1)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    a.out.mkdir(parents=True, exist_ok=True)
    log = (a.out / "metrics.jsonl").open("a" if a.resume else "w", encoding="utf-8") if rank == 0 else None
    start = time.perf_counter()
    interval_start = start
    previous_eval_step = start_step
    checkpoint_io_s_since_eval = 0.0
    for step in range(start_step + 1, a.steps + 1):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(accum):
            x, y = sample_batch(train, a.batch_size, seq, rng, device)
            sync = nullcontext() if world == 1 or micro == accum - 1 else net.no_sync()
            with sync, ctx():
                _, loss = net(x, y)
                (loss / accum).backward()
            loss_sum += loss.detach().item() / accum
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        if (a.eval_every > 0 and step % a.eval_every == 0) or step in eval_steps or step == a.steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if rank == 0:
                train_interval_s = time.perf_counter() - interval_start - checkpoint_io_s_since_eval
                with ctx():
                    val_loss = evaluate(raw_model, val, a.batch_size, seq, a.eval_seed, device,
                                        n_batches=a.eval_batches)
                elapsed = time.perf_counter() - start
                rec = {"step": step, "tokens_seen": step * global_tokens, "train_loss_rank0": loss_sum,
                       "val_loss": val_loss, "train_tokens_per_s": (step - previous_eval_step) * global_tokens / max(train_interval_s, 1e-9),
                       "elapsed_s_including_eval": elapsed, "world_size": world, "params": raw_model.count_parameters(),
                       "seed": a.seed, "config": a.config, "seq_len": seq, "global_tokens": global_tokens,
                       "eval_seed": a.eval_seed,
                       "eval_batches": a.eval_batches,
                       "batch_size_per_gpu": a.batch_size, "tokenizer_sha256": tokenizer_sha256,
                       "segment_start_step": start_step}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                interval_start = time.perf_counter()
                previous_eval_step = step
                checkpoint_io_s_since_eval = 0.0
            if world > 1:
                dist.barrier()
        if (a.save_every > 0 and step % a.save_every == 0) or step == a.steps:
            local_rng_state = rng.bit_generator.state
            if world > 1:
                all_rng_states = [None] * world
                dist.all_gather_object(all_rng_states, local_rng_state)
            else:
                all_rng_states = [local_rng_state]
            if rank == 0:
                save_start = time.perf_counter()
                temp_path = a.out / "latest.tmp.pt"
                save_checkpoint(temp_path, raw_model, optimizer=optimizer.state_dict(),
                                rank_rng_states=all_rng_states, step=step, global_tokens=global_tokens,
                                world_size=world, seq_len=seq, tokenizer_sha256=tokenizer_sha256,
                                seed=a.seed, eval_seed=a.eval_seed, eval_batches=a.eval_batches)
                os.replace(temp_path, a.out / "latest.pt")
                checkpoint_io_s_since_eval += time.perf_counter() - save_start
    if rank == 0:
        save_checkpoint(a.out / "base.pt", raw_model, step=a.steps, tokens_seen=a.steps * global_tokens,
                        seed=a.seed, eval_seed=a.eval_seed)
        log.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
