"""Matched attention and replicated multi-GPU inference benchmarks."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from .model import load_checkpoint
from .tokenizer import ByteBPE, EOS


def bench_attention(args):
    if not torch.cuda.is_available():
        raise RuntimeError("attention benchmark needs CUDA")
    from .triton_attention import causal_attention
    device = "cuda"
    q = torch.randn(args.batch, args.heads, args.seq_len, args.head_dim, device=device, dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    actual = causal_attention(q, k, v)
    error = (actual.float() - reference.float()).abs()
    if not torch.allclose(actual, reference, atol=2e-2, rtol=2e-2):
        raise AssertionError(f"attention mismatch: max_abs={error.max().item()}")
    results = {}
    for name, fn in (("sdpa", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)),
                     ("triton", lambda: causal_attention(q, k, v))):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            fn()
        torch.cuda.synchronize()
        results[name + "_ms"] = (time.perf_counter() - start) * 1000 / args.iterations
    return {"kind": "attention_forward", "gpu": torch.cuda.get_device_name(), "shape": list(q.shape),
            "max_abs_error": error.max().item(), **results, "triton_speedup": results["sdpa_ms"] / results["triton_ms"]}


@torch.inference_mode()
def generate(model, ids, new_tokens):
    for _ in range(new_tokens):
        logits = model(ids[:, -model.cfg.max_seq_len:])
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat((ids, next_id), dim=-1)
    return ids


def bench_inference(args):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if world > 1:
        if device.startswith("cuda"):
            torch.cuda.set_device(local)
        dist.init_process_group("nccl" if device.startswith("cuda") else "gloo")
    model, _ = load_checkpoint(args.base, device)
    model.eval()
    tok = ByteBPE.load(args.tokenizer)
    if tok.size != model.cfg.vocab_size:
        raise ValueError("tokenizer/model vocab mismatch")
    prompt = tok.encode(args.prompt) or [EOS]
    if len(prompt) > model.cfg.max_seq_len:
        prompt = prompt[-model.cfg.max_seq_len:]
    x = torch.tensor([prompt] * args.requests_per_gpu, device=device)
    generate(model, x, 2)  # warmup
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    start = time.perf_counter()
    generate(model, x, args.new_tokens)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if world > 1:
        t = torch.tensor([elapsed], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        elapsed = t.item()
    result = {"kind": "replicated_inference_full_recompute", "world_size": world,
              "requests_per_gpu": args.requests_per_gpu, "new_tokens": args.new_tokens,
              "prompt_tokens": len(prompt), "wall_s": elapsed,
              "aggregate_generated_tokens_per_s": world * args.requests_per_gpu * args.new_tokens / elapsed,
              "per_request_batch_latency_s": elapsed,
              "checkpoint_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
              "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
              "device": torch.cuda.get_device_name(local) if device.startswith("cuda") else "CPU"}
    if world > 1:
        dist.destroy_process_group()
    return result if rank == 0 else None


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="kind", required=True)
    a = sub.add_parser("attention")
    a.add_argument("--batch", type=int, default=2)
    a.add_argument("--heads", type=int, default=12)
    a.add_argument("--seq-len", type=int, default=512)
    a.add_argument("--head-dim", type=int, default=64)
    a.add_argument("--iterations", type=int, default=100)
    a.add_argument("--out", type=Path)
    b = sub.add_parser("inference")
    b.add_argument("--base", type=Path, required=True)
    b.add_argument("--tokenizer", type=Path, required=True)
    b.add_argument("--prompt", default="Explain attention in one sentence.")
    b.add_argument("--requests-per-gpu", type=int, default=1)
    b.add_argument("--new-tokens", type=int, default=32)
    b.add_argument("--out", type=Path)
    args = p.parse_args()
    result = bench_attention(args) if args.kind == "attention" else bench_inference(args)
    if result is not None:
        print(json.dumps(result))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
