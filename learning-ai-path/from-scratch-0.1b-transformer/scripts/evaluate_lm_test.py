"""Evaluate every next-token target in a packed held-out split exactly once."""

import argparse
import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from fundamental.model import load_checkpoint
from fundamental.tokenizer import ByteBPE
from fundamental.train import load_split


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.batch_size < 1 or a.seq_len < 1:
        raise ValueError("positive batch and sequence lengths required")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, payload = load_checkpoint(a.checkpoint, device)
    metadata_keys = {"config", "step", "tokens_seen", "seed", "eval_seed", "stage",
                     "root_base_sha256", "parent_sha256", "steps", "skipped_zero_variance",
                     "sampled_tokens", "elapsed_s"}
    metadata = {key: value for key, value in payload.items() if key in metadata_keys}
    model.eval()
    tok = ByteBPE.load(a.tokenizer)
    if tok.size != model.cfg.vocab_size or a.seq_len > model.cfg.max_seq_len:
        raise ValueError("tokenizer vocabulary or context mismatch")
    data = load_split(a.data, a.split)
    if len(data) < 2:
        raise ValueError("split has no next-token targets")
    full_windows, remainder = divmod(len(data) - 1, a.seq_len)
    amp = device == "cuda" and torch.cuda.is_bf16_supported()
    ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    total_nll = 0.0
    total_targets = 0
    started = time.perf_counter()
    with torch.inference_mode(), ctx():
        for first in range(0, full_windows, a.batch_size):
            starts = np.arange(first, min(first + a.batch_size, full_windows)) * a.seq_len
            offsets = np.arange(a.seq_len + 1)
            windows = np.asarray(data[starts[:, None] + offsets[None, :]], dtype=np.int64)
            batch = torch.from_numpy(windows).to(device)
            _, loss = model(batch[:, :-1], batch[:, 1:])
            count = batch[:, 1:].numel()
            total_nll += float(loss) * count
            total_targets += count
        if remainder:
            start = full_windows * a.seq_len
            tail = torch.from_numpy(np.asarray(data[start:], dtype=np.int64)).to(device)[None, :]
            _, loss = model(tail[:, :-1], tail[:, 1:])
            total_nll += float(loss) * remainder
            total_targets += remainder
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {"split": a.split, "checkpoint": str(a.checkpoint),
              "checkpoint_sha256": sha256(a.checkpoint), "tokenizer_sha256": sha256(a.tokenizer),
              "tokens_in_split": len(data), "targets_evaluated": total_targets,
              "nll_nats": total_nll, "loss_nats_per_token": total_nll / total_targets,
              "perplexity": math.exp(total_nll / total_targets),
              "seq_len": a.seq_len, "batch_size": a.batch_size,
              "elapsed_s": elapsed, "targets_per_s": total_targets / elapsed,
              "device": torch.cuda.get_device_name() if device == "cuda" else "CPU",
              "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              "checkpoint_metadata": metadata}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
