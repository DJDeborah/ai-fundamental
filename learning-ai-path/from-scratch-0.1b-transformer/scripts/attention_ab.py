"""Alternating CUDA-event A/B timing for Triton causal attention vs SDPA.

Measures forward-only GPU time with identical Q/K/V. The result is not a
training or end-to-end generation speedup because the custom kernel lacks a
backward pass and the measurement starts after Q/K/V already exist.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F

from fundamental.triton_attention import causal_attention


def elapsed_ms(fn, iterations: int):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / iterations


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(args.seed)
    q = torch.randn(args.batch, args.heads, args.seq_len, args.head_dim,
                    device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    funcs = {
        "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        "triton": lambda: causal_attention(q, k, v),
    }
    ref, actual = funcs["sdpa"](), funcs["triton"]()
    error = float((ref.float() - actual.float()).abs().max())
    torch.testing.assert_close(actual, ref, atol=2e-2, rtol=2e-2)
    for _ in range(30):
        funcs["sdpa"]()
        funcs["triton"]()
    torch.cuda.synchronize()
    rows = []
    for repetition in range(args.repetitions):
        order = ("sdpa", "triton") if repetition % 2 == 0 else ("triton", "sdpa")
        times = {name: elapsed_ms(funcs[name], args.iterations) for name in order}
        row = {"repetition": repetition, "order": list(order), "seq_len": args.seq_len,
               "batch": args.batch, "heads": args.heads, "head_dim": args.head_dim,
               "iterations": args.iterations, "dtype": "float16", "gpu": torch.cuda.get_device_name(),
               "max_abs_error": error, "sdpa_event_ms": times["sdpa"],
               "triton_event_ms": times["triton"],
               "speedup": times["sdpa"] / times["triton"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"seq_len": args.seq_len, "repetitions": args.repetitions,
                      "speedup_median": statistics.median(row["speedup"] for row in rows),
                      "speedup_min": min(row["speedup"] for row in rows),
                      "speedup_max": max(row["speedup"] for row in rows)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--repetitions", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
