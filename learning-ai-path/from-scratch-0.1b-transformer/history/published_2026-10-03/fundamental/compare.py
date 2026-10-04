"""Check matched conditions and compute measured 1→k speedups."""
import argparse
import json
from pathlib import Path


def last_record(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty log: {path}")
    return rows[-1]


def compare(a, b, kind, load_mode="per-gpu"):
    if kind == "train":
        keys = ("config", "params", "seq_len", "global_tokens", "tokenizer_sha256", "seed", "tokens_seen")
        metric = "train_tokens_per_s"
    else:
        if load_mode not in {"per-gpu", "fixed-total"}:
            raise ValueError("load_mode must be per-gpu or fixed-total")
        keys = ("kind", "checkpoint_sha256", "prompt_sha256", "prompt_tokens", "new_tokens")
        if load_mode == "per-gpu":
            keys += ("requests_per_gpu",)
        metric = "aggregate_generated_tokens_per_s"
    differences = {key: [a.get(key), b.get(key)] for key in keys if a.get(key) != b.get(key)}
    if differences:
        raise ValueError(f"unmatched benchmark conditions: {differences}")
    if a["world_size"] != 1 or b["world_size"] <= 1:
        raise ValueError("pass single-GPU log first and multi-GPU log second")
    if kind == "inference" and load_mode == "fixed-total" and \
            a["requests_per_gpu"] != b["world_size"] * b["requests_per_gpu"]:
        raise ValueError("fixed-total inference needs the same total request count")
    speedup = b[metric] / a[metric]
    return {"kind": kind, "single_world_size": 1, "multi_world_size": b["world_size"],
            "load_mode": load_mode if kind == "inference" else "fixed-global-tokens",
            "single_throughput": a[metric], "multi_throughput": b[metric],
            "throughput_unit": "train tokens/s" if kind == "train" else "generated tokens/s",
            "speedup": speedup, "parallel_efficiency": speedup / b["world_size"],
            "single_val_loss": a.get("val_loss"), "multi_val_loss": b.get("val_loss")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["train", "inference"], required=True)
    p.add_argument("--single", type=Path, required=True)
    p.add_argument("--multi", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--load-mode", choices=["per-gpu", "fixed-total"], default="per-gpu",
                   help="inference: scale total requests with GPUs, or hold total requests constant")
    args = p.parse_args()
    result = compare(last_record(args.single), last_record(args.multi), args.kind, args.load_mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
