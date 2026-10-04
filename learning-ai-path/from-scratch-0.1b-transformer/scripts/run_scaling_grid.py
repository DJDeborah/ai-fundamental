"""Run a resumable 3-model x 3-seed scaling experiment on one GPU.

Each run evaluates at the same optimizer steps and uses a common validation
sampling seed. A failed run stops the grid; rerunning resumes its checkpoint.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_rows(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_resume(run_dir: Path, steps: int):
    rows = metric_rows(run_dir / "metrics.jsonl")
    if rows and rows[-1]["step"] == steps and (run_dir / "base.pt").exists():
        return "complete", None
    checkpoint = run_dir / "latest.pt"
    if not checkpoint.exists():
        return "fresh", None
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_step = int(saved["step"])
    if saved_step >= steps:
        raise RuntimeError(f"latest.pt reached step {saved_step} but base.pt is missing: {run_dir}")
    retained = [row for row in rows if row["step"] <= saved_step]
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row) + "\n")
    return "resume", checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=["small", "medium", "base100m"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--global-tokens", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=36621)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=[2441, 9766, 36621])
    parser.add_argument("--eval-seed", type=int, default=10042)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--max-hours", type=float, default=90)
    parser.add_argument("--min-free-gb", type=float, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("scaling grid requires a CUDA GPU")
    if args.steps < 1 or any(step < 1 or step > args.steps for step in args.eval_steps):
        raise ValueError("invalid step schedule")
    if args.global_tokens % (args.batch_size * args.seq_len):
        raise ValueError("global_tokens must be divisible by batch_size * seq_len")
    data_manifest = args.data / "manifest.json"
    if not data_manifest.exists():
        raise FileNotFoundError(data_manifest)
    data = json.loads(data_manifest.read_text(encoding="utf-8"))
    if data["train"]["tokens"] < args.global_tokens * args.steps:
        raise ValueError("training split is smaller than the largest measured token budget")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "grid_manifest.json"
    manifest = {
        "started_at_utc": now(), "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "data_manifest_sha256": sha256_file(data_manifest),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "code_sha256": {name: sha256_file(Path(__file__).resolve().parents[1] / name)
                        for name in ("fundamental/model.py", "fundamental/train.py",
                                     "fundamental/tokenizer.py")},
        "recipe": {"configs": args.configs, "seeds": args.seeds,
                   "batch_size": args.batch_size, "seq_len": args.seq_len,
                   "global_tokens": args.global_tokens, "steps": args.steps,
                   "eval_steps": args.eval_steps, "eval_seed": args.eval_seed,
                   "eval_batches": args.eval_batches,
                   "save_every": args.save_every},
        "runs": {},
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (previous["recipe"] != manifest["recipe"] or
            previous["tokenizer_sha256"] != manifest["tokenizer_sha256"] or
            previous["data_manifest_sha256"] != manifest["data_manifest_sha256"] or
            previous["code_sha256"] != manifest["code_sha256"]):
            raise ValueError("grid recipe or data changed; choose a new output directory")
        manifest = previous
    started = time.monotonic()
    for config in args.configs:
        for seed in args.seeds:
            run_id = f"{config}_seed{seed}"
            run_dir = args.out / run_id
            if args.dry_run:
                rows = metric_rows(run_dir / "metrics.jsonl")
                mode = "complete" if rows and rows[-1]["step"] == args.steps and (run_dir / "base.pt").exists() else "resume" if (run_dir / "latest.pt").exists() else "fresh"
                checkpoint = run_dir / "latest.pt" if mode == "resume" else None
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                mode, checkpoint = prepare_resume(run_dir, args.steps)
            if mode == "complete":
                print(f"skip complete {run_id}", flush=True)
                continue
            if time.monotonic() - started > args.max_hours * 3600:
                raise RuntimeError("grid wall-time cap reached before starting next run")
            free_gb = shutil.disk_usage(args.out).free / 1e9
            if free_gb < args.min_free_gb:
                raise RuntimeError(f"only {free_gb:.1f} GB free; stop before writing more checkpoints")
            command = [sys.executable, "-m", "fundamental.train",
                       "--data", str(args.data), "--tokenizer", str(args.tokenizer),
                       "--config", config, "--batch-size", str(args.batch_size),
                       "--seq-len", str(args.seq_len), "--global-tokens", str(args.global_tokens),
                       "--steps", str(args.steps), "--eval-every", "0",
                       "--eval-steps", *map(str, args.eval_steps),
                       "--eval-seed", str(args.eval_seed), "--seed", str(seed),
                       "--eval-batches", str(args.eval_batches),
                       "--save-every", str(args.save_every), "--out", str(run_dir)]
            if checkpoint:
                command.extend(["--resume", str(checkpoint)])
            print(json.dumps({"run": run_id, "mode": mode, "command": command}), flush=True)
            if args.dry_run:
                continue
            manifest["runs"][run_id] = {"status": "running", "started_at_utc": now(), "mode": mode}
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            began = time.monotonic()
            with (run_dir / "driver.log").open("a", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            manifest["runs"][run_id].update({"status": "complete" if result.returncode == 0 else "failed",
                                              "returncode": result.returncode,
                                              "elapsed_s": time.monotonic() - began,
                                              "finished_at_utc": now()})
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps({"run": run_id, **manifest["runs"][run_id]}), flush=True)
            if result.returncode:
                raise RuntimeError(f"{run_id} failed; inspect {run_dir / 'driver.log'}")
    print(json.dumps({"grid": "complete", "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
