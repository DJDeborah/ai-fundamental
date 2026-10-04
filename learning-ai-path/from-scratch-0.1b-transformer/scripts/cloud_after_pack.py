"""Wait for the full token pack, run a target-size probe, then start the scaling grid.

This script is designed for an AutoDL terminal detached from the browser. Its
stage log is append-only, and the grid runner resumes from checkpoints.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def event(stage: str, **fields):
    print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                      "stage": stage, **fields}), flush=True)


def run(command: list[str], log_path: Path):
    event("start", command=command, log=str(log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    event("finish", returncode=result.returncode, log=str(log_path))
    if result.returncode:
        raise RuntimeError(f"command failed: inspect {log_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=Path("/root/autodl-tmp/corpus"))
    p.add_argument("--runs", type=Path, default=Path("/root/autodl-tmp/runs"))
    p.add_argument("--wait-hours", type=float, default=3)
    a = p.parse_args()
    tokens = a.corpus / "tokens_8192"
    manifest_path = tokens / "manifest.json"
    started = time.monotonic()
    event("waiting_for_pack", manifest=str(manifest_path))
    while not manifest_path.exists():
        if time.monotonic() - started > a.wait_hours * 3600:
            raise TimeoutError(f"token pack did not complete; inspect {a.corpus / 'token_pack.log'}")
        time.sleep(30)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event("pack_ready", counts=manifest)
    tokenizer = a.corpus / "tokenizer_8192.json"
    if not tokenizer.exists() or manifest["train"]["tokens"] < 300_000_000:
        raise RuntimeError("missing tokenizer or fewer than 300M training tokens")
    probe = a.runs / "full100m_probe"
    if not (probe / "base.pt").exists():
        run([sys.executable, "-m", "fundamental.train", "--data", str(tokens),
             "--tokenizer", str(tokenizer), "--config", "base100m",
             "--batch-size", "1", "--seq-len", "1024", "--global-tokens", "8192",
             "--steps", "20", "--eval-every", "10", "--seed", "42",
             "--out", str(probe)], probe / "driver.log")
    event("probe_ready", metrics=str(probe / "metrics.jsonl"))
    run([sys.executable, "scripts/run_scaling_grid.py", "--data", str(tokens),
         "--tokenizer", str(tokenizer), "--out", str(a.runs / "scaling_grid_v1"),
         "--max-hours", "90", "--min-free-gb", "8"],
        a.runs / "scaling_grid_v1" / "driver.log")
    event("grid_ready", manifest=str(a.runs / "scaling_grid_v1" / "grid_manifest.json"))


if __name__ == "__main__":
    main()
