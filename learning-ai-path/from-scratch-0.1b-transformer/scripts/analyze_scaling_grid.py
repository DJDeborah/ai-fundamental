"""Validate a complete 3x3x3 grid and fit the held-out scaling surface."""

import argparse
import hashlib
import json
from pathlib import Path

from fundamental.scaling import fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target-tokens", type=int, default=2_000_000_000)
    p.add_argument("--bootstrap-samples", type=int, default=200)
    a = p.parse_args()
    manifest_path = a.grid / "grid_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recipe = manifest["recipe"]
    rows = []
    files = []
    for config in recipe["configs"]:
        for seed in recipe["seeds"]:
            name = f"{config}_seed{seed}"
            if manifest["runs"].get(name, {}).get("status") != "complete":
                raise RuntimeError(f"grid run not complete: {name}")
            path = a.grid / name / "metrics.jsonl"
            entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if [row["step"] for row in entries] != recipe["eval_steps"]:
                raise ValueError(f"unexpected validation schedule in {path}")
            for row in entries:
                expected = {"config": config, "seed": seed,
                            "eval_seed": recipe["eval_seed"],
                            "eval_batches": recipe["eval_batches"],
                            "batch_size_per_gpu": recipe["batch_size"],
                            "seq_len": recipe["seq_len"],
                            "global_tokens": recipe["global_tokens"],
                            "tokenizer_sha256": manifest["tokenizer_sha256"]}
                for key, value in expected.items():
                    if row.get(key) != value:
                        raise ValueError(f"{path}: {key} mismatch")
                if row["tokens_seen"] != row["step"] * recipe["global_tokens"]:
                    raise ValueError(f"{path}: token count mismatch")
            rows.extend(entries)
            files.append(str(path))
    sizes = {row["config"]: row["params"] for row in rows}
    target_params = sizes["base100m"]
    result = fit(rows, bootstrap_samples=a.bootstrap_samples,
                 target_params=target_params, target_tokens=a.target_tokens)
    result["provenance"] = {
        "grid_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "data_manifest_sha256": manifest["data_manifest_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "validation_tokens_per_point": recipe["eval_batches"] * recipe["batch_size"] * recipe["seq_len"],
        "metric_files": files, "raw_observations": len(rows), "model_sizes": sizes,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(a.out), "holdout": result["holdout"],
                      "extrapolation": result["extrapolation"],
                      "bootstrap": result["bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
