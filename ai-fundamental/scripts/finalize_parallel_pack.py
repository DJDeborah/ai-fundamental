"""Validate a completed parallel pack and publish it to the standard path.

The manifest is moved last; consumers only start after every binary is present.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-manifest", type=Path, required=True)
    p.add_argument("--parallel-dir", type=Path, required=True)
    p.add_argument("--standard-dir", type=Path, required=True)
    a = p.parse_args()
    if (a.standard_dir / "manifest.json").exists():
        raise RuntimeError("standard pack is already published")
    clean = json.loads(a.clean_manifest.read_text(encoding="utf-8"))
    packed = json.loads((a.parallel_dir / "manifest.json").read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        source = a.parallel_dir / f"{split}.bin"
        record = packed[split]
        if record["documents"] != clean["counts"][split]:
            raise ValueError(f"{split} document count disagrees with cleaning manifest")
        if source.stat().st_size != record["tokens"] * 2:
            raise ValueError(f"{split} uint16 file length disagrees with token count")
        if record["dtype"] != "uint16":
            raise ValueError("unexpected token dtype")
    a.standard_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        os.replace(a.parallel_dir / f"{split}.bin", a.standard_dir / f"{split}.bin")
    os.replace(a.parallel_dir / "manifest.json", a.standard_dir / "manifest.json")
    print(json.dumps({"published": str(a.standard_dir), "summary": packed}), flush=True)


if __name__ == "__main__":
    main()
