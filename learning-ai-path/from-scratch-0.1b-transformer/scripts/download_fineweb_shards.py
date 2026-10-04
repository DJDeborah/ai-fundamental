"""Download the two exact FineWeb-Edu sample/10BT shards used in the report.

Uses requests' environment proxy settings; on AutoDL, run
`source /etc/network_turbo` in the same shell before this command.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import requests


SHARDS = {
    "000": "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871",
    "001": "3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693",
}
ROOT = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(out: Path, shard: str):
    expected = SHARDS[shard]
    path = out / f"fineweb_edu_{shard}.parquet"
    url = f"{ROOT}/{shard}_00000.parquet?download=true"
    if path.exists() and digest(path) == expected:
        return {"shard": shard, "url": url, "path": str(path),
                "bytes": path.stat().st_size, "sha256": expected, "status": "verified_existing"}
    partial = path.with_suffix(".parquet.part")
    partial.unlink(missing_ok=True)
    h = hashlib.sha256()
    try:
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        h.update(chunk)
                        handle.write(chunk)
        if h.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {shard}: got {h.hexdigest()}")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return {"shard": shard, "url": url, "path": str(path),
            "bytes": path.stat().st_size, "sha256": expected, "status": "downloaded"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shards", nargs="+", choices=sorted(SHARDS), default=sorted(SHARDS))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for shard in a.shards:
        if a.dry_run:
            print(json.dumps({"shard": shard, "url": f"{ROOT}/{shard}_00000.parquet?download=true",
                              "expected_sha256": SHARDS[shard]}))
        else:
            print(json.dumps(download(a.out, shard)), flush=True)


if __name__ == "__main__":
    main()
