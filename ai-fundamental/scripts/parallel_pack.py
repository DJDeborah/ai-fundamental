"""Pack clean JSONL into uint16 token streams using independent byte ranges.

Each worker begins at a JSONL line boundary and writes its own part. Parts are
concatenated in source order, so the output matches the serial packer byte for
byte. This changes throughput only; it does not change token IDs or splits.
"""

import argparse
import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from fundamental.tokenizer import ByteBPE, EOS


def ranges(path: Path, workers: int):
    size = path.stat().st_size
    boundaries = [0]
    with path.open("rb") as handle:
        for index in range(1, workers):
            handle.seek(size * index // workers)
            handle.readline()  # Move to the beginning of the next complete row.
            boundaries.append(handle.tell())
    boundaries.append(size)
    return [(boundaries[i], boundaries[i + 1]) for i in range(workers)]


def pack_range(source: str, tokenizer: str, start: int, end: int, part: str):
    tok = ByteBPE.load(Path(tokenizer))
    documents = tokens = 0
    with Path(source).open("rb") as read, Path(part).open("wb") as write:
        read.seek(start)
        while read.tell() < end:
            line = read.readline()
            if not line:
                break
            text = json.loads(line)["text"]
            ids = tok.encode(text, eos=True)
            np.asarray(ids, dtype="<u2").tofile(write)
            documents += 1
            tokens += len(ids)
    return documents, tokens


def pack_split(source: Path, tokenizer: Path, destination: Path, workers: int):
    if not source.exists():
        return {"documents": 0, "tokens": 0, "dtype": "uint16", "format": "raw little-endian .bin"}
    parts = ranges(source, workers)
    with tempfile.TemporaryDirectory(prefix="token-parts-", dir=destination.parent) as temp:
        part_paths = [Path(temp) / f"{index:04}.bin" for index in range(len(parts))]
        counts = [None] * len(parts)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(pack_range, str(source), str(tokenizer), start, end, str(part_paths[i])): i
                    for i, (start, end) in enumerate(parts)}
            for future in as_completed(jobs):
                index = jobs[future]
                counts[index] = future.result()
                print(json.dumps({"split": source.stem, "part": index, "documents": counts[index][0],
                                  "tokens": counts[index][1]}), flush=True)
        combined = destination.with_suffix(".tmp.bin")
        try:
            with combined.open("wb") as output:
                for part in part_paths:
                    with part.open("rb") as handle:
                        shutil.copyfileobj(handle, output, length=8 * 1024 * 1024)
            combined.replace(destination)
        finally:
            combined.unlink(missing_ok=True)
    return {"documents": sum(x[0] for x in counts), "tokens": sum(x[1] for x in counts),
            "dtype": "uint16", "format": "raw little-endian .bin"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.workers < 1:
        raise ValueError("workers must be positive")
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        summary[split] = pack_split(a.input_dir / f"{split}.jsonl", a.tokenizer,
                                    a.out / f"{split}.bin", a.workers)
    (a.out / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
