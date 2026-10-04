"""Take a deterministic min-hash document sample from the clean train split.

Keeping the smallest hashes makes the sample independent of input row order and
ensures tokenizer fitting never reads validation or test documents.
"""

import argparse
import hashlib
import heapq
import json
from pathlib import Path


def sample(input_path: Path, out: Path, target_bytes: int) -> dict:
    if target_bytes < 1:
        raise ValueError("target_bytes must be positive")
    heap = []
    total_bytes = seen = 0
    source_hash = hashlib.sha256()
    with input_path.open("rb") as handle:
        for raw_line in handle:
            source_hash.update(raw_line)
            row = json.loads(raw_line)
            text = row["text"]
            key = int(row["sha256"], 16)
            size = len(text.encode("utf-8"))
            seen += 1
            heapq.heappush(heap, (-key, seen, row, size))
            total_bytes += size
            while total_bytes > target_bytes:
                _, _, _, removed_size = heapq.heappop(heap)
                total_bytes -= removed_size
    chosen = sorted(heap, key=lambda item: -item[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for _, _, row, _ in chosen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "input": str(input_path),
        "input_sha256": source_hash.hexdigest(),
        "documents_seen": seen,
        "selection": "lowest SHA-256 normalized-text digests within UTF-8 text-byte budget",
        "target_text_bytes": target_bytes,
        "selected_documents": len(chosen),
        "selected_text_bytes": total_bytes,
        "output": str(out),
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-bytes", type=int, default=20_000_000)
    args = parser.parse_args()
    print(json.dumps(sample(args.input, args.out, args.target_bytes), indent=2))


if __name__ == "__main__":
    main()
