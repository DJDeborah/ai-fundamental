"""Stream raw JSONL/text to deduplicated, document-disjoint splits."""
import argparse
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def normalize(text: str) -> str:
    text = html.unescape(text).replace("\x00", "")
    text = re.sub(r"<[^>]{1,500}>", " ", text)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def simhash(text: str) -> int:
    words = re.findall(r"\w+", text.lower())
    shingles = [" ".join(words[i:i+5]) for i in range(max(1, len(words)-4))]
    digests = b"".join(hashlib.blake2b(gram.encode(), digest_size=8).digest() for gram in set(shingles))
    bits = np.unpackbits(np.frombuffer(digests, dtype=np.uint8)).reshape(-1, 64)
    majority = bits.sum(axis=0) * 2 >= bits.shape[0]
    return int.from_bytes(np.packbits(majority).tobytes(), "big")


def split_for(digest: str) -> str:
    n = int(digest[:8], 16) % 1000
    return "train" if n < 980 else "val" if n < 990 else "test"


def iter_raw(path: Path, text_license: str | None):
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix in {".jsonl", ".txt"})
    for file in files:
        if file.suffix == ".txt":
            if not text_license:
                raise ValueError(".txt requires --text-license")
            yield {"text": file.read_text(encoding="utf-8"), "source": str(file), "license": text_license}
        elif file.suffix == ".jsonl":
            with file.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{file}:{line_no}: {e}") from e
                    yield obj


def clean(input_path: Path, out: Path, licenses: set[str], text_license: str | None, min_chars: int):
    if not input_path.exists():
        raise FileNotFoundError(f"input corpus does not exist: {input_path}")
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    exact = set()
    # Four 16-bit buckets give candidate retrieval; exact Hamming check avoids false positives.
    buckets = defaultdict(set)
    handles = {s: (out / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "val", "test")}
    try:
        for row in iter_raw(input_path, text_license):
            counts["raw"] += 1
            if not all(isinstance(row.get(k), str) for k in ("text", "source", "license")):
                counts["missing_metadata"] += 1
                continue
            if row["license"] not in licenses:
                counts["license_excluded"] += 1
                continue
            body = normalize(row["text"])
            if len(body) < min_chars:
                counts["short"] += 1
                continue
            digest = hashlib.sha256(body.encode()).hexdigest()
            if digest in exact:
                counts["exact_duplicate"] += 1
                continue
            signature = simhash(body)
            candidates = set().union(*(buckets[(i, (signature >> (i*16)) & 65535)] for i in range(4)))
            if any((signature ^ old).bit_count() <= 3 for old in candidates):
                counts["near_duplicate"] += 1
                continue
            exact.add(digest)
            for i in range(4):
                buckets[(i, (signature >> (i*16)) & 65535)].add(signature)
            split = split_for(digest)
            handles[split].write(json.dumps({"text": body, "source": row["source"], "license": row["license"], "sha256": digest}, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for f in handles.values():
            f.close()
    manifest = {"counts": dict(counts), "allowed_licenses": sorted(licenses), "split": "sha256(normalized text) mod 1000: 980/10/10", "min_chars": min_chars}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--licenses", nargs="+", required=True)
    p.add_argument("--text-license")
    p.add_argument("--min-chars", type=int, default=20)
    a = p.parse_args()
    print(json.dumps(clean(a.input, a.out, set(a.licenses), a.text_license, a.min_chars), ensure_ascii=False))


if __name__ == "__main__":
    main()
