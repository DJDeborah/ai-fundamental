"""Handwritten byte-level BPE. IDs 0..255 are bytes, 256 is EOS, 257 is PAD."""
import argparse
import heapq
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np


EOS, PAD = 256, 257
PIECES = re.compile(r"\S+|\s+", re.UNICODE)


def text_pieces(text: str):
    for m in PIECES.finditer(text):
        yield tuple(m.group().encode("utf-8"))


def iter_texts(file: Path):
    with file.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


class ByteBPE:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = merges or []
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = [bytes([i]) for i in range(256)] + [b"", b""]
        for left, right in self.merges:
            self.vocab.append(self.vocab[left] + self.vocab[right])

    @property
    def size(self):
        return len(self.vocab)

    def encode(self, text: str, eos: bool = False) -> list[int]:
        output = []
        for piece in text_pieces(text):
            output.extend(self._encode_piece(piece))
        if eos:
            output.append(EOS)
        return output

    @lru_cache(maxsize=200_000)
    def _encode_piece(self, piece: tuple[int, ...]) -> tuple[int, ...]:
        ids = list(piece)
        while len(ids) > 1:
            choices = ((self.ranks[p], j) for j, p in enumerate(zip(ids, ids[1:])) if p in self.ranks)
            best = min(choices, default=None)
            if best is None:
                break
            _, j = best
            ids[j:j+2] = [258 + self.ranks[(ids[j], ids[j+1])]]
        return tuple(ids)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids if i not in (EOS, PAD)).decode("utf-8", errors="replace")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": "byte-bpe-v1", "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj["format"] != "byte-bpe-v1":
            raise ValueError("unsupported tokenizer format")
        return cls([tuple(pair) for pair in obj["merges"]])


def train_bpe(texts, vocab_size: int, max_bytes: int | None = None) -> ByteBPE:
    if not 258 <= vocab_size <= 65535:
        raise ValueError("vocab size must be in [258, 65535]")
    words = Counter()
    used = 0
    for text in texts:
        for piece in text_pieces(text):
            used += len(piece)
            if max_bytes is not None and used > max_bytes:
                break
            words[piece] += 1
        if max_bytes is not None and used > max_bytes:
            break
    seqs = [list(w) for w in words]
    weights = list(words.values())
    where = defaultdict(set)
    counts = Counter()
    for i, seq in enumerate(seqs):
        for pair, n in Counter(zip(seq, seq[1:])).items():
            counts[pair] += n * weights[i]
            where[pair].add(i)
    heap = [(-n, pair) for pair, n in counts.items()]
    heapq.heapify(heap)
    merges = []
    while len(merges) + 258 < vocab_size and heap:
        neg, pair = heapq.heappop(heap)
        if -neg != counts[pair] or counts[pair] <= 0:
            continue
        new_id = 258 + len(merges)
        affected = list(where[pair])
        changed = set()
        for i in affected:
            old = seqs[i]
            old_pairs = Counter(zip(old, old[1:]))
            for p, n in old_pairs.items():
                counts[p] -= n * weights[i]
                where[p].discard(i)
                changed.add(p)
                heapq.heappush(heap, (-counts[p], p))
            new = []
            j = 0
            while j < len(old):
                if j + 1 < len(old) and (old[j], old[j+1]) == pair:
                    new.append(new_id)
                    j += 2
                else:
                    new.append(old[j])
                    j += 1
            seqs[i] = new
            for p, n in Counter(zip(new, new[1:])).items():
                counts[p] += n * weights[i]
                where[p].add(i)
                changed.add(p)
                heapq.heappush(heap, (-counts[p], p))
        for p in changed:
            if counts[p] <= 0:
                counts.pop(p, None)
                where.pop(p, None)
        if len(heap) > max(10000, 4 * len(counts)):
            heap = [(-n, p) for p, n in counts.items()]
            heapq.heapify(heap)
        merges.append(pair)
    return ByteBPE(merges)


def pack(input_dir: Path, out: Path, tok: ByteBPE):
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        src = input_dir / f"{split}.jsonl"
        documents = tokens = 0
        with (out / f"{split}.bin").open("wb") as handle:
            if src.exists():
                for text in iter_texts(src):
                    ids = tok.encode(text, eos=True)
                    np.asarray(ids, dtype="<u2").tofile(handle)
                    documents += 1
                    tokens += len(ids)
        summary[split] = {"documents": documents, "tokens": tokens, "dtype": "uint16", "format": "raw little-endian .bin"}
    (out / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("train")
    a.add_argument("--input", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--vocab-size", type=int, default=8192)
    a.add_argument("--max-bytes", type=int, default=20_000_000)
    b = sub.add_parser("pack")
    b.add_argument("--tokenizer", type=Path, required=True)
    b.add_argument("--input-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("encode")
    c.add_argument("--tokenizer", type=Path, required=True)
    c.add_argument("text")
    args = p.parse_args()
    if args.command == "train":
        tok = train_bpe(iter_texts(args.input), args.vocab_size, args.max_bytes)
        tok.save(args.out)
        print(json.dumps({"vocab_size": tok.size, "out": str(args.out)}))
    elif args.command == "pack":
        print(json.dumps(pack(args.input_dir, args.out, ByteBPE.load(args.tokenizer))))
    else:
        tok = ByteBPE.load(args.tokenizer)
        ids = tok.encode(args.text, eos=True)
        print(json.dumps({"ids": ids, "roundtrip": tok.decode(ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
