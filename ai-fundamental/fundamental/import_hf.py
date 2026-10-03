"""Optional streaming adapter: Hugging Face dataset rows -> raw JSONL.

The caller must check and provide the dataset's actual license string.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--license", required=True)
    p.add_argument("--max-docs", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the optional dependency: python -m pip install datasets") from exc
    stream = load_dataset(a.dataset, name=a.config, split=a.split, streaming=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    with a.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(stream):
            if saved >= a.max_docs:
                break
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            source = row.get("url") or row.get("id") or f"{a.dataset}/{a.config}/{a.split}#{i}"
            f.write(json.dumps({"text": text, "source": str(source), "license": a.license}, ensure_ascii=False) + "\n")
            saved += 1
    print(json.dumps({"saved_documents": saved, "out": str(a.out)}))


if __name__ == "__main__":
    main()
