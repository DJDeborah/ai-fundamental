"""Stream an official FineWeb-Edu Parquet shard into the project's raw JSONL format.

This conversion preserves source IDs and records the input file hash. The output is
still raw for this project: fundamental.data performs our split and deduplication.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, out: Path, *, source_url: str, expected_sha256: str | None, max_docs: int | None) -> dict:
    if max_docs is not None and max_docs < 1:
        raise ValueError("max_docs must be positive")
    actual_sha256 = sha256_file(source)
    if expected_sha256 and actual_sha256 != expected_sha256.lower():
        raise ValueError(f"Parquet SHA-256 mismatch: {actual_sha256}")
    parquet = pq.ParquetFile(source)
    columns = set(parquet.schema_arrow.names)
    if "text" not in columns or not ({"url", "id"} & columns):
        raise ValueError(f"expected text and url/id columns; found {sorted(columns)}")
    selected = [name for name in ("text", "url", "id") if name in columns]
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    total_rows = saved = empty = missing_source = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for batch in parquet.iter_batches(batch_size=1024, columns=selected):
                table = batch.to_pydict()
                for position in range(batch.num_rows):
                    row_index = total_rows
                    total_rows += 1
                    body = table["text"][position]
                    if not isinstance(body, str) or not body.strip():
                        empty += 1
                        continue
                    url = table.get("url", [None] * batch.num_rows)[position]
                    source_id = table.get("id", [None] * batch.num_rows)[position]
                    source_name = url or source_id
                    if not source_name:
                        missing_source += 1
                        continue
                    record = {
                        "text": body,
                        "source": str(source_name),
                        "license": "odc-by",
                        "source_id": source_id,
                        "dataset": "HuggingFaceFW/fineweb-edu",
                        "config": "sample-10BT",
                        "parquet_file": source.name,
                        "row_index_in_shard": row_index,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    saved += 1
                    if max_docs is not None and saved >= max_docs:
                        break
                if max_docs is not None and saved >= max_docs:
                    break
        temporary.replace(out)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "license_label": "odc-by",
        "source_url": source_url,
        "input": str(source),
        "input_bytes": source.stat().st_size,
        "input_sha256": actual_sha256,
        "parquet_rows": parquet.metadata.num_rows,
        "selection": "sequential rows of this Parquet shard" if max_docs else "all rows of this Parquet shard",
        "max_docs": max_docs,
        "rows_read": total_rows,
        "saved_documents": saved,
        "empty_text_rows": empty,
        "missing_source_rows": missing_source,
        "output": str(out),
        "output_bytes": out.stat().st_size,
        "output_sha256": sha256_file(out),
        "converted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--max-docs", type=int)
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.out, source_url=args.source_url,
                             expected_sha256=args.expected_sha256,
                             max_docs=args.max_docs), indent=2))


if __name__ == "__main__":
    main()
