"""Parse a Common Crawl WET archive into provenance-rich raw JSONL.

WET contains plaintext already extracted by Common Crawl; it is closer to a
crawl dump than FineWeb-Edu Parquet, but it is not the original HTML WARC.
"""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def skip_bytes(handle, count: int):
    while count:
        chunk = handle.read(min(count, 1024 * 1024))
        if not chunk:
            raise ValueError("truncated WET record body")
        count -= len(chunk)


def records(path: Path, max_body_bytes: int):
    with gzip.open(path, "rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            if not line.startswith(b"WARC/"):
                raise ValueError(f"unexpected WET record marker: {line[:80]!r}")
            headers = {}
            while True:
                line = handle.readline()
                if not line:
                    raise ValueError("truncated WET headers")
                if not line.strip():
                    break
                name, separator, value = line.partition(b":")
                if not separator:
                    raise ValueError(f"malformed WET header: {line[:80]!r}")
                headers[name.decode("ascii").lower()] = value.strip().decode("utf-8", errors="replace")
            length = int(headers["content-length"])
            if length < 0:
                raise ValueError("negative WET Content-Length")
            if length > max_body_bytes:
                skip_bytes(handle, length)
                yield headers, None
            else:
                body = handle.read(length)
                if len(body) != length:
                    raise ValueError("truncated WET record body")
                yield headers, body


def convert(source: Path, out: Path, source_url: str, *, max_records: int,
            language: str, max_body_bytes: int) -> dict:
    if max_records < 1 or max_body_bytes < 1:
        raise ValueError("max_records and max_body_bytes must be positive")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    counts = Counter()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as target:
            for headers, body in records(source, max_body_bytes):
                if counts["conversion_records_read"] >= max_records:
                    break
                counts["warc_records_read"] += 1
                if headers.get("warc-type") != "conversion":
                    counts["non_conversion"] += 1
                    continue
                counts["conversion_records_read"] += 1
                if body is None:
                    counts["oversized"] += 1
                    continue
                detected = headers.get("warc-identified-content-language", "")
                if language and language not in detected.lower().replace(" ", "").split(","):
                    counts["language_excluded"] += 1
                    continue
                url = headers.get("warc-target-uri")
                if not url:
                    counts["missing_url"] += 1
                    continue
                text = body.decode("utf-8", errors="replace")
                counts["replacement_characters"] += text.count("\ufffd")
                if len(text.strip()) < 20:
                    counts["short"] += 1
                    continue
                target.write(json.dumps({
                    "text": text,
                    "source": url,
                    "license": "commoncrawl-access-terms",
                    "record_id": headers.get("warc-record-id"),
                    "warc_date": headers.get("warc-date"),
                    "language_header": detected,
                    "archive": source.name,
                }, ensure_ascii=False) + "\n")
                counts["saved"] += 1
        temporary.replace(out)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "source_url": source_url,
        "source_file": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "format": "Common Crawl WET, extracted plaintext in WARC records",
        "license_label": "commoncrawl-access-terms (does not license individual webpage content)",
        "selection": f"first {max_records} conversion records, then {language or 'all'} language filter",
        "max_body_bytes": max_body_bytes,
        "counts": dict(counts),
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
    parser.add_argument("--max-records", type=int, default=10_000)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--max-body-bytes", type=int, default=2_000_000)
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.out, args.source_url,
                             max_records=args.max_records, language=args.language,
                             max_body_bytes=args.max_body_bytes), indent=2))


if __name__ == "__main__":
    main()
