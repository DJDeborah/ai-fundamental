"""Download a small FineWeb-Edu pilot through the official Dataset Viewer API.

This is a convenience fallback when the training instance cannot reach the Hub.
It exports the first N rows of sample-10BT; it is not a random evaluation sample.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
SPLIT = "train"
LICENSE = "odc-by"
API = "https://datasets-server.huggingface.co/rows"
HUB_API = "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu"


def get_json(url: str, attempts: int = 4) -> dict:
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "ai-fundamental-research-pilot/1.0"})
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"request failed after {attempts} attempts: {url}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def download(out: Path, count: int) -> dict:
    if count < 1:
        raise ValueError("count must be positive")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    downloaded = 0
    total_rows = None
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            while downloaded < count:
                length = min(100, count - downloaded)
                url = API + "?" + urlencode({"dataset": DATASET, "config": CONFIG,
                                              "split": SPLIT, "offset": downloaded,
                                              "length": length})
                payload = get_json(url)
                rows = payload.get("rows")
                if payload.get("partial") or not isinstance(rows, list) or len(rows) != length:
                    raise ValueError(f"incomplete page at offset {downloaded}: {url}")
                total_rows = payload.get("num_rows_total")
                for index, item in enumerate(rows, downloaded):
                    if item.get("row_idx") != index or item.get("truncated_cells"):
                        raise ValueError(f"out-of-order or truncated row {index}")
                    row = item["row"]
                    body = row.get("text")
                    if not isinstance(body, str) or not body.strip():
                        raise ValueError(f"empty text in row {index}")
                    source = row.get("url") or row.get("id")
                    if not source:
                        raise ValueError(f"missing URL and ID in row {index}")
                    record = {"text": body, "source": str(source), "license": LICENSE,
                              "source_id": row.get("id"), "row_index": index,
                              "dataset": DATASET, "config": CONFIG}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                downloaded += length
                print(f"downloaded {downloaded}/{count}", flush=True)
        if downloaded != count:
            raise AssertionError("download count mismatch")
        temporary.replace(out)
    finally:
        if temporary.exists():
            temporary.unlink()

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    hub = get_json(HUB_API)
    manifest = {"dataset": DATASET, "config": CONFIG, "split": SPLIT,
                "selection": f"first {count} rows by Dataset Viewer row index",
                "rows": count, "dataset_num_rows_total": total_rows,
                "license_label": LICENSE,
                "dataset_card": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
                "dataset_hub_sha_at_download": hub.get("sha"),
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "api": API, "file": out.name, "bytes": out.stat().st_size,
                "sha256": digest, "truncated_rows": 0}
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("raw/fineweb_edu_pilot.jsonl"))
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(download(args.out, args.count), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
