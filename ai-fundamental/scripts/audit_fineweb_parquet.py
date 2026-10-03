"""Summarize source metadata for the exact FineWeb-Edu shards in use."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def audit(paths: list[Path]) -> dict:
    languages = Counter()
    token_counts = []
    language_scores = []
    quality_scores = []
    shards = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        needed = [name for name in ("language", "language_score", "token_count", "score") if name in columns]
        if not needed:
            raise ValueError(f"no audit columns in {path}")
        shards.append({"path": str(path), "rows": parquet.metadata.num_rows})
        for batch in parquet.iter_batches(batch_size=8192, columns=needed):
            data = batch.to_pydict()
            languages.update(str(x) for x in data.get("language", []))
            token_counts.extend(int(x) for x in data.get("token_count", []) if x is not None)
            language_scores.extend(float(x) for x in data.get("language_score", []) if x is not None)
            quality_scores.extend(float(x) for x in data.get("score", []) if x is not None)

    def summary(values):
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {"count": len(values), "mean": float(array.mean()),
                "p01": float(np.quantile(array, 0.01)), "p50": float(np.median(array)),
                "p99": float(np.quantile(array, 0.99))}

    result = {"shards": shards, "total_rows": sum(s["rows"] for s in shards),
              "languages": dict(languages),
              "source_reported_token_count_sum": sum(token_counts),
              "token_count": summary(token_counts),
              "language_score": summary(language_scores),
              "quality_score": summary(quality_scores)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
