"""Draw measured scaling points, the held-out point, and labeled extrapolation."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fundamental.scaling import predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="PNG path; also writes PDF")
    p.add_argument("--continuation", type=Path, help="later measured JSONL for the held-out model")
    a = p.parse_args()
    result = json.loads(a.result.read_text(encoding="utf-8"))
    observations = result["observations"]
    params = sorted({int(x["params"]) for x in observations})
    colors = ["#4377A8", "#359A7F", "#B8553E"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    holdout = result["holdout"]
    for n, color in zip(params, colors):
        items = sorted((x for x in observations if x["params"] == n), key=lambda x: x["tokens"])
        xs = np.array([x["tokens"] for x in items]) / 1e6
        ys = np.array([x["mean_val_loss"] for x in items])
        errors = np.array([x["sd_val_loss"] or 0.0 for x in items])
        ax.errorbar(xs, ys, yerr=errors, fmt="o", color=color, capsize=3,
                    label=f"N={n/1e6:.1f}M, 3 seeds", zorder=3)
        grid = np.geomspace(xs.min(), xs.max(), 100)
        ax.plot(grid, [predict(n, x * 1e6, result) for x in grid],
                color=color, linewidth=1.3, alpha=0.85)
    ax.scatter(holdout["tokens"] / 1e6, holdout["actual_loss"], s=170,
               marker="o", facecolors="none", edgecolors="black", linewidths=1.5,
               label="held-out largest (N,D)", zorder=5)
    target = result.get("extrapolation")
    if target:
        base_n = target["params"]
        start = max(x["tokens"] for x in observations) / 1e6
        extension = np.geomspace(start, target["tokens"] / 1e6, 100)
        ax.plot(extension, [predict(base_n, x * 1e6, result) for x in extension],
                linestyle="--", linewidth=1.3, color=colors[params.index(base_n)])
        ax.scatter(target["tokens"] / 1e6, target["predicted_loss"],
                   marker="*", s=155, color="#A5395A", zorder=5,
                   label="2B-token extrapolation (unverified)")
    if a.continuation:
        later = [json.loads(line) for line in a.continuation.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if any(row["params"] != holdout["params"] or row["tokens_seen"] <= holdout["tokens"]
               for row in later):
            raise ValueError("continuation must extend the held-out model and token budget")
        ax.scatter([row["tokens_seen"] / 1e6 for row in later],
                   [row["val_loss"] for row in later], marker="D", s=50,
                   color="#232323", zorder=6, label="seed 42 continuation (measured)")
    ax.set_xscale("log")
    ax.set_xlabel("Training tokens D (millions, log scale)")
    ax.set_ylabel("Validation cross-entropy (nats/token)")
    ax.set_title("Local scaling fit with a held-out largest point")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.with_suffix(".pdf"))
    plt.close(fig)
    print(json.dumps({"png": str(a.out), "pdf": str(a.out.with_suffix('.pdf'))}))


if __name__ == "__main__":
    main()
