"""Plot held-out scaling residuals from saved cloud observations.

Usage: python scripts/plot_final_analysis.py
The chart deliberately separates the three-seed grid from the one-seed
continuation and marks predictions as predictions.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "report" / "data" / "cloud"
FIG = ROOT / "report" / "figures"


def predict(n, d, fit):
    return (fit["E"] + fit["A"] * (n / 1e6) ** -fit["alpha"]
            + fit["B"] * (d / 1e6) ** -fit["beta"])


def main():
    fit = json.loads((DATA / "scaling_fit.json").read_text(encoding="utf-8"))
    held = fit["holdout"]
    groups = sorted({int(r["params"]) for r in fit["observations"]})
    colors = ["#326b9e", "#23806b", "#b04f39"]
    fig, ax = plt.subplots(figsize=(7.2, 4.1), layout="constrained")
    for n, color in zip(groups, colors):
        rows = sorted((r for r in fit["observations"] if int(r["params"]) == n),
                      key=lambda r: r["tokens"])
        for row in rows:
            x = row["tokens"] / 1e6
            residual = predict(n, row["tokens"], fit) - row["mean_val_loss"]
            is_held = n == held["params"] and row["tokens"] == held["tokens"]
            ax.errorbar(x, residual, yerr=row["sd_val_loss"], fmt="o" if not is_held else "D",
                        mfc="white" if is_held else color, mec=color, color=color,
                        capsize=3, markersize=8 if is_held else 6, zorder=3)
        ax.plot([], [], "o", color=color, label=f"{n/1e6:.1f}M parameters")
    later = [json.loads(line) for line in (DATA / "base100m_2b_metrics_partial.jsonl")
             .read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in later:
        residual = predict(row["params"], row["tokens_seen"], fit) - row["val_loss"]
        ax.scatter(row["tokens_seen"] / 1e6, residual, marker="*", color="#7d427d",
                   s=150, zorder=4, label="500M: one seed, external check")
    ax.axhline(0, color="#333333", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([20, 80, 300, 500], ["20", "80", "300", "500"])
    ax.set_xlabel("Training tokens (millions, log scale)")
    ax.set_ylabel("Prediction minus measured loss (nats/token)")
    ax.set_title("Scaling residuals expose optimistic uncertainty")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.annotate("held-out 3-seed mean", xy=(300, .031184), xytext=(100, .027),
                fontsize=8, arrowprops={"arrowstyle": "->", "color": "#444444"})
    ax.text(.01, .02, "Error bars: across-seed SD (grid); star has no seed SD",
            transform=ax.transAxes, fontsize=8, color="#444444")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "scaling_residuals.png", dpi=300)
    fig.savefig(FIG / "scaling_residuals.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
