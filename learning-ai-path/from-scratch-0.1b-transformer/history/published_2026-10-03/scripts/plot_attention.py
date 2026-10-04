"""Plot raw alternating CUDA-event forward timings without smoothing away runs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="PNG path; a PDF is written beside it")
    a = p.parse_args()
    groups = defaultdict(list)
    for line in a.input.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups[int(row["seq_len"])].append(float(row["speedup"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
    lengths = sorted(groups)
    for x in lengths:
        ys = groups[x]
        offsets = np.linspace(-0.055, 0.055, len(ys))
        ax.scatter(np.full(len(ys), np.log2(x)) + offsets, ys, color="#4176A7",
                   alpha=0.55, s=25, linewidths=0, zorder=2)
        ax.scatter(np.log2(x), np.median(ys), color="#B84B32", marker="D", s=58,
                   edgecolors="white", linewidths=0.6, zorder=3)
    ax.axhline(1, color="#666666", linestyle="--", linewidth=1)
    ax.set_xticks([np.log2(x) for x in lengths], [str(x) for x in lengths])
    ax.set_xlabel("Sequence length T")
    ax.set_ylabel("SDPA time / Triton time")
    ax.set_title("RTX 4090 causal attention forward (CUDA events)")
    ax.set_ylim(0.75, 1.36)
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.99, 0.02, "Each dot: 300 calls; diamond: median of 10 paired runs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#444444")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.with_suffix(".pdf"))
    plt.close(fig)
    print(json.dumps({"png": str(a.out), "pdf": str(a.out.with_suffix('.pdf'))}))


if __name__ == "__main__":
    main()
