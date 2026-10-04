"""Fit L(N,D) = E + A (N/1e6)^-alpha + B (D/1e6)^-beta.

Small educational grid search, not a claim of universal scaling behavior.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _fit_means(points, holdout):
    keys = [nd for nd in points if nd != holdout]
    n = np.array([nd[0] / 1e6 for nd in keys])
    d = np.array([nd[1] / 1e6 for nd in keys])
    y = np.array([points[nd] for nd in keys])
    best = None
    for alpha in np.linspace(0.05, 1.2, 47):
        for beta in np.linspace(0.05, 1.2, 47):
            x = np.column_stack((np.ones_like(n), n**-alpha, d**-beta))
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            if np.any(coef < 0):
                continue
            rmse = float(np.sqrt(np.mean((x @ coef - y)**2)))
            if best is None or rmse < best[0]:
                best = (rmse, float(alpha), float(beta), coef)
    if best is None:
        raise ValueError("no nonnegative fit; inspect data and model family")
    rmse, alpha, beta, (e, a, b) = best
    return {"E": float(e), "A": float(a), "B": float(b), "alpha": alpha,
            "beta": beta, "fit_rmse": rmse, "train_points": len(keys)}


def fit(rows, *, bootstrap_samples=0, target_params=None, target_tokens=None, seed=42):
    grouped = defaultdict(dict)
    for index, row in enumerate(rows):
        key = (int(row["params"]), int(row["tokens_seen"]))
        run_seed = row.get("seed", f"row-{index}")
        if run_seed in grouped[key]:
            raise ValueError(f"duplicate seed {run_seed} at N,D={key}")
        grouped[key][run_seed] = float(row["val_loss"])
    points = {key: float(np.mean(list(seed_losses.values()))) for key, seed_losses in grouped.items()}
    if len({n for n, _ in points}) < 3 or len({d for _, d in points}) < 3:
        raise ValueError("need at least three model sizes and three token budgets")
    if len(points) < 8:
        raise ValueError("need at least eight distinct N,D observations")
    holdout = max(points, key=lambda nd: nd[0] * nd[1])
    fitted = _fit_means(points, holdout)
    hn, hd = holdout
    pred = predict(hn, hd, fitted)
    fitted["holdout"] = {"params": hn, "tokens": hd, "actual_loss": points[holdout],
                          "predicted_loss": pred, "absolute_error": abs(pred - points[holdout])}
    fitted["observations"] = [
        {"params": n, "tokens": d, "seeds": sorted(str(s) for s in grouped[n, d]),
         "replicates": len(grouped[n, d]), "mean_val_loss": points[n, d],
         "sd_val_loss": float(np.std(list(grouped[n, d].values()), ddof=1))
         if len(grouped[n, d]) > 1 else None}
        for n, d in sorted(points)
    ]
    if target_params is not None and target_tokens is not None:
        fitted["extrapolation"] = {
            "params": target_params, "tokens": target_tokens,
            "predicted_loss": predict(target_params, target_tokens, fitted),
            "caution": "prediction uncertainty grows outside measured N,D range",
        }
    if bootstrap_samples:
        if bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be nonnegative")
        rng = np.random.default_rng(seed)
        holdout_predictions = []
        target_predictions = []
        failed = 0
        model_groups = defaultdict(list)
        for key in grouped:
            model_groups[key[0]].append(key)
        paired = all(len({frozenset(grouped[key]) for key in keys}) == 1
                     for keys in model_groups.values())
        for _ in range(bootstrap_samples):
            if paired:
                sampled = {}
                for keys in model_groups.values():
                    seeds = list(grouped[keys[0]])
                    selected = rng.choice(seeds, size=len(seeds), replace=True)
                    for key in keys:
                        sampled[key] = float(np.mean([grouped[key][seed] for seed in selected]))
            else:
                sampled = {key: float(rng.choice(list(values.values()), len(values), replace=True).mean())
                           for key, values in grouped.items()}
            try:
                trial = _fit_means(sampled, holdout)
            except ValueError:
                failed += 1
                continue
            holdout_predictions.append(predict(hn, hd, trial))
            if target_params is not None and target_tokens is not None:
                target_predictions.append(predict(target_params, target_tokens, trial))
        if len(holdout_predictions) < max(20, bootstrap_samples // 2):
            raise ValueError("too many bootstrap fits failed; scaling surface is unstable")
        fitted["bootstrap"] = {
            "requested": bootstrap_samples, "successful": len(holdout_predictions), "failed": failed,
            "scheme": "paired_seed_trajectories_by_model" if paired else "independent_cell_replicates",
            "holdout_prediction_95pct": np.quantile(holdout_predictions, [0.025, 0.975]).tolist(),
        }
        if target_predictions:
            fitted["bootstrap"]["target_prediction_95pct"] = np.quantile(
                target_predictions, [0.025, 0.975]).tolist()
    return fitted


def predict(params, tokens, fitted):
    return fitted["E"] + fitted["A"] * (params/1e6)**-fitted["alpha"] + fitted["B"] * (tokens/1e6)**-fitted["beta"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", type=Path, required=True, help="training metrics JSONL files")
    p.add_argument("--target-params", type=int)
    p.add_argument("--target-tokens", type=int)
    p.add_argument("--bootstrap-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = []
    for file in args.input:
        rows.extend(json.loads(line) for line in file.read_text().splitlines() if line.strip())
    result = fit(rows, bootstrap_samples=args.bootstrap_samples,
                 target_params=args.target_params, target_tokens=args.target_tokens,
                 seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
