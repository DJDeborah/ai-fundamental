"""Print compact, verifiable result lines for the final report tutorial.

This script reads saved artifacts and runs one CPU-only deterministic environment
example. It does not start training, access the cloud, or use a GPU.
"""

import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundamental.env import ArithmeticChainEnv
from fundamental.tokenizer import ByteBPE


CLOUD = ROOT / "report" / "data" / "cloud"


def read_json(name):
    return json.loads((CLOUD / name).read_text(encoding="utf-8"))


def read_jsonl(name):
    return [json.loads(line) for line in (CLOUD / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False))


def main():
    pilot = json.loads((ROOT / "clean_fineweb_pilot" / "manifest.json").read_text(encoding="utf-8"))
    clean = read_json("fineweb_clean_manifest.json")
    packed = read_json("full_token_manifest.json")
    emit("clean_pilot", **pilot["counts"])
    emit("clean_full", **clean["counts"])
    emit("pack_full", train_tokens=packed["train"]["tokens"],
         val_tokens=packed["val"]["tokens"], test_tokens=packed["test"]["tokens"])

    pilot_tok = ByteBPE.load(ROOT / "tokenizer_fineweb_pilot_2048.json")
    sample = "Hello, world!"
    ids = pilot_tok.encode(sample, eos=True)
    emit("bpe_pilot", vocab_size=pilot_tok.size, ids=ids, roundtrip=pilot_tok.decode(ids))
    tok_path = CLOUD / "tokenizer_8192.json"
    formal_tok = ByteBPE.load(tok_path)
    emit("bpe_formal", vocab_size=formal_tok.size,
         sha256=hashlib.sha256(tok_path.read_bytes()).hexdigest())

    grid = read_json("scaling_grid_v2_manifest_final.json")
    fit = read_json("scaling_fit.json")
    emit("model", params=fit["provenance"]["model_sizes"],
         world_sizes=sorted({row["world_size"] for config in grid["recipe"]["configs"]
                             for seed in grid["recipe"]["seeds"]
                             for row in read_jsonl(f"scaling_{config}_seed{seed}_metrics.jsonl")}))
    for row in read_jsonl("scaling_base100m_seed42_metrics.jsonl"):
        emit("train_base100m_seed42", step=row["step"], tokens_seen=row["tokens_seen"],
             val_loss=round(row["val_loss"], 6),
             train_tokens_per_s=round(row["train_tokens_per_s"]))
    for row in read_jsonl("base100m_2b_metrics_partial.jsonl"):
        emit("continuation_measured", step=row["step"], tokens_seen=row["tokens_seen"],
             val_loss=round(row["val_loss"], 6))

    held = fit["holdout"]
    emit("scaling_holdout", train_cells=fit["train_points"],
         predicted=round(held["predicted_loss"], 6), actual=round(held["actual_loss"], 6),
         absolute_error=round(held["absolute_error"], 6))
    emit("scaling_extrapolation_unverified", target_tokens=fit["extrapolation"]["tokens"],
         predicted=round(fit["extrapolation"]["predicted_loss"], 6))

    attention = read_jsonl("attention_event_ab.jsonl")
    for length in sorted({row["seq_len"] for row in attention}):
        rows = [row for row in attention if row["seq_len"] == length]
        emit("attention_forward", seq_len=length, pairs=len(rows),
             speedup_median=round(statistics.median(row["speedup"] for row in rows), 6),
             max_abs_error=max(row["max_abs_error"] for row in rows))

    for row in read_json("posttrain_probe_eval.json"):
        emit("posttrain_short_probe", branch=row["stage"], correct=row["correct"], n=row["n"])

    env = ArithmeticChainEnv(seed=0, horizon=4)
    obs, action = env.observation(), env.oracle_action()
    result = env.step(action)
    emit("agent_oracle_cpu_example", observation=obs, action=action, **result)
    emit("two_gpu_comparison", measured=False, reason="all saved grid logs have world_size=1")


if __name__ == "__main__":
    main()
