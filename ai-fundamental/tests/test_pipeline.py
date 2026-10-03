import json

import numpy as np
import pytest

from fundamental.data import clean
from fundamental.compare import compare
from fundamental.env import ArithmeticChainEnv
from fundamental.posttrain import make_data
from fundamental.scaling import fit
from fundamental.tokenizer import ByteBPE, EOS, pack, train_bpe


def test_bpe_unicode_roundtrip(tmp_path):
    tok = train_bpe(["hello 世界\n", "hello! 世界"], 280)
    for text in ("hello 世界\n", "new 🐈 bytes", "  spaces\tand\nlines"):
        assert tok.decode(tok.encode(text, eos=True)) == text
    path = tmp_path / "tok.json"
    tok.save(path)
    assert ByteBPE.load(path).encode("hello 世界") == tok.encode("hello 世界")


def test_clean_and_pack(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [
        {"text": "A unique paragraph about tokenizer construction and tests 123.", "source": "a", "license": "CC0"},
        {"text": "A unique paragraph about tokenizer construction and tests 123.", "source": "b", "license": "CC0"},
        {"text": "A different paragraph about attention kernels and tests 456.", "source": "c", "license": "CC0"},
        {"text": "Excluded material with unspecified reuse license.", "source": "d", "license": "X"},
    ]
    raw.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    out = tmp_path / "clean"
    result = clean(raw, out, {"CC0"}, None, 20)
    assert result["counts"]["exact_duplicate"] == 1
    assert result["counts"]["license_excluded"] == 1
    train = out / "train.jsonl"
    # pack itself handles empty splits too.
    tok = train_bpe([r["text"] for r in rows], 280)
    summary = pack(out, tmp_path / "tokens", tok)
    assert sum(x["documents"] for x in summary.values()) >= 1
    ids = np.memmap(tmp_path / "tokens" / "train.bin", dtype=np.uint16, mode="r")
    assert ids.dtype == np.uint16
    assert all(i < tok.size or i == EOS for i in ids)


def test_clean_rejects_missing_input(tmp_path):
    out = tmp_path / "clean"
    with pytest.raises(FileNotFoundError, match="input corpus does not exist"):
        clean(tmp_path / "missing.jsonl", out, {"CC0"}, None, 20)
    assert not out.exists()


def test_oracle_completes_long_horizon():
    env = ArithmeticChainEnv(99, 16)
    for _ in range(16):
        assert env.step(env.oracle_action())["valid"]
    assert env.success
    bad = ArithmeticChainEnv(99, 16)
    assert not bad.step("ADD 999")["valid"]
    assert bad.done and not bad.success


def test_scaling_fit_with_heldout():
    rows = []
    for n in (1e6, 10e6, 100e6):
        for d in (1e6, 10e6, 100e6):
            rows.append({"params": int(n), "tokens_seen": int(d),
                         "val_loss": 1.1 + 0.4 * (n/1e6)**-0.5 + 0.3 * (d/1e6)**-0.5})
    result = fit(rows)
    assert result["holdout"]["absolute_error"] < 0.01
    with pytest.raises(ValueError):
        fit(rows[:3])


def test_scaling_keeps_replicates_and_bootstraps():
    rows = []
    for n in (1e6, 10e6, 100e6):
        for d in (1e6, 10e6, 100e6):
            center = 1.1 + 0.4 * (n/1e6)**-0.5 + 0.3 * (d/1e6)**-0.5
            for seed, offset in ((11, -0.01), (12, 0.0), (13, 0.01)):
                rows.append({"params": int(n), "tokens_seen": int(d),
                             "seed": seed, "val_loss": center + offset})
    result = fit(rows, bootstrap_samples=20, target_params=200_000_000,
                 target_tokens=200_000_000)
    assert all(point["replicates"] == 3 for point in result["observations"])
    assert result["holdout"]["absolute_error"] < 0.01
    assert len(result["bootstrap"]["holdout_prediction_95pct"]) == 2
    assert result["bootstrap"]["scheme"] == "paired_seed_trajectories_by_model"
    with pytest.raises(ValueError, match="duplicate seed"):
        fit(rows + [rows[0]])


def test_research_posttrain_split_is_disjoint(tmp_path):
    make_data(tmp_path, "research")
    def prompts(name):
        return {json.loads(line)["prompt"] for line in (tmp_path / f"{name}.jsonl").read_text().splitlines()}
    train, heldout, ood = prompts("sft"), prompts("eval"), prompts("eval_ood")
    assert len(train) + len(heldout) == 1000
    assert len(ood) == 200
    assert not (train & heldout or train & ood or heldout & ood)
    assert train == prompts("dpo") == prompts("rlvr")


def test_compare_rejects_unmatched_conditions():
    single = {"world_size": 1, "kind": "replicated_inference_full_recompute",
              "checkpoint_sha256": "same", "prompt_sha256": "same", "prompt_tokens": 8,
              "new_tokens": 16, "requests_per_gpu": 1, "aggregate_generated_tokens_per_s": 10}
    multi = {**single, "world_size": 4, "aggregate_generated_tokens_per_s": 30}
    result = compare(single, multi, "inference")
    assert result["speedup"] == 3
    assert result["parallel_efficiency"] == 0.75
    with pytest.raises(ValueError):
        compare(single, {**multi, "checkpoint_sha256": "different"}, "inference")
    fixed_load = compare({**single, "requests_per_gpu": 4}, multi,
                         "inference", load_mode="fixed-total")
    assert fixed_load["load_mode"] == "fixed-total"
    with pytest.raises(ValueError):
        compare(single, multi, "inference", load_mode="fixed-total")
