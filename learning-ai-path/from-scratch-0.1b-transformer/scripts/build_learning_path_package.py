"""Build the publishable 0.1B learning-path bundle from a fixed allowlist.

The raw FineWeb documents and cloud 0.1B checkpoints are deliberately absent:
they were never archived locally for redistribution. The saved measurements,
trained tokenizer, plots, report, source and a historical published snapshot
are included. No network or GPU operation is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LM_SCRIPTS = (
    "analyze_scaling_grid.py", "attention_ab.py", "audit_fineweb_parquet.py",
    "build_final_pdf.py", "build_learning_path_package.py", "cloud_after_pack.py",
    "download_fineweb_pilot.py", "download_fineweb_shards.py", "evaluate_lm_test.py",
    "finalize_parallel_pack.py", "jupyter_pull.py", "jupyter_remote.py",
    "make_autodl_bundle.py", "parallel_pack.py", "parquet_to_raw_jsonl.py",
    "plot_attention.py", "plot_final_analysis.py", "plot_scaling.py",
    "run_scaling_grid.py", "sample_tokenizer_train.py", "tutorial_results.py",
    "wet_to_raw_jsonl.py",
)
ROOT_FILES = (
    "AUTODL_START_HERE.md", "BEGINNER_TUTORIAL.md", "DATA_PILOT_README.md",
    "LESSONS.md", "requirements.txt", "requirements-data.txt",
    "requirements-report.txt", "tokenizer.json", "tokenizer_fineweb_pilot_2048.json",
    "tokenizer_fineweb_pilot_2048_repeat.json", "tokenizer_fineweb_pilot_8192.json",
    "raw/demo.jsonl", "raw/fineweb_edu_pilot.manifest.json",
    "runs/chat_toy.pt", "output/pdf/0.1B_transformer_final_report.pdf",
    "report/FINAL_REPORT.md", "report/neurips_style.md",
)
SECRET_PATTERNS = {
    "GitHub credential": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "API credential": r"\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}\b",
    "private key": r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----",
    "Jupyter credential": r"jupyter-autodl-[A-Za-z0-9-]{20,}",
    "credential URL": r"https?://[^\s/:]+:[^\s/@]{6,}@",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".sh"}
BLOCKED_PARTS = {".pytest_cache", "__pycache__", ".git", ".venv"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files() -> list[Path]:
    paths = {ROOT / name for name in ROOT_FILES}
    paths.update((ROOT / "fundamental").glob("*.py"))
    paths.update((ROOT / "tests").glob("*.py"))
    paths.update((ROOT / "tasks").glob("*.jsonl"))
    paths.update(p for p in (ROOT / "runs").glob("*")
                 if p.is_file() and p.suffix.lower() in {".json", ".jsonl"})
    paths.update(ROOT / "scripts" / name for name in LM_SCRIPTS)
    paths.update(p for p in (ROOT / "report/data").rglob("*") if p.is_file())
    paths.update(p for p in (ROOT / "report/figures").glob("*") if p.is_file())
    paths.update(p for p in (ROOT / "report/evidence").glob("*.png") if p.is_file())
    paths.update(ROOT / name for name in (
        "clean/manifest.json", "tokens/manifest.json",
        "clean_fineweb_pilot/manifest.json", "tokens_fineweb_pilot_2048/manifest.json",
        "tokens_fineweb_pilot_8192/manifest.json"))
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing bundle inputs: " + ", ".join(sorted(missing)))
    return sorted(paths)


def scan_text(path: Path, relative: str) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    content = path.read_text(encoding="utf-8", errors="replace")
    for label, pattern in SECRET_PATTERNS.items():
        if re.search(pattern, content, re.IGNORECASE):
            raise RuntimeError(f"Secret scan needs review: {relative}: {label}")


def write_package(out: Path, old: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for source in selected_files():
        rel = source.relative_to(ROOT)
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # This is the exact state of the first public Git commit, not a recreated
    # intermediate training checkpoint or a claim that every edit was saved.
    historic = out / "history/published_2026-10-03"
    for folder, glob in (("fundamental", "*.py"), ("tests", "*.py")):
        for source in (old / folder).glob(glob):
            target = historic / folder / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in LM_SCRIPTS:
        source = old / "scripts" / name
        if source.is_file():
            target = historic / "scripts" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in ("report/FINAL_REPORT.md", "BEGINNER_TUTORIAL.md"):
        source = old / name
        if source.is_file():
            target = historic / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    readme = """# Learning AI Path: From-scratch 0.1B Transformer

Start with [the beginner walkthrough](BEGINNER_TUTORIAL.md), then read the
[final report](report/FINAL_REPORT.md). The report places implementation
excerpts, measured log lines and analysis beside each research stage. The
[12-page PDF](output/pdf/0.1B_transformer_final_report.pdf) contains the same
research narrative and three plots. All current implementation files are in
[`fundamental/`](fundamental/) and [`scripts/`](scripts/); tests are in
[`tests/`](tests/).

## Learning order

| Stage | Main code | Evidence |
|---|---|---|
| Raw WET / FineWeb, cleaning | `scripts/wet_to_raw_jsonl.py`, `fundamental/data.py` | `report/data/cloud/*manifest.json` |
| Byte BPE tokenizer | `fundamental/tokenizer.py` | `tokenizer_fineweb_pilot_8192.json`, token manifest |
| Decoder Transformer / 0.1B | `fundamental/model.py`, `fundamental/train.py` | nine seed metrics, partial 500M continuation |
| Triton causal attention | `fundamental/triton_attention.py`, `scripts/attention_ab.py` | CUDA Event JSONL, attention plot |
| Scaling law | `fundamental/scaling.py`, `scripts/analyze_scaling_grid.py` | held-out fit JSON, two plots |
| SFT, DPO, RLVR | `fundamental/posttrain.py`, `fundamental/evaluate.py` | short failed probe, generated tasks |
| Agent and self-judge | `fundamental/env.py`, `fundamental/agent.py` | oracle example, no successful learned long-horizon result |

Run `python scripts/tutorial_results.py` to reprint the saved result lines
without GPU or cloud rental. With the project dependencies installed, run
`python -m pytest -q tests` for the CPU checks; the Triton checks require a
supported NVIDIA GPU and Linux/CUDA setup.

## What is and is not in this package

This package contains the full **current project source**, saved metrics,
manifests, trained tokenizer, generated figures, report, toy chat checkpoint,
and the first public code/report snapshot in [`history/`](history/). The
chronological research log is [`report/neurips_style.md`](report/neurips_style.md).

The original FineWeb article text, full token streams, and formal 0.1B model
checkpoints were not archived locally for publication and are **not** in this
bundle. `runs/chat_toy.pt` is only a small teaching checkpoint; it is not the
0.1B model. The 2B validation, matched two-GPU benchmarks, and successful
long-horizon Agent results were not measured. Use the download/preparation
scripts and recorded hashes when re-running on licensed data.

FineWeb-Edu source and usage conditions are documented in the report. No
project-wide software license has been declared; the dataset and third-party
materials do not inherit a license from this repository.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    history = """# Historical versions and evidence

- `published_2026-10-03/` is a byte-for-byte copy of the 0.1B source and
  report present in the first public repository commit `09297a2`.
- `report/neurips_style.md` is the chronological experiment notebook; it
  includes interrupted runs and revisions, with dates and links to raw logs.
- `report/data/cloud/*initial*`, `*progress*` and `*final*` are the actual
  saved run manifests. `report/data/attention_4090_round*.jsonl` and
  `attention_4090_confirm_1024.jsonl` preserve earlier attention rounds.
- Git history of the containing GitHub repository remains available. The
  source workspace had no `.git` directory, so no other local commits or
  unsaved intermediate code states are represented here.

The current code/report at package root supersede the historical snapshot.
Historical results are retained as evidence, not merged into final estimates.
"""
    (out / "history/README.md").write_text(history, encoding="utf-8")

    manifest = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()
                       and p.name != "BUNDLE_MANIFEST.json"
                       and not BLOCKED_PARTS.intersection(p.relative_to(out).parts)):
        rel = path.relative_to(out).as_posix()
        scan_text(path, rel)
        manifest[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (out / "BUNDLE_MANIFEST.json").write_text(
        json.dumps({"files": manifest, "exclusions": [
            "FineWeb source text and token streams", "cloud 0.1B checkpoints",
            "credentials, local environments, voice/spatial project files"]},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = out.parent / (out.name + "-complete.zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipout:
        for path in sorted(p for p in out.rglob("*") if p.is_file()
                           and not BLOCKED_PARTS.intersection(p.relative_to(out).parts)):
            zipout.write(path, f"{out.name}/{path.relative_to(out).as_posix()}")
    with zipfile.ZipFile(archive) as zin:
        bad = zin.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failed: {bad}")
    print(json.dumps({"directory": str(out), "files": len(manifest),
                      "zip": str(archive), "zip_bytes": archive.stat().st_size,
                      "zip_sha256": sha256(archive)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--old-checkout", type=Path, required=True)
    args = parser.parse_args()
    write_package(args.out.resolve(), args.old_checkout.resolve())


if __name__ == "__main__":
    main()
