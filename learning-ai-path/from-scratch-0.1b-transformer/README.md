# Learning AI Path: From-scratch 0.1B Transformer

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
