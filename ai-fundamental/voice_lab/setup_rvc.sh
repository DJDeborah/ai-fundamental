#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# A separate environment: the language-model environment keeps its original Torch.
if [[ "${RVC_REUSE_IMAGE_TORCH:-0}" == "1" ]]; then
  python -m venv --system-site-packages .rvc-venv
else
  python -m venv .rvc-venv
fi
.rvc-venv/bin/python -m pip install --upgrade pip
if [[ "${RVC_REUSE_IMAGE_TORCH:-0}" == "1" ]]; then
  # Explicit opt-in for the verified new image; both arms share this runtime.
  .rvc-venv/bin/python -c 'import torch; assert torch.__version__ == "2.8.0+cu128", torch.__version__'
  .rvc-venv/bin/python -m pip install torchaudio==2.8.0 --no-deps --index-url https://download.pytorch.org/whl/cu128
else
  .rvc-venv/bin/python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
fi
.rvc-venv/bin/python -m pip install -r requirements-rvc.txt
command -v ffmpeg >/dev/null || { apt-get update; apt-get install -y ffmpeg; }
.rvc-venv/bin/python -m ipykernel install --user --name rvc-lab --display-name 'RVC Comparison'
