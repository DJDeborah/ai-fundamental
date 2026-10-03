#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -c "import torch; assert torch.__version__.split('+')[0] == '2.8.0', 'Use the PyTorch 2.8.0 CUDA 12.8 image'; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
apt-get update
apt-get install -y ffmpeg libsndfile1 build-essential
python -m venv --system-site-packages .voice-venv
source .voice-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-cloud.txt
python -m ipykernel install --user --name voice-lab --display-name "Voice Lab"
python -m voicelab doctor --require-gpu
python -m pip freeze > runs/environment.lock.txt
echo "Next terminal session: cd into voice-lab and source .voice-venv/bin/activate"
