#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash setup_rvc.sh
.rvc-venv/bin/python preflight_rvc.py
.rvc-venv/bin/python -m voicelab.rvc_compare "$@"
