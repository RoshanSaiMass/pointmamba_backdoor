#!/usr/bin/env bash
# Run inside the existing pointmamba-a100 Conda environment on an allocated A100.
set -euo pipefail
cd "$(dirname "$0")/.."
python -c "import ctypes, torch; assert torch.__version__.split('+')[0]=='2.6.0', 'Expected existing torch 2.6.0 environment'; assert torch.cuda.is_available(), 'Request a GPU allocation first'; from mamba_ssm import Mamba"
python -m pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-official.txt
python -m pip check
python scripts/check_official.py
