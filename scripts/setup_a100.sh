#!/usr/bin/env bash
# Run in a CUDA 12.4 development image (nvcc and a compatible NVIDIA driver).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvcc >/dev/null || { echo 'CUDA development toolkit (nvcc) is required to build Mamba.' >&2; exit 1; }
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'pip==25.0.1' 'setuptools==75.8.0' 'wheel==0.45.1' 'packaging==24.2' 'ninja==1.11.1.3'
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
# Limit parallelism to avoid exhausting host RAM during CUDA extension compilation.
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST=8.0
python -m pip install 'causal-conv1d==1.5.0.post8' --no-build-isolation
python -m pip install 'mamba-ssm==2.2.4' --no-build-isolation
python -m pip check
OMP_NUM_THREADS=4 python -m pytest -q
python scripts/check_cuda.py
python -m pip freeze > a100-installed.txt
