#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pointmamba-matplotlib}"
python -m pytest -q
python ablation.py --config configs/smoke.yaml --output runs/smoke_core
python generate_figures.py --config configs/smoke.yaml --checkpoint runs/smoke_core/curve_spiral_seed7/last.pt --per-class 1 --output figures/smoke_core
