#!/bin/bash
#SBATCH --job-name=official-pointmamba
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=official-pointmamba-%j.out
#SBATCH --error=official-pointmamba-%j.err
set -euo pipefail
cd /export/home/achyut/Sarvesh/pointmamba_backdoor
module load cuda/cuda-12.4
source /export/home/achyut/miniconda3/etc/profile.d/conda.sh
conda activate pointmamba-a100
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
bash scripts/install_official.sh
python -u ablation.py --config configs/official.yaml --suite core --output runs/official_modelnet10_core
for curve in spiral rose lissajous; do
    python generate_figures.py --config configs/official.yaml --checkpoint "runs/official_modelnet10_core/curve_${curve}_seed42/last.pt" --output "figures/official_${curve}_checkpoint"
done
python report.py --root runs/official_modelnet10_core
