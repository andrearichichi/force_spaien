#!/bin/bash
#SBATCH --job-name=forcesapien_definitive
#SBATCH -A IscrC_AReGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=production_generation_%j.log
#SBATCH --error=production_generation_%j.err
set -euo pipefail
RUN_ROOT=/leonardo_work/IscrC_EditGS/andrea/force_spaien
RUN_PYTHON=/leonardo_work/IscrC_EditGS/andrea/FORCEARTGS/.venv/bin/python
: "${FORCESAPIEN_OUTPUT_ROOT:?Set FORCESAPIEN_OUTPUT_ROOT to a new explicit output path}"
cd "$RUN_ROOT"
"$RUN_PYTHON" scripts/generate_definitive_dataset.py \
  --dataset-root final_dataset \
  --output-root "$FORCESAPIEN_OUTPUT_ROOT"
