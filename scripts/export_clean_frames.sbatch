#!/bin/bash
#SBATCH --job-name=forcesapien_clean_export
#SBATCH -A IscrC_AReGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/tmp/forcesapien_clean_export_%j.log
#SBATCH --error=/tmp/forcesapien_clean_export_%j.err
set -euo pipefail
RUN_ROOT=/leonardo_work/IscrC_EditGS/andrea/force_spaien
RUN_PYTHON=/leonardo_work/IscrC_EditGS/andrea/FORCEARTGS/.venv/bin/python
cd "$RUN_ROOT"
"$RUN_PYTHON" scripts/export_clean_frames.py
