#!/bin/bash
#SBATCH --job-name=forcesapien_realistic
#SBATCH --account=iscrc_editgs
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/forcesapien_realistic_%j.out
#SBATCH --error=logs/forcesapien_realistic_%j.err

set -euo pipefail
cd /leonardo_work/IscrC_EditGS/andrea/force_spaien

/leonardo_work/IscrC_EditGS/andrea/FORCEARTGS/.venv/bin/python scripts/run_forcesapien_batch_final_dataset.py \
  --dataset_root final_dataset \
  --output_root outputs \
  --object_ids 10211 10905 11100 45135 100109 101917 102255 103111 103706 103776 \
  --force-application-mode external_link_force \
  --force-policy realistic_response_calibration \
  --realistic-response-params configs/realistic_response_params.yaml \
  --contact-overrides configs/manual_contact_overrides.yaml \
  --manual-contact-required true \
  --adaptive-duration true \
  --min-sim-duration-s 6.0 \
  --max-sim-duration-s 15.0 \
  --settle-qdot-threshold 0.002 \
  --settle-window-s 0.5 \
  --post-settle-hold-s 1.0 \
  --force-duration-s 0.2 \
  --fps 30 \
  --end-hold-seconds 0 \
  --end-hold-mode never \
  --output-suffix realistic_manual_contact_adaptive_check \
  --python-executable /leonardo_work/IscrC_EditGS/andrea/FORCEARTGS/.venv/bin/python
