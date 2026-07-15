#!/bin/bash
#SBATCH --job-name=forcesapien_global_phys
#SBATCH --account=iscrc_editgs
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/forcesapien_global_phys_%j.out
#SBATCH --error=logs/forcesapien_global_phys_%j.err

set -euo pipefail
cd /leonardo_work/IscrC_EditGS/andrea/force_spaien

/leonardo_work/IscrC_EditGS/andrea/FORCEARTGS/.venv/bin/python scripts/run_forcesapien_batch_final_dataset.py \
  --dataset_root final_dataset \
  --output_root outputs \
  --object_ids 10211 10905 11100 45135 100109 101917 102255 103111 103706 103776 \
  --force-application-mode external_link_force \
  --force-policy fixed_global_physics \
  --force-magnitude 5.0 \
  --force-duration-s 0.2 \
  --joint-damping 1.0 \
  --joint-friction 0.15 \
  --contact-overrides configs/manual_contact_overrides.yaml \
  --manual-contact-required true \
  --adaptive-duration true \
  --min-sim-duration-s 6.0 \
  --max-sim-duration-s 20.0 \
  --settle-qdot-threshold 0.002 \
  --settle-window-s 0.5 \
  --post-settle-hold-s 1.0 \
  --fps 30 \
  --end-hold-seconds 0 \
  --end-hold-mode never \
  --output-suffix manual_contact_global_physics_adaptive_check \
  --python-executable /leonardo_work/IscrC_EditGS/andrea/FORCEARTGS/.venv/bin/python
