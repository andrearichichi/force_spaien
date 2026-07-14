#!/bin/bash
#SBATCH --job-name=forcesapien_final_5n
#SBATCH --account=iscrc_editgs
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/andrea/force_spaien/logs/forcesapien_final_5n_%j.out
#SBATCH --error=/leonardo_work/IscrC_EditGS/andrea/force_spaien/logs/forcesapien_final_5n_%j.err

set -euo pipefail

ROOT=/leonardo_work/IscrC_EditGS/andrea/force_spaien
PYTHON=/leonardo_work/IscrC_EditGS/andrea/FORCEARTGS/.venv/bin/python
VULKAN_RUNTIME="$ROOT/.vulkan_runtime"

cd "$ROOT"
mkdir -p logs outputs outputs/forcesapien_physics_mode_audit

export LD_LIBRARY_PATH="$VULKAN_RUNTIME/lib:${LD_LIBRARY_PATH:-}"
export SAPIEN_VULKAN_LIBRARY_PATH="$VULKAN_RUNTIME/lib/libvulkan.so.1"
module load cuda/12.2 >/dev/null 2>&1 || true

echo "Running on host: $(hostname)"
echo "Started at: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

"$PYTHON" scripts/run_forcesapien_batch_final_dataset.py \
  --dataset_root final_dataset \
  --output_root outputs \
  --object_ids 10211 10905 11100 101917 102255 103111 103706 103776 \
  --force \
  --force-application-mode external_link_force \
  --force-magnitude 5.0 \
  --force-duration-s 0.2 \
  --sim-duration-s 6.0 \
  --fps 30 \
  --joint-damping 0.5 \
  --joint-friction 0.05 \
  --end-hold-seconds 0 \
  --end-hold-mode never \
  --output-suffix external_force_5n_physics_check \
  --python-executable "$PYTHON"

echo "Finished at: $(date --iso-8601=seconds)"
