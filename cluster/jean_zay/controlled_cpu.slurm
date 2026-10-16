#!/usr/bin/env bash
#SBATCH --job-name=vge-controlled
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --array=0-19%4
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from the repository root}/cluster/jean_zay/env_cpu.sh"
RUN_NAME="${RUN_NAME:-controlled-${SLURM_ARRAY_JOB_ID}}"
SHARD_SIZE="${VGE_CONTROLLED_SHARD_SIZE:-10}"
OFFSET="$((SLURM_ARRAY_TASK_ID * SHARD_SIZE))"
OUT="$VGE_OUTPUT_ROOT/$RUN_NAME/shards/$(printf '%03d' "$SLURM_ARRAY_TASK_ID")"
python scripts/generate_pairs.py --config configs/controlled.yaml --output "$OUT" --patch-offset "$OFFSET" --patch-limit "$SHARD_SIZE"
python scripts/compute_distances.py --config configs/controlled.yaml --run "$OUT"
python scripts/complete_shard.py --shard "$OUT"

