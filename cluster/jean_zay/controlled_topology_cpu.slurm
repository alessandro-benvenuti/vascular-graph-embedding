#!/usr/bin/env bash
#SBATCH --job-name=vge-controlled-topology
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-19%4
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from the repository root}/cluster/jean_zay/env_cpu.sh"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
SHARD="$VGE_OUTPUT_ROOT/$RUN_NAME/shards/$(printf '%03d' "$SLURM_ARRAY_TASK_ID")"
python scripts/compute_topology.py --run "$SHARD"
python scripts/complete_shard.py --shard "$SHARD"
