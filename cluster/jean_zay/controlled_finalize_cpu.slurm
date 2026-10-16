#!/usr/bin/env bash
#SBATCH --job-name=vge-controlled-finalize
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from the repository root}/cluster/jean_zay/env_cpu.sh"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
OUT="$VGE_OUTPUT_ROOT/$RUN_NAME"
python scripts/merge_shards.py --run "$OUT" --expected-shards "${VGE_CONTROLLED_SHARDS:-20}"
python scripts/analyse_distances.py --run "$OUT"

