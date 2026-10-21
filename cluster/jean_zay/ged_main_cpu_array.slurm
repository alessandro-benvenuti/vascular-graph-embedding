#!/usr/bin/env bash
#SBATCH --job-name=vge-ged-main
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --array=0-1199%64
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/cluster/jean_zay/env_ged_cpu.sh"
GED_RUN_DIR="${GED_RUN_DIR:?GED_RUN_DIR is required}"
python scripts/run_ged_task.py --run "$GED_RUN_DIR" --task-id "$SLURM_ARRAY_TASK_ID"
