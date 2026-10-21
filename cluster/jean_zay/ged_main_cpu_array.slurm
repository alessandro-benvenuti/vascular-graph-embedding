#!/usr/bin/env bash
#SBATCH --job-name=vge-ged-main
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --array=0-199%64
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/cluster/jean_zay/env_ged_cpu.sh"
GED_RUN_DIR="${GED_RUN_DIR:?GED_RUN_DIR is required}"
GED_TASK_COUNT="${GED_TASK_COUNT:?GED_TASK_COUNT is required}"
GED_TASKS_PER_ARRAY="${GED_TASKS_PER_ARRAY:-6}"
FIRST_TASK="$((SLURM_ARRAY_TASK_ID * GED_TASKS_PER_ARRAY))"
LAST_TASK="$((FIRST_TASK + GED_TASKS_PER_ARRAY - 1))"
if (( LAST_TASK >= GED_TASK_COUNT )); then LAST_TASK="$((GED_TASK_COUNT - 1))"; fi
for TASK_ID in $(seq "$FIRST_TASK" "$LAST_TASK"); do
  python scripts/run_ged_task.py --run "$GED_RUN_DIR" --task-id "$TASK_ID"
done
