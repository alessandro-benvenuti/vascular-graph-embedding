#!/usr/bin/env bash
#SBATCH --job-name=vge-ged-analysis
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/cluster/jean_zay/env_ged_cpu.sh"
GED_RUN_DIR="${GED_RUN_DIR:?GED_RUN_DIR is required}"
python scripts/analyze_ged.py --run "$GED_RUN_DIR"
