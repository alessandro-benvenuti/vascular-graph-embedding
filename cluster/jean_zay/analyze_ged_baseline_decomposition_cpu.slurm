#!/usr/bin/env bash
#SBATCH --job-name=vge-ged-decompose
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from repository root}/cluster/jean_zay/env_ged_cpu.sh"
MAIN_RUN="${MAIN_RUN:?MAIN_RUN is required}"
OVERWRITE_ARGS=()
if [[ "${GED_DECOMPOSITION_OVERWRITE:-NO}" == "YES" ]]; then OVERWRITE_ARGS+=(--overwrite); fi
python scripts/analyze_ged_baseline_decomposition.py --main-run "$MAIN_RUN" "${OVERWRITE_ARGS[@]}"
