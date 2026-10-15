#!/usr/bin/env bash
#SBATCH --job-name=vge-solver
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
set -euo pipefail
source "${VGE_REPO_DIR:-$WORK/projects/vascular-graph-embedding}/cluster/jean_zay/env_cpu.sh"
module load gurobi/13.0.1 2>/dev/null || true
RUN_NAME="${RUN_NAME:?Set RUN_NAME to an existing generated run name}"
python scripts/benchmark_correspondence.py --run "$VGE_OUTPUT_ROOT/$RUN_NAME" --limit "${VGE_SOLVER_LIMIT:-25}" --timeout "${VGE_SOLVER_TIMEOUT:-30}"

