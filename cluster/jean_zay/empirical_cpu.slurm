#!/usr/bin/env bash
#SBATCH --job-name=vge-empirical
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=19:59:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
set -euo pipefail
source "${VGE_REPO_DIR:-$WORK/projects/vascular-graph-embedding}/cluster/jean_zay/env_cpu.sh"
RUN_NAME="${RUN_NAME:-empirical-${SLURM_JOB_ID}}"; OUT="$VGE_OUTPUT_ROOT/$RUN_NAME"
python scripts/generate_pairs.py --config configs/empirical.yaml --output "$OUT"
python scripts/compute_distances.py --config configs/empirical.yaml --run "$OUT"
python scripts/analyse_distances.py --run "$OUT"

