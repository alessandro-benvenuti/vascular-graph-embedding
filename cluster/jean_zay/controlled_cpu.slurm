#!/usr/bin/env bash
#SBATCH --job-name=vge-controlled
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
source "${SLURM_SUBMIT_DIR:?Submit from the repository root}/cluster/jean_zay/env_cpu.sh"
RUN_NAME="${RUN_NAME:-controlled-${SLURM_JOB_ID}}"; OUT="$VGE_OUTPUT_ROOT/$RUN_NAME"
python scripts/generate_pairs.py --config configs/controlled.yaml --output "$OUT"
python scripts/compute_distances.py --config configs/controlled.yaml --run "$OUT"
python scripts/analyse_distances.py --run "$OUT"

