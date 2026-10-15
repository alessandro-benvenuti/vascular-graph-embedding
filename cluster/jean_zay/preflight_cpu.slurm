#!/usr/bin/env bash
#SBATCH --job-name=vge-preflight
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
set -euo pipefail
source "${VGE_REPO_DIR:-$WORK/projects/vascular-graph-embedding}/cluster/jean_zay/env_cpu.sh"
python scripts/preflight.py --config configs/smoke.yaml --output "$TMPDIR/vge-preflight-pair.npz"

