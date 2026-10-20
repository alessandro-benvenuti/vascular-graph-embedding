#!/usr/bin/env bash
#SBATCH --job-name=vge-topology-associations
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
set -euo pipefail

source "${SLURM_SUBMIT_DIR:?Submit from the repository root}/cluster/jean_zay/env_cpu.sh"

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
RUN="$VGE_OUTPUT_ROOT/$RUN_NAME"

python scripts/analyse_topology_associations.py --run "$RUN"
