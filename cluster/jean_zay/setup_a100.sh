#!/usr/bin/env bash
set -euo pipefail
module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0
bash "$(dirname "$0")/setup_cpu.sh"

