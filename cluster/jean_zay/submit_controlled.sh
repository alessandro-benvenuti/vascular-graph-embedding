#!/usr/bin/env bash
set -euo pipefail
SUBMIT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SUBMIT_SCRIPT_DIR/../.." && pwd -P)"
ROOT="${VGE_OUTPUT_ROOT:-$WORK/analysis/vascular-graph-embedding}"
mkdir -p "$ROOT/job_ids"
cd "$REPO"
RUN_NAME="${RUN_NAME:-controlled-$(date +%Y%m%d-%H%M%S)}"
ARRAY_JOB="$(sbatch --parsable --export=ALL,RUN_NAME="$RUN_NAME" cluster/jean_zay/controlled_cpu.slurm)"
FINAL_JOB="$(sbatch --parsable --dependency="afterok:$ARRAY_JOB" --export=ALL,RUN_NAME="$RUN_NAME" cluster/jean_zay/controlled_finalize_cpu.slurm)"
printf 'run_name=%s\narray_job=%s\nfinalize_job=%s\n' "$RUN_NAME" "$ARRAY_JOB" "$FINAL_JOB" | tee "$ROOT/job_ids/$RUN_NAME.txt"
