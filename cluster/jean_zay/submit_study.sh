#!/usr/bin/env bash
set -euo pipefail
REPO="${VGE_REPO_DIR:-$WORK/projects/vascular-graph-embedding}"
ROOT="${VGE_OUTPUT_ROOT:-$WORK/analysis/vascular-graph-embedding}"
mkdir -p "$ROOT/job_ids"
cd "$REPO"
stamp="$(date +%Y%m%d-%H%M%S)"
job="$(sbatch --parsable --export=ALL,RUN_NAME="smoke-$stamp" cluster/jean_zay/smoke_cpu.slurm)"
printf '%s\n' "$job" | tee "$ROOT/job_ids/smoke-$stamp.txt"
echo "Smoke submitted as $job. Inspect it before submitting main stages."

