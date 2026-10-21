#!/usr/bin/env bash
set -euo pipefail
if [[ "${ALLOW_MAIN:-}" != "YES" ]]; then
  echo 'Main submission is intentionally disabled. Review smoke and pilot, then rerun with ALLOW_MAIN=YES.' >&2
  exit 2
fi
source cluster/jean_zay/env_ged_cpu.sh
RUN_NAME="${RUN_NAME:-ged-main-$(date +%Y%m%d-%H%M%S)}"
GED_RUN_DIR="$VGE_GED_OUTPUT_ROOT/$RUN_NAME"; export GED_RUN_DIR
python scripts/prepare_ged_stage.py --config configs/ged_main.yaml --run "$GED_RUN_DIR"
TASKS="$(( $(wc -l < "$GED_RUN_DIR/task_manifest.csv") - 1 ))"; LAST_TASK="$((TASKS - 1))"
SOLVE_JOB="$(sbatch --parsable --array="0-${LAST_TASK}%64" --output="$VGE_GED_LOG_ROOT/%x-%A_%a.out" --error="$VGE_GED_LOG_ROOT/%x-%A_%a.out" --export=ALL,GED_RUN_DIR="$GED_RUN_DIR" cluster/jean_zay/ged_main_cpu_array.slurm)"; SOLVE_JOB="${SOLVE_JOB%%;*}"
ANALYSIS_JOB="$(sbatch --parsable --dependency="afterok:$SOLVE_JOB" --output="$VGE_GED_LOG_ROOT/%x-%j.out" --error="$VGE_GED_LOG_ROOT/%x-%j.out" --export=ALL,GED_RUN_DIR="$GED_RUN_DIR" cluster/jean_zay/analyze_ged_cpu.slurm)"; ANALYSIS_JOB="${ANALYSIS_JOB%%;*}"
printf 'run=%s\nsolve_job=%s\nanalysis_job=%s\n' "$RUN_NAME" "$SOLVE_JOB" "$ANALYSIS_JOB" | tee "$GED_RUN_DIR/job_ids.txt"
