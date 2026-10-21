#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${1:?usage: check_run.sh RUN_DIR}"
source "$RUN_DIR/job_ids.txt"
sacct -j "$solve_job,$analysis_job" --format=JobID,JobName%28,State,Elapsed,ExitCode
find "$RUN_DIR/results" -name 'task_*.json' | wc -l
if [[ -f "$RUN_DIR/COMPLETED.json" ]]; then
  cat "$RUN_DIR/COMPLETED.json"; cat "$RUN_DIR/summary.json"
fi
