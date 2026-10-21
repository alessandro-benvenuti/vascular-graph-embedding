#!/usr/bin/env bash
set -euo pipefail
MAIN_RUN="${1:-${MAIN_RUN:-}}"
if [[ -z "$MAIN_RUN" ]]; then echo "usage: check_ged_baseline_decomposition.sh MAIN_RUN" >&2; exit 2; fi
source "$MAIN_RUN/baseline_decomposition_job.txt"
sacct -j "$job_id" --format=JobID,JobName%28,State,Elapsed,ExitCode
find "$output_dir" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort || true
if [[ -f "$output_dir/COMPLETED.json" ]]; then cat "$output_dir/COMPLETED.json"; fi
