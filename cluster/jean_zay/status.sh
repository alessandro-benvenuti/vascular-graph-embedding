#!/usr/bin/env bash
set -euo pipefail
ROOT="${VGE_OUTPUT_ROOT:-$WORK/analysis/vascular-graph-embedding}"
echo "Recorded job IDs:"
find "$ROOT/job_ids" -type f -maxdepth 1 -print -exec sed -n '1p' {} \; 2>/dev/null || true
echo "Queue:"
squeue -u "$USER" -o '%.18i %.24j %.10T %.10M %.6D %R'
echo "Recent completion markers:"
find "$ROOT" -name COMPLETED.json -mtime -7 -print 2>/dev/null | tail -20

