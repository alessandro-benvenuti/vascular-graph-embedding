#!/usr/bin/env bash
set -euo pipefail
source cluster/jean_zay/env_ged_cpu.sh
MAIN_RUN="${MAIN_RUN:?Set MAIN_RUN to the completed main GED run}"
for REQUIRED in COMPLETED.json resolved_config.yaml task_manifest.csv; do
  if [[ ! -f "$MAIN_RUN/$REQUIRED" ]]; then
    echo "missing required input: $MAIN_RUN/$REQUIRED" >&2
    exit 2
  fi
done
if [[ ! -d "$MAIN_RUN/results" ]]; then echo "missing required input: $MAIN_RUN/results" >&2; exit 2; fi
OUTPUT_DIR="$MAIN_RUN/baseline_decomposition"
if [[ -f "$OUTPUT_DIR/COMPLETED.json" && "${GED_DECOMPOSITION_OVERWRITE:-NO}" != "YES" ]]; then
  echo "completed decomposition exists: $OUTPUT_DIR; set GED_DECOMPOSITION_OVERWRITE=YES to replace it" >&2
  exit 2
fi
if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" && "${GED_DECOMPOSITION_OVERWRITE:-NO}" != "YES" ]]; then
  echo "nonempty incomplete decomposition exists: $OUTPUT_DIR; inspect it or set GED_DECOMPOSITION_OVERWRITE=YES" >&2
  exit 2
fi
JOB_ID="$(sbatch --parsable \
  --output="$VGE_GED_LOG_ROOT/%x-%j.out" \
  --error="$VGE_GED_LOG_ROOT/%x-%j.out" \
  --export=ALL,MAIN_RUN="$MAIN_RUN",GED_DECOMPOSITION_OVERWRITE="${GED_DECOMPOSITION_OVERWRITE:-NO}" \
  cluster/jean_zay/analyze_ged_baseline_decomposition_cpu.slurm)"
JOB_ID="${JOB_ID%%;*}"
printf 'main_run=%s\noutput_dir=%s\njob_id=%s\n' "$MAIN_RUN" "$OUTPUT_DIR" "$JOB_ID" | tee "$MAIN_RUN/baseline_decomposition_job.txt"
