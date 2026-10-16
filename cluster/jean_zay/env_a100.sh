#!/usr/bin/env bash
# This file is sourced by both interactive shells and strict-mode Slurm jobs.
# Do not mutate the caller's shell options here.
module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0
VGE_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VGE_REPO_CANDIDATE="${SLURM_SUBMIT_DIR:-$VGE_ENV_SCRIPT_DIR/../..}"
export VGE_REPO_DIR="$(cd "$VGE_REPO_CANDIDATE" && pwd -P)"
if [[ ! -f "$VGE_REPO_DIR/pyproject.toml" || ! -d "$VGE_REPO_DIR/src/vascular_graph_embedding" ]]; then
  echo "Invalid VGE repository root: $VGE_REPO_DIR" >&2
  return 1
fi
export VGE_OUTPUT_ROOT="${VGE_OUTPUT_ROOT:-$WORK/analysis/vascular-graph-embedding}"
export VGE_ENV_DIR="${VGE_ENV_DIR:-$WORK/environments/vascular-graph-embedding-cpu}"
export PATH="$VGE_ENV_DIR/bin:$PATH"
export PYTHONPATH="$VGE_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$VGE_REPO_DIR"

