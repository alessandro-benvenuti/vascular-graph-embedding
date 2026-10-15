#!/usr/bin/env bash
# This file is sourced by both interactive shells and strict-mode Slurm jobs.
# Do not mutate the caller's shell options here.
module purge
module load pytorch-gpu/py3/2.3.0
VGE_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VGE_REPO_CANDIDATE="${SLURM_SUBMIT_DIR:-$VGE_ENV_SCRIPT_DIR/../..}"
export VGE_REPO_DIR="$(git -C "$VGE_REPO_CANDIDATE" rev-parse --show-toplevel)"
export VGE_OUTPUT_ROOT="${VGE_OUTPUT_ROOT:-$WORK/analysis/vascular-graph-embedding}"
export VGE_ENV_DIR="${VGE_ENV_DIR:-$WORK/environments/vascular-graph-embedding-cpu}"
export PATH="$VGE_ENV_DIR/bin:$PATH"
export PYTHONPATH="$VGE_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$VGE_REPO_DIR"

