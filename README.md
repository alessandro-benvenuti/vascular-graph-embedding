# Vascular graph corruption-distance study

This repository generates reusable pairs of clean training graphs and reproducible controlled or empirical corruptions, then evaluates pluggable graph costs. The first plugin implements the proposed correspondence-conditioned soft geometric edit **cost**. It is a surrogate, not a proven metric or exact graph edit distance.

The fixed-slot representation stores normalized D/H/W coordinates, node-presence probabilities `p`, symmetric conditional edge probabilities `q`, and effective adjacency `p_i p_j q_ij`. Stable node identities and GT correspondence are independent of array positions. VTP coordinates are explicitly interpreted as voxel or normalized coordinates; the legacy SyntheticMRI convention is `voxel_index / axis_size`.

## Local validation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest
python scripts/generate_pairs.py --config configs/smoke.yaml --output /tmp/vge-smoke --dry-run
python scripts/generate_pairs.py --config configs/smoke.yaml --output /tmp/vge-smoke --synthetic-fixtures
python scripts/compute_distances.py --config configs/smoke.yaml --run /tmp/vge-smoke
python scripts/analyse_distances.py --run /tmp/vge-smoke
python scripts/benchmark_correspondence.py --run /tmp/vge-smoke --limit 25
```

Generated datasets, runs, logs, environments and checkpoints belong outside Git. Pair archives are immutable inputs to distance plugins, so adding a plugin does not regenerate corruptions. Completed studies contain `COMPLETED.json` and are never silently overwritten.

## Study stages

1. `preflight.py` checks paths, split, dependencies, optional solver availability, and graph round-trip.
2. `select_patches.py` selects training patches with size/topology strata and a patient-load tie break.
3. `generate_pairs.py` records requested/realized edits, seeds, intensities, ownership and rejection reasons.
4. `compute_distances.py` exports raw terms, denominators, normalized terms, total cost, background baseline, and signed change from that baseline.
5. `analyse_distances.py` writes topology diagnostics, clustered summaries, plots, report, summary, and completion marker.
6. `benchmark_correspondence.py` certifies tiny exhaustive cases. Timeout incumbents are not exact labels. Gurobi remains an optional backend to be added behind the solver interface after its Jean Zay license preflight passes.

The current local generator uses deterministic synthetic fixtures for offline validation. The Jean Zay data paths are declared in configuration and preflighted; cluster results have not been run or validated here. Validation-derived empirical profiles are a development input. Evaluation on those same validation patients is not an untouched final assessment.

See [docs/study_design.md](docs/study_design.md) and [docs/jean_zay.md](docs/jean_zay.md).

