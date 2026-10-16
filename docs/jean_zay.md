# Jean Zay commands

No cluster job was run during local implementation. Run the smoke stage and provide its output before treating the cluster pipeline as validated.

First checkout:

```bash
cd "$WORK/projects"
git clone https://github.com/alessandro-benvenuti/vascular-graph-embedding.git
cd vascular-graph-embedding
git switch codex/corruption-distance-study
bash cluster/jean_zay/setup_cpu.sh
```

Subsequent update:

```bash
cd "$WORK/projects/vascular-graph-embedding"
git switch codex/corruption-distance-study
git pull --ff-only origin codex/corruption-distance-study
bash cluster/jean_zay/setup_cpu.sh
```

Preflight and smoke submission:

```bash
cd "$WORK/projects/vascular-graph-embedding"
source cluster/jean_zay/env_cpu.sh
sbatch cluster/jean_zay/preflight_cpu.slurm
RUN_NAME="smoke-$(date +%Y%m%d-%H%M%S)" sbatch cluster/jean_zay/smoke_cpu.slurm
bash cluster/jean_zay/status.sh
```

Main stages and optional solver:

```bash
RUN_NAME="controlled-$(date +%Y%m%d-%H%M%S)" bash cluster/jean_zay/submit_controlled.sh
RUN_NAME="empirical-$(date +%Y%m%d-%H%M%S)" sbatch cluster/jean_zay/empirical_cpu.slurm
RUN_NAME="solver-$(date +%Y%m%d-%H%M%S)" sbatch cluster/jean_zay/solver_cpu.slurm
bash cluster/jean_zay/status.sh
```

The controlled submission records both the 20-task array job and its `afterok` finalize job. Each array task processes ten of the shared 200-patch selection, with a default concurrency cap of four. The finalize job refuses to aggregate unless all 20 `SHARD_COMPLETED.json` markers exist, checks pair-ID uniqueness and distance coverage, then writes the study-level report and completion marker. Pair payloads remain in per-shard ZIP archives so new distance plugins can reuse them without regenerating corruptions.

`submit_study.sh` submits smoke first and records every job ID under `$WORK/analysis/vascular-graph-embedding/job_ids`. Ordinary corruption studies are CPU-only. The A100 template is preparation for future accelerated plugins/encoders and is not used by the normal pipeline.

