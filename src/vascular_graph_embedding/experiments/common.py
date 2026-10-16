from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vascular_graph_embedding.data.storage import atomic_json


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle: return yaml.safe_load(handle)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def git_sha(repository: Path) -> str:
    """Resolve HEAD even on compute images where the git executable is absent."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_entry = repository / ".git"
        if git_entry.is_file():
            value = git_entry.read_text().strip()
            if value.startswith("gitdir:"):
                git_entry = (repository / value.split(":", 1)[1].strip()).resolve()
        head = git_entry / "HEAD"
        if not head.is_file():
            return "unknown"
        value = head.read_text().strip()
        if not value.startswith("ref:"):
            return value
        reference = value.split(":", 1)[1].strip()
        loose = git_entry / reference
        if loose.is_file():
            return loose.read_text().strip()
        packed = git_entry / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line and not line.startswith(("#", "^")):
                    sha, name = line.split(" ", 1)
                    if name == reference:
                        return sha
        return "unknown"


def prepare_run(config: dict[str, Any], output: str | Path) -> Path:
    output = Path(output)
    if (output / "COMPLETED.json").exists(): raise FileExistsError(f"completed run will not be overwritten: {output}")
    output.mkdir(parents=True, exist_ok=True)
    temporary=output/"resolved_config.yaml.tmp";temporary.write_text(yaml.safe_dump(config,sort_keys=True));temporary.replace(output/"resolved_config.yaml")
    repository = Path(os.environ.get("VGE_REPO_DIR", Path.cwd())).resolve()
    atomic_json(output/"run_metadata.json", {"schema":"vascular-run/v1","config_hash":config_hash(config),"git_sha":git_sha(repository),"python":platform.python_version(),"numpy":np.__version__,"seed":config.get("seed"),"source_paths":config.get("data",{}),"normalization":"D/H/W index divided by axis size","execution_environment":"jean_zay_slurm" if os.environ.get("SLURM_JOB_ID") else "local","slurm_job_id":os.environ.get("SLURM_JOB_ID"),"validation_development_warning":"validation-derived empirical profiles are development input; those patients are not untouched final evaluation"})
    return output

