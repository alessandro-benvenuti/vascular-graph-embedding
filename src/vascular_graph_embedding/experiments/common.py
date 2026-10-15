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


def prepare_run(config: dict[str, Any], output: str | Path) -> Path:
    output = Path(output)
    if (output / "COMPLETED.json").exists(): raise FileExistsError(f"completed run will not be overwritten: {output}")
    output.mkdir(parents=True, exist_ok=True)
    temporary=output/"resolved_config.yaml.tmp";temporary.write_text(yaml.safe_dump(config,sort_keys=True));temporary.replace(output/"resolved_config.yaml")
    try: sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: sha="unknown"
    atomic_json(output/"run_metadata.json", {"schema":"vascular-run/v1","config_hash":config_hash(config),"git_sha":sha,"python":platform.python_version(),"numpy":np.__version__,"seed":config.get("seed"),"source_paths":config.get("data",{}),"normalization":"D/H/W index divided by axis size","validation_development_warning":"validation-derived empirical profiles are development input; those patients are not untouched final evaluation"})
    return output

