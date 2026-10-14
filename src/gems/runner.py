"""The experiment runner: one public function, `run_experiment`, used by the CLI and notebooks."""

import json
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gems.raster import read_feature_bands, read_grid, read_labels
from gems.submission import validate_submission, write_submission

DEFAULTS: dict[str, Any] = {
    "run": {"name": "run", "iteration": "V1"},
    "data": {
        "features": "data/gems-geodawn-numerical-features.tif",
        "labels": "data/existing_faults.tif",
        "bands": None,  # None = every band in the features raster
    },
    "output": {"runs_dir": "runs"},
}

PLACEHOLDER_PROBABILITY = 0.5  # stand-in for a model until the V1 model lands


@dataclass
class RunResult:
    run_dir: Path
    submission_path: Path
    scores: dict[str, Any] = field(default_factory=dict)


class InvalidSubmissionError(RuntimeError):
    def __init__(self, path: Path, problems: list[str]):
        self.path, self.problems = path, problems
        super().__init__(f"Submission {path} is invalid:\n- " + "\n- ".join(problems))


def load_config(config: str | Path | dict) -> dict[str, Any]:
    """Read a TOML config (or take a dict) and fill in defaults."""
    if isinstance(config, dict):
        raw = config
    else:
        with open(config, "rb") as f:
            raw = tomllib.load(f)
    resolved = {section: dict(values) for section, values in DEFAULTS.items()}
    for section, values in raw.items():
        if isinstance(values, dict):
            resolved.setdefault(section, {}).update(values)
        else:
            resolved[section] = values
    return resolved


def run_experiment(config: str | Path | dict) -> RunResult:
    """Run one experiment end to end: load data, predict, write and validate a submission."""
    cfg = load_config(config)
    started = datetime.now().astimezone()
    t0 = time.perf_counter()
    run_dir = _new_run_dir(Path(cfg["output"]["runs_dir"]), cfg["run"]["name"], started)
    _log(f"Run folder: {run_dir}")

    features_path, labels_path = Path(cfg["data"]["features"]), Path(cfg["data"]["labels"])
    _log("Loading feature bands and known-fault labels")
    grid = read_grid(features_path)
    bands, names = read_feature_bands(features_path, cfg["data"]["bands"])
    faults, study_area = read_labels(labels_path)
    if study_area.shape != grid.shape:
        raise ValueError(f"Label grid {study_area.shape} does not match feature grid {grid.shape}")
    cfg["data"]["bands"] = names

    _log("Predicting (placeholder: constant inside the study area)")
    probabilities = np.full(grid.shape, np.nan, dtype=np.float32)
    probabilities[study_area] = PLACEHOLDER_PROBABILITY

    _log("Writing submission")
    submission_path = write_submission(run_dir / "submission.tif", probabilities, grid)
    problems = validate_submission(submission_path, features_path, labels_path)
    if problems:
        raise InvalidSubmissionError(submission_path, problems)

    ended = datetime.now().astimezone()
    commit, dirty = _git_state()
    (run_dir / "config.toml").write_text(_to_toml(cfg), encoding="utf-8")
    metadata = {
        "run_name": cfg["run"]["name"],
        "iteration": cfg["run"]["iteration"],
        "git_commit": commit,
        "git_dirty": dirty,
        "start_time": started.isoformat(),
        "end_time": ended.isoformat(),
        "runtime_seconds": round(time.perf_counter() - t0, 3),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _log(f"Done in {metadata['runtime_seconds']} s: {submission_path}")
    return RunResult(run_dir=run_dir, submission_path=submission_path)


def _log(message: str) -> None:
    print(f"[gems] {message}", flush=True)


def _new_run_dir(runs_dir: Path, name: str, started: datetime) -> Path:
    stem = f"{started:%Y%m%d-%H%M%S}_{name}"
    run_dir, n = runs_dir / stem, 1
    while run_dir.exists():
        n += 1
        run_dir = runs_dir / f"{stem}-{n}"
    run_dir.mkdir(parents=True)
    return run_dir


def _git_state() -> tuple[str | None, bool | None]:
    """Commit hash and dirty flag of the code that is running."""
    cwd = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def _to_toml(cfg: dict[str, Any]) -> str:
    """Minimal TOML writer for the resolved config (tables of scalars and lists)."""

    def value(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(value(x) for x in v) + "]"
        return json.dumps(str(v))  # JSON string escaping is valid TOML basic-string escaping

    lines: list[str] = []
    for section, values in cfg.items():
        if not isinstance(values, dict):
            lines.append(f"{section} = {value(values)}")
            continue
        lines.append(f"\n[{section}]")
        lines.extend(f"{k} = {value(v)}" for k, v in values.items() if v is not None)
    return "\n".join(lines).lstrip() + "\n"
