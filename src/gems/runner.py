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

from gems.features import compute_features
from gems.folds import assign_spatial_folds, local_scores, training_region
from gems.model import build_model
from gems.postprocess import TUNING_NOTE, tune_post_processing
from gems.preview import save_preview
from gems.raster import read_feature_bands, read_grid, read_labels
from gems.results import append_result
from gems.sampling import sample_training_pixels
from gems.submission import validate_submission, write_submission

DEFAULTS: dict[str, Any] = {
    "run": {"name": "run", "iteration": "V1"},
    "data": {
        "features": "data/gems-geodawn-numerical-features.tif",
        "labels": "data/existing_faults.tif",
        "bands": None,  # None = every band in the features raster
    },
    "output": {"runs_dir": "runs"},
    "folds": {"block_size": 20_000.0, "n_folds": 5, "seed": 0},  # block_size in metres
    "sampling": {"negative_ratio": 10.0, "seed": 0},  # non-fault pixels per known-fault pixel
    "model": {  # HistGradientBoostingClassifier keyword arguments
        "learning_rate": 0.1,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "early_stopping": False,
        "random_state": 0,
    },
    # Post-processing search space: every combination is tried, plus the identity.
    "postprocess": {
        "thresholds": [round(0.05 * i, 2) for i in range(20)],  # 0.0, 0.05, ..., 0.95
        "powers": [0.5, 1.0, 2.0],
        "scales": [1.0, 1.5, 2.0, 3.0, 5.0],
    },
}


RESULTS_CSV = "results.csv"  # the shared results table, inside runs_dir


@dataclass
class RunResult:
    run_dir: Path
    submission_path: Path
    scores: dict[str, Any] = field(default_factory=dict)
    out_of_fold: np.ndarray | None = None  # raw out-of-fold fault probabilities, NaN outside study area
    folds: np.ndarray | None = None  # spatial fold of each pixel, -1 outside the study area


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
    """Run one experiment end to end: load data, predict, write and validate a submission.

    The run folder gets the resolved config.toml first and metadata.json last, also when the
    run fails (status "failed" and the error); the failure is then re-raised. Only completed
    runs get a row in the results CSV.
    """
    cfg = load_config(config)
    started = datetime.now().astimezone()
    t0 = time.perf_counter()
    run_dir = _new_run_dir(Path(cfg["output"]["runs_dir"]), cfg["run"]["name"], started)
    _log(f"Run folder: {run_dir}")
    _write_config(run_dir, cfg)
    commit, dirty = _git_state()
    metadata: dict[str, Any] = {
        "run_name": cfg["run"]["name"],
        "iteration": cfg["run"]["iteration"],
        "status": "running",
        "git_commit": commit,
        "git_dirty": dirty,
        "start_time": started.isoformat(),
    }
    try:
        result = _run(cfg, run_dir, metadata)
    except BaseException as e:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(e).__name__}: {e}"
        _write_metadata(run_dir, metadata, t0)
        raise
    metadata["status"] = "completed"
    _write_metadata(run_dir, metadata, t0)
    append_result(
        Path(cfg["output"]["runs_dir"]) / RESULTS_CSV,
        {
            "timestamp": metadata["start_time"],
            "run_name": metadata["run_name"],
            "iteration": metadata["iteration"],
            "run_dir": run_dir.name,
            "git_commit": commit,
            "git_dirty": dirty,
            **result.scores,
        },
    )
    _log(f"Done in {metadata['runtime_seconds']} s: {result.submission_path}")
    return result


def _run(cfg: dict[str, Any], run_dir: Path, metadata: dict[str, Any]) -> RunResult:
    """The experiment itself; adds scores and post-processing parameters to `metadata`."""
    features_path, labels_path = Path(cfg["data"]["features"]), Path(cfg["data"]["labels"])
    _log("Loading feature bands and known-fault labels")
    grid = read_grid(features_path)
    bands, names = read_feature_bands(features_path, cfg["data"]["bands"])
    known_faults, study_area = read_labels(labels_path)
    if study_area.shape != grid.shape:
        raise ValueError(f"Label grid {study_area.shape} does not match feature grid {grid.shape}")
    if cfg["data"]["bands"] != names:
        cfg["data"]["bands"] = names
        _write_config(run_dir, cfg)  # record the bands actually used

    features = compute_features(bands, names)
    del bands

    folds = assign_spatial_folds(study_area, pixel_size=grid.res[0], **cfg["folds"])
    out_of_fold = _out_of_fold_predictions(
        features, known_faults, folds, model_params=cfg["model"], **cfg["sampling"]
    )
    fold_scores = local_scores(out_of_fold, known_faults, folds)
    scores = {
        "mean_local_score_raw": float(np.mean(fold_scores)),
        "fold_local_scores_raw": fold_scores,
    }
    metadata["scores"] = scores
    _log(f"Mean local score (raw): {_score_summary(fold_scores)}")

    _log("Tuning post-processing on the out-of-fold predictions")
    post_processing = tune_post_processing(out_of_fold, known_faults, folds, cfg["postprocess"])
    post_scores = local_scores(post_processing(out_of_fold), known_faults, folds)
    scores["mean_local_score_postprocessed"] = float(np.mean(post_scores))
    scores["fold_local_scores_postprocessed"] = post_scores
    metadata["postprocess"] = post_processing.as_dict()
    metadata["postprocess_note"] = TUNING_NOTE
    _log(f"Tuned post-processing: {post_processing.as_dict()}")
    _log(f"Mean local score (post-processed): {_score_summary(post_scores)}")

    _log("Final fit on the whole study area")
    train = _training_pixels(known_faults, study_area, **cfg["sampling"], sample_key=0)
    raw_probabilities = build_model(cfg["model"]).fit(features, known_faults, train).predict(features, study_area)
    probabilities = post_processing(raw_probabilities)

    _log("Writing submission")
    submission_path = write_submission(run_dir / "submission.tif", probabilities, grid)
    problems = validate_submission(submission_path, features_path, labels_path)
    if problems:
        raise InvalidSubmissionError(submission_path, problems)

    _log("Saving preview map")
    save_preview(run_dir / "preview.png", probabilities, known_faults, study_area, grid, title=run_dir.name)
    return RunResult(run_dir, submission_path, scores, out_of_fold=out_of_fold, folds=folds)


def _write_config(run_dir: Path, cfg: dict[str, Any]) -> None:
    (run_dir / "config.toml").write_text(_to_toml(cfg), encoding="utf-8")


def _write_metadata(run_dir: Path, metadata: dict[str, Any], t0: float) -> None:
    """Write metadata.json with the end time and runtime filled in."""
    metadata.update(
        end_time=datetime.now().astimezone().isoformat(),
        runtime_seconds=round(time.perf_counter() - t0, 3),
    )
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _out_of_fold_predictions(
    features: np.ndarray,
    known_faults: np.ndarray,
    folds: np.ndarray,
    *,
    model_params: dict[str, Any],
    negative_ratio: float,
    seed: int,
) -> np.ndarray:
    """Train one model per spatial fold and predict its held-out region."""
    out_of_fold = np.full(folds.shape, np.nan, dtype=np.float32)
    n_folds = int(folds.max()) + 1
    for k in range(n_folds):
        held_out = folds == k
        train = _training_pixels(
            known_faults, training_region(folds, k), negative_ratio=negative_ratio, seed=seed, sample_key=k + 1
        )
        _log(
            f"Fold {k + 1}/{n_folds}: training on {int(train.sum()):,} pixels "
            f"({int((train & known_faults).sum()):,} known-fault), predicting {int(held_out.sum()):,}"
        )
        model = build_model(model_params).fit(features, known_faults, train)
        out_of_fold[held_out] = model.predict(features, held_out)[held_out]
    return out_of_fold


def _training_pixels(
    known_faults: np.ndarray, eligible: np.ndarray, *, negative_ratio: float, seed: int, sample_key: int
) -> np.ndarray:
    """All known faults in `eligible` plus seeded non-fault samples; one random stream per fit."""
    rng = np.random.default_rng([seed, sample_key])
    return sample_training_pixels(known_faults, eligible, negative_ratio=negative_ratio, rng=rng)


def _score_summary(fold_scores: list[float]) -> str:
    per_fold = ", ".join(f"{s:.4f}" for s in fold_scores)
    return f"{np.mean(fold_scores):.4f} (folds: {per_fold})"


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
