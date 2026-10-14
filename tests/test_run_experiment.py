import json
import tomllib
from datetime import datetime

import numpy as np
import rasterio

from conftest import COLS, ORIGIN_X, ORIGIN_Y, PIXEL, ROWS
from gems.runner import run_experiment
from gems.submission import validate_submission


def test_run_produces_a_valid_submission(synthetic):
    result = run_experiment(synthetic.config)

    assert result.submission_path.exists()
    assert validate_submission(result.submission_path, synthetic.features, synthetic.labels) == []


def test_submission_matches_the_reference_grid_and_challenge_format(synthetic):
    result = run_experiment(synthetic.config)

    with rasterio.open(result.submission_path) as src:
        assert src.crs.to_epsg() == 32611
        assert src.res == (PIXEL, PIXEL)
        assert tuple(src.bounds) == (ORIGIN_X, ORIGIN_Y - ROWS * PIXEL, ORIGIN_X + COLS * PIXEL, ORIGIN_Y)
        assert src.shape == (ROWS, COLS)
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        values = src.read(1)

    assert np.array_equal(np.isnan(values), ~synthetic.study_area)
    inside = values[synthetic.study_area]
    assert inside.min() >= 0 and inside.max() <= 1


def test_run_folder_records_config_metadata_and_submission(synthetic):
    result = run_experiment(synthetic.config)

    assert result.run_dir.parent == synthetic.runs_dir
    assert result.run_dir.name.endswith("_synthetic")
    assert result.submission_path.parent == result.run_dir

    with open(result.run_dir / "config.toml", "rb") as f:
        saved = tomllib.load(f)
    assert saved["run"]["name"] == "synthetic"
    assert saved["data"]["bands"] == ["mag_anom", "iso_grav_anom_slope", "geod_shearrate"]

    meta = json.loads((result.run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert isinstance(meta["git_commit"], str) and len(meta["git_commit"]) == 40
    assert isinstance(meta["git_dirty"], bool)
    assert datetime.fromisoformat(meta["start_time"]) <= datetime.fromisoformat(meta["end_time"])
    assert meta["runtime_seconds"] >= 0


def test_repeated_runs_get_separate_folders(synthetic):
    first = run_experiment(synthetic.config)
    second = run_experiment(synthetic.config)

    assert first.run_dir != second.run_dir
    assert first.submission_path.exists() and second.submission_path.exists()


def test_run_accepts_an_in_memory_config(synthetic):
    with open(synthetic.config, "rb") as f:
        config = tomllib.load(f)

    result = run_experiment(config)

    assert validate_submission(result.submission_path, synthetic.features, synthetic.labels) == []
