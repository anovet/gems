import csv
import json
import tomllib
from datetime import datetime

import numpy as np
import pytest
import rasterio

from conftest import COLS, ORIGIN_X, ORIGIN_Y, PIXEL, ROWS
from gems.dti import score_dti
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


def test_run_reports_per_fold_and_mean_local_scores(synthetic):
    result = run_experiment(synthetic.config)

    folds = result.scores["fold_local_scores_raw"]
    mean = result.scores["mean_local_score_raw"]
    assert len(folds) == 5
    assert all(0.0 <= s <= 1.0 for s in folds)
    assert mean == pytest.approx(sum(folds) / 5)
    meta = json.loads((result.run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["scores"]["fold_local_scores_raw"] == folds
    assert meta["scores"]["mean_local_score_raw"] == mean


def test_local_score_beats_an_all_zeros_prediction(synthetic):
    zeros = np.zeros(synthetic.study_area.shape, dtype=np.float32)
    all_zeros = score_dti(zeros, synthetic.faults, synthetic.study_area).dti

    result = run_experiment(synthetic.config)

    assert result.scores["mean_local_score_raw"] > all_zeros + 0.1


def test_every_study_area_pixel_gets_one_spatial_fold_and_one_out_of_fold_prediction(synthetic):
    result = run_experiment(synthetic.config)

    study = synthetic.study_area
    assert set(np.unique(result.folds[study]).tolist()) == {0, 1, 2, 3, 4}
    assert np.all(result.folds[~study] == -1)
    assert np.array_equal(np.isfinite(result.out_of_fold), study)
    inside = result.out_of_fold[study]
    assert inside.min() >= 0 and inside.max() <= 1


def test_same_seed_gives_identical_scores(synthetic):
    first = run_experiment(synthetic.config)
    second = run_experiment(synthetic.config)

    assert first.scores == second.scores


def test_each_run_appends_one_row_to_the_results_table(synthetic):
    results_csv = synthetic.runs_dir / "results.csv"
    assert not results_csv.exists()

    first = run_experiment(synthetic.config)
    second = run_experiment(synthetic.config)

    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    row = rows[1]
    assert row["run_name"] == "synthetic"
    assert row["iteration"] == "V1"
    assert row["run_dir"] == second.run_dir.name
    assert len(row["git_commit"]) == 40
    assert row["git_dirty"] in {"True", "False"}
    datetime.fromisoformat(row["timestamp"])
    assert float(row["mean_local_score_raw"]) == pytest.approx(second.scores["mean_local_score_raw"])
    folds = [float(s) for s in row["fold_local_scores_raw"].split(";")]
    assert folds == pytest.approx(second.scores["fold_local_scores_raw"])
    assert row["leaderboard_score"] == ""
    assert rows[0]["run_dir"] == first.run_dir.name
