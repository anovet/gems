import numpy as np
import rasterio

from gems.cli import main
from gems.submission import validate_submission


def _write(path, values, src_profile_path, dtype="float32", nodata=np.nan):
    with rasterio.open(src_profile_path) as ref:
        crs, transform = ref.crs, ref.transform
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0], count=1,
        dtype=dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(values.astype(dtype), 1)
    return path


def _good_values(synthetic):
    values = np.full(synthetic.study_area.shape, np.nan, dtype=np.float64)
    values[synthetic.study_area] = 0.3
    return values


def test_valid_file_has_no_problems(synthetic, tmp_path):
    path = _write(tmp_path / "ok.tif", _good_values(synthetic), synthetic.features)

    assert validate_submission(path, synthetic.features, synthetic.labels) == []


def test_wrong_dtype_is_reported(synthetic, tmp_path):
    path = _write(tmp_path / "f64.tif", _good_values(synthetic), synthetic.features, dtype="float64")

    problems = validate_submission(path, synthetic.features, synthetic.labels)

    assert any("float64" in p and "float32" in p for p in problems)


def test_values_outside_unit_interval_are_reported(synthetic, tmp_path):
    values = _good_values(synthetic)
    values[synthetic.study_area] = 1.5
    path = _write(tmp_path / "big.tif", values, synthetic.features)

    problems = validate_submission(path, synthetic.features, synthetic.labels)

    assert any("outside [0, 1]" in p for p in problems)


def test_values_outside_the_study_area_are_reported(synthetic, tmp_path):
    values = _good_values(synthetic)
    values[~synthetic.study_area] = 0.0
    path = _write(tmp_path / "border.tif", values, synthetic.features)

    problems = validate_submission(path, synthetic.features, synthetic.labels)

    assert any("outside the study area" in p for p in problems)


def test_cli_validate_reports_problems_and_exit_code(synthetic, tmp_path, capsys):
    good = _write(tmp_path / "ok.tif", _good_values(synthetic), synthetic.features)
    bad_values = _good_values(synthetic)
    bad_values[synthetic.study_area] = -0.2
    bad = _write(tmp_path / "bad.tif", bad_values, synthetic.features)
    refs = ["--features", str(synthetic.features), "--labels", str(synthetic.labels)]

    assert main(["validate", str(good), *refs]) == 0
    assert "valid" in capsys.readouterr().out.lower()

    assert main(["validate", str(bad), *refs]) == 1
    assert "outside [0, 1]" in capsys.readouterr().out


def test_cli_run_completes_on_synthetic_config(synthetic, capsys):
    assert main(["run", "--config", str(synthetic.config)]) == 0

    run_dirs = [p for p in synthetic.runs_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "submission.tif").exists()
