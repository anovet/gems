# GEMS

Our entry to the DOE GEMS Prize: a fault probability map of the GeoDAWN region, scored by the distance-weighted Tversky index (DTI) against new faults. Vocabulary is defined in [`GLOSSARY.md`](GLOSSARY.md), decisions in [`docs/adr/`](docs/adr/), and the plan and its known limitations in [`docs/roadmap.md`](docs/roadmap.md).

## Setup

### Data files

The `data/` folder isn't tracked by Git because the files are too large for GitHub. The main features raster alone is about 400 MB. Download these files from the GEMS Prize competition data download page and save them in `data/`:

- `gems-geodawn-numerical-features.tif`: the 19 feature bands of the GeoDAWN region (EPSG:32611, 100 m)
- `existing_faults.tif`: known-fault labels (1 = known fault, 0 = no fault, -1 = outside the study area)
- `example_submission.tif`: sample submission, showing the format
- `tnm_items.json`: The National Map item listing, for the 1 m DEM downloads (not used until V4)

### Python environment

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```
uv sync
```

This installs Python 3.14, the dependencies and the `gems` command. Check it with `uv run gems --help`.

Tests use synthetic data only and never read `data/`: `uv run pytest`.

## Running an experiment

```
uv run gems run --config configs/v1.toml
```

Run it from the repo root: paths in the config (`[data]` rasters, `[output] runs_dir`) are relative to the current directory. To try a variation, copy the config, edit it (feature bands, spatial folds, sampling, model, post-processing, run name and iteration) and run the copy. Save configs as UTF-8 without a byte-order mark; Windows PowerShell 5.1's `Set-Content -Encoding utf8` adds one and the TOML parser rejects it.

The run prints progress messages, scores the out-of-fold predictions on the spatial folds, tunes the post-processing, retrains on all known faults, writes the submission and validates it. It exits with an error if the submission is invalid.

The same run from a notebook, through the same code path:

```python
from gems.runner import run_experiment

result = run_experiment("configs/v1.toml")  # or a dict of config sections
result.scores, result.run_dir, result.submission_path
```

## Where run outputs live

Everything under `runs/` is git-ignored.

- **Run folder**, `runs/<YYYYMMDD-HHMMSS>_<name>/`, one per run, never overwritten:
  - `config.toml`: the resolved config, with defaults filled in, written as the run starts
  - `metadata.json`: status (`completed`, or `failed` with the error), git commit and dirty flag, start and end time, runtime, per-fold and mean local scores (raw and post-processed), and the tuned post-processing parameters; written even if the run fails
  - `submission.tif`: the submission to upload
  - `preview.png`: the fault probability map with known faults overlaid and a zoomed inset
- **Results CSV**, `runs/results.csv`: one row per completed run, with columns `timestamp`, `run_name`, `iteration`, `run_dir` (the run folder's name), `git_commit`, `git_dirty`, `mean_local_score_raw`, `fold_local_scores_raw`, `mean_local_score_postprocessed`, `fold_local_scores_postprocessed` (per-fold scores are `;`-separated) and `leaderboard_score`, which starts empty.

## Validating a GeoTIFF

Check any GeoTIFF before spending one of the 3 weekly submissions on it:

```
uv run gems validate path/to/file.tif
```

It checks CRS, resolution, bounds, shape, band count, dtype, the value range [0, 1], and that NaN falls exactly outside the study area. It prints `valid submission.` (exit code 0) or lists each problem (exit code 1). The reference rasters default to the `data/` files; override them with `--features` and `--labels`.

## Recording a leaderboard score

Uploads are manual, through the challenge website. After the leaderboard score appears:

1. Open `runs/results.csv` and find the row whose `run_dir` is the name of the run folder holding the `submission.tif` you uploaded.
2. Type the score into that row's `leaderboard_score` column and save the file. Keep it as plain CSV.
3. Copy the score into the iteration record as well (see below), since `runs/` is not in git.

## Iterations, branches and tags

Each iteration (V1, V2, …) is planned in [`docs/roadmap.md`](docs/roadmap.md) and follows the same convention:

1. Develop it on its own branch and merge it to `main`.
2. Set `[run] iteration` in its config so results group by iteration.
3. Write its iteration record in `docs/iterations/vN.md`, starting from [`docs/iterations/TEMPLATE.md`](docs/iterations/TEMPLATE.md), and link it from the roadmap.
4. Tag the commit that produced the iteration's chosen submission by hand: `git tag v1 <commit>` (then `v2`, …). The commit hash is in the run's `metadata.json` and the results CSV.

The final submission is chosen by the rule in [ADR 0005](docs/adr/0005-final-submission-selection-rule.md): best local score, unless its leaderboard score is clearly worse than the runner-up's.
