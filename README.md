# GEMS

## Data

The `data/` folder isn't tracked by Git because the files are too large for GitHub. The main features raster alone is about 400 MB. To set up the project, download these files from the GEMS Prize competition data download page and save them in `data/`:

- `gems-geodawn-numerical-features.tif`: GeoDAWN training features (EPSG 32611, 100 m)
- `existing_faults.tif`: USGS quaternary fault labels (raster)
- `example_submission.tif`: sample submission format
- `tnm_items.json`: The National Map item listing (for DEM downloads)

## Setup and usage

Requires [uv](https://docs.astral.sh/uv/). `uv sync` installs the Python 3.14 dependencies and the `gems` command.

- Run an experiment: `uv run gems run --config configs/v1.toml`. Each run gets its own git-ignored folder `runs/<timestamp>_<name>/` containing the resolved config, `metadata.json` (git commit, dirty flag, start/end time, runtime) and `submission.tif`. The run validates its own submission and fails if the submission is invalid.
- Validate any GeoTIFF before uploading it: `uv run gems validate path/to/file.tif`.
- From a notebook: `from gems.runner import run_experiment; result = run_experiment("configs/v1.toml")`.
- Tests (synthetic data only, they never read `data/`): `uv run pytest`.
