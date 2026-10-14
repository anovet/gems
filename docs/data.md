# Data facts and gotchas

Checked against the real files in `data/` on 2026-10-14. `README.md` lists the files and where to download them; this page covers what isn't obvious from opening them.

## The shared grid

All three rasters share one grid: EPSG:32611 (UTM 11N), 100 m pixels, 3730 rows × 3292 cols, bounds (left, bottom, right, top) = 243350, 4135550, 572550, 4508550. A submission must match this grid exactly.

## Feature bands (`gems-geodawn-numerical-features.tif`)

- There are 19 float32 bands. Each band has a name in its `band_name` tag; read bands by name, never by index. The per-band `data_category` and `description` tags explain what each band measures. Some descriptions hedge (for example `tc` is "Tilt angle or total curvature", and `deq_n100a15` is "Distance to earthquake"). `docs/agents/background.md` describes the earthquake bands as earthquake *density*. Treat the descriptions as hints.
- Nodata is the most negative float32 (about −3.4028e38), not NaN. Convert it to NaN when reading, or it will look like an extreme measurement.
- Value ranges differ by many orders of magnitude between bands (the earthquake bands reach about 5e6, while most bands stay below 1). This matters for any model that isn't tree-based.

## Labels and the study area (`existing_faults.tif`)

- The labels are int8: 1 = known fault, 0 = no fault, −1 = nodata.
- The **study area** is the label raster's valid pixels: 5,167,373 pixels, 42% of the grid. Known faults cover 60,988 pixels, which is 1.2% of the study area.
- The feature and label masks almost, but not exactly, agree. 3,061 study-area pixels have no feature data at all (another 12 are missing only `tc`), so they must still receive a prediction from NaN inputs. Another 1,540 pixels have features but lie outside the study area; they must be NaN in a submission.

## The sample submission (`example_submission.tif`)

It is the known-fault labels cast to float32, with NaN outside the study area. The problem page calls it a prediction of "total fault absence", but it isn't. It scores 0 only because known faults are excluded from the Initial Prize Round scoring.

## Corrections to `docs/agents/background.md`

That file is copied from the challenge pages and doesn't match the files we were given:

- It names the features raster `training_features.tif`; the actual file is `gems-geodawn-numerical-features.tif`.
- It names the DEM link list `1m_DEM_links.csv`; the actual file is `tnm_items.json`, a JSON array of 894 URLs.
- Its "Data Files" list describes `gems-geodawn-numerical-features.tif` twice. The second entry is really `tnm_items.json`.
- The competition homepage gives the deadline as Dec 3, 2026 at 11:59 p.m. UTC, but the official rules say 5:00 p.m. ET. Plan for the earlier time.
- Leaderboard submissions are limited to 3 per week.
