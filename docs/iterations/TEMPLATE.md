# VN: short title

Copy this file to `vN.md` (e.g. `v1.md`) and link it from the iterations table in [the roadmap](../roadmap.md).

- **Branch / tag:** `vN` branch, submission commit tagged `vN`
- **Run folder:** `runs/<timestamp>_<name>/`
- **Config:** `configs/vN.toml`

## What we tried

What changed relative to the previous iteration: features, model, post-processing, data.

## Why it should help

The geological or statistical reason to expect a better local score. For example, what a fault trace or fault scarp looks like in these feature bands, or what the model could not see before.

## Scores

| | Mean | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|---|
| Local score, raw | | | | | | |
| Local score, post-processed | | | | | | |

- **Tuned post-processing parameters:**
- **Leaderboard score:** (date submitted)
- **Previous iteration, for comparison:** local score, leaderboard score

How much did post-processing contribute? How large is the gap between local and leaderboard scores, and what might explain it (ADR 0001)?

## Preview maps

The run's preview image, plus anything else worth looking at. Copy images into `docs/iterations/vN/`; the run folder itself is not in git.

## What we learned

About the geology, the data, the model and the metric.

## Next ideas

Candidates for the next iteration, checked against the known limitations in [the roadmap](../roadmap.md#known-limitations--improvement-ideas).
