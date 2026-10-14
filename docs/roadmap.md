# Roadmap

The plan for our entry, from the steel thread to the final submission. Deadline: Dec 3, 2026 (plan for 5 p.m. ET). The leaderboard allows 3 submissions a week.

Each iteration is one planned, measurable improvement judged by its local score (ADR 0001), and it ends in a submission and an iteration record in [`iterations/`](iterations/) (start from [the template](iterations/TEMPLATE.md)).

## Iterations

| Iteration | What it adds | Record |
|---|---|---|
| **V1**, steel thread | Per-pixel gradient-boosted trees on the 19 raw feature bands; spatial folds; our own DTI scorer; post-processing tuned on the local score; a validated submission. | _not yet written_ |
| **V2** | Multi-scale neighbourhood and lineament features: local gradients, edge and lineament filters, focal statistics over about 300 m, 1 km and 3 km. | |
| **V3** | A U-Net like the [reference solution](https://github.com/drivendataorg/gems-prize-reference-solution), trained on the local GPU and compared against V2. | |
| **V4** | Features derived from the 1 m DEM: slope, curvature, hillshade and scarp-detection filters, aggregated to 100 m. | |

## Stretch goals

- **Ensembling**: combine the fault probability maps of several iterations or models.
- **Revisit known-fault handling** ([ADR 0002](adr/0002-keep-predictions-on-known-faults.md)): we keep model predictions on known-fault pixels. Forcing them to 1 for the Final Prize Round was deferred, not rejected.

## Known limitations / improvement ideas

These are deliberate shortcuts. Raise them first when discussing how to improve the model or pipeline.

- **Per-pixel features only** ([ADR 0003](adr/0003-steel-thread-uses-per-pixel-features-only.md)). V1 looks at each pixel in isolation. A fault trace is a linear feature, and a single pixel cannot see a line, a lineament or a fault scarp. This is the first candidate for improvement, planned for V2 (neighbourhood features) and V3 (U-Net).
- **No 1 m DEM** ([ADR 0004](adr/0004-defer-1m-dem.md)). Fault scarps are the most direct surface evidence of fault traces, and the DEM is probably the largest untapped source of signal. It is deferred to V4 because ingesting it is a data-engineering project of its own.
- **Post-processing tuning is mildly optimistic.** The same spatial folds that are scored also choose the post-processing parameters; there is no nested cross-validation.
- **The local score measures known faults, not new faults** ([ADR 0001](adr/0001-spatial-folds-are-the-primary-quality-signal.md)). New faults may be subtler, so expect a gap between local and leaderboard scores. Investigate the gap rather than tuning to the leaderboard.

## Final-submission selection rule

From [ADR 0005](adr/0005-final-submission-selection-rule.md): choose the submission with the best local score, unless its leaderboard score is clearly worse than the runner-up's. Exactly one final submission is scored in both the Initial and Final Prize Rounds, and it is chosen without seeing private scores. The results CSV (see the README) puts both scores side by side.
