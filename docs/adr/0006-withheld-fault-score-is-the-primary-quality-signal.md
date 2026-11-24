# The withheld-fault score, not spatial folds, is the primary quality signal

Supersedes ADR 0001 as the measure we optimise against. V1 scored a local score of 0.161 (spatial folds) but a leaderboard score of 0.0122. Spatial folds score *known* faults in ground the model never saw. The leaderboard does the reverse: it scores *new* faults, with known faults excluded, after the model has trained on every known fault, including those in the scored ground.

We now judge every change by the **withheld-fault score**, which imitates the leaderboard. Each round works like this:
- Known-fault traces (8-connected components) are dealt into 5 groups, in 10 km cluster blocks.
- For each group, a model trains on the whole study area with that group labelled as non-fault.
- Its map is scored only on that group, with the other known faults excluded.

Every score is reported next to the score of a constant map of 1, the no-skill baseline. Post-processing is tuned on the withheld-fault score. The spatial-fold local score is still reported, so runs stay comparable with V1.

## Considered options

Each option was scored with the V1 model and post-processing (2026-11-24, issue #11):

| Scheme | V1 score | Constant map of 1 |
|---|---|---|
| Spatial folds (ADR 0001) | 0.161 | 0.058 |
| Traces withheld one by one | 0.052 | 0.012 |
| Traces withheld in 3 km clusters | 0.038 | 0.012 |
| **Traces withheld in 10 km clusters** | **0.016** | 0.012 |
| Leaderboard | 0.0122 | – |

We chose 10 km clusters because they come closest to the leaderboard. Withheld traces then lie about 3 km from the nearest given fault on average, which fits new faults being ones nobody has mapped near known ones. Traces withheld one by one sit inside known fault zones, which flatters a model that has learned "fault zones" rather than fault traces.

## Consequences

- Scores from before this ADR are not comparable. V1 has been re-run under it.
- A run trains 5 more models (on the whole study area) and takes a few minutes instead of under one.
- ADR 0005's selection rule now reads "best withheld-fault score".
- The cluster size is a config setting. Revisit it if the leaderboard score of the constant map shows a very different density of new faults.
