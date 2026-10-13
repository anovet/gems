# Spatial folds, not the leaderboard, are the primary quality signal

We judge every change by its **local score**: the DTI on held-out known faults, averaged over 5 **spatial folds** of roughly 20 km blocks. A random pixel split would put adjacent pixels of the same fault trace on both sides of the split and grossly overstate quality. The leaderboard allows only 3 submissions a week and covers only the public test chunk, so it serves as a cross-check, not the signal we optimise against.

## Consequences

- The local score measures how well we recover *known* faults in unseen ground. That only stands in for the real target, *new* faults, which may be subtler (smaller offsets, weaker topographic expression). A gap between local and leaderboard scores is expected and is worth investigating, not "fixing" by tuning to the leaderboard.
- Changing the fold scheme invalidates comparisons with earlier runs, so it should be rare and deliberate.
