# GEMS

Our entry to the DOE GEMS Prize: predicting where geologic faults lie at the surface across the GeoDAWN region of western Nevada and eastern California. Faults matter because they channel geothermal fluids, but faults, not geothermal resources, are what we predict.

## Geology

**Fault**:
A fracture in the Earth's crust along which the rock on either side has measurably moved.
_Avoid_: Crack, fracture (unqualified)

**Fault trace**:
The line where a fault plane meets the ground surface; this is the prediction target.
_Avoid_: Fault line, surface fault

**Fault zone**:
A cluster of closely spaced, related fault traces treated as one structure.

**Fault scarp**:
A step in the land surface created by movement on a fault; the main topographic evidence of a fault trace.

**Lineament**:
A straight or gently curved linear feature in a map or image (topography, magnetics, gravity) that may, but need not, mark a fault trace.

**Quaternary fault**:
A fault that has moved within roughly the last 2.6 million years; the USGS fault database that supplies our labels catalogs these.

## Data

**GeoDAWN region**:
The fixed study area of the challenge: the extent of the GeoDAWN airborne magnetic and radiometric surveys, gridded at 100 m.
_Avoid_: AOI, the map, the tile

**Study area**:
The pixels of the GeoDAWN region that have data; pixels outside it are nodata and are neither predicted nor scored.
_Avoid_: Valid mask (in prose), footprint

**Feature band**:
One of the provided gridded measurements of the region (for example magnetic anomaly, gravity slope, strain rate, earthquake density).
_Avoid_: Layer, channel, input

**Known fault**:
A fault trace already in the public USGS / INGENIOUS fault database, supplied to us as training labels.
_Avoid_: Existing fault, training fault, ground truth (unqualified)

**New fault**:
A fault trace identified by the challenge's experts that is absent from the known faults; the hidden test labels consist of these.
_Avoid_: Hidden fault, test fault, unknown fault

**DEM**:
A digital elevation model: a grid of ground-surface heights, here the optional 1 m lidar-derived USGS tiles covering the region.

## Prediction and scoring

**Fault probability map**:
A raster over the GeoDAWN region giving, for each pixel, our confidence from 0 to 1 that a fault trace passes through it.
_Avoid_: Heatmap, mask, prediction (unqualified)

**Submission**:
A fault probability map in the exact format the challenge accepts, uploaded to the leaderboard.

**Distance-weighted Tversky index (DTI)**:
The challenge metric: an overlap score between a fault probability map and labelled fault traces that forgives positional error up to 300 m and penalises missed faults four times more heavily than false alarms.
_Avoid_: Tversky score, accuracy, IoU

**Distance tolerance**:
The 300 m radius within which a prediction near a labelled fault trace earns partial credit, linearly less the farther away it is.
_Avoid_: Buffer, kernel (in prose)

**Initial Prize Round**:
The first scoring of the final submission, against the private test set of new faults.

**Final Prize Round**:
The second scoring of the same final submission, against a label set the experts expand after reviewing every team's submissions.

**Public test set** / **Private test set**:
The two disjoint chunks of the GeoDAWN region whose new faults score submissions; the public chunk drives the visible leaderboard, the private chunk decides the Initial Prize Round.

## Our process

**Local score**:
The DTI we compute ourselves on held-out known faults, averaged across spatial folds; our primary measure of progress.
_Avoid_: CV score, validation accuracy

**Post-processing**:
The tuned mapping from a model's raw fault probabilities to the values submitted, chosen to maximise the local score; local scores are reported both raw and post-processed.
_Avoid_: Transform, calibration

**Leaderboard score**:
The DTI the challenge reports for a submission on the public test set.
_Avoid_: Public score, test score

**Spatial fold**:
One of the groups of contiguous blocks of the region held out together, so that a model is never evaluated on ground adjacent to what it trained on.
_Avoid_: Fold (unqualified), split

**Iteration**:
One planned, measurable improvement to the entry (V1, V2, …), ending in a submission and a written record of what was learned.
_Avoid_: Version, release, sprint

**Steel thread**:
The first iteration (V1): the simplest end-to-end path from feature bands to a valid submission with a local score.
_Avoid_: MVP, baseline (when meaning the whole pipeline)
