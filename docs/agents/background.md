# Intro

Welcome to the Geologic Enhanced Mapping System (GEMS) Prize Challenge!
U.S. geothermal resources can be harnessed for power production and heating and cooling without importing fuel. One major hurdle in this process is the identification of new hidden geothermal prospects. The Geologic Enhanced Mapping System (GEMS) Prize, presented by the U.S. Department of Energy’s Office of Geothermal, will prioritize efforts to accelerate the ongoing, detailed geologic mapping of the United States, with a focus on locating previously unknown deposits of critical minerals and geothermal resources.

The GEMS Prize seeks to address the challenge of creating an algorithm that will generate enhanced geologic fault datasets to support geologic and geophysical mapping and modeling that will assist geothermal and critical mineral studies. This work is vital to accelerate discoveries of new, commercially viable hidden geothermal systems while reducing the exploration and development risks for all geothermal resources.

The same submission is scored twice — once against a private expert-labeled test set for the Initial Prize Round, and again against an expanded label set built from expert review of all submissions for the Final Prize Round. See the competition structure section of the problem description for a full walkthrough.

A reference solution is available at https://github.com/drivendataorg/gems-prize-reference-solution

Use of test data
A unique feature of this challenge is that there is spatial overlap between the training dataset (the USGS quaternary fault dataset) and the test dataset (newly identified faults within the GeoDAWN region). For this challenge, you may use the provided set of faults for training purposes. Participants will be evaluated based on their performance predicting faults contained in a newly labeled, private set of faults.

Use of external data
The use of external data is encouraged in this competition, provided you or your team possess the necessary licenses to use the data for this challenge. All data used must be shareable with the challenge organizers to allow for independent result verification. For more information, see the complete rules document.

# Problem description

For this challenge, your task is to develop models and algorithms that provide accurate information about the presence of structures that are indicative of geothermal resources—namely, geological faults.

A unique feature of this challenge is that ground truth data for currently known faults is publicly available! While detailed, it is known that this set of faults is not complete and may even contain some inaccurate data.

For this challenge, we have consulted with fault experts who have manually identified faults that are not contained within the current public USGS database. These new faults will comprise the test dataset for the initial prize round of the challenge. After the initial prize round, an expert panel will use submitted predictions to update fault labels for the full region. Submissions will then be rescored against the entire updated label set for the final round awards.

## Competition Structure

This competition has a unique prize structure to address the challenge of incomplete and potentially inaccurate ground truth data, and to encourage participants to submit predictions for what they truly believe are faults in the region as opposed to simply optimizing for the existing labels.

For this competition, the sponsors have compiled a set of newly identified faults in the GeoDAWN region that are not included in the existing USGS fault database. These new faults have been manually labeled by fault experts and comprise the test dataset for the competition. The GeoDAWN region is chunked and split into a public test set and a private test set. Competitors' performance against the public test set is shown on the public leaderboard during the competition.

In the Initial Prize Round, submissions are evaluated by their performance on the private test set at the close of the competition. An expert panel then uses submitted predictions to update fault labels for the full region. Submissions are rescored against the entire updated label set for the Final Prize Round awards. Competitors must choose a single submission for scoring across both rounds before the deadline, without knowing their private test set performance.

## Datasets

The challenge focuses on the region covered by the GeoDAWN study. GeoDAWN (Geoscience Data Acquisition for Western Nevada) consists of a set of high-resolution magnetic and radiometric surveys conducted to support geologic and geophysical mapping and modeling.

The datasets provided by this challenge come from the United States Geological Survey and from the Great Basin Center for Geothermal Energy’s INGENIOUS project. We will provide some of the data for this challenge directly via the data download page, including data from the GeoDAWN study and the ground truth fault vector and raster data from the USGS. We will also provide a list of URLs where the high-resolution Digital Elevation Model (DEM) data can be downloaded.

### Provided Features

On the data download page, you will find a GeoTIFF file called training_features.tif. This GeoTIFF is in a projected coordinate system for UTM zone 11N (EPSG 32611) at 100m resolution. The GeoTIFF has many layers:

Surface conductivity and depth to conductive base surface
Detrended elevation and the slope of detrended elevation
Dilatation rate, shear strain rate, and the second invariant of the strain rate tensor
Isostatic gravity anomaly and the slope of the isostatic gravity anomaly
Magnetics including reduced-to-pole magnetic anomaly, total magnetic intensity, the vertical and horizontal slope of total magnetic intensity, and the top-of-crustal magnetic source depth estimate
Density of earthquakes

In addition, you will find a CSV file called 1m_DEM_links.csv that contains links where DEM data at 1m resolution can be downloaded.

### Labels

The labels for this challenge come from the USGS quaternary fault maps and from INGENIOUS. Labels are provided in both vector and raster formats on the data download page.

### External datasets

Participants are allowed to use any additional data sources, provided that the participants possess a license that permits the data to be used in this challenge and shared with the sponsor for evaluation purposes. For more information, see the complete rules document.

## Performance metric

The performance metric for this competition is a distance-weighted Tversky index. The Tversky index is a similarity metric parameterized by coefficients alpha and beta representing the penalty term applied, respectively, to false positives and false negatives.

For this challenge, you will submit a GeoTIFF raster with fault predictions represented as pixel-wise probabilities or confidence scores between 0 and 1. Rasterization is lossy and can induce off‑by‑one errors near pixel boundaries. In addition, portions of the existing fault data may be misaligned from the true location of the surface fault, which is the prediction target. To mitigate these effects, we weight the contributions of true positives, false negatives, and false positives by the distance to the nearest ground truth pixel using a linear (triangular) kernel with 300 m support.

### Mathematical representation

Let 𝑝(𝑥) ∈[0,1] denote the predicted probability of a fault at pixel 𝑥, and let 𝑔(𝑥) denote the ground truth label of pixel 𝑥.

The Tversky index 𝑇𝐼 is defined as:

TI(𝛼,𝛽)=
∑𝑥𝑝(𝑥)𝑔(𝑥)
∑𝑥𝑝(𝑥)𝑔(𝑥)+𝛼∑𝑥𝑝(𝑥)(1−𝑔(𝑥))+𝛽∑𝑥(1−𝑝(𝑥))𝑔(𝑥)

where 𝛼 and 𝛽 are non-negative parameters that control the penalty applied to false positives and false negatives, respectively.

To compute a distance-weighted Tversky index, we define a triangular kernel 𝑘 as
𝑘(𝑑)=(1−𝑑/𝑅)+=max(1−
𝑑
𝑅

,0).

where the range 𝑅 is 300 meters (i.e., 3 pixels at 100m resolution). This kernel is linearly larger when distances are smaller and vice versa.

For probabilistic predictions, we compute each term in the Tversky index in a distance-weighted manner:

distance-weightedtruepositivecountTP𝑤=
∑
𝑔∈𝐺

max
𝑥:𝑑(𝑥,𝑔)≤𝑅
𝑝(𝑥)𝑘(𝑑(𝑥,𝑔))
distance-weightedfalsepositivecountFP𝑤=
∑
𝑥:𝑝(𝑥)>0
𝑝(𝑥)[1−
max
𝑔∈𝐺
𝑘(𝑑(𝑥,𝑔))]
distance-weightedfalsenegativecountFN𝑤=
∑
𝑔∈𝐺
[1−
max
𝑥:𝑑(𝑥,𝑔)≤𝑅
𝑝(𝑥)𝑘(𝑑(𝑥,𝑔))]

and the distance-weighted Tversky index is given by

DTI(𝛼,𝛽)=
TP𝑤
TP𝑤+𝛼FP𝑤+𝛽FN𝑤+𝜖

For this competition, we set 𝛼 =0.2 and 𝛽 =0.8, which reduces the penalty for false positive predictions and increases the penalty for false negative predictions.

## Submission Format

For this competition, you will submit a GeoTIFF file containing your predictions for all faults in the region.

Your submitted GeoTIFF should meet the following requirements:

Your submission is in the same projected coordinate reference system as the training data (projected coordinate system for UTM zone 11N, EPSG 32611)
Your submission is at the same resolution as the training data (100m)
Your submission has the same bounds as the training data, and data outside the bounds is null or nan.
Your submission contains a single layer with datatype of 32-bit float (float32) with values between 0 and 1 indicating the confidence or probability of fault presence, with higher values indicating higher probability.
A sample submission that predicts total fault absence is provided for your reference on the data download page. You can use this as a template to ensure that your submission is correctly formatted.

# About

## About the data

The U.S. Geological Survey (USGS) and the Department of Energy (DOE) have collaborated to acquire high-resolution airborne magnetic and radiometric data over northern and western Nevada and eastern California to support geologic and geophysical mapping and modeling that will assist geothermal and critical mineral studies. The surveys, referred to as GeoDAWN (Geoscience Data Acquisition for Western Nevada), span areas of major resource potential associated with the Walker Lane and western Great Basin.

They were conducted under the USGS’s Earth Mapping Resources Initiative (EarthMRI), with support from the DOE’s Office of Geothermal (OG), and involved acquisition of aeroradiometric and aeromagnetic data that provide key information on surface geology and soil composition and subsurface structure and geology, respectively. Coordinated with this effort was the collection of airborne lidar (light detection and ranging) data (conducted through the USGS 3D Elevation Program (3DEP)) that yielded detailed surface topographic models of the terrain over a similar extent spanned by the geophysical surveys.

## About the task

A geological fault is a fracture or discontinuity in Earth’s crust where two blocks of rock have undergone measurable displacement, whether vertical, horizontal, or a combination of both. These fractures can range from tiny features to vast systems spanning hundreds of miles, such as plate boundary faults that generate major earthquakes. The plane along which the displacement occurs is called the fault plane, and where this plane intersects the surface is known as the fault trace. When multiple related fractures exist in close proximity, geologists refer to this as a fault zone.

Geologists detect and delineate faults using a combination of traditional field observations, geophysical surveys, and advanced remote sensing and imaging techniques. In the field, they search for surface expressions such as fault scarps, offset rock layers, and polished slickensides. Subsurface methods like seismic reflection profiling can visualize hidden fault planes, especially where dense geophysical contrasts exist. Gravity and magnetic surveys also help infer the presence of faults through density or magnetic anomalies. From above, remote sensing data and elevation data reveal linear features and subtle ground movements indicative of fault activity. Furthermore, modern computational methods including edge detection, Hough transforms, and deep learning on seismic or topographic datasets further enhance automated fault mapping.

Detecting faults accurately is vital across multiple domains. In earthquake science and hazard assessment, recognizing active faults—those that have moved within the past ~11,700 years—is crucial for predicting seismic risks and enforcing safe land-use planning near fault lines. In engineering and construction, identifying fault zones guides the placement of infrastructure, ensuring stability and mitigating ground rupture hazards. Faults also play key roles in natural resource processes: they can act as conduits for fluids, concentrating minerals, hydrocarbons, geothermal energy, and groundwater, thus guiding exploration and extraction strategies.

## Additional information

Mattéo, L., Manighetti, I., Tarabalka, Y., Gaucel, J.-M., van den Ende, M., Mercier, A., et al. (2021). Automatic fault mapping in remote optical images and topographic data with deep learning. Journal of Geophysical Research: Solid Earth, 126, e2020JB021269. https://doi.org/10.1029/2020JB021269
Hermant, B., Kiersnowski, L., & Bellanger, M. (2025, February). Using deep learning to map Quaternary faults in Western USA. In Proceedings of the 50th Workshop on Geothermal Reservoir Engineering (Stanford, CA). Stanford University. https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf

# Relevant Links

https://www.drivendata.org/competitions/306/competition-doe-gems/page/966/
https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and
https://gbcge.org/current-projects/ingenious/
https://www.herox.com/GEMSPrize

# Data Files

data\example_submission.tif: A file demonstrating the required submission format containing known quaternary faults from INGENIOUS. Submitting this file will score zero because known faults are excluded from evaluation.

data\gems-geodawn-numerical-features.tif: Numerical feature data

data\existing_faults.tif Training fault labels from GDR / Ingenious - Quaternary faults v2

data\gems-geodawn-numerical-features.tif: Digital elevation model links JSON with links to download digital elevation model (DEM) data at 1m resolution directly from the USGS TNM.
