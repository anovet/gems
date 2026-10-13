---
status: accepted (deferred to V4)
---

# Defer the 1 m DEM; use only the provided 100 m feature bands until V4

The challenge supplies 894 links to 1 m lidar DEM tiles, likely hundreds of GB in total. Fault scarps are the most direct surface evidence of fault traces, so terrain derivatives of this DEM (slope, curvature, hillshade and scarp-detection filters, aggregated to 100 m) are probably the largest single source of untapped signal. Ingesting it is a data-engineering project of its own, so it waits until the modelling pipeline is proven. The provided bands already include detrended elevation and its slope at 100 m. Anyone proposing improvements should raise this as a major candidate.
