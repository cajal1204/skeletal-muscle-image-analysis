# Image-analysis methods

This document describes the analysis workflows included in this repository. Numeric values are the default settings in the scripts. Calibration and fluorescence thresholds should match the image set being analyzed.

## Image preparation and segmentation

Fluorescence channels are exported as TIFF images. Myofibers are segmented using Cellpose-SAM with a cell probability threshold of 1 and otherwise default settings. The segmentation channel is selected according to the image set. Nuclear segmentation is performed with StarDist before running these scripts. The repository begins with the exported masks and fluorescence images.

## Fiber areas

The two DAPI assignment scripts retain raw areas and create eroded-area columns. Their main `fiber_area_px` and `fiber_area_um2` columns use the eroded area. The erosion distance is derived from a reference pixel width of 1.2886 µm divided by the configured pixel width and rounded to an integer; it equals one pixel at the supplied calibration.

`calculate_csas.py`, `measure_fibers.py`, the RFP-only script, and the combined RFP/eMHC script measure the input label areas without additional erosion. Record any preprocessing applied to input masks. Area conversion is pixel count multiplied by the square of the configured pixel width in µm. The configured calibration is generally 1.2886 µm/pixel; the command-line combined script requires an optional calibration argument for µm² outputs. Set calibration using the image scale bar.

## DAPI assignment and central nucleation

Both DAPI assignment scripts use the fiber label beneath the rounded nucleus centroid, require at least 50% of nuclear area to overlap that fiber, and apply size and fiber-QC filters. The supplied reference nuclear-size limits differ: 15–500 pixels in `assign_nuclei_to_fibers.py` and 10–175 in the eMHC counterpart, with physical-area scaling when calibration changes.

The initial central classification requires centroid distance **≥8 µm** from the computed fiber boundary. A local laminin check uses a 1 µm neighborhood and a maximum local-to-membrane signal ratio of 0.6. An additional permissive classification combines segmented central nuclei with DAPI core signal: the source uses a threshold at the 95th percentile of nonzero DAPI pixels and requires at least 10 positive core pixels and at least 1% core coverage, subject to fiber QC. The chosen strict or permissive output column must be recorded for each analysis.

The dystrophin-associated assignment script includes minimum fiber area and laminin membrane coverage filters. The eMHC counterpart does not apply those same filters, although it retains the laminin core-to-membrane and manual-exclusion checks. The numeric settings are listed near the top of each script.

## RFP nucleus assignment

`assign_rfp_nuclei_to_fibers_ipax3dko.py` assigns each RFP-labeled nucleus by rounded centroid with no minimum-overlap filter. The combined RFP/eMHC script instead selects the fiber with greatest nuclear overlap and defaults to a minimum fraction of 0.5. Both RFP workflows count assigned nuclei per fiber and identify fibers with at least one. The code assumes that the supplied nuclear labels represent RFP-positive objects; it does not independently establish fluorescence specificity.

## Dystrophin scoring

`score_dystrophin_by_fiber.py` measures an internal membrane band with a default width of 3 µm. An optional exterior band is disabled. Fixed-mode pixel intensity cutoff is 50; pixels must exceed the cutoff. A fiber is positive if band QC passes, at least 50% of band pixels exceed the cutoff, and the background-corrected band mean is at least 0. The fixed negative baseline is 15. A negative-fiber-derived threshold mode is also available. These values depend on the acquisition/export intensity scale.

## Mean-based eMHC scoring

`score_emhc_by_fiber.py` and `analyze_rfp_nuclei_and_emhc_by_fiber.py` default to a 3-pixel boundary exclusion. If fewer than **5 core pixels** remain, the full fiber is scored. Fixed-mode classification requires a scored-region mean ≥20 and background-corrected mean ≥0, with negative mean 15 and a nonempty scored region. An alternative threshold can be calculated from negative fibers. Bright-pixel coverage is reported but is not a minimum-percentage positivity requirement in these versions.

## Review and export

The collection includes QC overlays, individual-fiber views, interactive artifact exclusion, and marker/central-nucleation correction tools. Available filters vary by workflow. CSV tables contain image/fiber measurements and classifications; correction records and final tables should be retained together. Image-level measurements can be combined for subsequent analysis. Record the final QC filters, selected classification columns, and biological-replicate grouping used in downstream analyses.
