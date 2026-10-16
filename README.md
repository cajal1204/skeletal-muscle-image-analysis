# Skeletal Muscle Image Analysis

Python tools for quantifying myofibers, nuclear localization, and marker expression in skeletal muscle immunofluorescence images.

This repository contains the image-analysis and visual-review tools used by Ethan Murphy in the Perlingeiro Lab at the University of Minnesota. The scripts use previously generated segmentation masks to analyze fiber cross-sectional area (CSA), central nucleation, RFP-positive nuclei, dystrophin, and embryonic myosin heavy chain (eMHC).

## What the tools do

- Measure fiber number and cross-sectional area.
- Assign nuclei to fibers and identify central nuclei.
- Count RFP-positive nuclei within individual fibers.
- Classify dystrophin and eMHC expression.
- Generate labeled overlays and individual-fiber views.
- Support manual artifact exclusion and classification correction.
- Export per-fiber, per-nucleus, and image-level CSV tables.

## Getting started

1. Before running these scripts, segment myofibers with [Cellpose-SAM](https://github.com/MouseLand/cellpose) and nuclei with [StarDist](https://github.com/stardist/stardist), as needed for the selected analysis.
2. Prepare aligned fluorescence channels and exported label masks, and install the [analysis dependencies](docs/SETUP.md#analysis-dependencies).
3. Choose a workflow below and set the input paths, image calibration, and analysis thresholds.
4. Run the analysis, review the results, and export the final tables.

## Choose a workflow

| Analysis | Scripts | Guide |
| --- | --- | --- |
| DAPI nuclear localization and dystrophin expression | `assign_nuclei_to_fibers.py` → `score_dystrophin_by_fiber.py` | [Run this workflow](docs/SETUP.md#dapi-plus-dystrophin-one-image) |
| DAPI nuclear localization and eMHC expression | `assign_nuclei_to_fibers_emhc.py` → `score_emhc_by_fiber.py` | [Run this workflow](docs/SETUP.md#dapi-plus-emhc-one-image) |
| RFP-positive nuclear counts | `assign_rfp_nuclei_to_fibers_ipax3dko.py` | [Run this workflow](docs/SETUP.md#rfp-counts-only) |
| RFP-positive nuclear counts and eMHC expression | `analyze_rfp_nuclei_and_emhc_by_fiber.py` | [Run this workflow](docs/SETUP.md#combined-rfp-and-mean-based-emhc) |
| Fiber CSA, one image | `calculate_csas.py` | [Run this workflow](docs/SETUP.md#csa-only-workflows) |
| Fiber CSA and counts, multiple images | `measure_fibers.py` → `summary.py` | [Run this workflow](docs/SETUP.md#csa-only-workflows) |

The workflows are intended for different image sets. Their filenames, settings, and measurement rules are documented in the guides; they are not interchangeable stages of one pipeline.

## Inputs and outputs

Each image folder contains aligned fluorescence channels and segmentation masks. Masks use **0 for background** and a unique positive integer for each fiber or nucleus. Use label images, not colored segmentation overlays. Preserve the fluorescence intensity scale and record the pixel width in µm.

Outputs include CSV tables of fiber measurements, nuclear assignments, marker classifications, and QC flags, together with review images where applicable. Keep each fiber ID associated with its source image when combining tables.

## Documentation

| Document | Contents |
| --- | --- |
| [Setup and execution](docs/SETUP.md) | Dependencies, input names, commands, and outputs |
| [Detailed methods](docs/METHODS.md) | Segmentation, measurement, classification, and review procedures |
| [Script inventory](docs/SCRIPT_INVENTORY.md) | All 23 scripts and their purposes |

Most scripts are configured through the user-settings section near the top of the file. The combined RFP/eMHC script also provides command-line options. 

## Software

Analysis uses NumPy, pandas, SciPy, scikit-image, Matplotlib, Pillow, and tifffile. Cellpose-SAM and StarDist are upstream prerequisites; this repository covers analysis after segmentation. Fiji is optional for opening individual-fiber review images.

## Attribution and reuse

Custom analysis scripts: Ethan Murphy, Perlingeiro Lab, University of Minnesota. Please cite the applicable Cellpose-SAM and StarDist publications when using these tools. 

