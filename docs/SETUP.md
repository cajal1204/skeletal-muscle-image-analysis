# Running the analysis scripts

## Prior steps

Generate fiber label masks with Cellpose-SAM and, for nuclear analyses, nucleus label masks with StarDist before running these scripts. In our workflow, Cellpose-SAM used a cell probability threshold of 1 with otherwise default settings. The fluorescence channel used for fiber segmentation depended on the image set. Export the masks and fluorescence channels for downstream analysis. Installation and operation of the segmentation software are covered by the [Cellpose](https://github.com/MouseLand/cellpose) and [StarDist](https://github.com/stardist/stardist) documentation.

## Analysis dependencies

The custom scripts use NumPy, pandas, SciPy, scikit-image, Matplotlib, Pillow, and tifffile. Install these in your Python environment from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Prepare images

Use aligned 2D images and integer label masks with matching shape. Keep one image's channels and masks together. Fluorescence should preferably be exported as individual grayscale TIFF channels. Some scripts accept RGB images but choose different channels, so do not assume arbitrary RGB exports are interchangeable.

Keep original data and final results separately from reruns. Several scripts overwrite tables with fixed filenames. Run only the relevant workflow in a working copy of the image folder.

## DAPI plus dystrophin, one image

Required files: `fibers.tif`, `nuclei.tif` (DAPI labels), `laminin.tif`, `dapi.tif`, and `dystrophin.tif`.

Copy the needed scripts into that image's working folder. Edit their top-level user settings, including `pixel_size_um` in both assignment and scoring scripts. From that folder run:

```bash
python assign_nuclei_to_fibers.py
python score_dystrophin_by_fiber.py
```

The first script writes `image02_nucleus_assignments.csv` and `image02_fiber_summary.csv`; the second writes `image02_fiber_summary_with_dystrophin.csv`. `image02` is a fixed filename prefix, not an automatically detected image ID.

Optional review tools: `make_fiber_id_overlay.py`, `make_all_fibers_central_dystrophin_overlay.py`, `make_qc_overlay_on_image.py`, and `lookup_fiber_by_id.py`. Correction tools: `toggle_dystrophin_status.py`, `toggle_manual_fiber_exclusion.py`, and `mark_fiber_not_centrally_nucleated.py`.

## DAPI plus eMHC, one image

Use `fibers.tif`, DAPI `nuclei.tif`, `laminin.tif`, `dapi.tif`, and `emhc.tif`. Copy the matching scripts into the working image folder and configure the settings:

```bash
python assign_nuclei_to_fibers_emhc.py
python score_emhc_by_fiber.py
```

Outputs are `image02_nucleus_assignments.csv`, `image02_fiber_summary.csv`, `image02_fiber_summary_with_emhc.csv`, and, by default, `image02_emhc_scored_region_labels.tif`. Use the corresponding `_emhc.py` lookup, overlay, and correction tools. This workflow measures bright-pixel coverage but does not use a minimum coverage percentage to decide eMHC positivity.

## RFP counts only

Set `root_dir`, `fiber_mask_name`, `nuclei_mask_name`, and calibration in `assign_rfp_nuclei_to_fibers_ipax3dko.py`. Actual defaults require `membrane_cp_masks.png` and `nuclei.tif` containing RFP nucleus labels. Use the filenames configured in the user-settings section. Run from the experiment root if retaining `root_dir = Path(".").resolve()`:

```bash
python /absolute/path/to/repository/scripts/assign_rfp_nuclei_to_fibers_ipax3dko.py
```

Results are collected under `ipax3dko_rfp_nucleus_results/`. This script does not apply a 50% overlap requirement.

## Combined RFP and mean-based eMHC

Required per image: `laminin_cp_masks.png`, `rfp_stardist_labels.tif`, and `emhc.tif`. The script also recognizes several alternate nucleus-mask extensions and the historical `starfist` spelling. From the repository root:

```bash
python scripts/analyze_rfp_nuclei_and_emhc_by_fiber.py --help
python scripts/analyze_rfp_nuclei_and_emhc_by_fiber.py /absolute/path/to/experiment --no-scored-region-mask
```

The second command uses the supplied defaults and omits only the optional QC mask output. Supply `--pixel-size-um` with the actual pixel width to add calibrated areas. It is one value per invocation: process datasets with different calibration separately. Other options control filenames, thresholds, overlap, and output location.

Outputs include per-image `combined_fiber_summary.csv` and `combined_rfp_nucleus_assignments.csv`, plus collected results under `combined_rfp_emhc_results/`. Inspect the reported errors and number of successfully processed folders; batch completion can include skipped failures.

## CSA-only workflows

For `calculate_csas.py`, place `fibers.tif` in a working folder, copy the script there, set `pixel_size_um`, and run `python calculate_csas.py`. Output: `fiber_cross_sectional_areas.csv`.

For batch measurement, set `ROOT_FOLDER`, `MASK_FILENAME`, and `PIXEL_SIZE_UM` in `measure_fibers.py`. The default mask name is `dystrophin_cp_masks.png`. Run it, then configure the same root in `summary.py` and run that script. Outputs include `dystrophin_fiber_cross_sectional_areas.csv`, `individual_image_CSA_csvs/`, and `dystrophin_fiber_counts_by_image.csv`. Minimum-area QC and border flags are reported; verify which filtered columns were actually used downstream. These scripts do not add mask erosion.

## Manual review

Lookup and correction tools prompt for fiber IDs in a terminal. Lookup tools include a Windows-specific Fiji launcher: set `open_png_in_fiji = False` on other platforms or set the correct Fiji executable on Windows. Saved overlays can be inspected separately.

Preserve exclusion lists, manual-correction records, backups, and the final corrected CSV tables. Rerunning assignment or scoring can overwrite corrected results; not all correction logs are automatically reapplied. Recreate overlays after corrections and document the final filtering and biological-replicate aggregation outside these image-level scripts.

