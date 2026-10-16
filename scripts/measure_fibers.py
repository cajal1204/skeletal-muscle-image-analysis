print("Starting dystrophin fiber CSA analysis...", flush=True)

from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io
from skimage.measure import label as connected_components
from skimage.measure import regionprops_table


# USER SETTINGS

# Top-level folder containing the individual image folders.
ROOT_FOLDER = Path(
    r"C:/Users/emurp/Documents/0000"
)

# Name of the Cellpose mask inside each image folder.
MASK_FILENAME = "dystrophin_cp_masks.png"

# Combined CSV containing fibers from every image.
OUTPUT_CSV = (
    ROOT_FOLDER /
    "dystrophin_fiber_cross_sectional_areas.csv"
)

# All individual-image CSVs will be placed here.
INDIVIDUAL_CSV_FOLDER = (
    ROOT_FOLDER /
    "individual_image_CSA_csvs"
)

# Pixel size of the images before any resizing.
# Change this if the images were collected or resized differently.
PIXEL_SIZE_UM = 1.2886

# Fibers are retained regardless of this setting.
# This setting only controls the passes_min_area_qc column.
#
# Use 0 to flag every fiber as passing.
# Use 100, for example, to flag objects under 100 pixels as failing.
MIN_FIBER_AREA_PX = 0


# HELPER FUNCTIONS

def load_fiber_labels(mask_path):
    """
    Load a Cellpose fiber mask as a two-dimensional label image.

    Expected format:
        0 = background
        1, 2, 3, ... = individual fibers

    If the input is binary, connected components are labeled
    automatically.
    """

    mask = io.imread(mask_path)
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask, but this file has "
            f"shape {mask.shape}:\n{mask_path}"
        )

    if not np.all(np.isfinite(mask)):
        raise ValueError(
            f"Mask contains NaN or infinite values:\n{mask_path}"
        )

    # Confirm that floating-point masks contain integer labels.
    if np.issubdtype(mask.dtype, np.floating):
        if not np.allclose(mask, np.rint(mask)):
            raise ValueError(
                f"Mask contains non-integer label values:\n"
                f"{mask_path}"
            )

        mask = np.rint(mask)

    mask = mask.astype(np.int64)

    if np.any(mask < 0):
        raise ValueError(
            f"Mask contains negative labels:\n{mask_path}"
        )

    positive_labels = np.unique(mask[mask > 0])

    # If there is only one nonzero value, the image may be binary.
    # Convert disconnected foreground regions into separate labels.
    if len(positive_labels) == 1:
        mask = connected_components(
            mask > 0,
            connectivity=1,
        )

    return mask


def measure_fibers(mask_path):
    """
    Measure every labeled fiber in one mask.

    Returns one DataFrame row per fiber.
    """

    fiber_labels = load_fiber_labels(mask_path)

    properties = regionprops_table(
        fiber_labels,
        properties=[
            "label",
            "area",
            "centroid",
        ],
    )

    fibers = pd.DataFrame(properties)

    fibers = fibers.rename(
        columns={
            "label": "fiber_id",
            "area": "fiber_area_px",
            "centroid-0": "fiber_centroid_y_px",
            "centroid-1": "fiber_centroid_x_px",
        }
    )

    # These conversions also work when the DataFrame is empty.
    fibers["fiber_id"] = fibers["fiber_id"].astype(int)
    fibers["fiber_area_px"] = fibers["fiber_area_px"].astype(int)

    # In a transverse muscle section, the segmented two-dimensional
    # fiber area is the fiber cross-sectional area.
    fibers["pixel_size_um"] = PIXEL_SIZE_UM

    fibers["fiber_cross_sectional_area_um2"] = (
        fibers["fiber_area_px"] *
        PIXEL_SIZE_UM ** 2
    )

    # Retain every fiber but record whether it passes area QC.
    fibers["passes_min_area_qc"] = (
        fibers["fiber_area_px"] >=
        MIN_FIBER_AREA_PX
    )

    # Identify labels touching any image boundary.
    border_labels = np.unique(
        np.concatenate(
            [
                fiber_labels[0, :],
                fiber_labels[-1, :],
                fiber_labels[:, 0],
                fiber_labels[:, -1],
            ]
        )
    )

    border_labels = set(
        border_labels[
            border_labels > 0
        ].astype(int)
    )

    fibers["touches_image_border"] = (
        fibers["fiber_id"].isin(border_labels)
    )

    return fibers


# FIND MASK FILES

if not ROOT_FOLDER.exists():
    raise FileNotFoundError(
        f"Root folder does not exist:\n{ROOT_FOLDER}"
    )

mask_paths = sorted(
    ROOT_FOLDER.rglob(MASK_FILENAME)
)

if not mask_paths:
    raise FileNotFoundError(
        f"No files named '{MASK_FILENAME}' were found beneath:\n"
        f"{ROOT_FOLDER}"
    )

print(
    f"Found {len(mask_paths)} mask file(s).",
    flush=True,
)


# CREATE OUTPUT FOLDERS

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

INDIVIDUAL_CSV_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# PROCESS EVERY IMAGE

all_results = []

# Used to detect duplicate image-folder names.
used_csv_names = set()

for mask_number, mask_path in enumerate(
    mask_paths,
    start=1,
):
    relative_mask_path = mask_path.relative_to(
        ROOT_FOLDER
    )

    relative_folder = mask_path.parent.relative_to(
        ROOT_FOLDER
    )

    # The folder containing the mask is treated as the image name.
    image_name = mask_path.parent.name

    print(
        f"[{mask_number}/{len(mask_paths)}] "
        f"Processing {relative_mask_path}...",
        flush=True,
    )

    fibers = measure_fibers(mask_path)

    # Add image-identification columns.
    fibers.insert(
        0,
        "sample",
        image_name,
    )

    fibers.insert(
        1,
        "folder",
        str(relative_folder),
    )

    fibers.insert(
        2,
        "mask_file",
        str(relative_mask_path),
    )

    # ========================================================
    # Save this image's individual CSV
    # ========================================================

    individual_csv = (
        INDIVIDUAL_CSV_FOLDER /
        f"{image_name}.csv"
    )

    # Windows filenames are case-insensitive, so use casefold()
    # when checking for conflicting output names.
    output_name_key = individual_csv.name.casefold()

    if output_name_key in used_csv_names:
        raise ValueError(
            f"More than one image folder has the name "
            f"'{image_name}'.\n"
            f"The individual CSV filenames would conflict."
        )

    used_csv_names.add(output_name_key)

    fibers.to_csv(
        individual_csv,
        index=False,
    )

    # Add this image to the combined results.
    all_results.append(fibers)

    print(
        f"  Measured {len(fibers)} fibers.",
        flush=True,
    )

    print(
        f"  Individual CSV: {individual_csv}",
        flush=True,
    )


# CREATE COMBINED TABLE

results = pd.concat(
    all_results,
    ignore_index=True,
)

if not results.empty:
    results = results.sort_values(
        [
            "sample",
            "mask_file",
            "fiber_id",
        ]
    ).reset_index(drop=True)

    # Make sure a fiber was not accidentally added more than once.
    duplicate_fibers = results.duplicated(
        subset=[
            "sample",
            "mask_file",
            "fiber_id",
        ],
        keep=False,
    )

    if duplicate_fibers.any():
        duplicate_table = results.loc[
            duplicate_fibers,
            [
                "sample",
                "mask_file",
                "fiber_id",
            ],
        ]

        print(
            "Duplicated fibers:",
            flush=True,
        )

        print(
            duplicate_table.to_string(index=False),
            flush=True,
        )

        raise RuntimeError(
            "One or more fibers were added to the "
            "combined table more than once."
        )


# SAVE COMBINED CSV

results.to_csv(
    OUTPUT_CSV,
    index=False,
)


# FINAL SUMMARY

print("", flush=True)
print("Analysis finished.", flush=True)

print(
    f"Images processed: {len(mask_paths)}",
    flush=True,
)

print(
    f"Total fibers measured: {len(results)}",
    flush=True,
)

print(
    f"Combined CSV:\n{OUTPUT_CSV}",
    flush=True,
)

print(
    f"Individual-image CSV folder:\n"
    f"{INDIVIDUAL_CSV_FOLDER}",
    flush=True,
)