print("Starting dystrophin scoring...", flush=True)

import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io
from scipy.ndimage import distance_transform_edt


# User inputs

fiber_mask_path = Path("fibers.tif")
dystrophin_image_path = Path("dystrophin.tif")
fiber_summary_path = Path("image02_fiber_summary.csv")

output_csv_path = Path("image02_fiber_summary_with_dystrophin.csv")

# Only used if threshold_mode = "negative_fibers"
negative_fibers_path = Path("negative_fibers.csv")


# Threshold mode

# Options:
# "fixed" = use a previously calibrated threshold
# "negative_fibers" = calculate threshold from negative_fibers.csv
threshold_mode = "fixed"

# Use these only if threshold_mode = "fixed"
fixed_dystrophin_threshold = 50
fixed_negative_baseline_median = 15


# Scale settings

# Use the actual pixel size for THIS image.
#
# Important:
# If Fiji says "1 pixel = 0.762 microns", then:
#     pixel_size_um = 0.762
#
# If Fiji says "0.776 pixels / micron", then:
#     pixel_size_um = 1 / 0.776 = 1.2886
#
# Full-size example:
#pixel_size_um = 1.2886
#
# Half-size example:
# pixel_size_um = 2.5766
#
# Current image:
pixel_size_um = 1.2886

# Original standard scale used when you first chose pixel-based cutoffs.
standard_pixel_size_um = 1.2886

print(f"Pixel size: {pixel_size_um:.6f} um/px", flush=True)
print(f"Area conversion factor: {pixel_size_um ** 2:.6f} um2/px", flush=True)


# Membrane band settings

# Inner band just inside the fiber border.
# This is defined in microns, then converted to pixels.
inner_membrane_band_width_um = 3.0

# Outer band just outside the fiber border.
# This captures real dystrophin signal that may sit slightly outside
# the assigned laminin/fiber mask.
include_outer_membrane_band = False
outer_membrane_band_width_um = 2.0


# Scale-aware membrane-band QC settings

# Original minimum membrane-band pixel cutoff at the standard scale.
# This gets converted into a physical area, then back to current-image pixels.
standard_min_membrane_band_pixels = 10

min_membrane_band_area_um2 = (
    standard_min_membrane_band_pixels
    * (standard_pixel_size_um ** 2)
)

min_membrane_band_pixels = (
    min_membrane_band_area_um2
    / (pixel_size_um ** 2)
)

# At minimum, require at least 1 membrane-band pixel.
min_membrane_band_pixels = max(1, min_membrane_band_pixels)

print(
    f"Minimum membrane-band QC cutoff: "
    f"{min_membrane_band_area_um2:.2f} um2 = "
    f"{min_membrane_band_pixels:.2f} px at current scale",
    flush=True
)


# Dystrophin scoring settings

# Only used if threshold_mode = "negative_fibers".
# Lower = more permissive.
# Higher = stricter.
negative_fiber_threshold_percentile = 90

# Fiber-level dystrophin positivity cutoff.
# Positive if at least this percent of combined membrane pixels are above threshold.
positive_coverage_cutoff_pct = 50.0

# Extra safeguard:
# Mean membrane intensity must be above the negative baseline by this much.
# Set to 0.0 if you mostly want coverage-based classification.
min_dystrophin_mean_bg_corrected = 0.0


# Helper functions

def as_bool(series):
    """
    Robustly convert a pandas column to boolean.
    Handles booleans, True/False strings, and 1/0.
    """
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        })
        .fillna(False)
        .astype(bool)
    )


def convert_um_to_px(distance_um, pixel_size_um, min_px=1):
    """
    Convert a physical distance in microns to pixels for this image.
    """
    px = int(round(distance_um / pixel_size_um))
    return max(min_px, px)


def safe_mean(values):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


# Load data

print("Loading files...", flush=True)

fiber_labels = io.imread(fiber_mask_path)
dystrophin = io.imread(dystrophin_image_path)
fibers_df = pd.read_csv(fiber_summary_path)

if fiber_labels.ndim > 2:
    fiber_labels = fiber_labels[..., 0]

if dystrophin.ndim > 2:
    # If this is RGB, take channel 0 as red.
    # Ideally, dystrophin.tif should already be red-only grayscale.
    dystrophin = dystrophin[..., 0]

fiber_labels = fiber_labels.astype(np.int32)
dystrophin = dystrophin.astype(float)

print(f"Fiber mask shape: {fiber_labels.shape}", flush=True)
print(f"Dystrophin image shape: {dystrophin.shape}", flush=True)

if fiber_labels.shape != dystrophin.shape:
    raise ValueError(
        f"Fiber mask and dystrophin image have different shapes: "
        f"{fiber_labels.shape} vs {dystrophin.shape}"
    )

if "fiber_id" not in fibers_df.columns:
    raise ValueError("fiber_summary file must contain a column named 'fiber_id'.")

max_label = int(fiber_labels.max())
print(f"Maximum fiber label: {max_label}", flush=True)


# Create inner membrane band

print("Creating inner membrane band...", flush=True)

inner_membrane_band_width_px = convert_um_to_px(
    inner_membrane_band_width_um,
    pixel_size_um,
    min_px=1
)

print(
    f"Inner membrane band width: "
    f"{inner_membrane_band_width_um} um = "
    f"{inner_membrane_band_width_px} px",
    flush=True
)

# Boundary is any pixel where the fiber label changes compared with a neighbor.
# This treats boundaries between adjacent labeled fibers as borders too.
boundary = np.zeros_like(fiber_labels, dtype=bool)

boundary[:-1, :] |= fiber_labels[:-1, :] != fiber_labels[1:, :]
boundary[1:, :] |= fiber_labels[1:, :] != fiber_labels[:-1, :]
boundary[:, :-1] |= fiber_labels[:, :-1] != fiber_labels[:, 1:]
boundary[:, 1:] |= fiber_labels[:, 1:] != fiber_labels[:, :-1]

# Background counts as boundary/not-fiber.
boundary |= fiber_labels == 0

# Distance from each internal fiber pixel to the nearest border.
distance_to_border_px_map = distance_transform_edt(~boundary)

# Inner band: labeled fiber pixels close to border.
# distance > 0 avoids exact boundary pixels.
inner_membrane_band = (
    (fiber_labels > 0)
    & (distance_to_border_px_map > 0)
    & (distance_to_border_px_map <= inner_membrane_band_width_px)
)

inner_labels = fiber_labels[inner_membrane_band].astype(np.int32)
inner_red = dystrophin[inner_membrane_band]

print(f"Inner membrane band pixels: {len(inner_red)}", flush=True)


# Create optional outer membrane band

if include_outer_membrane_band:
    print("Creating outer membrane band...", flush=True)

    outer_membrane_band_width_px = convert_um_to_px(
        outer_membrane_band_width_um,
        pixel_size_um,
        min_px=1
    )

    print(
        f"Outer membrane band width: "
        f"{outer_membrane_band_width_um} um = "
        f"{outer_membrane_band_width_px} px",
        flush=True
    )

    # Background pixels are candidate outside-fiber pixels.
    background_mask = fiber_labels == 0

    # For each background pixel, find distance and index of nearest fiber pixel.
    # distance_transform_edt measures distance to the nearest zero pixel.
    # Since background_mask is True outside fibers and False inside fibers,
    # this gives distance from background to nearest fiber.
    distance_to_fiber_px, nearest_indices = distance_transform_edt(
        background_mask,
        return_indices=True
    )

    nearest_fiber_labels = fiber_labels[
        nearest_indices[0],
        nearest_indices[1]
    ]

    # Outer band: background pixels close to a labeled fiber.
    # Each outside pixel is assigned to the nearest fiber.
    outer_membrane_band = (
        background_mask
        & (distance_to_fiber_px > 0)
        & (distance_to_fiber_px <= outer_membrane_band_width_px)
        & (nearest_fiber_labels > 0)
    )

    outer_labels = nearest_fiber_labels[outer_membrane_band].astype(np.int32)
    outer_red = dystrophin[outer_membrane_band]

    print(f"Outer membrane band pixels: {len(outer_red)}", flush=True)

else:
    outer_membrane_band_width_px = 0
    outer_labels = np.array([], dtype=np.int32)
    outer_red = np.array([], dtype=float)


# Combine inner and outer bands

print("Combining membrane bands...", flush=True)

labels_in_band = np.concatenate([inner_labels, outer_labels])
red_in_band = np.concatenate([inner_red, outer_red])

print(f"Combined membrane band pixels: {len(red_in_band)}", flush=True)

if len(red_in_band) == 0:
    raise ValueError("No membrane-band pixels found. Check fiber mask and band settings.")


# Estimate dystrophin threshold

print("Estimating dystrophin threshold...", flush=True)

if threshold_mode == "fixed":
    dystrophin_threshold = fixed_dystrophin_threshold
    negative_baseline_median = fixed_negative_baseline_median
    threshold_method = "fixed_threshold"

    print(f"Using fixed dystrophin threshold: {dystrophin_threshold:.3f}", flush=True)
    print(f"Using fixed negative baseline median: {negative_baseline_median:.3f}", flush=True)

elif threshold_mode == "negative_fibers":
    print("Estimating dystrophin threshold from negative fibers...", flush=True)

    if not negative_fibers_path.exists():
        raise FileNotFoundError(
            "negative_fibers.csv not found. Create this file with a column named 'fiber_id'."
        )

    negative_df = pd.read_csv(negative_fibers_path)

    if "fiber_id" not in negative_df.columns:
        raise ValueError("negative_fibers.csv must contain a column named 'fiber_id'.")

    negative_fiber_ids = set(negative_df["fiber_id"].dropna().astype(int))

    if len(negative_fiber_ids) == 0:
        raise ValueError("negative_fibers.csv contains no fiber IDs.")

    # Important:
    # The negative threshold is calculated using the same combined membrane band
    # that will be used for scoring all fibers.
    negative_band_pixels = red_in_band[
        np.isin(labels_in_band, list(negative_fiber_ids))
    ]

    if len(negative_band_pixels) < 100:
        raise ValueError(
            f"Only {len(negative_band_pixels)} membrane pixels came from negative fibers. "
            "Add more fiber IDs to negative_fibers.csv."
        )

    negative_baseline_median = np.median(negative_band_pixels)

    dystrophin_threshold = np.percentile(
        negative_band_pixels,
        negative_fiber_threshold_percentile
    )

    threshold_method = f"negative_fibers_p{negative_fiber_threshold_percentile}"

    print(f"Negative fiber membrane pixels: {len(negative_band_pixels)}", flush=True)
    print(f"Negative-fiber baseline median: {negative_baseline_median:.3f}", flush=True)
    print(
        f"Dystrophin threshold "
        f"({negative_fiber_threshold_percentile}th percentile of negative fibers): "
        f"{dystrophin_threshold:.3f}",
        flush=True
    )

else:
    raise ValueError(
        f"Unknown threshold_mode: {threshold_mode}. "
        "Use 'fixed' or 'negative_fibers'."
    )


# Aggregate dystrophin signal per fiber

print("Scoring dystrophin per fiber...", flush=True)

# Combined band measurements
combined_band_px = np.bincount(labels_in_band, minlength=max_label + 1)
combined_sum_red = np.bincount(
    labels_in_band,
    weights=red_in_band,
    minlength=max_label + 1
)

combined_positive_pixels = red_in_band > dystrophin_threshold

combined_positive_px = np.bincount(
    labels_in_band[combined_positive_pixels],
    minlength=max_label + 1
)

combined_mean_red = np.full(max_label + 1, np.nan)
combined_coverage_pct = np.full(max_label + 1, np.nan)

combined_valid = combined_band_px > 0

combined_mean_red[combined_valid] = (
    combined_sum_red[combined_valid]
    / combined_band_px[combined_valid]
)

combined_coverage_pct[combined_valid] = (
    100
    * combined_positive_px[combined_valid]
    / combined_band_px[combined_valid]
)


# Inner band measurements, useful for troubleshooting
inner_band_px = np.bincount(inner_labels, minlength=max_label + 1)
inner_sum_red = np.bincount(
    inner_labels,
    weights=inner_red,
    minlength=max_label + 1
)

inner_positive_pixels = inner_red > dystrophin_threshold

inner_positive_px = np.bincount(
    inner_labels[inner_positive_pixels],
    minlength=max_label + 1
)

inner_mean_red = np.full(max_label + 1, np.nan)
inner_coverage_pct = np.full(max_label + 1, np.nan)

inner_valid = inner_band_px > 0

inner_mean_red[inner_valid] = (
    inner_sum_red[inner_valid]
    / inner_band_px[inner_valid]
)

inner_coverage_pct[inner_valid] = (
    100
    * inner_positive_px[inner_valid]
    / inner_band_px[inner_valid]
)


# Outer band measurements, useful for troubleshooting
if include_outer_membrane_band and len(outer_red) > 0:
    outer_band_px = np.bincount(outer_labels, minlength=max_label + 1)
    outer_sum_red = np.bincount(
        outer_labels,
        weights=outer_red,
        minlength=max_label + 1
    )

    outer_positive_pixels = outer_red > dystrophin_threshold

    outer_positive_px = np.bincount(
        outer_labels[outer_positive_pixels],
        minlength=max_label + 1
    )

    outer_mean_red = np.full(max_label + 1, np.nan)
    outer_coverage_pct = np.full(max_label + 1, np.nan)

    outer_valid = outer_band_px > 0

    outer_mean_red[outer_valid] = (
        outer_sum_red[outer_valid]
        / outer_band_px[outer_valid]
    )

    outer_coverage_pct[outer_valid] = (
        100
        * outer_positive_px[outer_valid]
        / outer_band_px[outer_valid]
    )

else:
    outer_band_px = np.zeros(max_label + 1, dtype=int)
    outer_positive_px = np.zeros(max_label + 1, dtype=int)
    outer_mean_red = np.full(max_label + 1, np.nan)
    outer_coverage_pct = np.full(max_label + 1, np.nan)


# Build dystrophin output table

dystrophin_df = pd.DataFrame({
    "fiber_id": np.arange(max_label + 1),

    # Scale metadata
    "dystrophin_pixel_size_um": pixel_size_um,
    "dystrophin_standard_pixel_size_um": standard_pixel_size_um,

    # Band definition metadata
    "dystrophin_inner_band_width_um": inner_membrane_band_width_um,
    "dystrophin_inner_band_width_px": inner_membrane_band_width_px,
    "dystrophin_outer_band_included": include_outer_membrane_band,
    "dystrophin_outer_band_width_um": outer_membrane_band_width_um,
    "dystrophin_outer_band_width_px": outer_membrane_band_width_px,

    # QC cutoff metadata
    "dystrophin_min_membrane_band_pixels_used": min_membrane_band_pixels,
    "dystrophin_min_membrane_band_area_um2": min_membrane_band_area_um2,

    # Combined inner + outer measurements used for final classification
    "dystrophin_membrane_band_px": combined_band_px,
    "dystrophin_membrane_mean": combined_mean_red,
    "dystrophin_membrane_mean_bg_corrected": (
        combined_mean_red - negative_baseline_median
    ),
    "dystrophin_positive_px": combined_positive_px,
    "dystrophin_membrane_coverage_pct": combined_coverage_pct,

    # Inner-only troubleshooting columns
    "dystrophin_inner_band_px": inner_band_px,
    "dystrophin_inner_mean": inner_mean_red,
    "dystrophin_inner_positive_px": inner_positive_px,
    "dystrophin_inner_coverage_pct": inner_coverage_pct,

    # Outer-only troubleshooting columns
    "dystrophin_outer_band_px": outer_band_px,
    "dystrophin_outer_mean": outer_mean_red,
    "dystrophin_outer_positive_px": outer_positive_px,
    "dystrophin_outer_coverage_pct": outer_coverage_pct,
})

# Remove background row
dystrophin_df = dystrophin_df[dystrophin_df["fiber_id"] > 0].copy()

dystrophin_df["dystrophin_threshold_used"] = dystrophin_threshold
dystrophin_df["dystrophin_negative_baseline_median"] = negative_baseline_median
dystrophin_df["dystrophin_threshold_method"] = threshold_method

dystrophin_df["dystrophin_band_definition"] = np.where(
    include_outer_membrane_band,
    f"combined_inner_{inner_membrane_band_width_um}um_outer_{outer_membrane_band_width_um}um",
    f"inner_only_{inner_membrane_band_width_um}um"
)

dystrophin_df["dystrophin_qc_pass"] = (
    dystrophin_df["dystrophin_membrane_band_px"].fillna(0)
    >= min_membrane_band_pixels
)

dystrophin_df["dystrophin_positive"] = (
    dystrophin_df["dystrophin_qc_pass"]
    & (
        dystrophin_df["dystrophin_membrane_coverage_pct"]
        >= positive_coverage_cutoff_pct
    )
    & (
        dystrophin_df["dystrophin_membrane_mean_bg_corrected"]
        >= min_dystrophin_mean_bg_corrected
    )
)

dystrophin_df["dystrophin_class"] = np.where(
    dystrophin_df["dystrophin_positive"],
    "positive",
    "negative"
)


# Print quick summary

print("Dystrophin combined coverage summary:", flush=True)
print(dystrophin_df["dystrophin_membrane_coverage_pct"].describe(), flush=True)

print(
    f"Dystrophin-positive fibers: "
    f"{dystrophin_df['dystrophin_positive'].sum()} / {len(dystrophin_df)}",
    flush=True
)

print("Inner coverage summary:", flush=True)
print(dystrophin_df["dystrophin_inner_coverage_pct"].describe(), flush=True)

if include_outer_membrane_band:
    print("Outer coverage summary:", flush=True)
    print(dystrophin_df["dystrophin_outer_coverage_pct"].describe(), flush=True)


# Merge with existing fiber summary

print("Merging with fiber summary...", flush=True)

merged_df = fibers_df.merge(
    dystrophin_df,
    on="fiber_id",
    how="left"
)

# Final analysis-ready flag:
# passed previous fiber QC and dystrophin QC.
if "fiber_qc_pass" in merged_df.columns:
    merged_df["analysis_qc_pass"] = (
        as_bool(merged_df["fiber_qc_pass"])
        & as_bool(merged_df["dystrophin_qc_pass"])
    )
else:
    merged_df["analysis_qc_pass"] = as_bool(merged_df["dystrophin_qc_pass"])


# Save output

print("Saving output...", flush=True)

merged_df.to_csv(output_csv_path, index=False)

print(f"Saved: {output_csv_path}", flush=True)
print("Done.", flush=True)