print("Starting embryonic myosin heavy chain (eMHC) scoring...", flush=True)

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage import io


# =============================================================================
# USER SETTINGS
# =============================================================================

# Required inputs. All images must have the same height and width.
fiber_mask_path = Path("fibers.tif")
emhc_image_path = Path("emhc.tif")
fiber_summary_path = Path("image02_fiber_summary.csv")

# Outputs.
output_csv_path = Path("image02_fiber_summary_with_emhc.csv")
scored_region_mask_output_path = Path("image02_emhc_scored_region_labels.tif")

# Save a label image showing exactly which part of each fiber was scored.
# Overlay this file on emhc.tif in Fiji to check the scoring regions.
save_scored_region_mask = True


# Core definition
# ----------------
# Exclude this many pixels from the INSIDE of every fiber border. The pixels
# left after this inward erosion are the cytoplasmic core used for scoring.
#
# A value of 3 excludes the membrane-adjacent three-pixel rim.
core_border_exclusion_px = 3

# A fiber normally uses its eroded core. If fewer than this many core pixels
# remain, the script scores all pixels in that fiber instead.
min_core_pixels = 5


# Threshold mode
# --------------
# "fixed": use a previously calibrated mean core-intensity threshold.
# "negative_fibers": estimate the threshold from negative_fibers.csv.
threshold_mode = "fixed"

# Used only when threshold_mode = "fixed".
# These are starting placeholders; calibrate them for the bit depth and
# acquisition settings used for your eMHC images.
fixed_emhc_core_mean_threshold = 20.0
fixed_negative_core_mean = 15.0

# Used only when threshold_mode = "negative_fibers".
# The CSV must have a column named fiber_id containing confidently negative
# fibers from this image (or a representative calibration image).
negative_fibers_path = Path("negative_fibers.csv")

# A fiber is called positive when its mean core intensity is at or above this
# percentile of the negative fibers' mean core intensities. Higher is stricter.
negative_fiber_mean_percentile = 95.0

# Optional additional requirement. The background-corrected mean must be at
# least this value. Leave at 0.0 to classify primarily from the main threshold.
min_emhc_core_mean_bg_corrected = 0.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def as_bool(series):
    """Robustly convert booleans, True/False strings, yes/no, and 1/0."""
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


def load_2d_image(path, image_kind):
    """
    Load a 2-D label or intensity image.

    For an RGB/RGBA eMHC image, the green channel is selected. For an RGB/RGBA
    fiber mask, the first channel is selected. Grayscale TIFFs are preferred.
    Ambiguous 3-D stacks are rejected instead of silently choosing a Z plane.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    image = np.squeeze(io.imread(path))

    if image.ndim == 2:
        return image

    channel_index = 1 if image_kind == "emhc" else 0

    if image.ndim == 3 and image.shape[-1] in (3, 4):
        selected = image[..., channel_index]
        channel_name = "green" if image_kind == "emhc" else "first"
        print(
            f"{path.name} is RGB/RGBA; using the {channel_name} channel.",
            flush=True,
        )
        return selected

    if image.ndim == 3 and image.shape[0] in (3, 4):
        selected = image[channel_index, ...]
        channel_name = "green" if image_kind == "emhc" else "first"
        print(
            f"{path.name} is channel-first; using the {channel_name} channel.",
            flush=True,
        )
        return selected

    raise ValueError(
        f"{path} has unsupported shape {image.shape}. Export it as a 2-D "
        "grayscale TIFF, or as a standard RGB/RGBA image."
    )


def check_nonnegative_integer_labels(labels):
    """Make sure fibers.tif is a labeled mask, not a fluorescence image."""
    if not np.all(np.isfinite(labels)):
        raise ValueError("fibers.tif contains NaN or infinite values.")

    if np.any(labels < 0):
        raise ValueError("fibers.tif contains negative labels.")

    rounded = np.rint(labels)
    if not np.allclose(labels, rounded):
        raise ValueError(
            "fibers.tif contains non-integer values. It must be a label image "
            "with 0 for background and one positive integer ID per fiber."
        )

    return rounded.astype(np.int64)


# =============================================================================
# LOAD AND VALIDATE INPUTS
# =============================================================================

if core_border_exclusion_px < 0:
    raise ValueError("core_border_exclusion_px must be 0 or greater.")

if min_core_pixels < 1:
    raise ValueError("min_core_pixels must be at least 1.")

if not fiber_summary_path.exists():
    raise FileNotFoundError(f"Required file not found: {fiber_summary_path}")

print("Loading files...", flush=True)

fiber_labels = load_2d_image(fiber_mask_path, "fiber_mask")
emhc = load_2d_image(emhc_image_path, "emhc")
fibers_df = pd.read_csv(fiber_summary_path)

fiber_labels = check_nonnegative_integer_labels(fiber_labels)
emhc = emhc.astype(np.float64)

if not np.all(np.isfinite(emhc)):
    raise ValueError("emhc.tif contains NaN or infinite values.")

if fiber_labels.shape != emhc.shape:
    raise ValueError(
        "fibers.tif and emhc.tif have different shapes: "
        f"{fiber_labels.shape} versus {emhc.shape}. They must be registered "
        "images with identical pixel dimensions."
    )

if "fiber_id" not in fibers_df.columns:
    raise ValueError(
        f"{fiber_summary_path} must contain a column named 'fiber_id'."
    )

if fibers_df["fiber_id"].isna().any():
    raise ValueError("The fiber_id column in the fiber summary contains blanks.")

summary_ids = pd.to_numeric(fibers_df["fiber_id"], errors="coerce")
if summary_ids.isna().any() or not np.allclose(summary_ids, np.rint(summary_ids)):
    raise ValueError("The fiber_id column must contain integer fiber labels.")

fibers_df["fiber_id"] = np.rint(summary_ids).astype(np.int64)

max_label = int(fiber_labels.max())
if max_label == 0:
    raise ValueError("fibers.tif contains no labeled fibers (no values above 0).")

print(f"Fiber mask shape: {fiber_labels.shape}", flush=True)
print(f"eMHC image shape: {emhc.shape}", flush=True)
print(f"Maximum fiber label: {max_label}", flush=True)


# =============================================================================
# DEFINE THE CYTOPLASMIC CORE
# =============================================================================

print(
    f"Creating fiber cores after excluding {core_border_exclusion_px} px "
    "from each internal border...",
    flush=True,
)

# A boundary pixel is any pixel whose label differs from a 4-connected neighbor.
# This detects fiber/background edges and boundaries between touching fibers.
boundary = np.zeros_like(fiber_labels, dtype=bool)
boundary[:-1, :] |= fiber_labels[:-1, :] != fiber_labels[1:, :]
boundary[1:, :] |= fiber_labels[1:, :] != fiber_labels[:-1, :]
boundary[:, :-1] |= fiber_labels[:, :-1] != fiber_labels[:, 1:]
boundary[:, 1:] |= fiber_labels[:, 1:] != fiber_labels[:, :-1]
boundary |= fiber_labels == 0

# Each internal pixel receives its Euclidean distance to the nearest boundary.
distance_to_border_px = distance_transform_edt(~boundary)

if core_border_exclusion_px == 0:
    core_mask = fiber_labels > 0
else:
    core_mask = (
        (fiber_labels > 0)
        & (distance_to_border_px > core_border_exclusion_px)
    )

core_labels = fiber_labels[core_mask]
core_green = emhc[core_mask]

print(f"Total pixels remaining in eroded cores: {len(core_green)}", flush=True)


# =============================================================================
# AGGREGATE eMHC SIGNAL PER FIBER
# =============================================================================

print("Measuring green intensity in each fiber core...", flush=True)

fiber_px = np.bincount(
    fiber_labels[fiber_labels > 0],
    minlength=max_label + 1,
)

core_px = np.bincount(core_labels, minlength=max_label + 1)
core_sum = np.bincount(
    core_labels,
    weights=core_green,
    minlength=max_label + 1,
)
core_sum_sq = np.bincount(
    core_labels,
    weights=np.square(core_green),
    minlength=max_label + 1,
)

core_mean = np.full(max_label + 1, np.nan)
core_std = np.full(max_label + 1, np.nan)
core_fraction_pct = np.full(max_label + 1, np.nan)

has_core = core_px > 0
has_fiber = fiber_px > 0

core_mean[has_core] = core_sum[has_core] / core_px[has_core]

# Population SD is included as a heterogeneity/troubleshooting measurement.
core_variance = np.zeros(max_label + 1, dtype=float)
core_variance[has_core] = (
    core_sum_sq[has_core] / core_px[has_core]
    - np.square(core_mean[has_core])
)
core_variance = np.maximum(core_variance, 0.0)
core_std[has_core] = np.sqrt(core_variance[has_core])

core_fraction_pct[has_fiber] = (
    100.0 * core_px[has_fiber] / fiber_px[has_fiber]
)


# Select the actual scoring region for every fiber. Normally this is the core;
# when fewer than min_core_pixels remain, it is the entire labeled fiber.
use_full_fiber_fallback = has_fiber & (core_px < min_core_pixels)

scored_region_mask = (
    (fiber_labels > 0)
    & (
        core_mask
        | use_full_fiber_fallback[fiber_labels]
    )
)

scored_region_label_image = np.where(
    scored_region_mask,
    fiber_labels,
    0,
)
scored_labels = fiber_labels[scored_region_mask]
scored_green = emhc[scored_region_mask]

scored_px = np.bincount(scored_labels, minlength=max_label + 1)
scored_sum = np.bincount(
    scored_labels,
    weights=scored_green,
    minlength=max_label + 1,
)
scored_sum_sq = np.bincount(
    scored_labels,
    weights=np.square(scored_green),
    minlength=max_label + 1,
)

scored_mean = np.full(max_label + 1, np.nan)
scored_std = np.full(max_label + 1, np.nan)
has_scored_region = scored_px > 0

scored_mean[has_scored_region] = (
    scored_sum[has_scored_region]
    / scored_px[has_scored_region]
)

scored_variance = np.zeros(max_label + 1, dtype=float)
scored_variance[has_scored_region] = (
    scored_sum_sq[has_scored_region] / scored_px[has_scored_region]
    - np.square(scored_mean[has_scored_region])
)
scored_variance = np.maximum(scored_variance, 0.0)
scored_std[has_scored_region] = np.sqrt(
    scored_variance[has_scored_region]
)

print(
    f"Fibers using the full-fiber fallback: "
    f"{int(use_full_fiber_fallback.sum())}",
    flush=True,
)
print(f"Total pixels in final scoring regions: {len(scored_green)}", flush=True)


# =============================================================================
# ESTIMATE THE FIBER-LEVEL POSITIVITY THRESHOLD
# =============================================================================

if threshold_mode == "fixed":
    emhc_mean_threshold = float(fixed_emhc_core_mean_threshold)
    negative_mean = float(fixed_negative_core_mean)
    threshold_method = "fixed_scored_mean_threshold"

    print(
        f"Using fixed eMHC mean-intensity threshold: "
        f"{emhc_mean_threshold:.3f}",
        flush=True,
    )
    print(
        f"Using fixed negative mean: {negative_mean:.3f}",
        flush=True,
    )

elif threshold_mode == "negative_fibers":
    if not negative_fibers_path.exists():
        raise FileNotFoundError(
            f"{negative_fibers_path} was not found. Create it with a "
            "column named 'fiber_id', or change threshold_mode to 'fixed'."
        )

    negative_df = pd.read_csv(negative_fibers_path)
    if "fiber_id" not in negative_df.columns:
        raise ValueError(
            f"{negative_fibers_path} must contain a column named 'fiber_id'."
        )

    negative_ids = pd.to_numeric(
        negative_df["fiber_id"], errors="coerce"
    ).dropna()

    if len(negative_ids) == 0 or not np.allclose(
        negative_ids, np.rint(negative_ids)
    ):
        raise ValueError(
            f"{negative_fibers_path} must contain valid integer fiber IDs."
        )

    negative_ids = np.unique(np.rint(negative_ids).astype(np.int64))
    valid_negative_ids = negative_ids[
        (negative_ids > 0)
        & (negative_ids <= max_label)
        & (scored_px[np.clip(negative_ids, 0, max_label)] > 0)
    ]

    if len(valid_negative_ids) < 3:
        raise ValueError(
            "Fewer than 3 listed negative fibers are present in fibers.tif. "
            "Add more valid negative fiber IDs."
        )

    negative_fiber_means = scored_mean[valid_negative_ids]
    negative_mean = float(np.median(negative_fiber_means))
    emhc_mean_threshold = float(
        np.percentile(
            negative_fiber_means,
            negative_fiber_mean_percentile,
        )
    )
    threshold_method = (
        f"negative_fiber_scored_means_p{negative_fiber_mean_percentile:g}"
    )

    print(
        f"Valid negative fibers used: {len(valid_negative_ids)} / "
        f"{len(negative_ids)}",
        flush=True,
    )
    print(
        f"Median negative-fiber scored mean: {negative_mean:.3f}",
        flush=True,
    )
    print(
        f"eMHC threshold ({negative_fiber_mean_percentile:g}th percentile "
        f"of negative-fiber scored means): {emhc_mean_threshold:.3f}",
        flush=True,
    )

else:
    raise ValueError(
        f"Unknown threshold_mode '{threshold_mode}'. Use 'fixed' or "
        "'negative_fibers'."
    )

if not np.isfinite(emhc_mean_threshold):
    raise ValueError("The calculated eMHC threshold is not finite.")


# Coverage is supplementary: it reports how much of the selected scoring region
# is at or above the fiber-level mean threshold. Final classification is based
# on mean intensity, not coverage.
above_threshold = scored_green >= emhc_mean_threshold
scored_above_threshold_px = np.bincount(
    scored_labels[above_threshold],
    minlength=max_label + 1,
)

scored_above_threshold_pct = np.full(max_label + 1, np.nan)
scored_above_threshold_pct[has_scored_region] = (
    100.0
    * scored_above_threshold_px[has_scored_region]
    / scored_px[has_scored_region]
)


# =============================================================================
# BUILD AND MERGE THE OUTPUT TABLE
# =============================================================================

emhc_df = pd.DataFrame({
    "fiber_id": np.arange(max_label + 1, dtype=np.int64),
    "emhc_fiber_px": fiber_px,
    "emhc_core_border_exclusion_px": core_border_exclusion_px,
    "emhc_min_core_pixels_used": min_core_pixels,
    "emhc_core_px": core_px,
    "emhc_core_fraction_of_fiber_pct": core_fraction_pct,
    "emhc_core_mean": core_mean,
    "emhc_core_std": core_std,
    "emhc_used_full_fiber_fallback": use_full_fiber_fallback,
    "emhc_scored_px": scored_px,
    "emhc_scored_mean": scored_mean,
    "emhc_scored_mean_bg_corrected": scored_mean - negative_mean,
    "emhc_scored_std": scored_std,
    "emhc_scored_px_at_or_above_threshold": scored_above_threshold_px,
    "emhc_scored_area_at_or_above_threshold_pct": scored_above_threshold_pct,
})

# Remove the background row.
emhc_df = emhc_df[emhc_df["fiber_id"] > 0].copy()

emhc_df["emhc_scoring_region"] = np.where(
    emhc_df["emhc_used_full_fiber_fallback"],
    "full_fiber_fallback",
    "core",
)
emhc_df.loc[emhc_df["emhc_scored_px"] == 0, "emhc_scoring_region"] = "none"

emhc_df["emhc_mean_threshold_used"] = emhc_mean_threshold
emhc_df["emhc_negative_mean"] = negative_mean
emhc_df["emhc_threshold_method"] = threshold_method

emhc_df["emhc_qc_pass"] = emhc_df["emhc_scored_px"] > 0

emhc_df["emhc_positive"] = (
    emhc_df["emhc_qc_pass"]
    & (emhc_df["emhc_scored_mean"] >= emhc_mean_threshold)
    & (
        emhc_df["emhc_scored_mean_bg_corrected"]
        >= min_emhc_core_mean_bg_corrected
    )
)

# Every label in fibers.tif has a scoring region. "Unscored" is retained only
# as protection against missing/empty IDs in an input summary.
emhc_df["emhc_class"] = np.select(
    [
        ~emhc_df["emhc_qc_pass"],
        emhc_df["emhc_positive"],
    ],
    ["unscored", "positive"],
    default="negative",
)

print("eMHC scored-mean summary:", flush=True)
print(
    emhc_df.loc[emhc_df["emhc_qc_pass"], "emhc_scored_mean"].describe(),
    flush=True,
)
print(
    f"eMHC-positive fibers: {int(emhc_df['emhc_positive'].sum())} / "
    f"{int(emhc_df['emhc_qc_pass'].sum())} QC-passing fibers",
    flush=True,
)
print(
    f"Unscored IDs absent from the mask: "
    f"{int((~emhc_df['emhc_qc_pass']).sum())}",
    flush=True,
)

print("Merging eMHC measurements with the fiber summary...", flush=True)

merged_df = fibers_df.merge(emhc_df, on="fiber_id", how="left")

# A summary row whose ID is absent from fibers.tif has no measurable region.
merged_df["emhc_qc_pass"] = as_bool(merged_df["emhc_qc_pass"])
merged_df["emhc_positive"] = as_bool(merged_df["emhc_positive"])
merged_df["emhc_used_full_fiber_fallback"] = as_bool(
    merged_df["emhc_used_full_fiber_fallback"]
)
merged_df["emhc_class"] = merged_df["emhc_class"].fillna("unscored")

# Final analysis-ready flag: the fiber passed both previous morphology QC (if
# present) and this script's eMHC core QC.
if "fiber_qc_pass" in merged_df.columns:
    merged_df["analysis_qc_pass"] = (
        as_bool(merged_df["fiber_qc_pass"])
        & merged_df["emhc_qc_pass"]
    )
else:
    merged_df["analysis_qc_pass"] = merged_df["emhc_qc_pass"]


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

print("Saving output...", flush=True)
merged_df.to_csv(output_csv_path, index=False)
print(f"Saved CSV: {output_csv_path}", flush=True)

if save_scored_region_mask:
    output_dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
    io.imsave(
        scored_region_mask_output_path,
        scored_region_label_image.astype(output_dtype),
        check_contrast=False,
    )
    print(
        f"Saved scored-region QC image: {scored_region_mask_output_path}",
        flush=True,
    )

print("Done.", flush=True)
