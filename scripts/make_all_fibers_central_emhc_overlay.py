import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, segmentation


# Input files

laminin_image_path = Path("laminin.tif")
dapi_image_path = Path("dapi.tif")
emhc_image_path = Path("emhc.tif")

fiber_mask_path = Path("fibers.tif")
nucleus_csv_path = Path("image02_nucleus_assignments.csv")
fiber_summary_path = Path("image02_fiber_summary_with_emhc.csv")

output_path = Path("image02_all_fibers_central_nuclei_emhc_overlay.png")


# Display settings

# Use only nuclei that passed nucleus_qc_pass.
# This makes the overlay match your final central nucleus calls.
show_only_qc_passing_nuclei = True

# Which nucleus column should define central nuclei?
# Recommended: "central_nucleus"
# Troubleshooting option: "central_nucleus_raw"
central_column_for_display = "central_nucleus"

# Which fiber QC column should control whether a fiber is shown in color?
# Recommended: "analysis_qc_pass" after emhc scoring.
# If not present, the script falls back to "fiber_qc_pass".
fiber_qc_column_preference = "analysis_qc_pass"

# If True, fibers that fail QC are shown as gray dotted outlines.
# If False, failed-QC fibers are hidden.
show_failed_qc_fibers_gray = True

# Dotted boundary density.
# Larger number = fewer boundary dots.
boundary_dot_stride = 3

# Dot sizes
fiber_boundary_dot_size = 0.12
nucleus_dot_size = 3.0

# Full-resolution-style output
output_dpi = 300


# Colors

# Fiber outline colors
emhc_positive_color = "lime"
emhc_negative_color = "magenta"
failed_or_unknown_color = "gray"

# Nucleus colors
central_nucleus_color = "orange"
noncentral_nucleus_color = "cyan"


# Helper functions

def as_bool_series(series):
    """
    Robustly convert a pandas column to boolean.
    Handles True/False, true/false strings, 1/0, yes/no.
    """
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def scale_using_full_image(img, low=1, high=99.5):
    """
    Scale an image using full-image percentiles.
    This avoids artificially boosting weak channels in local regions.
    """
    p_low, p_high = np.percentile(img, (low, high))

    if p_high <= p_low:
        return np.zeros_like(img, dtype=float)

    scaled = (img - p_low) / (p_high - p_low)
    return np.clip(scaled, 0, 1)


# Load data

print("Loading data...", flush=True)

laminin = io.imread(laminin_image_path)
dapi = io.imread(dapi_image_path)
emhc = io.imread(emhc_image_path)
fiber_labels = io.imread(fiber_mask_path)

nuclei_df = pd.read_csv(nucleus_csv_path)
fibers_df = pd.read_csv(fiber_summary_path)

if laminin.ndim > 2:
    laminin = laminin[..., 0]

if dapi.ndim > 2:
    print(f"DAPI image has multiple channels: {dapi.shape}", flush=True)
    dapi = dapi[..., 2]

if emhc.ndim > 2:
    # eMHC is the green channel in an RGB export.
    emhc = emhc[..., 1]

if fiber_labels.ndim > 2:
    fiber_labels = fiber_labels[..., 0]

laminin = laminin.astype(float)
dapi = dapi.astype(float)
emhc = emhc.astype(float)
fiber_labels = fiber_labels.astype(np.int32)

shape_ref = fiber_labels.shape

for name, arr in [
    ("laminin", laminin),
    ("dapi", dapi),
    ("emhc", emhc),
]:
    if arr.shape != shape_ref:
        raise ValueError(
            f"{name} image shape {arr.shape} does not match fiber mask shape {shape_ref}"
        )

print(f"Image shape: {shape_ref}", flush=True)
print(f"Fibers in mask: {len(np.unique(fiber_labels)) - 1}", flush=True)
print(f"Nuclei in CSV: {len(nuclei_df)}", flush=True)


# Build RGB background image

print("Building merged image...", flush=True)

laminin_disp = scale_using_full_image(laminin, low=1, high=99.5)
dapi_disp = scale_using_full_image(dapi, low=1, high=99.5)
emhc_disp = scale_using_full_image(emhc, low=1, high=99.99)

# Laminin = red
# eMHC = green
# DAPI = blue
rgb = np.dstack([
    laminin_disp,
    emhc_disp,
    dapi_disp,
])


# Prepare fiber status

print("Preparing fiber eMHC status...", flush=True)

if "fiber_id" not in fibers_df.columns:
    raise ValueError("fiber_summary file must contain a fiber_id column.")

if "emhc_positive" not in fibers_df.columns:
    raise ValueError("fiber_summary file must contain emhc_positive column.")

fibers_df["fiber_id"] = fibers_df["fiber_id"].astype(int)
fibers_df["emhc_positive_bool"] = as_bool_series(
    fibers_df["emhc_positive"]
)

# Choose QC column
if fiber_qc_column_preference in fibers_df.columns:
    qc_column = fiber_qc_column_preference
elif "fiber_qc_pass" in fibers_df.columns:
    qc_column = "fiber_qc_pass"
else:
    qc_column = None

if qc_column is not None:
    fibers_df["fiber_display_qc_pass"] = as_bool_series(fibers_df[qc_column])
    print(f"Using fiber QC column: {qc_column}", flush=True)
else:
    fibers_df["fiber_display_qc_pass"] = True
    print("No fiber QC column found. Showing all fibers as QC-passing.", flush=True)

positive_ids = set(
    fibers_df.loc[
        fibers_df["fiber_display_qc_pass"] & fibers_df["emhc_positive_bool"],
        "fiber_id"
    ].astype(int)
)

negative_ids = set(
    fibers_df.loc[
        fibers_df["fiber_display_qc_pass"] & ~fibers_df["emhc_positive_bool"],
        "fiber_id"
    ].astype(int)
)

failed_or_unknown_ids = set(
    fibers_df.loc[
        ~fibers_df["fiber_display_qc_pass"],
        "fiber_id"
    ].astype(int)
)

print(f"eMHC-positive QC-passing fibers: {len(positive_ids)}", flush=True)
print(f"eMHC-negative QC-passing fibers: {len(negative_ids)}", flush=True)
print(f"Failed-QC fibers: {len(failed_or_unknown_ids)}", flush=True)


# Prepare dotted fiber boundaries

print("Finding fiber boundaries...", flush=True)

all_boundaries = segmentation.find_boundaries(fiber_labels, mode="inner")

boundary_y, boundary_x = np.where(all_boundaries)
boundary_fiber_ids = fiber_labels[boundary_y, boundary_x].astype(int)

# Create dotted pattern
dot_keep = ((boundary_x + boundary_y) % boundary_dot_stride) == 0

boundary_y = boundary_y[dot_keep]
boundary_x = boundary_x[dot_keep]
boundary_fiber_ids = boundary_fiber_ids[dot_keep]

positive_boundary = np.isin(boundary_fiber_ids, list(positive_ids))
negative_boundary = np.isin(boundary_fiber_ids, list(negative_ids))
failed_boundary = np.isin(boundary_fiber_ids, list(failed_or_unknown_ids))

print(f"Boundary dots kept: {len(boundary_x)}", flush=True)


# Prepare nuclei

print("Preparing nuclei...", flush=True)

required_nucleus_cols = [
    "nucleus_x",
    "nucleus_y",
    "assigned_fiber_id",
]

for col in required_nucleus_cols:
    if col not in nuclei_df.columns:
        raise ValueError(f"nucleus CSV must contain column: {col}")

if central_column_for_display not in nuclei_df.columns:
    raise ValueError(
        f"Column '{central_column_for_display}' not found in nucleus CSV.\n"
        f"Available columns are:\n{list(nuclei_df.columns)}"
    )

nuclei_display = nuclei_df.copy()

# Keep only nuclei assigned to real fibers
nuclei_display = nuclei_display[
    nuclei_display["assigned_fiber_id"].fillna(0).astype(int) > 0
].copy()

# Keep only nuclei assigned to QC-passing fibers, if possible
if qc_column is not None:
    qc_passing_ids = positive_ids | negative_ids
    nuclei_display = nuclei_display[
        nuclei_display["assigned_fiber_id"].astype(int).isin(qc_passing_ids)
    ].copy()

# Optional: keep only nuclei that passed nucleus QC
if show_only_qc_passing_nuclei and "nucleus_qc_pass" in nuclei_display.columns:
    nuclei_display = nuclei_display[
        as_bool_series(nuclei_display["nucleus_qc_pass"])
    ].copy()

nuclei_display["central_display_bool"] = as_bool_series(
    nuclei_display[central_column_for_display]
)

central_nuclei = nuclei_display[nuclei_display["central_display_bool"]].copy()
noncentral_nuclei = nuclei_display[~nuclei_display["central_display_bool"]].copy()

print(f"Displayed nuclei: {len(nuclei_display)}", flush=True)
print(f"Displayed central nuclei: {len(central_nuclei)}", flush=True)
print(f"Displayed non-central nuclei: {len(noncentral_nuclei)}", flush=True)


# Plot

print("Plotting overlay...", flush=True)

height, width = fiber_labels.shape

# Set figure size so output roughly preserves image aspect ratio.
fig_width = width / output_dpi
fig_height = height / output_dpi

plt.figure(figsize=(fig_width, fig_height), dpi=output_dpi)
plt.imshow(rgb)

# Dotted fiber outlines
if np.any(positive_boundary):
    plt.scatter(
        boundary_x[positive_boundary],
        boundary_y[positive_boundary],
        s=fiber_boundary_dot_size,
        c=emhc_positive_color,
        alpha=0.95,
        linewidths=0,
        label="eMHC-positive fiber outline"
    )

if np.any(negative_boundary):
    plt.scatter(
        boundary_x[negative_boundary],
        boundary_y[negative_boundary],
        s=fiber_boundary_dot_size,
        c=emhc_negative_color,
        alpha=0.95,
        linewidths=0,
        label="eMHC-negative fiber outline"
    )

if show_failed_qc_fibers_gray and np.any(failed_boundary):
    plt.scatter(
        boundary_x[failed_boundary],
        boundary_y[failed_boundary],
        s=fiber_boundary_dot_size,
        c=failed_or_unknown_color,
        alpha=0.35,
        linewidths=0,
        label="Failed-QC / excluded fiber outline"
    )

# Nuclei
if len(noncentral_nuclei) > 0:
    plt.scatter(
        noncentral_nuclei["nucleus_x"],
        noncentral_nuclei["nucleus_y"],
        s=nucleus_dot_size,
        c=noncentral_nucleus_color,
        alpha=0.90,
        linewidths=0,
        label="Non-central nuclei"
    )

if len(central_nuclei) > 0:
    plt.scatter(
        central_nuclei["nucleus_x"],
        central_nuclei["nucleus_y"],
        s=nucleus_dot_size * 1.8,
        c=central_nucleus_color,
        alpha=0.95,
        linewidths=0,
        label=f"Central nuclei ({central_column_for_display})"
    )

plt.title(
    "All-fiber QC overlay: nuclei centrality and eMHC status",
    fontsize=8
)

plt.axis("off")

# Legend can be large on full-resolution images, so keep it small.
plt.legend(
    loc="upper right",
    fontsize=4,
    markerscale=4,
    framealpha=0.65
)

plt.tight_layout(pad=0)

print(f"Saving overlay to: {output_path}", flush=True)
plt.savefig(output_path, dpi=output_dpi, bbox_inches="tight", pad_inches=0)
plt.close()

print("Done.", flush=True)
