import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, segmentation, exposure
from skimage.measure import regionprops_table


# Input files


background_image_path = Path("emhc.tif")
fiber_mask_path = Path("fibers.tif")
nucleus_csv_path = Path("image02_nucleus_assignments.csv")
fiber_csv_path = Path("image02_fiber_summary_with_emhc.csv")

output_overlay_path = Path("image02_qc_overlay_on_emhc.png")



# Display settings


# Set this to True if you want fiber ID numbers on the plot.
show_fiber_ids = True

# Only label fibers that have at least one central nucleus.
label_only_centrally_nucleated_fibers = True

# Dot size for central nuclei.
central_dot_size = 1

# Boundary transparency.
boundary_alpha = 0.9



# Helper functions


def as_bool(series):
    """Convert common CSV boolean representations safely."""
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )



# Load data


print("Loading data...", flush=True)

background = io.imread(background_image_path)
fiber_labels = io.imread(fiber_mask_path)
nuclei_df = pd.read_csv(nucleus_csv_path)
fibers_df = pd.read_csv(fiber_csv_path)

if background.ndim > 2:
    # eMHC is the green channel in an RGB export.
    background = background[..., 1]

if fiber_labels.ndim > 2:
    fiber_labels = fiber_labels[..., 0]

background = background.astype(float)
fiber_labels = fiber_labels.astype(np.int32)

print(f"Background shape: {background.shape}", flush=True)
print(f"Fiber mask shape: {fiber_labels.shape}", flush=True)

if background.shape != fiber_labels.shape:
    raise ValueError(
        f"Background image and fiber mask have different shapes: "
        f"{background.shape} vs {fiber_labels.shape}"
    )



# Normalize background for viewing


# This rescales brightness for display only.
# It does not change your quantitative measurements.
p_low, p_high = np.percentile(background, (1, 99.5))
background_display = exposure.rescale_intensity(
    background,
    in_range=(p_low, p_high),
    out_range=(0, 1)
)



# Fiber boundaries


print("Finding fiber boundaries...", flush=True)

boundaries = segmentation.find_boundaries(fiber_labels, mode="inner")



# Select central nuclei


central_df = nuclei_df[
    as_bool(nuclei_df["nucleus_qc_pass"])
    & as_bool(nuclei_df["central_nucleus"])
].copy()

print(f"Central nuclei: {len(central_df)}", flush=True)



# Get centrally nucleated fibers


central_fiber_ids = sorted(central_df["assigned_fiber_id"].dropna().astype(int).unique())

central_fibers_df = fibers_df[
    fibers_df["fiber_id"].isin(central_fiber_ids)
].copy()

print(f"Fibers with central nuclei: {len(central_fibers_df)}", flush=True)



# Plot overlay


print("Making overlay...", flush=True)

plt.figure(figsize=(14, 12))

# Original eMHC/background image
plt.imshow(background_display, cmap="gray")

# Fiber boundaries shown on top
boundary_y, boundary_x = np.where(boundaries)
plt.scatter(
    boundary_x,
    boundary_y,
    s=0.01,
    alpha=boundary_alpha,
    label="Fiber boundaries"
)

# Central nuclei
plt.scatter(
    central_df["nucleus_x"],
    central_df["nucleus_y"],
    s=central_dot_size,
    label="Central nuclei"
)

# Optional fiber ID labels
if show_fiber_ids:
    if label_only_centrally_nucleated_fibers:
        label_df = central_fibers_df
    else:
        label_df = fibers_df

    for _, row in label_df.iterrows():
        plt.text(
            row["fiber_x"],
            row["fiber_y"],
            str(int(row["fiber_id"])),
            fontsize=2,
            ha="center",
            va="center",
            color = "red"
        )

plt.title("QC Overlay: Central Nuclei on eMHC Image")
plt.axis("off")
plt.legend(loc="upper right", markerscale=3)
plt.tight_layout()

print(f"Saving overlay to {output_overlay_path}...", flush=True)
plt.savefig(output_overlay_path, dpi=700, bbox_inches="tight")
plt.close()

print("Done.", flush=True)
