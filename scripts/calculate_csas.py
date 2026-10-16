print("Starting fiber area measurement...", flush=True)

import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io
from skimage.measure import label, regionprops_table


# User settings

fiber_mask_path = Path("fibers.tif")
output_csv_path = Path("fiber_cross_sectional_areas.csv")

# Change this to the pixel size of your image.
pixel_size_um = 1.2886


# Load fiber mask

print("Loading fibers.tif...", flush=True)

fiber_image = io.imread(fiber_mask_path)

# If the TIFF was saved as RGB, use its first channel.
if fiber_image.ndim > 2:
    fiber_image = fiber_image[..., 0]

if fiber_image.ndim != 2:
    raise ValueError(
        f"Expected a 2D fiber mask, but the image shape is {fiber_image.shape}"
    )


# Determine whether the image is binary or already labeled

positive_values = np.unique(fiber_image[fiber_image > 0])

if len(positive_values) <= 1:
    print("Binary mask detected; labeling connected fibers...", flush=True)

    # connectivity=1 uses edge-connected pixels.
    # Change to connectivity=2 if diagonally touching pixels should be connected.
    fiber_labels = label(fiber_image > 0, connectivity=1)
else:
    print("Labeled fiber mask detected...", flush=True)
    fiber_labels = fiber_image.astype(np.int32)


# Measure fiber cross-sectional areas

fiber_properties = regionprops_table(
    fiber_labels,
    properties=["label", "area"],
)

fibers_df = pd.DataFrame(fiber_properties)

fibers_df = fibers_df.rename(
    columns={
        "label": "fiber_id",
        "area": "fiber_area_px",
    }
)

# Convert pixel area to square micrometers.
fibers_df["fiber_area_um2"] = (
    fibers_df["fiber_area_px"] * pixel_size_um**2
)


# Save results

fibers_df.to_csv(output_csv_path, index=False)

print(f"Measured {len(fibers_df)} fibers.", flush=True)
print(
    f"Median fiber area: {fibers_df['fiber_area_um2'].median():.2f} um²",
    flush=True,
)
print(f"Results saved to: {output_csv_path}", flush=True)
print("Finished.", flush=True)