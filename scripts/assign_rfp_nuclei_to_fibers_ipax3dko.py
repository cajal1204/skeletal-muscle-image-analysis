"""Assign the RFP nuclei in nuclei.tif to fibers in fibers.tif.

This is a simplified adaptation of assign_nuclei_to_fibers_emhc.py.

Important: nuclei.tif was created from rfp.tif, so every labeled object in
nuclei.tif is already considered an RFP nucleus. This script never opens or
uses rfp.tif.

Assignment follows the original script: the fiber label at each nucleus
centroid is used as the assigned fiber. There is no overlap threshold.
"""

print("Starting RFP nucleus-to-fiber assignment...", flush=True)

from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io
from skimage.measure import regionprops_table


# User settings

root_dir = Path(".").resolve()
fiber_mask_name = "membrane_cp_masks.png"
nuclei_mask_name = "nuclei.tif"

# Retained from the original script for area measurements.
pixel_size_um = 1.2886

output_dir = root_dir / "ipax3dko_rfp_nucleus_results"


def load_2d_label_image(path):
    labels = np.squeeze(io.imread(path))
    if labels.ndim > 2:
        labels = labels[..., 0]
    if labels.ndim != 2:
        raise ValueError(
            f"Expected a 2D label image at {path}, but found shape {labels.shape}."
        )
    return labels.astype(np.int32)


def measure_fibers(fiber_labels):
    props = regionprops_table(
        fiber_labels,
        properties=["label", "area", "centroid"],
    )
    fibers = pd.DataFrame(props).rename(
        columns={
            "label": "fiber_id",
            "area": "fiber_area_px",
            "centroid-0": "fiber_y",
            "centroid-1": "fiber_x",
        }
    )
    fibers["fiber_area_um2"] = fibers["fiber_area_px"] * (pixel_size_um**2)
    return fibers


def measure_nuclei(nuclei_labels):
    props = regionprops_table(
        nuclei_labels,
        properties=["label", "area", "centroid"],
    )
    nuclei = pd.DataFrame(props).rename(
        columns={
            "label": "nucleus_id",
            "area": "nucleus_area_px",
            "centroid-0": "nucleus_y",
            "centroid-1": "nucleus_x",
        }
    )
    nuclei["nucleus_area_um2"] = nuclei["nucleus_area_px"] * (pixel_size_um**2)
    return nuclei


def analyze_image_folder(folder, fiber_path, nuclei_path):
    image_id = folder.relative_to(root_dir).as_posix()
    print(f"Processing: {image_id}", flush=True)

    fiber_labels = load_2d_label_image(fiber_path)
    nuclei_labels = load_2d_label_image(nuclei_path)

    if fiber_labels.shape != nuclei_labels.shape:
        raise ValueError(
            "Fiber and nucleus masks have different shapes: "
            f"{fiber_labels.shape} vs {nuclei_labels.shape}"
        )

    fibers = measure_fibers(fiber_labels)
    nuclei = measure_nuclei(nuclei_labels)

    # Assign nuclei exactly as in the original script: read the fiber label at
    # the rounded nucleus centroid. A value of 0 means the centroid is outside
    # all fibers.
    if len(nuclei) > 0:
        nucleus_y = np.rint(nuclei["nucleus_y"].to_numpy()).astype(int)
        nucleus_x = np.rint(nuclei["nucleus_x"].to_numpy()).astype(int)

        nucleus_y = np.clip(nucleus_y, 0, fiber_labels.shape[0] - 1)
        nucleus_x = np.clip(nucleus_x, 0, fiber_labels.shape[1] - 1)

        nuclei["assigned_fiber_id"] = fiber_labels[nucleus_y, nucleus_x]
    else:
        nuclei["assigned_fiber_id"] = pd.Series(dtype=np.int32)

    nuclei["inside_fiber"] = nuclei["assigned_fiber_id"] > 0
    nuclei["assignment_status"] = np.where(
        nuclei["inside_fiber"],
        "assigned",
        "centroid_outside_fiber",
    )

    assigned_nuclei = nuclei[nuclei["inside_fiber"]].copy()

    nucleus_counts = assigned_nuclei.groupby("assigned_fiber_id").size()
    nucleus_ids = assigned_nuclei.groupby("assigned_fiber_id")["nucleus_id"].agg(
        lambda values: ";".join(str(int(value)) for value in sorted(values))
    )

    fibers["rfp_nucleus_count"] = (
        fibers["fiber_id"].map(nucleus_counts).fillna(0).astype(int)
    )
    fibers["rfp_nucleus_ids"] = fibers["fiber_id"].map(nucleus_ids).fillna("")
    fibers["has_rfp_nucleus"] = fibers["rfp_nucleus_count"] > 0

    fibers.insert(0, "image_id", image_id)
    nuclei.insert(0, "image_id", image_id)

    fibers = fibers[
        [
            "image_id",
            "fiber_id",
            "rfp_nucleus_count",
            "rfp_nucleus_ids",
            "has_rfp_nucleus",
            "fiber_area_px",
            "fiber_area_um2",
            "fiber_y",
            "fiber_x",
        ]
    ]

    nuclei = nuclei[
        [
            "image_id",
            "nucleus_id",
            "assigned_fiber_id",
            "inside_fiber",
            "assignment_status",
            "nucleus_area_px",
            "nucleus_area_um2",
            "nucleus_y",
            "nucleus_x",
        ]
    ]

    assigned_count = int(nuclei["inside_fiber"].sum())
    summary = {
        "image_id": image_id,
        "total_fibers": int(len(fibers)),
        "fibers_with_rfp_nuclei": int(fibers["has_rfp_nucleus"].sum()),
        "fibers_without_rfp_nuclei": int((~fibers["has_rfp_nucleus"]).sum()),
        "total_rfp_nuclei": int(len(nuclei)),
        "assigned_rfp_nuclei": assigned_count,
        "unassigned_rfp_nuclei": int(len(nuclei) - assigned_count),
    }

    return fibers, nuclei, summary


# Find and analyze every image subfolder.

fiber_paths = sorted(root_dir.rglob(fiber_mask_name))

if not fiber_paths:
    raise FileNotFoundError(
        f"No {fiber_mask_name} files were found under {root_dir}."
    )

all_fiber_tables = []
all_nucleus_tables = []
summary_rows = []

for fiber_path in fiber_paths:
    folder = fiber_path.parent
    nuclei_path = folder / nuclei_mask_name

    if not nuclei_path.exists():
        print(f"Skipping {folder.name}: missing {nuclei_mask_name}", flush=True)
        continue

    try:
        fibers, nuclei, summary = analyze_image_folder(
            folder,
            fiber_path,
            nuclei_path,
        )
    except Exception as error:
        print(f"ERROR in {folder.name}: {type(error).__name__}: {error}", flush=True)
        continue

    all_fiber_tables.append(fibers)
    all_nucleus_tables.append(nuclei)
    summary_rows.append(summary)


if not all_fiber_tables:
    raise RuntimeError("No image folders were processed successfully.")

all_fibers = pd.concat(all_fiber_tables, ignore_index=True)

nonempty_nucleus_tables = [table for table in all_nucleus_tables if len(table) > 0]
if nonempty_nucleus_tables:
    all_nuclei = pd.concat(nonempty_nucleus_tables, ignore_index=True)
else:
    all_nuclei = all_nucleus_tables[0].iloc[0:0].copy()

all_fibers = all_fibers.sort_values(["image_id", "fiber_id"]).reset_index(drop=True)
all_nuclei = all_nuclei.sort_values(["image_id", "nucleus_id"]).reset_index(drop=True)

with_rfp_nuclei = all_fibers[all_fibers["has_rfp_nucleus"]].copy()
without_rfp_nuclei = all_fibers[~all_fibers["has_rfp_nucleus"]].copy()

summary = pd.DataFrame(summary_rows)
overall = {
    "image_id": "ALL_IMAGES",
    "total_fibers": int(len(all_fibers)),
    "fibers_with_rfp_nuclei": int(len(with_rfp_nuclei)),
    "fibers_without_rfp_nuclei": int(len(without_rfp_nuclei)),
    "total_rfp_nuclei": int(len(all_nuclei)),
    "assigned_rfp_nuclei": int(all_nuclei["inside_fiber"].sum()),
    "unassigned_rfp_nuclei": int((~all_nuclei["inside_fiber"]).sum()),
}
summary = pd.concat([pd.DataFrame([overall]), summary], ignore_index=True)


# Save simple CSV outputs. UTF-8 with BOM opens cleanly in Excel.

output_dir.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    output_dir / "analysis_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

with_rfp_nuclei.to_csv(
    output_dir / "fibers_with_rfp_nuclei.csv",
    index=False,
    encoding="utf-8-sig",
)

without_rfp_nuclei.to_csv(
    output_dir / "fibers_without_rfp_nuclei.csv",
    index=False,
    encoding="utf-8-sig",
)

all_nuclei.to_csv(
    output_dir / "rfp_nucleus_assignments.csv",
    index=False,
    encoding="utf-8-sig",
)


print(f"Results saved in: {output_dir}", flush=True)
print(f"Fibers with RFP nuclei: {len(with_rfp_nuclei)}", flush=True)
print(f"Fibers without RFP nuclei: {len(without_rfp_nuclei)}", flush=True)
print(f"Assigned RFP nuclei: {int(all_nuclei['inside_fiber'].sum())}", flush=True)
print("Done.", flush=True)
