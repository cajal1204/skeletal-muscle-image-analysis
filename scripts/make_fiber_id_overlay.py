import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from pathlib import Path
from skimage import io, segmentation, exposure



# Input files


background_image_path = Path("laminin.tif")
fiber_mask_path = Path("fibers.tif")

# Use the dystrophin-scored spreadsheet here, not the earlier fiber summary.
fiber_summary_path = Path("image02_fiber_summary_with_dystrophin.csv")

output_overlay_path = Path("image02_fiber_id_overlay_dystrophin_status.png")



# User options


# Choose which fibers to label:
#
# "all"
# "qc_pass_only"
# "central_only"
# "failed_qc_only"
# "failed_laminin_membrane_only"
# "failed_solid_core_only"
# "manual_excluded_only"
# "dystrophin_positive_only"
# "dystrophin_negative_only"
# "qc_pass_dystrophin_positive_only"
# "qc_pass_dystrophin_negative_only"
#
label_mode = "qc_pass_dystrophin_positive_only"

# How to write each label:
#
# "id"                  -> 3374
# "id_dystrophin"       -> 3374 D+
# "id_dystrophin_pct"   -> 3374 D+ 82.1%
#
label_text_mode = "id_dystrophin"

# Color label text by dystrophin status?
# True  = D+ red, D- cyan/blue, unknown white
# False = all labels white
color_labels_by_dystrophin = True

# Custom fiber IDs override label_mode when not None.
custom_fiber_ids = None
# Example:
# custom_fiber_ids = [101, 245, 876]

# Boundary display:
# True  = only show boundaries for selected/labeled fibers
# False = show boundaries for all segmented fibers
show_only_selected_boundaries = True

font_size = 1.5
boundary_alpha = 0.1
output_dpi = 600



# Helper functions


def require_column(df, column_name, mode_name):
    if column_name not in df.columns:
        raise ValueError(
            f"label_mode='{mode_name}' requires column '{column_name}', "
            f"but it was not found in {fiber_summary_path}."
        )


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


def get_dystrophin_status(row):
    """
    Return D+, D-, or D? for a fiber row.
    """
    if "dystrophin_positive" not in row.index or pd.isna(row["dystrophin_positive"]):
        return "D?"

    value = str(row["dystrophin_positive"]).strip().lower()

    if value in ["true", "1", "yes"]:
        return "D+"
    elif value in ["false", "0", "no"]:
        return "D-"
    else:
        return "D?"


def get_label_text(row):
    fiber_id = int(row["fiber_id"])
    status = get_dystrophin_status(row)

    if label_text_mode == "id":
        return str(fiber_id)

    elif label_text_mode == "id_dystrophin":
        return f"{fiber_id} {status}"

    elif label_text_mode == "id_dystrophin_pct":
        if "dystrophin_membrane_coverage_pct" in row.index and pd.notna(row["dystrophin_membrane_coverage_pct"]):
            return f"{fiber_id} {status} {row['dystrophin_membrane_coverage_pct']:.1f}%"
        else:
            return f"{fiber_id} {status}"

    else:
        raise ValueError(f"Unknown label_text_mode: {label_text_mode}")


def get_label_color(row):
    if not color_labels_by_dystrophin:
        return "white"

    status = get_dystrophin_status(row)

    if status == "D+":
        return "red"
    elif status == "D-":
        return "cyan"
    else:
        return "white"



# Load data


print("Loading data...", flush=True)

background = io.imread(background_image_path)
fiber_labels = io.imread(fiber_mask_path)
fibers_df = pd.read_csv(fiber_summary_path)

if background.ndim > 2:
    background = background[..., 0]

if fiber_labels.ndim > 2:
    fiber_labels = fiber_labels[..., 0]

background = background.astype(float)
fiber_labels = fiber_labels.astype(np.int32)

if background.shape != fiber_labels.shape:
    raise ValueError(
        f"Background and fiber mask have different shapes: "
        f"{background.shape} vs {fiber_labels.shape}"
    )

print(f"Background shape: {background.shape}", flush=True)
print(f"Fiber mask shape: {fiber_labels.shape}", flush=True)
print(f"Rows in fiber summary: {len(fibers_df)}", flush=True)



# Normalize background for display


print("Normalizing background for display...", flush=True)

p_low, p_high = np.percentile(background, (1, 99.5))

background_display = exposure.rescale_intensity(
    background,
    in_range=(p_low, p_high),
    out_range=(0, 1)
)



# Select which fibers to label


print(f"Selecting fibers using label_mode='{label_mode}'...", flush=True)

if custom_fiber_ids is not None:
    label_df = fibers_df[
        fibers_df["fiber_id"].astype(int).isin(custom_fiber_ids)
    ].copy()

elif label_mode == "all":
    label_df = fibers_df.copy()

elif label_mode == "qc_pass_only":
    require_column(fibers_df, "fiber_qc_pass", label_mode)
    label_df = fibers_df[as_bool(fibers_df["fiber_qc_pass"])].copy()

elif label_mode == "central_only":
    require_column(fibers_df, "has_central_nucleus", label_mode)
    label_df = fibers_df[as_bool(fibers_df["has_central_nucleus"])].copy()

elif label_mode == "failed_qc_only":
    require_column(fibers_df, "fiber_qc_pass", label_mode)
    label_df = fibers_df[~as_bool(fibers_df["fiber_qc_pass"])].copy()

elif label_mode == "failed_laminin_membrane_only":
    require_column(fibers_df, "laminin_membrane_signal_qc_pass", label_mode)
    label_df = fibers_df[
        ~as_bool(fibers_df["laminin_membrane_signal_qc_pass"])
    ].copy()

elif label_mode == "failed_solid_core_only":
    require_column(fibers_df, "laminin_solid_core_qc_pass", label_mode)
    label_df = fibers_df[
        ~as_bool(fibers_df["laminin_solid_core_qc_pass"])
    ].copy()

elif label_mode == "manual_excluded_only":
    require_column(fibers_df, "manual_exclude_fiber", label_mode)
    label_df = fibers_df[
        as_bool(fibers_df["manual_exclude_fiber"])
    ].copy()

elif label_mode == "dystrophin_positive_only":
    require_column(fibers_df, "dystrophin_positive", label_mode)
    label_df = fibers_df[
        as_bool(fibers_df["dystrophin_positive"])
    ].copy()

elif label_mode == "dystrophin_negative_only":
    require_column(fibers_df, "dystrophin_positive", label_mode)
    label_df = fibers_df[
        ~as_bool(fibers_df["dystrophin_positive"])
    ].copy()

elif label_mode == "qc_pass_dystrophin_positive_only":
    require_column(fibers_df, "fiber_qc_pass", label_mode)
    require_column(fibers_df, "dystrophin_positive", label_mode)
    label_df = fibers_df[
        as_bool(fibers_df["fiber_qc_pass"])
        & as_bool(fibers_df["dystrophin_positive"])
    ].copy()

elif label_mode == "qc_pass_dystrophin_negative_only":
    require_column(fibers_df, "fiber_qc_pass", label_mode)
    require_column(fibers_df, "dystrophin_positive", label_mode)
    label_df = fibers_df[
        as_bool(fibers_df["fiber_qc_pass"])
        & ~as_bool(fibers_df["dystrophin_positive"])
    ].copy()

else:
    raise ValueError(f"Unknown label_mode: {label_mode}")

print(f"Fibers to label: {len(label_df)}", flush=True)



# Compute boundaries


if show_only_selected_boundaries:
    print("Computing boundaries for selected fibers only...", flush=True)

    selected_ids = label_df["fiber_id"].dropna().astype(int).values

    display_labels = np.zeros_like(fiber_labels, dtype=np.int32)
    selected_mask = np.isin(fiber_labels, selected_ids)
    display_labels[selected_mask] = fiber_labels[selected_mask]

    boundaries = segmentation.find_boundaries(display_labels, mode="inner")

else:
    print("Computing boundaries for all fibers...", flush=True)
    boundaries = segmentation.find_boundaries(fiber_labels, mode="inner")



# Plot


print("Creating overlay...", flush=True)

plt.figure(figsize=(14, 12))
plt.imshow(background_display, cmap="gray")

boundary_y, boundary_x = np.where(boundaries)

plt.scatter(
    boundary_x,
    boundary_y,
    s=0.1,
    alpha=boundary_alpha
)

for _, row in label_df.iterrows():
    label_text = get_label_text(row)
    label_color = get_label_color(row)

    txt = plt.text(
        row["fiber_x"],
        row["fiber_y"],
        label_text,
        fontsize=font_size,
        color=label_color,
        ha="center",
        va="center",
        fontweight="bold"
    )

    txt.set_path_effects([
        path_effects.Stroke(linewidth=1.2, foreground="black"),
        path_effects.Normal()
    ])

plt.title(f"Fiber ID Overlay: {label_mode} | {label_text_mode}")
plt.axis("off")
plt.tight_layout()

print(f"Saving to {output_overlay_path}...", flush=True)

plt.savefig(
    output_overlay_path,
    dpi=output_dpi,
    bbox_inches="tight"
)

plt.close()

print("Done.", flush=True)