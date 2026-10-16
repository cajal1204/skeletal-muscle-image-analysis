import numpy as np
import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, segmentation


# Input files

laminin_image_path = Path("laminin.tif")
dapi_image_path = Path("dapi.tif")
dystrophin_image_path = Path("dystrophin.tif")

fiber_mask_path = Path("fibers.tif")
nucleus_csv_path = Path("image02_nucleus_assignments.csv")
fiber_summary_path = Path("image02_fiber_summary_with_dystrophin.csv")

# Fiji opening settings

open_png_in_fiji = True

fiji_exe_path = Path(r"C:\Users\emurp\Downloads\fiji-latest-win64-jdk\Fiji\fiji-windows-x64.exe")

# Display settings

padding = 120  # pixels around the fiber to show

# True = only show nuclei that passed nucleus_qc_pass.
# This makes the plot match final central_nucleus / has_central_nucleus calls.
show_only_qc_passing_nuclei = True

# Options:
# "central_nucleus" = final QC-filtered central nucleus call
# "central_nucleus_raw" = raw distance-based central call before QC filtering
central_column_for_display = "central_nucleus"

# If True, failed-QC nuclei assigned to the target fiber are shown faintly.
show_failed_qc_nuclei_faint = False


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

def open_image_in_fiji(image_path):
    """
    Open the saved PNG in Fiji through PowerShell Start-Process.
    This avoids Fiji/Java taking over the Anaconda Prompt.
    """
    if not open_png_in_fiji:
        return

    if not fiji_exe_path.exists():
        print(f"Fiji executable not found: {fiji_exe_path}", flush=True)
        print("PNG was saved, but it was not opened in Fiji.", flush=True)
        return

    image_path = image_path.resolve()

    def ps_quote(x):
        # Safely quote Windows paths for PowerShell
        return "'" + str(x).replace("'", "''") + "'"

    try:
        command = (
            f"Start-Process "
            f"-FilePath {ps_quote(fiji_exe_path)} "
            f"-ArgumentList {ps_quote(image_path)} "
            f"-WorkingDirectory {ps_quote(fiji_exe_path.parent)}"
        )

        subprocess.Popen(
            [
                "powershell",
                "-NoProfile",
                "-WindowStyle",
                "Hidden",
                "-Command",
                command
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
            close_fds=True
        )

        print(f"Opened in Fiji: {image_path.name}", flush=True)

    except Exception as e:
        print(f"Could not open image in Fiji: {e}", flush=True)

def scale_crop_using_full_image(full_img, crop_img, low=1, high=99.5):
    """
    Scale a crop using intensity percentiles from the full image.
    This prevents each crop from being independently stretched.
    """
    p_low, p_high = np.percentile(full_img, (low, high))

    if p_high <= p_low:
        return np.zeros_like(crop_img, dtype=float)

    scaled = (crop_img - p_low) / (p_high - p_low)
    return np.clip(scaled, 0, 1)


def make_lookup_image(target_fiber_id):
    """
    Create one lookup image for the requested fiber ID.
    """

    print(f"\nMaking lookup image for fiber {target_fiber_id}...", flush=True)

    # Find target fiber

    fiber_row_df = fibers_df[
        fibers_df["fiber_id"].astype(int) == int(target_fiber_id)
    ]

    if len(fiber_row_df) == 0:
        print(f"Fiber ID {target_fiber_id} was not found in the fiber summary.")
        return False

    if len(fiber_row_df) > 1:
        print(f"Fiber ID {target_fiber_id} appears more than once. Skipping.")
        return False

    fiber_row = fiber_row_df.iloc[0]

    fiber_mask = fiber_labels == target_fiber_id
    ys, xs = np.where(fiber_mask)

    if len(ys) == 0:
        print(f"No pixels found for fiber ID {target_fiber_id} in fibers.tif.")
        return False

    y_min = max(ys.min() - padding, 0)
    y_max = min(ys.max() + padding + 1, fiber_labels.shape[0])
    x_min = max(xs.min() - padding, 0)
    x_max = min(xs.max() + padding + 1, fiber_labels.shape[1])

    print(f"Cropping to y:{y_min}-{y_max}, x:{x_min}-{x_max}", flush=True)

    # Crop images

    laminin_crop = laminin[y_min:y_max, x_min:x_max]
    dapi_crop = dapi[y_min:y_max, x_min:x_max]
    dystrophin_crop = dystrophin[y_min:y_max, x_min:x_max]
    fiber_crop = fiber_labels[y_min:y_max, x_min:x_max]

    # Display scaling

    laminin_disp = scale_crop_using_full_image(
        laminin,
        laminin_crop,
        low=1,
        high=99.5
    )

    dapi_disp = scale_crop_using_full_image(
        dapi,
        dapi_crop,
        low=1,
        high=99.5
    )

    dystrophin_disp = scale_crop_using_full_image(
        dystrophin,
        dystrophin_crop,
        low=1,
        high=99.99
    )

    # Build merged RGB image

    # dystrophin = red
    # laminin = pale white
    # DAPI = blue
    laminin_weight = 0.45

    rgb = np.dstack([
        np.clip(dystrophin_disp + laminin_weight * laminin_disp, 0, 1),
        np.clip(laminin_weight * laminin_disp, 0, 1),
        np.clip(dapi_disp + laminin_weight * laminin_disp, 0, 1)
    ])

    # Boundaries

    all_boundaries = segmentation.find_boundaries(fiber_crop, mode="inner")
    target_boundary = segmentation.find_boundaries(
        fiber_crop == target_fiber_id,
        mode="inner"
    )

    # Nuclei in this crop

    crop_nuclei = nuclei_df[
        (nuclei_df["nucleus_x"] >= x_min) &
        (nuclei_df["nucleus_x"] < x_max) &
        (nuclei_df["nucleus_y"] >= y_min) &
        (nuclei_df["nucleus_y"] < y_max)
    ].copy()

    crop_nuclei["plot_x"] = crop_nuclei["nucleus_x"] - x_min
    crop_nuclei["plot_y"] = crop_nuclei["nucleus_y"] - y_min

    target_nuclei_all = crop_nuclei[
        crop_nuclei["assigned_fiber_id"].astype(int) == int(target_fiber_id)
    ].copy()

    # Split target nuclei into QC-passing and QC-failing groups
    if "nucleus_qc_pass" in target_nuclei_all.columns:
        target_nuclei_all["nucleus_qc_pass_bool"] = as_bool_series(
            target_nuclei_all["nucleus_qc_pass"]
        )

        target_nuclei_qc_pass = target_nuclei_all[
            target_nuclei_all["nucleus_qc_pass_bool"]
        ].copy()

        target_nuclei_qc_fail = target_nuclei_all[
            ~target_nuclei_all["nucleus_qc_pass_bool"]
        ].copy()

    else:
        print(
            "Warning: nucleus_qc_pass column not found. Showing all assigned nuclei.",
            flush=True
        )

        target_nuclei_qc_pass = target_nuclei_all.copy()
        target_nuclei_qc_fail = target_nuclei_all.iloc[0:0].copy()

    # Decide which nuclei to show normally
    if show_only_qc_passing_nuclei:
        target_nuclei = target_nuclei_qc_pass.copy()
    else:
        target_nuclei = target_nuclei_all.copy()

    # Check central display column
    if central_column_for_display not in nuclei_df.columns:
        raise ValueError(
            f"Column '{central_column_for_display}' was not found in nucleus CSV.\n"
            f"Available columns are:\n{list(nuclei_df.columns)}"
        )

    if len(target_nuclei) > 0:
        target_nuclei["central_display_bool"] = as_bool_series(
            target_nuclei[central_column_for_display]
        )

        central_target_nuclei = target_nuclei[
            target_nuclei["central_display_bool"]
        ].copy()

    else:
        central_target_nuclei = target_nuclei.copy()

    # Plot

    plt.figure(figsize=(9, 9))
    plt.imshow(rgb)

    # Faint boundaries of all fibers in crop
    by, bx = np.where(all_boundaries)
    plt.scatter(
        bx,
        by,
        s=0.1,
        alpha=0.20
    )

    # Strong boundary for target fiber
    ty, tx = np.where(target_boundary)
    plt.scatter(
        tx,
        ty,
        s=0.6,
        alpha=0.95,
        label="Target fiber boundary"
    )

    # Optional: show failed-QC nuclei faintly
    if show_failed_qc_nuclei_faint and len(target_nuclei_qc_fail) > 0:
        plt.scatter(
            target_nuclei_qc_fail["plot_x"],
            target_nuclei_qc_fail["plot_y"],
            s=18,
            alpha=0.25,
            label="Failed-QC nuclei assigned to target fiber"
        )

    # Nuclei assigned to target fiber
    if len(target_nuclei) > 0:
        plt.scatter(
            target_nuclei["plot_x"],
            target_nuclei["plot_y"],
            s=20,
            label="QC-passing nuclei in target fiber"
            if show_only_qc_passing_nuclei
            else "Nuclei in target fiber"
        )

    # Central nuclei assigned to target fiber
    if len(central_target_nuclei) > 0:
        plt.scatter(
            central_target_nuclei["plot_x"],
            central_target_nuclei["plot_y"],
            s=15,
            label=f"Central nuclei ({central_column_for_display})"
        )

    # Label target fiber

    fiber_x = fiber_row["fiber_x"] - x_min
    fiber_y = fiber_row["fiber_y"] - y_min

    plt.text(
        fiber_x,
        fiber_y,
        f"Fiber {target_fiber_id}",
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(facecolor="black", alpha=0.4, pad=2)
    )

    # Build title

    title_parts = [f"Fiber {target_fiber_id}"]

    if "fiber_area_um2" in fiber_row.index and pd.notna(fiber_row["fiber_area_um2"]):
        title_parts.append(f"area={fiber_row['fiber_area_um2']:.1f} µm²")
    elif "fiber_area_px" in fiber_row.index and pd.notna(fiber_row["fiber_area_px"]):
        title_parts.append(f"area={fiber_row['fiber_area_px']:.0f} px")

    if "fiber_qc_pass" in fiber_row.index:
        title_parts.append(f"fiber_qc={fiber_row['fiber_qc_pass']}")

    if "analysis_qc_pass" in fiber_row.index:
        title_parts.append(f"analysis_qc={fiber_row['analysis_qc_pass']}")

    if "has_central_nucleus" in fiber_row.index:
        title_parts.append(f"central_strict={fiber_row['has_central_nucleus']}")

    if "central_nucleation_loose" in fiber_row.index:
        title_parts.append(f"central_loose={fiber_row['central_nucleation_loose']}")

    if "central_nucleation_loose_source" in fiber_row.index:
        title_parts.append(
            f"loose_source={fiber_row['central_nucleation_loose_source']}"
        )

    if "total_nuclei" in fiber_row.index and pd.notna(fiber_row["total_nuclei"]):
        title_parts.append(f"nuclei={int(fiber_row['total_nuclei'])}")

    if "central_nuclei" in fiber_row.index and pd.notna(fiber_row["central_nuclei"]):
        title_parts.append(f"central_nuclei={int(fiber_row['central_nuclei'])}")

    if (
        "dystrophin_membrane_coverage_pct" in fiber_row.index
        and pd.notna(fiber_row["dystrophin_membrane_coverage_pct"])
    ):
        title_parts.append(
            f"dystrophin cov={fiber_row['dystrophin_membrane_coverage_pct']:.1f}%"
        )

    if "dystrophin_positive" in fiber_row.index:
        title_parts.append(f"dystrophin+={fiber_row['dystrophin_positive']}")

    if "dystrophin_class" in fiber_row.index:
        title_parts.append(f"class={fiber_row['dystrophin_class']}")

    if "dystrophin_manual_override" in fiber_row.index:
        title_parts.append(f"manual_override={fiber_row['dystrophin_manual_override']}")

    plt.title(" | ".join(title_parts))
    plt.axis("off")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    # Save output

    output_path = Path(f"fiber_{target_fiber_id}_lookup_multichannel.png")

    print(f"Saving to {output_path}...", flush=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}", flush=True)

    open_image_in_fiji(output_path)

    return True


# Load data once

print("Loading data...", flush=True)

laminin = io.imread(laminin_image_path)
dapi = io.imread(dapi_image_path)
dystrophin = io.imread(dystrophin_image_path)
fiber_labels = io.imread(fiber_mask_path)

nuclei_df = pd.read_csv(nucleus_csv_path)
fibers_df = pd.read_csv(fiber_summary_path)

if laminin.ndim > 2:
    laminin = laminin[..., 0]

if dapi.ndim > 2:
    print(f"DAPI image has multiple channels: {dapi.shape}", flush=True)
    dapi = dapi[..., 2]

if dystrophin.ndim > 2:
    dystrophin = dystrophin[..., 0]

if fiber_labels.ndim > 2:
    fiber_labels = fiber_labels[..., 0]

laminin = laminin.astype(float)
dapi = dapi.astype(float)
dystrophin = dystrophin.astype(float)
fiber_labels = fiber_labels.astype(np.int32)

# Check shapes
shape_ref = fiber_labels.shape

for name, arr in [
    ("laminin", laminin),
    ("dapi", dapi),
    ("dystrophin", dystrophin),
]:
    if arr.shape != shape_ref:
        raise ValueError(
            f"{name} image shape {arr.shape} does not match fiber mask shape {shape_ref}"
        )

print("Data loaded.", flush=True)
print(f"Available fibers in summary: {len(fibers_df)}", flush=True)


# Prompt loop

print("")
print("Enter a fiber ID to create a lookup image.")
print("You can enter one ID at a time:")
print("  1435")
print("Or several IDs separated by commas or spaces:")
print("  1435, 1068, 2950")
print("Press Enter with no ID to quit.")
print("")

created_ids = []

while True:
    user_input = input("Fiber ID(s) to look up: ").strip()

    if user_input == "":
        break

    cleaned_input = user_input.replace(",", " ")
    parts = cleaned_input.split()

    fiber_ids = []

    for part in parts:
        try:
            fiber_ids.append(int(part))
        except ValueError:
            print(f"Skipping invalid entry: {part}", flush=True)

    if len(fiber_ids) == 0:
        print("No valid fiber IDs entered.", flush=True)
        continue

    for fiber_id in fiber_ids:
        success = make_lookup_image(fiber_id)
        if success:
            created_ids.append(fiber_id)

print("")
print("Done.", flush=True)

if len(created_ids) > 0:
    print(f"Created {len(created_ids)} lookup image(s):", flush=True)
    print(created_ids, flush=True)
else:
    print("No lookup images were created.", flush=True)