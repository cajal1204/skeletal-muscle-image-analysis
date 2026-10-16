print("Starting script...", flush=True)

import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io
from skimage.measure import regionprops_table
from scipy.ndimage import distance_transform_edt


# User inputs

fiber_mask_path = Path("fibers.tif")
nuclei_mask_path = Path("nuclei.tif")
laminin_image_path = Path("laminin.tif")
dapi_image_path = Path("dapi.tif")

output_nucleus_csv = Path("image02_nucleus_assignments.csv")
output_fiber_csv = Path("image02_fiber_summary.csv")

# Manual exclusion files
manual_exclude_nuclei_path = Path("exclude_nuclei.csv")
manual_exclude_fibers_path = Path("exclude_fibers.csv")


# Scale and analysis settings

# Use the actual pixel size for the image (before any resizing). 
#
# Full-size example:
#pixel_size_um = 1.2886
#
#Current image:
pixel_size_um = 1.2886
#
# Half-size example:
#pixel_size_um = 1

standard_pixel_size_um = 1.2886
standard_min_fiber_area_px = 100

# Convert original fiber area cutoff to physical area then back to pixels.
min_fiber_area_um2 = standard_min_fiber_area_px * (standard_pixel_size_um ** 2)
min_fiber_area_px = min_fiber_area_um2 / (pixel_size_um ** 2)

# Nucleus is "central" if its centroid is at least this far from a fiber border.
central_core_distance_um = 8.0

# Nucleus must have at least this fraction of its area inside the assigned fiber.
min_nucleus_fiber_overlap_fraction = 0.50

# Nucleus size filters, also scaled by physical area.
standard_min_nucleus_area_px = 15
standard_max_nucleus_area_px = 500

min_nucleus_area_um2 = standard_min_nucleus_area_px * (standard_pixel_size_um ** 2)
max_nucleus_area_um2 = standard_max_nucleus_area_px * (standard_pixel_size_um ** 2)

min_nucleus_area_px = min_nucleus_area_um2 / (pixel_size_um ** 2)
max_nucleus_area_px = max_nucleus_area_um2 / (pixel_size_um ** 2)

print(f"Pixel size: {pixel_size_um:.6f} um/px", flush=True)
print(f"Area conversion factor: {pixel_size_um ** 2:.6f} um2/px", flush=True)
print(f"Min fiber area cutoff: {min_fiber_area_um2:.2f} um2", flush=True)
print(f"Min fiber area cutoff: {min_fiber_area_px:.2f} px", flush=True)
print(f"Min nucleus area cutoff: {min_nucleus_area_px:.2f} px", flush=True)
print(f"Max nucleus area cutoff: {max_nucleus_area_px:.2f} px", flush=True)


# Area erosion setting

# Convert that physical erosion distance to the current image scale.
standard_area_erosion_um = standard_pixel_size_um * 1.0

area_erosion_px_float = standard_area_erosion_um / pixel_size_um

# If the scaled erosion is less than 0.75 px, do not erode.
# This avoids over-eroding half-size images. I actually hate this deeply, so I'm fixing it.
# if area_erosion_px_float < 0:
#     area_erosion_px = 0
# else:
area_erosion_px = int(round(area_erosion_px_float))

print(
    f"Area erosion: {area_erosion_px} px "
    f"({area_erosion_px * pixel_size_um:.2f} um)",
    flush=True
)


# Laminin QC for central nucleus classification

# If a candidate central nucleus lies in a laminin-rich local region it will not count as a central nucleus.
use_laminin_central_nucleus_qc = True

# Radius around the nucleus centroid used to measure local laminin.
nucleus_laminin_local_radius_um = 1.0

# A candidate central nucleus fails if local laminin is more than this fraction of that fiber's laminin membrane mean.
# Lower = stricter. Higher = more permissive.
max_nucleus_laminin_local_to_membrane_ratio_for_central = 0.60


# DAPI core signal fallback settings

# This adds a looser fiber-level central nucleation call.
# It helps catch fibers where StarDist missed a dense DAPI blob/cloud in the core.
use_dapi_core_signal_fallback = True

# Core definition uses this distance from the fiber border.
dapi_core_distance_um = central_core_distance_um

# DAPI-positive pixels are defined using this percentile of nonzero DAPI pixels.
# Lower = more permissive.
# Higher = stricter.
dapi_signal_threshold_percentile = 95

# Minimum number of DAPI-positive pixels required in the fiber core.
min_dapi_core_positive_px = 10

# Minimum percent of the fiber core that must be DAPI-positive.
# Nuclei usually occupy a small fraction of the fiber core.
min_dapi_core_coverage_pct = 1.0


# Laminin solid-core artifact filter settings

# Width of the membrane band just inside the fiber border.
# This is where laminin should be strongest in a real fiber.
membrane_band_width_um = 3.0

# Distance from the border where the "core" starts.
# A real fiber should usually have relatively low laminin here.
core_start_distance_um = 10.0

# If the core laminin is more than this fraction of the membrane laminin,
# flag the object as a likely solid-core artifact.
max_laminin_core_to_membrane_ratio = 0.70

# If a fiber has too few core pixels, do not fail it using this filter.
min_core_pixels_for_laminin_qc = 20


# Laminin membrane signal QC settings

# Estimate laminin background from non-fiber/background pixels.
# A fiber membrane is considered laminin-positive if membrane pixels are above:
# background median + threshold_mad_factor * robust noise.
laminin_threshold_mad_factor = 0

# Require at least this percent of the membrane band to be laminin-positive.
min_laminin_membrane_coverage_pct = 30.0

# Require at least this many membrane pixels so tiny fragments do not pass.
min_laminin_membrane_pixels = 38


# Helper function

def read_exclusion_csv(path, id_column):
    """
    Reads a manual exclusion CSV safely.

    Expected format:
        id_column,exclude_reason

    Example:
        fiber_id,exclude_reason
        104,border artifact
    """
    if not path.exists():
        print(f"Exclusion file not found: {path}", flush=True)
        return pd.DataFrame(columns=[id_column, "exclude_reason"]), set()

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"Exclusion file is empty: {path}", flush=True)
        return pd.DataFrame(columns=[id_column, "exclude_reason"]), set()

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    print(f"Read exclusion file: {path}", flush=True)
    print(f"Columns found: {list(df.columns)}", flush=True)

    if id_column not in df.columns:
        raise ValueError(
            f"Expected column '{id_column}' was not found in {path}.\n"
            f"Columns found were: {list(df.columns)}\n"
            "Make sure the CSV has separate columns, not comma-separated text inside one Excel cell."
        )

    ids = set(df[id_column].dropna().astype(int))
    return df, ids


# Load images

print("Loading images...", flush=True)

fiber_labels = io.imread(fiber_mask_path)
nuclei_labels = io.imread(nuclei_mask_path)
laminin = io.imread(laminin_image_path)
dapi = io.imread(dapi_image_path)

if fiber_labels.ndim > 2:
    fiber_labels = fiber_labels[..., 0]

if nuclei_labels.ndim > 2:
    nuclei_labels = nuclei_labels[..., 0]

if laminin.ndim > 2:
    laminin = laminin[..., 0]

if dapi.ndim > 2:
    # If dapi.tif is RGB, DAPI is usually in the blue channel.
    print(f"DAPI image has multiple channels: {dapi.shape}", flush=True)
    dapi = dapi[..., 2]

fiber_labels = fiber_labels.astype(np.int32)
nuclei_labels = nuclei_labels.astype(np.int32)
laminin = laminin.astype(float)
dapi = dapi.astype(float)

print(f"Fiber mask shape: {fiber_labels.shape}", flush=True)
print(f"Nuclei mask shape: {nuclei_labels.shape}", flush=True)
print(f"Laminin image shape: {laminin.shape}", flush=True)
print(f"DAPI image shape: {dapi.shape}", flush=True)
print(f"Fiber labels: {len(np.unique(fiber_labels)) - 1}", flush=True)
print(f"Nuclei labels: {len(np.unique(nuclei_labels)) - 1}", flush=True)

if fiber_labels.shape != nuclei_labels.shape:
    raise ValueError(
        f"Fiber mask and nuclei mask have different shapes: "
        f"{fiber_labels.shape} vs {nuclei_labels.shape}"
    )

if laminin.shape != fiber_labels.shape:
    raise ValueError(
        f"Laminin image and fiber mask have different shapes: "
        f"{laminin.shape} vs {fiber_labels.shape}"
    )

if dapi.shape != fiber_labels.shape:
    raise ValueError(
        f"DAPI image and fiber mask have different shapes: "
        f"{dapi.shape} vs {fiber_labels.shape}"
    )


# Measure nuclei

print("Measuring nuclei...", flush=True)

nucleus_props = regionprops_table(
    nuclei_labels,
    properties=["label", "area", "centroid"],
)

nuclei_df = pd.DataFrame(nucleus_props)

nuclei_df = nuclei_df.rename(
    columns={
        "label": "nucleus_id",
        "centroid-0": "nucleus_y",
        "centroid-1": "nucleus_x",
        "area": "nucleus_area_px",
    }
)

nuclei_df["nucleus_area_um2"] = nuclei_df["nucleus_area_px"] * (pixel_size_um ** 2)

print(f"Measured nuclei: {len(nuclei_df)}", flush=True)


# Measure fibers

print("Measuring fibers...", flush=True)

fiber_props = regionprops_table(
    fiber_labels,
    properties=["label", "area", "centroid"],
)

fibers_df = pd.DataFrame(fiber_props)

fibers_df = fibers_df.rename(
    columns={
        "label": "fiber_id",
        "centroid-0": "fiber_y",
        "centroid-1": "fiber_x",
        "area": "fiber_area_px",
    }
)

fibers_df["fiber_area_um2"] = fibers_df["fiber_area_px"] * (pixel_size_um ** 2)

print(f"Measured fibers: {len(fibers_df)}", flush=True)


# Eroded fiber area measurement

print("Measuring eroded fiber areas...", flush=True)

boundary_for_area = np.zeros_like(fiber_labels, dtype=bool)

boundary_for_area[:-1, :] |= fiber_labels[:-1, :] != fiber_labels[1:, :]
boundary_for_area[1:, :] |= fiber_labels[1:, :] != fiber_labels[:-1, :]
boundary_for_area[:, :-1] |= fiber_labels[:, :-1] != fiber_labels[:, 1:]
boundary_for_area[:, 1:] |= fiber_labels[:, 1:] != fiber_labels[:, :-1]

boundary_for_area |= fiber_labels == 0

distance_to_border_px_map_for_area = distance_transform_edt(~boundary_for_area)

eroded_fiber_labels = fiber_labels.copy()

if area_erosion_px > 0:
    eroded_fiber_labels[distance_to_border_px_map_for_area <= area_erosion_px] = 0
else:
    print("Area erosion is 0 px for this image scale; using raw mask area.", flush=True)

eroded_props = regionprops_table(
    eroded_fiber_labels,
    properties=["label", "area"],
)

eroded_df = pd.DataFrame(eroded_props).rename(
    columns={
        "label": "fiber_id",
        "area": "fiber_area_px_eroded",
    }
)

eroded_df["fiber_area_um2_eroded"] = (
    eroded_df["fiber_area_px_eroded"] * (pixel_size_um ** 2)
)

fibers_df = fibers_df.rename(
    columns={
        "fiber_area_px": "fiber_area_px_raw",
        "fiber_area_um2": "fiber_area_um2_raw",
    }
)

fibers_df = fibers_df.merge(
    eroded_df,
    on="fiber_id",
    how="left"
)

fibers_df["fiber_area_px_eroded"] = fibers_df["fiber_area_px_eroded"].fillna(0)
fibers_df["fiber_area_um2_eroded"] = fibers_df["fiber_area_um2_eroded"].fillna(0)

# Main fiber area uses the eroded area.
# If area_erosion_px is 0, eroded area equals raw area.
fibers_df["fiber_area_px"] = fibers_df["fiber_area_px_eroded"]
fibers_df["fiber_area_um2"] = fibers_df["fiber_area_um2_eroded"]

print(
    f"Median raw fiber area: {fibers_df['fiber_area_um2_raw'].median():.2f} um2",
    flush=True
)

print(
    f"Median final fiber area: {fibers_df['fiber_area_um2'].median():.2f} um2",
    flush=True
)


# Fiber QC filtering

print("Applying fiber QC filters...", flush=True)

fibers_df["fiber_area_qc_pass"] = fibers_df["fiber_area_px"] >= min_fiber_area_px


# Laminin solid-core artifact filter

print("Applying laminin solid-core artifact filter...", flush=True)

boundary_for_fiber_qc = np.zeros_like(fiber_labels, dtype=bool)

boundary_for_fiber_qc[:-1, :] |= fiber_labels[:-1, :] != fiber_labels[1:, :]
boundary_for_fiber_qc[1:, :] |= fiber_labels[1:, :] != fiber_labels[:-1, :]
boundary_for_fiber_qc[:, :-1] |= fiber_labels[:, :-1] != fiber_labels[:, 1:]
boundary_for_fiber_qc[:, 1:] |= fiber_labels[:, 1:] != fiber_labels[:, :-1]

boundary_for_fiber_qc |= fiber_labels == 0

distance_to_border_px_map_for_fiber_qc = distance_transform_edt(~boundary_for_fiber_qc)

membrane_band_width_px = max(1, int(round(membrane_band_width_um / pixel_size_um)))
core_start_distance_px = max(1, int(round(core_start_distance_um / pixel_size_um)))

print(f"Laminin membrane band width: {membrane_band_width_px} px", flush=True)
print(f"Laminin core starts at: {core_start_distance_px} px from border", flush=True)

laminin_membrane_band = (
    (fiber_labels > 0)
    & (distance_to_border_px_map_for_fiber_qc > 0)
    & (distance_to_border_px_map_for_fiber_qc <= membrane_band_width_px)
)

laminin_core = (
    (fiber_labels > 0)
    & (distance_to_border_px_map_for_fiber_qc >= core_start_distance_px)
)

max_label = int(fiber_labels.max())

laminin_background_pixels = laminin[fiber_labels == 0]

laminin_background_median = np.median(laminin_background_pixels)
laminin_mad = np.median(np.abs(laminin_background_pixels - laminin_background_median))
laminin_robust_sigma = 1.4826 * laminin_mad

laminin_signal_threshold = (
    laminin_background_median
    + laminin_threshold_mad_factor * laminin_robust_sigma
)

print(f"Laminin background median: {laminin_background_median:.3f}", flush=True)
print(f"Laminin signal threshold: {laminin_signal_threshold:.3f}", flush=True)

membrane_labels = fiber_labels[laminin_membrane_band].astype(np.int32)
membrane_values = laminin[laminin_membrane_band]

core_labels = fiber_labels[laminin_core].astype(np.int32)
core_values = laminin[laminin_core]

membrane_px = np.bincount(membrane_labels, minlength=max_label + 1)
membrane_sum = np.bincount(membrane_labels, weights=membrane_values, minlength=max_label + 1)

core_px = np.bincount(core_labels, minlength=max_label + 1)
core_sum = np.bincount(core_labels, weights=core_values, minlength=max_label + 1)

membrane_positive = membrane_values >= laminin_signal_threshold

membrane_positive_px = np.bincount(
    membrane_labels[membrane_positive],
    minlength=max_label + 1
)

laminin_membrane_coverage_pct = np.full(max_label + 1, np.nan)

valid_membrane_for_coverage = membrane_px > 0
laminin_membrane_coverage_pct[valid_membrane_for_coverage] = (
    100
    * membrane_positive_px[valid_membrane_for_coverage]
    / membrane_px[valid_membrane_for_coverage]
)

membrane_mean = np.full(max_label + 1, np.nan)
core_mean = np.full(max_label + 1, np.nan)

valid_membrane = membrane_px > 0
valid_core = core_px > 0

membrane_mean[valid_membrane] = membrane_sum[valid_membrane] / membrane_px[valid_membrane]
core_mean[valid_core] = core_sum[valid_core] / core_px[valid_core]

core_to_membrane_ratio = np.full(max_label + 1, np.nan)
valid_ratio = valid_core & valid_membrane & (membrane_mean > 0)
core_to_membrane_ratio[valid_ratio] = core_mean[valid_ratio] / membrane_mean[valid_ratio]

laminin_qc_df = pd.DataFrame({
    "fiber_id": np.arange(max_label + 1),
    "laminin_membrane_px": membrane_px,
    "laminin_core_px": core_px,
    "laminin_membrane_mean": membrane_mean,
    "laminin_core_mean": core_mean,
    "laminin_core_to_membrane_ratio": core_to_membrane_ratio,
    "laminin_membrane_positive_px": membrane_positive_px,
    "laminin_membrane_coverage_pct": laminin_membrane_coverage_pct,
})

laminin_qc_df = laminin_qc_df[laminin_qc_df["fiber_id"] > 0].copy()

fibers_df = fibers_df.merge(
    laminin_qc_df,
    on="fiber_id",
    how="left"
)

fibers_df["laminin_solid_core_qc_pass"] = (
    (fibers_df["laminin_core_px"].fillna(0) < min_core_pixels_for_laminin_qc)
    | (
        fibers_df["laminin_core_to_membrane_ratio"].fillna(0)
        <= max_laminin_core_to_membrane_ratio
    )
)

fibers_df["laminin_solid_core_flag"] = ~fibers_df["laminin_solid_core_qc_pass"]

fibers_df["laminin_membrane_signal_qc_pass"] = (
    (fibers_df["laminin_membrane_px"].fillna(0) >= min_laminin_membrane_pixels)
    & (
        fibers_df["laminin_membrane_coverage_pct"].fillna(0)
        >= min_laminin_membrane_coverage_pct
    )
)

fibers_df["laminin_no_membrane_signal_flag"] = (
    ~fibers_df["laminin_membrane_signal_qc_pass"]
)


# Manual fiber exclusion

exclude_fibers_df, excluded_fiber_ids = read_exclusion_csv(
    manual_exclude_fibers_path,
    "fiber_id"
)

print(f"Manually excluded fibers: {len(excluded_fiber_ids)}", flush=True)

fibers_df["manual_exclude_fiber"] = (
    fibers_df["fiber_id"].astype(int).isin(excluded_fiber_ids)
)

if len(exclude_fibers_df) > 0 and "exclude_reason" in exclude_fibers_df.columns:
    reason_map = dict(
        zip(
            exclude_fibers_df["fiber_id"].dropna().astype(int),
            exclude_fibers_df["exclude_reason"].astype(str)
        )
    )

    fibers_df["fiber_exclude_reason"] = (
        fibers_df["fiber_id"].astype(int).map(reason_map).fillna("")
    )
else:
    fibers_df["fiber_exclude_reason"] = ""


# Final fiber QC status

fibers_df["fiber_qc_pass"] = (
    fibers_df["fiber_area_qc_pass"]
    & fibers_df["laminin_membrane_signal_qc_pass"]
    & fibers_df["laminin_solid_core_qc_pass"]
    & ~fibers_df["manual_exclude_fiber"]
)

print(f"Fibers passing area filter: {fibers_df['fiber_area_qc_pass'].sum()}", flush=True)
print(f"Fibers flagged as solid-core artifacts: {fibers_df['laminin_solid_core_flag'].sum()}", flush=True)
print(f"Fibers manually excluded: {fibers_df['manual_exclude_fiber'].sum()}", flush=True)
print(f"Fibers passing final QC: {fibers_df['fiber_qc_pass'].sum()}", flush=True)


# Assign each nucleus to a fiber using nucleus centroid

print("Assigning nuclei to fibers...", flush=True)

nucleus_y = np.rint(nuclei_df["nucleus_y"].values).astype(int)
nucleus_x = np.rint(nuclei_df["nucleus_x"].values).astype(int)

nucleus_y = np.clip(nucleus_y, 0, fiber_labels.shape[0] - 1)
nucleus_x = np.clip(nucleus_x, 0, fiber_labels.shape[1] - 1)

assigned_fiber_ids = fiber_labels[nucleus_y, nucleus_x]

nuclei_df["assigned_fiber_id"] = assigned_fiber_ids
nuclei_df["inside_fiber"] = nuclei_df["assigned_fiber_id"] > 0

print(f"Nuclei inside fibers: {nuclei_df['inside_fiber'].sum()}", flush=True)


# Central nucleus classification from segmented nuclei

print("Computing global fiber-border distance map...", flush=True)

boundary = np.zeros_like(fiber_labels, dtype=bool)

boundary[:-1, :] |= fiber_labels[:-1, :] != fiber_labels[1:, :]
boundary[1:, :] |= fiber_labels[1:, :] != fiber_labels[:-1, :]
boundary[:, :-1] |= fiber_labels[:, :-1] != fiber_labels[:, 1:]
boundary[:, 1:] |= fiber_labels[:, 1:] != fiber_labels[:, :-1]

boundary |= fiber_labels == 0

distance_to_border_px_map = distance_transform_edt(~boundary)

distance_to_border_px = distance_to_border_px_map[nucleus_y, nucleus_x]
distance_to_border_um = distance_to_border_px * pixel_size_um

nuclei_df["distance_to_fiber_border_px"] = np.where(
    nuclei_df["inside_fiber"],
    distance_to_border_px,
    np.nan
)

nuclei_df["distance_to_fiber_border_um"] = np.where(
    nuclei_df["inside_fiber"],
    distance_to_border_um,
    np.nan
)

nuclei_df["central_nucleus_raw"] = (
    nuclei_df["inside_fiber"]
    & (nuclei_df["distance_to_fiber_border_um"] >= central_core_distance_um)
)

print("Finished central nucleus classification.", flush=True)


# Nucleus QC filtering

print("Applying nucleus QC filters...", flush=True)

print("Computing nucleus-fiber overlap fractions...", flush=True)

nucleus_pixel_mask = nuclei_labels > 0

pixel_overlap_df = pd.DataFrame({
    "nucleus_id": nuclei_labels[nucleus_pixel_mask].astype(np.int32),
    "fiber_id_at_pixel": fiber_labels[nucleus_pixel_mask].astype(np.int32),
})

overlap_counts = (
    pixel_overlap_df
    .groupby(["nucleus_id", "fiber_id_at_pixel"])
    .size()
    .reset_index(name="overlap_px")
)

assigned_pairs = nuclei_df[["nucleus_id", "assigned_fiber_id"]].copy()
assigned_pairs["row_index"] = nuclei_df.index
assigned_pairs = assigned_pairs.rename(
    columns={"assigned_fiber_id": "fiber_id_at_pixel"}
)

assigned_overlap = assigned_pairs.merge(
    overlap_counts,
    on=["nucleus_id", "fiber_id_at_pixel"],
    how="left"
)

assigned_overlap["overlap_px"] = assigned_overlap["overlap_px"].fillna(0)
assigned_overlap = assigned_overlap.sort_values("row_index")

nuclei_df["overlap_px_with_assigned_fiber"] = assigned_overlap["overlap_px"].values
nuclei_df["fraction_nucleus_inside_assigned_fiber"] = (
    nuclei_df["overlap_px_with_assigned_fiber"] / nuclei_df["nucleus_area_px"]
)


# Check whether assigned fiber passes fiber QC
valid_fiber_ids = set(
    fibers_df.loc[fibers_df["fiber_qc_pass"], "fiber_id"].astype(int)
)

nuclei_df["assigned_fiber_qc_pass"] = (
    nuclei_df["assigned_fiber_id"].astype(int).isin(valid_fiber_ids)
)


# Manual nucleus exclusion
exclude_nuclei_df, excluded_nucleus_ids = read_exclusion_csv(
    manual_exclude_nuclei_path,
    "nucleus_id"
)

print(f"Manually excluded nuclei: {len(excluded_nucleus_ids)}", flush=True)

nuclei_df["manual_exclude_nucleus"] = (
    nuclei_df["nucleus_id"].astype(int).isin(excluded_nucleus_ids)
)

if len(exclude_nuclei_df) > 0 and "exclude_reason" in exclude_nuclei_df.columns:
    nucleus_reason_map = dict(
        zip(
            exclude_nuclei_df["nucleus_id"].dropna().astype(int),
            exclude_nuclei_df["exclude_reason"].astype(str)
        )
    )

    nuclei_df["nucleus_exclude_reason"] = (
        nuclei_df["nucleus_id"].astype(int).map(nucleus_reason_map).fillna("")
    )
else:
    nuclei_df["nucleus_exclude_reason"] = ""


# Semi-final nucleus QC pass/fail
nuclei_df["nucleus_area_qc_pass"] = (
    (nuclei_df["nucleus_area_px"] >= min_nucleus_area_px)
    & (nuclei_df["nucleus_area_px"] <= max_nucleus_area_px)
)

nuclei_df["nucleus_overlap_qc_pass"] = (
    nuclei_df["fraction_nucleus_inside_assigned_fiber"]
    >= min_nucleus_fiber_overlap_fraction
)

nuclei_df["nucleus_qc_pass"] = (
    nuclei_df["inside_fiber"]
    & nuclei_df["assigned_fiber_qc_pass"]
    & nuclei_df["nucleus_area_qc_pass"]
    & nuclei_df["nucleus_overlap_qc_pass"]
    & ~nuclei_df["manual_exclude_nucleus"]
)

# Final central nucleus call after QC, before laminin-local QC
nuclei_df["central_nucleus_before_laminin_qc"] = (
    nuclei_df["central_nucleus_raw"]
    & nuclei_df["nucleus_qc_pass"]
)

if use_laminin_central_nucleus_qc:
    print("Applying laminin-local QC to candidate central nuclei...", flush=True)

    nucleus_laminin_local_radius_px = max(
        1,
        int(round(nucleus_laminin_local_radius_um / pixel_size_um))
    )

    print(
        f"Nucleus-local laminin radius: "
        f"{nucleus_laminin_local_radius_um} um = "
        f"{nucleus_laminin_local_radius_px} px",
        flush=True
    )

    # Measure laminin in a small circular region around each nucleus centroid.
    nucleus_laminin_local_mean = np.full(len(nuclei_df), np.nan)
    nucleus_laminin_local_max = np.full(len(nuclei_df), np.nan)

    for i, (yy, xx) in enumerate(zip(nucleus_y, nucleus_x)):
        y0 = max(yy - nucleus_laminin_local_radius_px, 0)
        y1 = min(yy + nucleus_laminin_local_radius_px + 1, laminin.shape[0])
        x0 = max(xx - nucleus_laminin_local_radius_px, 0)
        x1 = min(xx + nucleus_laminin_local_radius_px + 1, laminin.shape[1])

        laminin_crop = laminin[y0:y1, x0:x1]

        yy_grid, xx_grid = np.ogrid[y0:y1, x0:x1]
        disk_mask = (
            (yy_grid - yy) ** 2
            + (xx_grid - xx) ** 2
            <= nucleus_laminin_local_radius_px ** 2
        )

        local_values = laminin_crop[disk_mask]

        if len(local_values) > 0:
            nucleus_laminin_local_mean[i] = np.mean(local_values)
            nucleus_laminin_local_max[i] = np.max(local_values)

    nuclei_df["nucleus_laminin_local_mean"] = nucleus_laminin_local_mean
    nuclei_df["nucleus_laminin_local_max"] = nucleus_laminin_local_max

    # Compare local laminin near the nucleus to the membrane laminin level
    # of that nucleus's assigned fiber.
    membrane_mean_map = (
        fibers_df
        .set_index("fiber_id")["laminin_membrane_mean"]
        .to_dict()
    )

    nuclei_df["assigned_fiber_laminin_membrane_mean"] = (
        nuclei_df["assigned_fiber_id"]
        .astype(int)
        .map(membrane_mean_map)
    )

    nuclei_df["nucleus_laminin_local_to_membrane_ratio"] = np.where(
        nuclei_df["assigned_fiber_laminin_membrane_mean"] > 0,
        (
            nuclei_df["nucleus_laminin_local_mean"]
            / nuclei_df["assigned_fiber_laminin_membrane_mean"]
        ),
        np.nan
    )

    # Pass if the local laminin around the nucleus is low relative to
    # the fiber's membrane laminin.
    #
    # NaN is treated as passing here because nuclei with bad assigned fibers
    # already fail nucleus_qc_pass through assigned_fiber_qc_pass.
    nuclei_df["nucleus_laminin_central_qc_pass"] = (
        nuclei_df["nucleus_laminin_local_to_membrane_ratio"].isna()
        | (
            nuclei_df["nucleus_laminin_local_to_membrane_ratio"]
            <= max_nucleus_laminin_local_to_membrane_ratio_for_central
        )
    )

else:
    nuclei_df["nucleus_laminin_local_mean"] = np.nan
    nuclei_df["nucleus_laminin_local_max"] = np.nan
    nuclei_df["assigned_fiber_laminin_membrane_mean"] = np.nan
    nuclei_df["nucleus_laminin_local_to_membrane_ratio"] = np.nan
    nuclei_df["nucleus_laminin_central_qc_pass"] = True


# Track which candidate central nuclei were removed only because of laminin.
nuclei_df["central_nucleus_laminin_excluded"] = (
    nuclei_df["central_nucleus_before_laminin_qc"]
    & ~nuclei_df["nucleus_laminin_central_qc_pass"]
)

# Final central nucleus call after all QC
nuclei_df["central_nucleus"] = (
    nuclei_df["central_nucleus_before_laminin_qc"]
    & nuclei_df["nucleus_laminin_central_qc_pass"]
)

print(f"Nuclei passing QC: {nuclei_df['nucleus_qc_pass'].sum()}", flush=True)
print(
    f"Candidate central nuclei before laminin QC: "
    f"{nuclei_df['central_nucleus_before_laminin_qc'].sum()}",
    flush=True
)
print(
    f"Central nuclei excluded due to local laminin: "
    f"{nuclei_df['central_nucleus_laminin_excluded'].sum()}",
    flush=True
)
print(f"Central nuclei after all QC: {nuclei_df['central_nucleus'].sum()}", flush=True)

print(
    f"Fibers flagged for weak/no laminin membrane signal: "
    f"{fibers_df['laminin_no_membrane_signal_flag'].sum()}",
    flush=True
)


# Summarize segmented nuclei per fiber

print("Summarizing per fiber...", flush=True)

nuclei_inside = nuclei_df[nuclei_df["nucleus_qc_pass"]].copy()

fiber_nuclei_summary = (
    nuclei_inside
    .groupby("assigned_fiber_id")
    .agg(
        total_nuclei=("nucleus_id", "count"),
        central_nuclei=("central_nucleus", "sum"),
    )
    .reset_index()
    .rename(columns={"assigned_fiber_id": "fiber_id"})
)

fibers_df = fibers_df.merge(
    fiber_nuclei_summary,
    on="fiber_id",
    how="left"
)

fibers_df["total_nuclei"] = fibers_df["total_nuclei"].fillna(0).astype(int)
fibers_df["central_nuclei"] = fibers_df["central_nuclei"].fillna(0).astype(int)
fibers_df["has_central_nucleus"] = fibers_df["central_nuclei"] > 0


# Optional loose central nucleation call using raw DAPI core signal

if use_dapi_core_signal_fallback:
    print("Measuring DAPI signal in fiber cores...", flush=True)

    dapi_core_distance_px = max(
        1,
        int(round(dapi_core_distance_um / pixel_size_um))
    )

    print(
        f"DAPI core starts at: {dapi_core_distance_px} px from border",
        flush=True
    )

    # Core = pixels inside fibers far enough from the border.
    dapi_core_mask = (
        (fiber_labels > 0)
        & (distance_to_border_px_map >= dapi_core_distance_px)
    )

    # Estimate DAPI threshold from nonzero DAPI pixels.
    # Avoids issue where empty background is exactly zero.
    nonzero_dapi = dapi[dapi > 0]

    if len(nonzero_dapi) == 0:
        raise ValueError("No nonzero DAPI pixels found. Check dapi.tif.")

    dapi_signal_threshold = np.percentile(
        nonzero_dapi,
        dapi_signal_threshold_percentile
    )

    print(
        f"DAPI signal threshold "
        f"({dapi_signal_threshold_percentile}th percentile of nonzero DAPI): "
        f"{dapi_signal_threshold:.3f}",
        flush=True
    )

    core_labels_for_dapi = fiber_labels[dapi_core_mask].astype(np.int32)
    core_dapi_values = dapi[dapi_core_mask]

    dapi_core_px = np.bincount(
        core_labels_for_dapi,
        minlength=max_label + 1
    )

    dapi_core_sum = np.bincount(
        core_labels_for_dapi,
        weights=core_dapi_values,
        minlength=max_label + 1
    )

    dapi_core_positive = core_dapi_values >= dapi_signal_threshold

    dapi_core_positive_px = np.bincount(
        core_labels_for_dapi[dapi_core_positive],
        minlength=max_label + 1
    )

    dapi_core_mean = np.full(max_label + 1, np.nan)
    dapi_core_coverage_pct = np.full(max_label + 1, np.nan)

    valid_dapi_core = dapi_core_px > 0

    dapi_core_mean[valid_dapi_core] = (
        dapi_core_sum[valid_dapi_core]
        / dapi_core_px[valid_dapi_core]
    )

    dapi_core_coverage_pct[valid_dapi_core] = (
        100
        * dapi_core_positive_px[valid_dapi_core]
        / dapi_core_px[valid_dapi_core]
    )

    dapi_core_df = pd.DataFrame({
        "fiber_id": np.arange(max_label + 1),
        "dapi_core_px": dapi_core_px,
        "dapi_core_mean": dapi_core_mean,
        "dapi_core_positive_px": dapi_core_positive_px,
        "dapi_core_coverage_pct": dapi_core_coverage_pct,
    })

    dapi_core_df = dapi_core_df[dapi_core_df["fiber_id"] > 0].copy()

    dapi_core_df["dapi_core_signal_threshold"] = dapi_signal_threshold
    dapi_core_df["dapi_core_threshold_method"] = (
        f"nonzero_dapi_p{dapi_signal_threshold_percentile}"
    )

    dapi_core_df["dapi_core_signal_flag"] = (
        (
            dapi_core_df["dapi_core_positive_px"].fillna(0)
            >= min_dapi_core_positive_px
        )
        & (
            dapi_core_df["dapi_core_coverage_pct"].fillna(0)
            >= min_dapi_core_coverage_pct
        )
    )

    fibers_df = fibers_df.merge(
        dapi_core_df,
        on="fiber_id",
        how="left"
    )

    fibers_df["dapi_core_signal_flag"] = (
        fibers_df["dapi_core_signal_flag"]
        .fillna(False)
        .astype(bool)
    )

    # Analysis version requires the fiber itself to pass QC.
    fibers_df["dapi_core_signal_flag_analysis"] = (
        fibers_df["fiber_qc_pass"]
        & fibers_df["dapi_core_signal_flag"]
    )

    # Looser central nucleation call:
    # True if the fiber has a QC-passing segmented central nucleus
    # OR has strong DAPI signal in the core.
    fibers_df["central_nucleation_loose"] = (
        fibers_df["fiber_qc_pass"]
        & (
            fibers_df["has_central_nucleus"]
            | fibers_df["dapi_core_signal_flag"]
        )
    )

    fibers_df["central_nucleation_loose_source"] = np.select(
        [
            fibers_df["has_central_nucleus"] & fibers_df["dapi_core_signal_flag_analysis"],
            fibers_df["has_central_nucleus"],
            fibers_df["dapi_core_signal_flag_analysis"],
        ],
        [
            "segmented_central_nucleus_and_dapi_core_signal",
            "segmented_central_nucleus",
            "dapi_core_signal_only",
        ],
        default="none"
    )

    print(
        f"Fibers with DAPI core signal: "
        f"{fibers_df['dapi_core_signal_flag'].sum()}",
        flush=True
    )

    print(
        f"QC-passing fibers with DAPI core signal: "
        f"{fibers_df['dapi_core_signal_flag_analysis'].sum()}",
        flush=True
    )

    print(
        f"Fibers with loose central nucleation: "
        f"{fibers_df['central_nucleation_loose'].sum()}",
        flush=True
    )

else:
    fibers_df["central_nucleation_loose"] = fibers_df["has_central_nucleus"]
    fibers_df["central_nucleation_loose_source"] = np.where(
        fibers_df["has_central_nucleus"],
        "segmented_central_nucleus",
        "none"
    )


# Save outputs

print("Saving CSV files...", flush=True)

nuclei_df.to_csv(output_nucleus_csv, index=False)
fibers_df.to_csv(output_fiber_csv, index=False)

print(f"Saved nucleus assignments to: {output_nucleus_csv}", flush=True)
print(f"Saved fiber summary to: {output_fiber_csv}", flush=True)
print("Done.", flush=True)