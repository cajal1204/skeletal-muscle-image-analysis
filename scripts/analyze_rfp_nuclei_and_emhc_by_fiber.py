#!/usr/bin/env python3
"""Assign RFP-positive nuclei to fibers and score eMHC in one workflow.

This script combines the two attached analyses:

1. Assign each pre-labeled RFP-positive nucleus to the fiber with which it has
   the greatest overlap.
2. Measure eMHC intensity in the cytoplasmic core of each fiber and classify
   the fiber as eMHC-positive or eMHC-negative.

Expected files in each image folder
-----------------------------------
* ``laminin_cp_masks.png``: labeled fiber mask (0 = background; one positive
  integer label per fiber)
* ``rfp_stardist_labels.tif`` (or ``rfp_starfist_labels.tif``): labeled
  RFP-positive nucleus mask
* ``emhc.tif``: eMHC intensity image

``rfp.tif`` and ``laminin.tif`` may also be present, but they are not read:
their derived label masks contain the information needed by this analysis.

The root supplied on the command line can be one image folder or a parent
folder containing many image folders. The default is the current folder.

In addition to the combined tables, the script creates
``combined_rfp_emhc_results/individual_image_csvs``. That folder contains one
complete fiber-summary CSV for every successfully processed image.

Typical use
-----------
Run from the image folder or experiment root:

    python -u analyze_rfp_nuclei_and_emhc_by_fiber.py

Use any nucleus/fiber overlap as sufficient for assignment:

    python -u analyze_rfp_nuclei_and_emhc_by_fiber.py --min-overlap 0

Add areas in square microns when one pixel is 0.65 microns wide:

    python -u analyze_rfp_nuclei_and_emhc_by_fiber.py --pixel-size-um 0.65

Required packages: numpy, pandas, and scipy, plus either scikit-image or Pillow.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

try:
    from skimage import io as skimage_io
    from skimage.measure import label as skimage_label
    from skimage.measure import regionprops_table as skimage_regionprops_table
except ImportError:  # Pillow/SciPy fallback for environments without scikit-image.
    skimage_io = None
    skimage_label = None
    skimage_regionprops_table = None
    from PIL import Image
    from scipy.ndimage import find_objects as scipy_find_objects
    from scipy.ndimage import generate_binary_structure
    from scipy.ndimage import label as scipy_label


DEFAULT_OUTPUT_DIR = "combined_rfp_emhc_results"
DEFAULT_FIBER_FILE = "laminin_cp_masks.png"
DEFAULT_EMHC_FILE = "emhc.tif"
NUCLEUS_MASK_STEMS = (
    "rfp_stardist_labels",
    "rfp_starfist_labels",  # Accept the common filename typo as well.
)
MASK_EXTENSIONS = {".tif", ".tiff", ".png"}

PER_IMAGE_FIBER_CSV = "combined_fiber_summary.csv"
PER_IMAGE_NUCLEUS_CSV = "combined_rfp_nucleus_assignments.csv"
PER_IMAGE_SCORED_MASK = "combined_emhc_scored_region_labels.tif"
INDIVIDUAL_IMAGE_CSV_DIR = "individual_image_csvs"


@dataclass(frozen=True)
class ImageFiles:
    folder: Path
    fiber_mask: Path
    nuclei_mask: Path | None
    emhc_image: Path | None


@dataclass(frozen=True)
class EmhcSettings:
    core_border_exclusion_px: int
    min_core_pixels: int
    threshold_mode: str
    fixed_threshold: float
    fixed_negative_mean: float
    negative_fibers_file: Path
    negative_percentile: float
    min_bg_corrected: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign labeled RFP nuclei to labeled muscle fibers and score "
            "eMHC intensity for the same fibers."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("."),
        help="One image folder or an experiment root to scan (default: current folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Combined-results directory. A relative path is placed under root "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--fiber-file",
        default=DEFAULT_FIBER_FILE,
        help=f"Fiber label-mask filename (default: {DEFAULT_FIBER_FILE}).",
    )
    parser.add_argument(
        "--nuclei-file",
        default=None,
        help=(
            "RFP nucleus label-mask filename. By default the script detects "
            "rfp_stardist_labels or rfp_starfist_labels with a TIFF/PNG extension."
        ),
    )
    parser.add_argument(
        "--emhc-file",
        default=DEFAULT_EMHC_FILE,
        help=f"eMHC intensity-image filename (default: {DEFAULT_EMHC_FILE}).",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.50,
        help=(
            "Minimum fraction of a nucleus inside its best fiber for assignment "
            "(default: 0.50; use 0 to accept any nonzero overlap)."
        ),
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="Optional pixel width in microns; adds fiber/nucleus area in um^2.",
    )

    emhc = parser.add_argument_group("eMHC scoring")
    emhc.add_argument(
        "--core-border-exclusion-px",
        type=int,
        default=3,
        help="Pixels excluded inward from each fiber border (default: 3).",
    )
    emhc.add_argument(
        "--min-core-pixels",
        type=int,
        default=5,
        help=(
            "Use the whole fiber if erosion leaves fewer pixels than this "
            "(default: 5)."
        ),
    )
    emhc.add_argument(
        "--threshold-mode",
        choices=("fixed", "negative_fibers"),
        default="fixed",
        help="How the eMHC-positive threshold is obtained (default: fixed).",
    )
    emhc.add_argument(
        "--fixed-emhc-threshold",
        type=float,
        default=20.0,
        help="Mean scored-region threshold in fixed mode (default: 20.0).",
    )
    emhc.add_argument(
        "--fixed-negative-mean",
        type=float,
        default=15.0,
        help="Negative mean for background correction in fixed mode (default: 15.0).",
    )
    emhc.add_argument(
        "--negative-fibers-file",
        type=Path,
        default=Path("negative_fibers.csv"),
        help=(
            "Per-image CSV containing a fiber_id column in negative_fibers mode "
            "(default: negative_fibers.csv in each image folder)."
        ),
    )
    emhc.add_argument(
        "--negative-fiber-percentile",
        type=float,
        default=95.0,
        help="Percentile of negative-fiber means used as threshold (default: 95).",
    )
    emhc.add_argument(
        "--min-bg-corrected",
        type=float,
        default=0.0,
        help="Optional minimum background-corrected scored mean (default: 0).",
    )

    outputs = parser.add_argument_group("outputs")
    outputs.add_argument(
        "--no-per-image-output",
        action="store_true",
        help=(
            "Do not write CSV/QC files back into each source image folder. "
            "The centrally collected individual-image CSVs are still created."
        ),
    )
    outputs.add_argument(
        "--no-scored-region-mask",
        action="store_true",
        help="Do not save the eMHC scored-region QC label image.",
    )
    outputs.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first bad image folder instead of recording and skipping it.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.min_overlap <= 1.0:
        raise ValueError("--min-overlap must be between 0 and 1.")
    if args.pixel_size_um is not None and args.pixel_size_um <= 0:
        raise ValueError("--pixel-size-um must be greater than 0.")
    if args.core_border_exclusion_px < 0:
        raise ValueError("--core-border-exclusion-px must be 0 or greater.")
    if args.min_core_pixels < 1:
        raise ValueError("--min-core-pixels must be at least 1.")
    if not 0.0 <= args.negative_fiber_percentile <= 100.0:
        raise ValueError("--negative-fiber-percentile must be between 0 and 100.")
    for option_name, value in (
        ("--fixed-emhc-threshold", args.fixed_emhc_threshold),
        ("--fixed-negative-mean", args.fixed_negative_mean),
        ("--min-bg-corrected", args.min_bg_corrected),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{option_name} must be finite.")


def accepted_names(requested_name: str) -> set[str]:
    """Return case-insensitive exact/TIF/TIFF alternatives for a filename."""
    requested = Path(requested_name)
    names = {requested.name.lower()}
    if requested.suffix.lower() in {".tif", ".tiff"}:
        names.update(
            {
                f"{requested.stem.lower()}.tif",
                f"{requested.stem.lower()}.tiff",
            }
        )
    elif requested.suffix == "":
        names.update(
            f"{requested.name.lower()}{extension}" for extension in MASK_EXTENSIONS
        )
    return names


def find_named_file(folder: Path, requested_name: str) -> Path | None:
    names = accepted_names(requested_name)
    matches = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.name.lower() in names
        ),
        key=lambda path: path.name.lower(),
    )
    if not matches:
        return None
    exact = [
        path for path in matches if path.name.lower() == Path(requested_name).name.lower()
    ]
    return exact[0] if exact else matches[0]


def find_nucleus_mask(folder: Path, requested_name: str | None) -> Path | None:
    if requested_name is not None:
        return find_named_file(folder, requested_name)

    matches = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in MASK_EXTENSIONS
            and path.stem.lower() in NUCLEUS_MASK_STEMS
        ),
        key=lambda path: (
            NUCLEUS_MASK_STEMS.index(path.stem.lower()),
            path.suffix.lower(),
            path.name.lower(),
        ),
    )
    return matches[0] if matches else None


def discover_image_folders(
    root: Path,
    fiber_file: str,
    nuclei_file: str | None,
    emhc_file: str,
) -> list[ImageFiles]:
    folders: Iterable[Path] = (
        root,
        *sorted(path for path in root.rglob("*") if path.is_dir()),
    )
    discovered: list[ImageFiles] = []
    for folder in folders:
        fiber_mask = find_named_file(folder, fiber_file)
        if fiber_mask is None:
            continue
        discovered.append(
            ImageFiles(
                folder=folder,
                fiber_mask=fiber_mask,
                nuclei_mask=find_nucleus_mask(folder, nuclei_file),
                emhc_image=find_named_file(folder, emhc_file),
            )
        )
    return discovered


def image_id_for(folder: Path, root: Path) -> str:
    relative = folder.relative_to(root)
    return root.name if str(relative) == "." else relative.as_posix()


def read_image(path: Path) -> np.ndarray:
    if skimage_io is not None:
        return np.asarray(skimage_io.imread(path))
    with Image.open(path) as image:
        return np.asarray(image)


def label_binary_mask(mask: np.ndarray) -> np.ndarray:
    if skimage_label is not None:
        return skimage_label(mask, connectivity=1).astype(np.int64)
    structure = generate_binary_structure(rank=2, connectivity=1)
    labels, _ = scipy_label(mask, structure=structure)
    return labels.astype(np.int64)


def load_mask_image(path: Path) -> tuple[np.ndarray, bool]:
    """Load a 2-D integer label mask and label it if it is binary."""
    raw = np.squeeze(read_image(path))

    if raw.ndim == 3 and raw.shape[-1] in (3, 4):
        channels = raw[..., :3]
        if not np.all(channels == channels[..., :1]):
            raise ValueError(
                f"{path.name} is a color image whose channels differ. It appears "
                "to be an overlay rather than an integer label mask. Export the "
                "actual labels as a grayscale PNG/TIFF."
            )
        raw = channels[..., 0]
    elif raw.ndim == 3 and raw.shape[0] in (3, 4):
        channels = raw[:3, ...]
        if not np.all(channels == channels[:1, ...]):
            raise ValueError(
                f"{path.name} is a color image whose channels differ. Export the "
                "actual integer label mask rather than a colored overlay."
            )
        raw = channels[0, ...]

    if raw.ndim != 2:
        raise ValueError(
            f"Expected a 2-D label mask at {path}, but its shape is {raw.shape}."
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"Label mask contains NaN or infinite values: {path}")
    if np.any(raw < 0):
        raise ValueError(f"Label mask contains negative values: {path}")

    rounded = np.rint(raw)
    if not np.allclose(raw, rounded):
        raise ValueError(
            f"{path.name} is not integer-valued. Confirm that it is a label "
            "mask rather than a raw fluorescence image."
        )

    labels = rounded.astype(np.int64, copy=False)
    positive_values = np.unique(labels[labels > 0])
    was_binary = len(positive_values) <= 1
    if len(positive_values) == 1:
        labels = label_binary_mask(labels > 0)
    return labels, was_binary


def load_emhc_image(path: Path) -> np.ndarray:
    """Load a 2-D eMHC image, using green for an RGB/RGBA image."""
    image = np.squeeze(read_image(path))
    if image.ndim == 2:
        selected = image
    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        selected = image[..., 1]
        print(f"  {path.name} is RGB/RGBA; using the green channel.", flush=True)
    elif image.ndim == 3 and image.shape[0] in (3, 4):
        selected = image[1, ...]
        print(f"  {path.name} is channel-first; using the green channel.", flush=True)
    else:
        raise ValueError(
            f"{path.name} has unsupported shape {image.shape}. Export a 2-D "
            "grayscale image or a standard RGB/RGBA image."
        )

    selected = selected.astype(np.float64, copy=False)
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"{path.name} contains NaN or infinite values.")
    return selected


def measure_objects(labels: np.ndarray, object_name: str) -> pd.DataFrame:
    renamed_columns = {
        "label": f"{object_name}_id",
        "area": f"{object_name}_area_px",
        "centroid-0": f"{object_name}_centroid_y_px",
        "centroid-1": f"{object_name}_centroid_x_px",
    }
    if skimage_regionprops_table is not None:
        properties = skimage_regionprops_table(
            labels,
            properties=("label", "area", "centroid"),
        )
        return pd.DataFrame(properties).rename(columns=renamed_columns)

    records: list[dict[str, float | int]] = []
    for object_id, object_slice in enumerate(scipy_find_objects(labels), start=1):
        if object_slice is None:
            continue
        local_y, local_x = np.nonzero(labels[object_slice] == object_id)
        if len(local_y) == 0:
            continue
        records.append(
            {
                f"{object_name}_id": object_id,
                f"{object_name}_area_px": int(len(local_y)),
                f"{object_name}_centroid_y_px": float(
                    local_y.mean() + object_slice[0].start
                ),
                f"{object_name}_centroid_x_px": float(
                    local_x.mean() + object_slice[1].start
                ),
            }
        )
    return pd.DataFrame(records, columns=list(renamed_columns.values()))


def best_fiber_overlaps(
    nuclei_labels: np.ndarray,
    fiber_labels: np.ndarray,
    nucleus_areas: dict[int, float],
    min_overlap: float,
) -> pd.DataFrame:
    """Find and evaluate the greatest fiber overlap for every nucleus."""
    flat_nuclei = nuclei_labels.ravel()
    positive_nucleus = flat_nuclei > 0
    nucleus_pixels = flat_nuclei[positive_nucleus].astype(np.int64, copy=False)
    fiber_pixels = fiber_labels.ravel()[positive_nucleus].astype(np.int64, copy=False)

    max_fiber_id = int(fiber_labels.max())
    multiplier = max_fiber_id + 1
    pair_codes = nucleus_pixels * multiplier + fiber_pixels
    unique_codes, counts = np.unique(pair_codes, return_counts=True)

    overlaps = pd.DataFrame(
        {
            "nucleus_id": (unique_codes // multiplier).astype(np.int64),
            "fiber_id": (unique_codes % multiplier).astype(np.int64),
            "overlap_px": counts.astype(np.int64),
        }
    )
    overlaps = overlaps[overlaps["fiber_id"] > 0].copy()
    nucleus_ids = np.array(sorted(nucleus_areas), dtype=np.int64)

    if overlaps.empty:
        best = pd.DataFrame(
            {
                "nucleus_id": nucleus_ids,
                "best_fiber_id": np.zeros(len(nucleus_ids), dtype=np.int64),
                "overlap_px_with_best_fiber": np.zeros(
                    len(nucleus_ids), dtype=np.int64
                ),
            }
        )
    else:
        best = (
            overlaps.sort_values(
                ["nucleus_id", "overlap_px", "fiber_id"],
                ascending=[True, False, True],
            )
            .drop_duplicates("nucleus_id", keep="first")
            .rename(
                columns={
                    "fiber_id": "best_fiber_id",
                    "overlap_px": "overlap_px_with_best_fiber",
                }
            )
        )
        best = pd.DataFrame({"nucleus_id": nucleus_ids}).merge(
            best, on="nucleus_id", how="left"
        )
        best["best_fiber_id"] = best["best_fiber_id"].fillna(0).astype(np.int64)
        best["overlap_px_with_best_fiber"] = (
            best["overlap_px_with_best_fiber"].fillna(0).astype(np.int64)
        )

    best["nucleus_area_px_for_assignment"] = best["nucleus_id"].map(nucleus_areas)
    best["fraction_nucleus_inside_best_fiber"] = (
        best["overlap_px_with_best_fiber"]
        / best["nucleus_area_px_for_assignment"]
    )
    best["assignment_pass"] = (
        (best["best_fiber_id"] > 0)
        & (best["overlap_px_with_best_fiber"] > 0)
        & (best["fraction_nucleus_inside_best_fiber"] >= min_overlap)
    )
    best["assigned_fiber_id"] = np.where(
        best["assignment_pass"], best["best_fiber_id"], 0
    ).astype(np.int64)
    best["assignment_status"] = np.select(
        [best["assignment_pass"], best["best_fiber_id"] == 0],
        ["assigned", "no_fiber_overlap"],
        default="overlap_below_threshold",
    )
    return best.drop(columns="nucleus_area_px_for_assignment")


def locate_negative_fibers_file(folder: Path, requested: Path) -> Path:
    if requested.is_absolute():
        return requested
    return folder / requested


def emhc_threshold(
    scored_mean: np.ndarray,
    scored_px: np.ndarray,
    max_label: int,
    folder: Path,
    settings: EmhcSettings,
) -> tuple[float, float, str]:
    if settings.threshold_mode == "fixed":
        return (
            settings.fixed_threshold,
            settings.fixed_negative_mean,
            "fixed_scored_mean_threshold",
        )

    negative_path = locate_negative_fibers_file(folder, settings.negative_fibers_file)
    if not negative_path.exists():
        raise FileNotFoundError(
            f"{negative_path} was not found. It is required when "
            "--threshold-mode negative_fibers is used."
        )

    negative_df = pd.read_csv(negative_path)
    if "fiber_id" not in negative_df.columns:
        raise ValueError(f"{negative_path} must contain a fiber_id column.")
    negative_ids = pd.to_numeric(negative_df["fiber_id"], errors="coerce").dropna()
    if len(negative_ids) == 0 or not np.allclose(negative_ids, np.rint(negative_ids)):
        raise ValueError(f"{negative_path} must contain valid integer fiber IDs.")

    negative_ids = np.unique(np.rint(negative_ids).astype(np.int64))
    valid_ids = negative_ids[
        (negative_ids > 0)
        & (negative_ids <= max_label)
        & (scored_px[np.clip(negative_ids, 0, max_label)] > 0)
    ]
    if len(valid_ids) < 3:
        raise ValueError(
            f"Only {len(valid_ids)} valid negative fibers were found in "
            f"{negative_path}; at least 3 are required."
        )

    negative_means = scored_mean[valid_ids]
    negative_mean = float(np.median(negative_means))
    threshold = float(np.percentile(negative_means, settings.negative_percentile))
    method = f"negative_fiber_scored_means_p{settings.negative_percentile:g}"
    return threshold, negative_mean, method


def score_emhc(
    fiber_labels: np.ndarray,
    emhc: np.ndarray,
    folder: Path,
    settings: EmhcSettings,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    """Measure and classify eMHC signal in each fiber's scoring region."""
    max_label = int(fiber_labels.max())

    boundary = np.zeros_like(fiber_labels, dtype=bool)
    boundary[:-1, :] |= fiber_labels[:-1, :] != fiber_labels[1:, :]
    boundary[1:, :] |= fiber_labels[1:, :] != fiber_labels[:-1, :]
    boundary[:, :-1] |= fiber_labels[:, :-1] != fiber_labels[:, 1:]
    boundary[:, 1:] |= fiber_labels[:, 1:] != fiber_labels[:, :-1]
    boundary |= fiber_labels == 0

    distance_to_border_px = distance_transform_edt(~boundary)
    if settings.core_border_exclusion_px == 0:
        core_mask = fiber_labels > 0
    else:
        core_mask = (
            (fiber_labels > 0)
            & (distance_to_border_px > settings.core_border_exclusion_px)
        )

    fiber_pixels = fiber_labels[fiber_labels > 0]
    core_labels = fiber_labels[core_mask]
    core_values = emhc[core_mask]

    fiber_px = np.bincount(fiber_pixels, minlength=max_label + 1)
    core_px = np.bincount(core_labels, minlength=max_label + 1)
    core_sum = np.bincount(
        core_labels, weights=core_values, minlength=max_label + 1
    )
    core_sum_sq = np.bincount(
        core_labels, weights=np.square(core_values), minlength=max_label + 1
    )

    core_mean = np.full(max_label + 1, np.nan)
    core_std = np.full(max_label + 1, np.nan)
    core_fraction_pct = np.full(max_label + 1, np.nan)
    has_core = core_px > 0
    has_fiber = fiber_px > 0
    core_mean[has_core] = core_sum[has_core] / core_px[has_core]
    core_variance = np.zeros(max_label + 1, dtype=float)
    core_variance[has_core] = (
        core_sum_sq[has_core] / core_px[has_core] - np.square(core_mean[has_core])
    )
    core_std[has_core] = np.sqrt(np.maximum(core_variance[has_core], 0.0))
    core_fraction_pct[has_fiber] = 100.0 * core_px[has_fiber] / fiber_px[has_fiber]

    use_full_fiber_fallback = has_fiber & (core_px < settings.min_core_pixels)
    scored_region_mask = (
        (fiber_labels > 0)
        & (core_mask | use_full_fiber_fallback[fiber_labels])
    )
    scored_region_labels = np.where(scored_region_mask, fiber_labels, 0)
    scored_labels = fiber_labels[scored_region_mask]
    scored_values = emhc[scored_region_mask]

    scored_px = np.bincount(scored_labels, minlength=max_label + 1)
    scored_sum = np.bincount(
        scored_labels, weights=scored_values, minlength=max_label + 1
    )
    scored_sum_sq = np.bincount(
        scored_labels, weights=np.square(scored_values), minlength=max_label + 1
    )
    scored_mean = np.full(max_label + 1, np.nan)
    scored_std = np.full(max_label + 1, np.nan)
    has_scored_region = scored_px > 0
    scored_mean[has_scored_region] = (
        scored_sum[has_scored_region] / scored_px[has_scored_region]
    )
    scored_variance = np.zeros(max_label + 1, dtype=float)
    scored_variance[has_scored_region] = (
        scored_sum_sq[has_scored_region] / scored_px[has_scored_region]
        - np.square(scored_mean[has_scored_region])
    )
    scored_std[has_scored_region] = np.sqrt(
        np.maximum(scored_variance[has_scored_region], 0.0)
    )

    threshold, negative_mean, threshold_method = emhc_threshold(
        scored_mean,
        scored_px,
        max_label,
        folder,
        settings,
    )
    if not np.isfinite(threshold) or not np.isfinite(negative_mean):
        raise ValueError("The calculated eMHC threshold/negative mean is not finite.")

    above_threshold = scored_values >= threshold
    scored_above_threshold_px = np.bincount(
        scored_labels[above_threshold], minlength=max_label + 1
    )
    scored_above_threshold_pct = np.full(max_label + 1, np.nan)
    scored_above_threshold_pct[has_scored_region] = (
        100.0
        * scored_above_threshold_px[has_scored_region]
        / scored_px[has_scored_region]
    )

    fiber_ids = np.flatnonzero(has_fiber).astype(np.int64)
    emhc_df = pd.DataFrame(
        {
            "fiber_id": fiber_ids,
            "emhc_fiber_px": fiber_px[fiber_ids],
            "emhc_core_border_exclusion_px": settings.core_border_exclusion_px,
            "emhc_min_core_pixels_used": settings.min_core_pixels,
            "emhc_core_px": core_px[fiber_ids],
            "emhc_core_fraction_of_fiber_pct": core_fraction_pct[fiber_ids],
            "emhc_core_mean": core_mean[fiber_ids],
            "emhc_core_std": core_std[fiber_ids],
            "emhc_used_full_fiber_fallback": use_full_fiber_fallback[fiber_ids],
            "emhc_scored_px": scored_px[fiber_ids],
            "emhc_scored_mean": scored_mean[fiber_ids],
            "emhc_scored_mean_bg_corrected": scored_mean[fiber_ids] - negative_mean,
            "emhc_scored_std": scored_std[fiber_ids],
            "emhc_scored_px_at_or_above_threshold": scored_above_threshold_px[
                fiber_ids
            ],
            "emhc_scored_area_at_or_above_threshold_pct": (
                scored_above_threshold_pct[fiber_ids]
            ),
        }
    )
    emhc_df["emhc_scoring_region"] = np.where(
        emhc_df["emhc_used_full_fiber_fallback"], "full_fiber_fallback", "core"
    )
    emhc_df.loc[emhc_df["emhc_scored_px"] == 0, "emhc_scoring_region"] = "none"
    emhc_df["emhc_mean_threshold_used"] = threshold
    emhc_df["emhc_negative_mean"] = negative_mean
    emhc_df["emhc_threshold_method"] = threshold_method
    emhc_df["emhc_qc_pass"] = emhc_df["emhc_scored_px"] > 0
    emhc_df["emhc_positive"] = (
        emhc_df["emhc_qc_pass"]
        & (emhc_df["emhc_scored_mean"] >= threshold)
        & (emhc_df["emhc_scored_mean_bg_corrected"] >= settings.min_bg_corrected)
    )
    emhc_df["emhc_class"] = np.select(
        [~emhc_df["emhc_qc_pass"], emhc_df["emhc_positive"]],
        ["unscored", "positive"],
        default="negative",
    )

    details = {
        "emhc_threshold_method": threshold_method,
        "emhc_mean_threshold_used": threshold,
        "emhc_negative_mean": negative_mean,
        "fibers_using_full_fiber_fallback": int(use_full_fiber_fallback.sum()),
    }
    return emhc_df, scored_region_labels, details


def add_rfp_assignments(
    fibers: pd.DataFrame,
    nuclei: pd.DataFrame,
    nuclei_labels: np.ndarray,
    fiber_labels: np.ndarray,
    min_overlap: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nucleus_areas = dict(
        zip(nuclei["nucleus_id"].astype(int), nuclei["nucleus_area_px"])
    )
    assignments = best_fiber_overlaps(
        nuclei_labels, fiber_labels, nucleus_areas, min_overlap
    )
    nuclei = nuclei.merge(assignments, on="nucleus_id", how="left")

    assigned = nuclei[nuclei["assignment_pass"]].copy()
    nucleus_counts = assigned.groupby("assigned_fiber_id").size()
    nucleus_ids = assigned.groupby("assigned_fiber_id")["nucleus_id"].agg(
        lambda values: ";".join(str(int(value)) for value in sorted(values))
    )

    fibers["rfp_nucleus_count"] = (
        fibers["fiber_id"].map(nucleus_counts).fillna(0).astype(int)
    )
    fibers["rfp_nucleus_ids"] = fibers["fiber_id"].map(nucleus_ids).fillna("")
    fibers["has_rfp_nucleus"] = fibers["rfp_nucleus_count"] > 0
    fibers["rfp_group"] = np.where(
        fibers["has_rfp_nucleus"], "with_rfp_nuclei", "without_rfp_nuclei"
    )
    return fibers, nuclei


def preferred_fiber_column_order(frame: pd.DataFrame) -> list[str]:
    first = [
        "image_id",
        "fiber_id",
        "fiber_area_px",
        "fiber_area_um2",
        "fiber_centroid_y_px",
        "fiber_centroid_x_px",
        "rfp_nucleus_count",
        "rfp_nucleus_ids",
        "has_rfp_nucleus",
        "rfp_group",
        "emhc_positive",
        "emhc_class",
        "analysis_qc_pass",
    ]
    present_first = [column for column in first if column in frame.columns]
    return present_first + [column for column in frame.columns if column not in present_first]


def preferred_nucleus_column_order(frame: pd.DataFrame) -> list[str]:
    first = [
        "image_id",
        "nucleus_id",
        "assigned_fiber_id",
        "assignment_status",
        "assignment_pass",
        "fraction_nucleus_inside_best_fiber",
        "overlap_px_with_best_fiber",
        "best_fiber_id",
        "nucleus_area_px",
        "nucleus_area_um2",
        "nucleus_centroid_y_px",
        "nucleus_centroid_x_px",
    ]
    return [column for column in first if column in frame.columns]


def analyze_image_folder(
    files: ImageFiles,
    root: Path,
    args: argparse.Namespace,
    settings: EmhcSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, object]]:
    if files.nuclei_mask is None:
        expected = args.nuclei_file or "rfp_stardist_labels.* / rfp_starfist_labels.*"
        raise FileNotFoundError(f"Missing RFP nucleus label mask ({expected}).")
    if files.emhc_image is None:
        raise FileNotFoundError(f"Missing {args.emhc_file}.")

    image_id = image_id_for(files.folder, root)
    print(f"Processing: {image_id}", flush=True)
    print(f"  Fiber mask: {files.fiber_mask.name}", flush=True)
    print(f"  RFP nucleus mask: {files.nuclei_mask.name}", flush=True)
    print(f"  eMHC image: {files.emhc_image.name}", flush=True)

    fiber_labels, fiber_was_binary = load_mask_image(files.fiber_mask)
    nuclei_labels, nuclei_was_binary = load_mask_image(files.nuclei_mask)
    emhc = load_emhc_image(files.emhc_image)
    if fiber_labels.shape != nuclei_labels.shape or fiber_labels.shape != emhc.shape:
        raise ValueError(
            "Input dimensions differ: "
            f"fiber mask {fiber_labels.shape}, nucleus mask {nuclei_labels.shape}, "
            f"eMHC image {emhc.shape}. All inputs must be registered."
        )
    if int(fiber_labels.max()) == 0:
        raise ValueError("The fiber mask contains no labeled fibers.")

    fibers = measure_objects(fiber_labels, "fiber")
    nuclei = measure_objects(nuclei_labels, "nucleus")
    if args.pixel_size_um is not None:
        area_factor = args.pixel_size_um**2
        fibers["fiber_area_um2"] = fibers["fiber_area_px"] * area_factor
        nuclei["nucleus_area_um2"] = nuclei["nucleus_area_px"] * area_factor

    fibers, nuclei = add_rfp_assignments(
        fibers,
        nuclei,
        nuclei_labels,
        fiber_labels,
        args.min_overlap,
    )
    emhc_df, scored_region_labels, emhc_details = score_emhc(
        fiber_labels, emhc, files.folder, settings
    )
    fibers = fibers.merge(emhc_df, on="fiber_id", how="left", validate="one_to_one")
    fibers["analysis_qc_pass"] = fibers["emhc_qc_pass"].fillna(False).astype(bool)

    fibers.insert(0, "image_id", image_id)
    nuclei.insert(0, "image_id", image_id)
    fibers = fibers[preferred_fiber_column_order(fibers)]
    nuclei = nuclei[preferred_nucleus_column_order(nuclei)]

    assigned_count = int(nuclei["assignment_pass"].sum())
    summary = {
        "image_id": image_id,
        "status": "processed",
        "message": "",
        "fiber_file": files.fiber_mask.name,
        "nuclei_file": files.nuclei_mask.name,
        "emhc_file": files.emhc_image.name,
        "fiber_mask_was_binary": fiber_was_binary,
        "nuclei_mask_was_binary": nuclei_was_binary,
        "total_fibers": int(len(fibers)),
        "fibers_with_rfp_nuclei": int(fibers["has_rfp_nucleus"].sum()),
        "fibers_without_rfp_nuclei": int((~fibers["has_rfp_nucleus"]).sum()),
        "total_rfp_nuclei": int(len(nuclei)),
        "assigned_rfp_nuclei": assigned_count,
        "unassigned_rfp_nuclei": int(len(nuclei) - assigned_count),
        "emhc_positive_fibers": int(fibers["emhc_positive"].sum()),
        "emhc_negative_fibers": int((fibers["emhc_class"] == "negative").sum()),
        "emhc_unscored_fibers": int((fibers["emhc_class"] == "unscored").sum()),
        "rfp_fibers_that_are_emhc_positive": int(
            (fibers["has_rfp_nucleus"] & fibers["emhc_positive"]).sum()
        ),
        "minimum_overlap_fraction": args.min_overlap,
        "pixel_size_um": args.pixel_size_um,
        **emhc_details,
    }
    return fibers, nuclei, scored_region_labels, summary


def save_label_tiff(path: Path, labels: np.ndarray) -> None:
    max_label = int(labels.max())
    dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
    output = labels.astype(dtype)
    if skimage_io is not None:
        skimage_io.imsave(path, output, check_contrast=False)
    else:
        Image.fromarray(output).save(path, format="TIFF")


def write_per_image_outputs(
    folder: Path,
    fibers: pd.DataFrame,
    nuclei: pd.DataFrame,
    scored_region_labels: np.ndarray,
    save_scored_mask: bool,
) -> None:
    fibers.to_csv(folder / PER_IMAGE_FIBER_CSV, index=False)
    nuclei.to_csv(folder / PER_IMAGE_NUCLEUS_CSV, index=False)
    if save_scored_mask:
        save_label_tiff(folder / PER_IMAGE_SCORED_MASK, scored_region_labels)


def error_summary(files: ImageFiles, root: Path, message: str) -> dict[str, object]:
    return {
        "image_id": image_id_for(files.folder, root),
        "status": "error",
        "message": message,
        "fiber_file": files.fiber_mask.name,
        "nuclei_file": files.nuclei_mask.name if files.nuclei_mask else "",
        "emhc_file": files.emhc_image.name if files.emhc_image else "",
    }


def overall_summary(
    fibers: pd.DataFrame,
    nuclei: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    assigned_count = int(nuclei["assignment_pass"].sum())
    return {
        "image_id": "ALL_IMAGES",
        "status": "processed",
        "message": "Combined totals from successfully processed image folders.",
        "total_fibers": int(len(fibers)),
        "fibers_with_rfp_nuclei": int(fibers["has_rfp_nucleus"].sum()),
        "fibers_without_rfp_nuclei": int((~fibers["has_rfp_nucleus"]).sum()),
        "total_rfp_nuclei": int(len(nuclei)),
        "assigned_rfp_nuclei": assigned_count,
        "unassigned_rfp_nuclei": int(len(nuclei) - assigned_count),
        "emhc_positive_fibers": int(fibers["emhc_positive"].sum()),
        "emhc_negative_fibers": int((fibers["emhc_class"] == "negative").sum()),
        "emhc_unscored_fibers": int((fibers["emhc_class"] == "unscored").sum()),
        "rfp_fibers_that_are_emhc_positive": int(
            (fibers["has_rfp_nucleus"] & fibers["emhc_positive"]).sum()
        ),
        "minimum_overlap_fraction": args.min_overlap,
        "pixel_size_um": args.pixel_size_um,
    }


def write_combined_outputs(
    output_dir: Path,
    summary: pd.DataFrame,
    fibers: pd.DataFrame,
    nuclei: pd.DataFrame,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "analysis_summary.csv": summary,
        "all_fibers_rfp_nuclei_emhc.csv": fibers,
        "fibers_with_rfp_nuclei.csv": fibers[fibers["has_rfp_nucleus"]],
        "fibers_without_rfp_nuclei.csv": fibers[~fibers["has_rfp_nucleus"]],
        "emhc_positive_fibers.csv": fibers[fibers["emhc_positive"]],
        "emhc_negative_fibers.csv": fibers[fibers["emhc_class"] == "negative"],
        "rfp_nucleus_assignments.csv": nuclei,
    }
    written: list[Path] = []
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        written.append(path)
    return written


def safe_image_csv_stem(image_id: str) -> str:
    """Convert an image ID or relative folder path into a Windows-safe stem."""
    # Preserve nested-folder information while avoiding path separators in the
    # output filename. For example, set_1/image_A becomes set_1__image_A.
    stem = image_id.replace("\\", "__").replace("/", "__")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = stem.strip(" ._")
    return stem or "image"


def write_individual_image_csvs(
    output_dir: Path,
    fibers: pd.DataFrame,
) -> tuple[Path, list[Path]]:
    """Collect one complete fiber-summary CSV per image in one folder."""
    individual_dir = output_dir / INDIVIDUAL_IMAGE_CSV_DIR
    individual_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    used_names: set[str] = set()
    for image_id, image_fibers in fibers.groupby("image_id", sort=True):
        base_stem = safe_image_csv_stem(str(image_id))
        filename = f"{base_stem}_fiber_summary.csv"

        # This is rarely needed, but protects against two different image IDs
        # becoming the same name after unsafe characters are replaced.
        counter = 2
        while filename.casefold() in used_names:
            filename = f"{base_stem}_{counter}_fiber_summary.csv"
            counter += 1
        used_names.add(filename.casefold())

        path = individual_dir / filename
        image_fibers.sort_values("fiber_id").to_csv(path, index=False)
        written.append(path)

    return individual_dir, written


def main() -> int:
    args = parse_args()
    validate_args(args)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Experiment root does not exist: {root}")

    if args.output_dir is None:
        output_dir = root / DEFAULT_OUTPUT_DIR
    else:
        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        output_dir = output_dir.resolve()

    settings = EmhcSettings(
        core_border_exclusion_px=args.core_border_exclusion_px,
        min_core_pixels=args.min_core_pixels,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.fixed_emhc_threshold,
        fixed_negative_mean=args.fixed_negative_mean,
        negative_fibers_file=args.negative_fibers_file.expanduser(),
        negative_percentile=args.negative_fiber_percentile,
        min_bg_corrected=args.min_bg_corrected,
    )

    print(f"Scanning experiment root: {root}", flush=True)
    print(
        "Using the existing fiber and RFP-nucleus masks; rfp.tif and "
        "laminin.tif will not be read.",
        flush=True,
    )
    image_folders = discover_image_folders(
        root,
        args.fiber_file,
        args.nuclei_file,
        args.emhc_file,
    )
    if not image_folders:
        raise FileNotFoundError(
            f"No folders containing {args.fiber_file} were found under {root}."
        )
    print(f"Found {len(image_folders)} image folder(s).", flush=True)

    fiber_frames: list[pd.DataFrame] = []
    nucleus_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for files in image_folders:
        try:
            fibers, nuclei, scored_labels, summary = analyze_image_folder(
                files, root, args, settings
            )
            if not args.no_per_image_output:
                write_per_image_outputs(
                    files.folder,
                    fibers,
                    nuclei,
                    scored_labels,
                    save_scored_mask=not args.no_scored_region_mask,
                )
        except Exception as exc:
            if args.fail_fast:
                raise
            message = f"{type(exc).__name__}: {exc}"
            print(
                f"ERROR in {image_id_for(files.folder, root)}: {message}",
                flush=True,
            )
            summary_rows.append(error_summary(files, root, message))
            continue

        fiber_frames.append(fibers)
        nucleus_frames.append(nuclei)
        summary_rows.append(summary)
        print(
            f"  Done: {len(fibers)} fibers; "
            f"{int(fibers['has_rfp_nucleus'].sum())} with RFP nuclei; "
            f"{int(fibers['emhc_positive'].sum())} eMHC-positive.",
            flush=True,
        )

    if not fiber_frames:
        errors = "; ".join(str(row.get("message", "")) for row in summary_rows)
        raise RuntimeError(f"No image folders were processed successfully. {errors}")

    all_fibers = pd.concat(fiber_frames, ignore_index=True)
    nonempty_nucleus_frames = [frame for frame in nucleus_frames if not frame.empty]
    if nonempty_nucleus_frames:
        all_nuclei = pd.concat(nonempty_nucleus_frames, ignore_index=True)
    else:
        all_nuclei = nucleus_frames[0].iloc[0:0].copy()
    all_fibers = all_fibers.sort_values(["image_id", "fiber_id"]).reset_index(
        drop=True
    )
    all_nuclei = all_nuclei.sort_values(["image_id", "nucleus_id"]).reset_index(
        drop=True
    )
    summary_rows.insert(0, overall_summary(all_fibers, all_nuclei, args))
    summary = pd.DataFrame(summary_rows)
    written = write_combined_outputs(output_dir, summary, all_fibers, all_nuclei)
    individual_dir, individual_csvs = write_individual_image_csvs(
        output_dir, all_fibers
    )

    print(f"Combined results saved in: {output_dir}", flush=True)
    for path in written:
        print(f"  {path.name}", flush=True)
    print(
        f"  {individual_dir.name}/ "
        f"({len(individual_csvs)} individual image CSVs)",
        flush=True,
    )
    print(
        f"Finished: {len(all_fibers)} fibers across {len(fiber_frames)} image(s); "
        f"{int(all_fibers['has_rfp_nucleus'].sum())} fibers with RFP nuclei; "
        f"{int(all_fibers['emhc_positive'].sum())} eMHC-positive fibers.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
