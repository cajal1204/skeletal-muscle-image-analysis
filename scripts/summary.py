print("Starting fiber-count summary...", flush=True)

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

# Same root folder used by the fiber measurement script.
ROOT_FOLDER = Path(
    r"C:/Users/emurp/Documents/0000"
)

# Folder created by the fiber measurement script.
INDIVIDUAL_CSV_FOLDER = (
    ROOT_FOLDER /
    "individual_image_CSA_csvs"
)

# Output summary CSV.
OUTPUT_CSV = (
    ROOT_FOLDER /
    "dystrophin_fiber_counts_by_image.csv"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def convert_to_boolean(series, column_name, csv_path):
    """Safely convert common TRUE/FALSE formats to Boolean values."""

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    converted = text.map(
        {
            "true": True,
            "t": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "f": False,
            "0": False,
            "no": False,
            "n": False,
        }
    )

    if converted.isna().any():
        bad_values = sorted(
            series[converted.isna()]
            .astype(str)
            .unique()
        )

        raise ValueError(
            f"Could not interpret some values in "
            f"'{column_name}' from:\n{csv_path}\n"
            f"Values found: {bad_values}"
        )

    return converted.astype(bool)


# ============================================================
# FIND INDIVIDUAL CSV FILES
# ============================================================

if not INDIVIDUAL_CSV_FOLDER.exists():
    raise FileNotFoundError(
        f"Individual CSV folder does not exist:\n"
        f"{INDIVIDUAL_CSV_FOLDER}"
    )

individual_csv_paths = sorted(
    INDIVIDUAL_CSV_FOLDER.glob("*.csv")
)

if not individual_csv_paths:
    raise FileNotFoundError(
        f"No CSV files were found in:\n"
        f"{INDIVIDUAL_CSV_FOLDER}"
    )

print(
    f"Found {len(individual_csv_paths)} individual-image CSVs.",
    flush=True,
)


# ============================================================
# SUMMARIZE EACH IMAGE
# ============================================================

summary_rows = []

required_columns = {
    "fiber_id",
    "fiber_cross_sectional_area_um2",
    "passes_min_area_qc",
    "touches_image_border",
}

for csv_number, csv_path in enumerate(
    individual_csv_paths,
    start=1,
):
    image_name = csv_path.stem

    print(
        f"[{csv_number}/{len(individual_csv_paths)}] "
        f"Reading {csv_path.name}...",
        flush=True,
    )

    fibers = pd.read_csv(csv_path)

    missing_columns = required_columns - set(fibers.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in:\n{csv_path}\n"
            f"Missing: {sorted(missing_columns)}"
        )

    # Remove completely identical rows if present.
    rows_before = len(fibers)
    fibers = fibers.drop_duplicates().copy()
    exact_duplicates_removed = rows_before - len(fibers)

    # A fiber ID should occur only once within an image.
    duplicated_ids = fibers["fiber_id"].duplicated(
        keep=False
    )

    if duplicated_ids.any():
        print(
            fibers.loc[
                duplicated_ids,
                ["fiber_id"],
            ].to_string(index=False),
            flush=True,
        )

        raise ValueError(
            f"Conflicting duplicate fiber IDs were found in:\n"
            f"{csv_path}"
        )

    if fibers.empty:
        summary_rows.append(
            {
                "image_name": image_name,
                "individual_csv": csv_path.name,
                "n_fibers_total": 0,
                "n_fibers_passing_min_area_qc": 0,
                "n_fibers_failing_min_area_qc": 0,
                "n_fibers_touching_border": 0,
                "n_fibers_not_touching_border": 0,
                "mean_csa_um2": np.nan,
                "median_csa_um2": np.nan,
                "min_csa_um2": np.nan,
                "max_csa_um2": np.nan,
                "exact_duplicate_rows_removed": (
                    exact_duplicates_removed
                ),
            }
        )

        print("  Fibers: 0", flush=True)
        continue

    fibers["passes_min_area_qc"] = convert_to_boolean(
        fibers["passes_min_area_qc"],
        "passes_min_area_qc",
        csv_path,
    )

    fibers["touches_image_border"] = convert_to_boolean(
        fibers["touches_image_border"],
        "touches_image_border",
        csv_path,
    )

    fibers["fiber_cross_sectional_area_um2"] = (
        pd.to_numeric(
            fibers["fiber_cross_sectional_area_um2"],
            errors="coerce",
        )
    )

    invalid_area = (
        ~np.isfinite(
            fibers["fiber_cross_sectional_area_um2"]
        )
        | (
            fibers["fiber_cross_sectional_area_um2"]
            <= 0
        )
    )

    if invalid_area.any():
        raise ValueError(
            f"{invalid_area.sum()} invalid CSA value(s) "
            f"were found in:\n{csv_path}"
        )

    n_fibers_total = fibers["fiber_id"].nunique()

    n_passing = int(
        fibers["passes_min_area_qc"].sum()
    )

    n_touching_border = int(
        fibers["touches_image_border"].sum()
    )

    csa = fibers[
        "fiber_cross_sectional_area_um2"
    ]

    summary_rows.append(
        {
            "image_name": image_name,
            "individual_csv": csv_path.name,
            "n_fibers_total": n_fibers_total,
            "n_fibers_passing_min_area_qc": n_passing,
            "n_fibers_failing_min_area_qc": (
                n_fibers_total - n_passing
            ),
            "n_fibers_touching_border": (
                n_touching_border
            ),
            "n_fibers_not_touching_border": (
                n_fibers_total -
                n_touching_border
            ),
            "mean_csa_um2": csa.mean(),
            "median_csa_um2": csa.median(),
            "min_csa_um2": csa.min(),
            "max_csa_um2": csa.max(),
            "exact_duplicate_rows_removed": (
                exact_duplicates_removed
            ),
        }
    )

    print(
        f"  Fibers: {n_fibers_total}",
        flush=True,
    )


# ============================================================
# CREATE AND SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    "image_name"
).reset_index(drop=True)

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("", flush=True)
print("Fiber-count summary finished.", flush=True)

print(
    f"Images summarized: {len(summary)}",
    flush=True,
)

print(
    f"Total fibers across all images: "
    f"{summary['n_fibers_total'].sum()}",
    flush=True,
)

print(
    f"Total fibers passing area QC: "
    f"{summary['n_fibers_passing_min_area_qc'].sum()}",
    flush=True,
)

print(
    f"Summary CSV:\n{OUTPUT_CSV}",
    flush=True,
)