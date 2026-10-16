import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil


# User inputs

nucleus_csv_path = Path("image02_nucleus_assignments.csv")

fiber_summary_paths = [
    Path("image02_fiber_summary.csv"),
    Path("image02_fiber_summary_with_emhc.csv"),
]

# Optional tracking file
manual_not_central_path = Path("manual_not_central_fibers.csv")

default_reason = "manual not central nucleus correction"

# If True, also force the looser DAPI-core central nucleation call to FALSE.
also_override_loose_central_nucleation = True


# Helper functions

def as_bool_series(series):
    """
    Convert a pandas column to boolean safely.
    """
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def make_backup(path, label):
    """
    Make a timestamped backup before modifying a file.
    """
    if not path.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(
        f"{path.stem}_backup_before_{label}_{timestamp}{path.suffix}"
    )

    shutil.copy2(path, backup_path)
    print(f"Backup saved as: {backup_path}", flush=True)


def ensure_nucleus_columns(nuclei_df):
    """
    Add manual central-nucleus correction columns if missing.
    """
    required = [
        "nucleus_id",
        "assigned_fiber_id",
        "central_nucleus",
    ]

    for col in required:
        if col not in nuclei_df.columns:
            raise ValueError(f"Nucleus CSV must contain column: {col}")

    new_cols = {
        "central_nucleus_manual_not_central": "FALSE",
        "central_nucleus_previous_value": "",
        "central_nucleus_manual_not_central_fiber_id": "",
        "central_nucleus_manual_not_central_reason": "",
        "central_nucleus_manual_not_central_timestamp": "",
    }

    for col, default in new_cols.items():
        if col not in nuclei_df.columns:
            nuclei_df[col] = default
        nuclei_df[col] = nuclei_df[col].astype("object")

    return nuclei_df


def load_manual_not_central(path):
    """
    Load manual_not_central_fibers.csv, or create an empty one.
    """
    if not path.exists():
        return pd.DataFrame(columns=["fiber_id", "reason", "timestamp"])

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["fiber_id", "reason", "timestamp"])

    df.columns = df.columns.astype(str).str.strip()

    if "fiber_id" not in df.columns:
        raise ValueError(
            f"{path} must contain a column named 'fiber_id'. "
            f"Columns found: {list(df.columns)}"
        )

    if "reason" not in df.columns:
        df["reason"] = ""

    if "timestamp" not in df.columns:
        df["timestamp"] = ""

    df = df[["fiber_id", "reason", "timestamp"]].copy()
    df = df.dropna(subset=["fiber_id"])
    df["fiber_id"] = df["fiber_id"].astype(int)

    return df


def mark_fiber_not_central_in_nuclei(nuclei_df, manual_df, fiber_id):
    """
    Set central_nucleus = False for all nuclei assigned to the given fiber.
    """
    fiber_id = int(fiber_id)
    timestamp = datetime.now().isoformat(timespec="seconds")

    assigned_mask = nuclei_df["assigned_fiber_id"].fillna(0).astype(int) == fiber_id

    if assigned_mask.sum() == 0:
        print(f"Fiber {fiber_id}: no assigned nuclei found in nucleus CSV.", flush=True)
    else:
        current_central_mask = assigned_mask & as_bool_series(nuclei_df["central_nucleus"])
        current_central_count = int(current_central_mask.sum())

        # Save previous central_nucleus values as text before changing them
        nuclei_df.loc[assigned_mask, "central_nucleus_previous_value"] = (
            nuclei_df.loc[assigned_mask, "central_nucleus"].astype(str)
        )

        # Mark all nuclei assigned to that fiber as part of a manual fiber-level correction
        nuclei_df.loc[assigned_mask, "central_nucleus_manual_not_central"] = "TRUE"
        nuclei_df.loc[assigned_mask, "central_nucleus_manual_not_central_fiber_id"] = str(fiber_id)
        nuclei_df.loc[assigned_mask, "central_nucleus_manual_not_central_reason"] = default_reason
        nuclei_df.loc[assigned_mask, "central_nucleus_manual_not_central_timestamp"] = timestamp

        # Main correction: no nucleus in this fiber should count as central
        nuclei_df.loc[assigned_mask, "central_nucleus"] = False

        print(
            f"Fiber {fiber_id}: set central_nucleus = False for "
            f"{assigned_mask.sum()} assigned nuclei "
            f"({current_central_count} were previously central).",
            flush=True
        )

    # Update tracking file
    new_row = pd.DataFrame({
        "fiber_id": [fiber_id],
        "reason": [default_reason],
        "timestamp": [timestamp],
    })

    manual_df = pd.concat([manual_df, new_row], ignore_index=True)
    manual_df = manual_df.drop_duplicates(subset=["fiber_id"], keep="last")
    manual_df["fiber_id"] = manual_df["fiber_id"].astype(int)

    return nuclei_df, manual_df


def update_fiber_summary_from_nuclei(summary_path, nuclei_df, corrected_fiber_ids):
    """
    Update central_nuclei and has_central_nucleus in a fiber summary CSV.
    Also optionally override central_nucleation_loose.
    """
    if not summary_path.exists():
        print(f"Summary file not found, skipping: {summary_path}", flush=True)
        return

    summary_df = pd.read_csv(summary_path)

    if "fiber_id" not in summary_df.columns:
        raise ValueError(f"{summary_path} must contain a fiber_id column.")

    # Recalculate central_nuclei from the corrected nucleus CSV
    nuclei_work = nuclei_df.copy()
    nuclei_work["assigned_fiber_id"] = nuclei_work["assigned_fiber_id"].fillna(0).astype(int)
    nuclei_work["central_nucleus_bool"] = as_bool_series(nuclei_work["central_nucleus"])

    central_counts = (
        nuclei_work[nuclei_work["central_nucleus_bool"]]
        .groupby("assigned_fiber_id")
        .size()
        .to_dict()
    )

    summary_df["central_nuclei"] = (
        summary_df["fiber_id"]
        .astype(int)
        .map(central_counts)
        .fillna(0)
        .astype(int)
    )

    summary_df["has_central_nucleus"] = summary_df["central_nuclei"] > 0

    # Add manual correction metadata
    if "central_nucleus_manual_override" not in summary_df.columns:
        summary_df["central_nucleus_manual_override"] = "FALSE"

    if "central_nucleus_manual_override_timestamp" not in summary_df.columns:
        summary_df["central_nucleus_manual_override_timestamp"] = ""

    if "central_nucleus_manual_override_reason" not in summary_df.columns:
        summary_df["central_nucleus_manual_override_reason"] = ""

    if "central_nucleus_manual_override_note" not in summary_df.columns:
        summary_df["central_nucleus_manual_override_note"] = ""

    for col in [
        "central_nucleus_manual_override",
        "central_nucleus_manual_override_timestamp",
        "central_nucleus_manual_override_reason",
        "central_nucleus_manual_override_note",
    ]:
        summary_df[col] = summary_df[col].astype("object")

    timestamp = datetime.now().isoformat(timespec="seconds")

    corrected_mask = summary_df["fiber_id"].astype(int).isin(corrected_fiber_ids)

    summary_df.loc[corrected_mask, "central_nucleus_manual_override"] = "TRUE"
    summary_df.loc[corrected_mask, "central_nucleus_manual_override_timestamp"] = timestamp
    summary_df.loc[corrected_mask, "central_nucleus_manual_override_reason"] = default_reason
    summary_df.loc[corrected_mask, "central_nucleus_manual_override_note"] = (
        "all nuclei assigned to this fiber forced to central_nucleus = False"
    )

    # Optional: also force loose central nucleation to False
    if also_override_loose_central_nucleation:
        if "central_nucleation_loose" in summary_df.columns:
            if "central_nucleation_loose_previous_value" not in summary_df.columns:
                summary_df["central_nucleation_loose_previous_value"] = ""

            summary_df["central_nucleation_loose_previous_value"] = (
                summary_df["central_nucleation_loose_previous_value"].astype("object")
            )

            summary_df.loc[corrected_mask, "central_nucleation_loose_previous_value"] = (
                summary_df.loc[corrected_mask, "central_nucleation_loose"].astype(str)
            )

            summary_df.loc[corrected_mask, "central_nucleation_loose"] = False

        if "central_nucleation_loose_source" in summary_df.columns:
            if "central_nucleation_loose_source_previous_value" not in summary_df.columns:
                summary_df["central_nucleation_loose_source_previous_value"] = ""

            summary_df["central_nucleation_loose_source_previous_value"] = (
                summary_df["central_nucleation_loose_source_previous_value"].astype("object")
            )

            summary_df.loc[corrected_mask, "central_nucleation_loose_source_previous_value"] = (
                summary_df.loc[corrected_mask, "central_nucleation_loose_source"].astype(str)
            )

            summary_df.loc[corrected_mask, "central_nucleation_loose_source"] = (
                "manual_not_central_correction"
            )

    summary_df.to_csv(summary_path, index=False)

    print(f"Updated summary file: {summary_path}", flush=True)


# Main script

print("Starting manual central-nucleus correction script...", flush=True)

if not nucleus_csv_path.exists():
    raise FileNotFoundError(f"Could not find: {nucleus_csv_path}")

# Backups
make_backup(nucleus_csv_path, "manual_not_central_correction")

for path in fiber_summary_paths:
    make_backup(path, "manual_not_central_correction")

make_backup(manual_not_central_path, "manual_not_central_correction")

# Load files
nuclei_df = pd.read_csv(nucleus_csv_path)
nuclei_df = ensure_nucleus_columns(nuclei_df)

manual_df = load_manual_not_central(manual_not_central_path)

print("")
print("Enter fiber IDs that should NOT count as centrally nucleated.")
print("This will set central_nucleus = False for all nuclei assigned to that fiber.")
print("You can enter one ID at a time:")
print("  1068")
print("Or several IDs separated by commas or spaces:")
print("  1068, 1204, 1450")
print("Press Enter with no number to save and quit.")
print("")

corrected_ids = []

while True:
    user_input = input("Fiber ID(s) to mark NOT centrally nucleated: ").strip()

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
        nuclei_df, manual_df = mark_fiber_not_central_in_nuclei(
            nuclei_df,
            manual_df,
            fiber_id
        )
        corrected_ids.append(int(fiber_id))

    # Save after each batch
    nuclei_df.to_csv(nucleus_csv_path, index=False)

    manual_df = manual_df.sort_values("fiber_id").reset_index(drop=True)
    manual_df.to_csv(manual_not_central_path, index=False)

    unique_corrected_ids = sorted(set(corrected_ids))

    for summary_path in fiber_summary_paths:
        update_fiber_summary_from_nuclei(
            summary_path,
            nuclei_df,
            unique_corrected_ids
        )

    print(f"Saved updated nucleus CSV: {nucleus_csv_path}", flush=True)
    print(f"Saved manual correction file: {manual_not_central_path}", flush=True)
    print("")

print("Done.", flush=True)

if len(corrected_ids) > 0:
    print(f"Marked {len(set(corrected_ids))} fiber(s) as NOT centrally nucleated:", flush=True)
    print(sorted(set(corrected_ids)), flush=True)
else:
    print("No fibers were corrected.", flush=True)
