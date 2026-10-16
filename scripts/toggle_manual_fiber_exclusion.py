import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil


# User inputs

summary_csv_path = Path("image02_fiber_summary_with_dystrophin.csv")
exclude_fibers_path = Path("exclude_fibers.csv")

default_exclude_reason = "manual toggle exclusion"


# Helper functions

def as_bool_value(x):
    """
    Convert common TRUE/FALSE values to Python bool.
    """
    if pd.isna(x):
        return False

    if isinstance(x, bool):
        return x

    x_str = str(x).strip().lower()

    if x_str in ["true", "1", "yes", "y", "t"]:
        return True

    if x_str in ["false", "0", "no", "n", "f", ""]:
        return False

    return False


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


def load_exclude_fibers(path):
    """
    Load exclude_fibers.csv, or create an empty version if it does not exist.
    """
    if not path.exists():
        return pd.DataFrame(columns=["fiber_id", "exclude_reason"])

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["fiber_id", "exclude_reason"])

    df.columns = df.columns.astype(str).str.strip()

    if "fiber_id" not in df.columns:
        raise ValueError(
            f"{path} must contain a column named 'fiber_id'. "
            f"Columns found: {list(df.columns)}"
        )

    if "exclude_reason" not in df.columns:
        df["exclude_reason"] = ""

    df = df[["fiber_id", "exclude_reason"]].copy()
    df = df.dropna(subset=["fiber_id"])
    df["fiber_id"] = df["fiber_id"].astype(int)

    return df


def ensure_summary_columns(df):
    """
    Add manual-exclusion-related columns if missing.
    """
    if "fiber_id" not in df.columns:
        raise ValueError("Summary CSV must contain a column named 'fiber_id'.")

    if "manual_exclude_fiber" not in df.columns:
        df["manual_exclude_fiber"] = False

    if "fiber_exclude_reason" not in df.columns:
        df["fiber_exclude_reason"] = ""

    if "manual_exclusion_toggle" not in df.columns:
        df["manual_exclusion_toggle"] = "FALSE"

    if "manual_exclusion_previous_state" not in df.columns:
        df["manual_exclusion_previous_state"] = ""

    if "manual_exclusion_toggle_timestamp" not in df.columns:
        df["manual_exclusion_toggle_timestamp"] = ""

    if "manual_exclusion_toggle_note" not in df.columns:
        df["manual_exclusion_toggle_note"] = ""

    # Avoid pandas dtype issues when writing mixed text/bool values.
    object_cols = [
        "manual_exclude_fiber",
        "fiber_exclude_reason",
        "manual_exclusion_toggle",
        "manual_exclusion_previous_state",
        "manual_exclusion_toggle_timestamp",
        "manual_exclusion_toggle_note",
    ]

    for col in object_cols:
        df[col] = df[col].astype("object")

    return df


def recompute_qc_columns(df):
    """
    Recompute fiber_qc_pass and analysis_qc_pass after changing manual exclusion.
    """

    manual_excluded = as_bool_series(df["manual_exclude_fiber"])

    # Recompute fiber_qc_pass if the component QC columns are available.
    component_cols = [
        "fiber_area_qc_pass",
        "laminin_membrane_signal_qc_pass",
        "laminin_solid_core_qc_pass",
    ]

    if all(col in df.columns for col in component_cols):
        df["fiber_qc_pass"] = (
            as_bool_series(df["fiber_area_qc_pass"])
            & as_bool_series(df["laminin_membrane_signal_qc_pass"])
            & as_bool_series(df["laminin_solid_core_qc_pass"])
            & ~manual_excluded
        )
    elif "fiber_qc_pass" in df.columns:
        # If we cannot fully recompute, at least force manually excluded fibers to fail.
        current_fiber_qc = as_bool_series(df["fiber_qc_pass"])
        df["fiber_qc_pass"] = current_fiber_qc & ~manual_excluded

        print(
            "Warning: Could not fully recompute fiber_qc_pass because some component "
            "QC columns were missing. Excluded fibers were forced to fiber_qc_pass = False.",
            flush=True
        )
    else:
        df["fiber_qc_pass"] = ~manual_excluded

    # Recompute analysis_qc_pass if dystrophin_qc_pass exists.
    if "dystrophin_qc_pass" in df.columns:
        df["analysis_qc_pass"] = (
            as_bool_series(df["fiber_qc_pass"])
            & as_bool_series(df["dystrophin_qc_pass"])
        )
    elif "analysis_qc_pass" in df.columns:
        df["analysis_qc_pass"] = as_bool_series(df["fiber_qc_pass"])

    return df


def toggle_manual_exclusion(summary_df, exclude_df, fiber_id):
    """
    Toggle manual exclusion status for one fiber.

    If currently not excluded:
        - add to exclude_fibers.csv
        - set manual_exclude_fiber TRUE
        - set fiber_qc_pass FALSE after recomputing

    If currently excluded:
        - remove from exclude_fibers.csv
        - set manual_exclude_fiber FALSE
        - recompute fiber_qc_pass
    """

    fiber_id = int(fiber_id)

    mask = summary_df["fiber_id"].astype(int) == fiber_id

    if mask.sum() == 0:
        print(f"Fiber {fiber_id} was not found in summary CSV.", flush=True)
        return summary_df, exclude_df, False

    if mask.sum() > 1:
        raise ValueError(f"Fiber ID {fiber_id} appears more than once in the summary CSV.")

    row_index = summary_df.index[mask][0]

    currently_excluded_in_summary = as_bool_value(
        summary_df.loc[row_index, "manual_exclude_fiber"]
    )

    currently_excluded_in_file = fiber_id in set(exclude_df["fiber_id"].astype(int))

    currently_excluded = currently_excluded_in_summary or currently_excluded_in_file

    timestamp = datetime.now().isoformat(timespec="seconds")

    if currently_excluded:
        # Toggle OFF: remove manual exclusion
        exclude_df = exclude_df[exclude_df["fiber_id"].astype(int) != fiber_id].copy()

        summary_df.loc[row_index, "manual_exclude_fiber"] = False
        summary_df.loc[row_index, "fiber_exclude_reason"] = ""

        summary_df.loc[row_index, "manual_exclusion_toggle"] = "TRUE"
        summary_df.loc[row_index, "manual_exclusion_previous_state"] = "excluded"
        summary_df.loc[row_index, "manual_exclusion_toggle_timestamp"] = timestamp
        summary_df.loc[row_index, "manual_exclusion_toggle_note"] = "manual unexclude toggle"

        print(f"Fiber {fiber_id}: manually excluded -> NOT manually excluded", flush=True)

    else:
        # Toggle ON: add manual exclusion
        new_row = pd.DataFrame({
            "fiber_id": [fiber_id],
            "exclude_reason": [default_exclude_reason],
        })

        exclude_df = pd.concat([exclude_df, new_row], ignore_index=True)
        exclude_df = exclude_df.drop_duplicates(subset=["fiber_id"], keep="last")
        exclude_df["fiber_id"] = exclude_df["fiber_id"].astype(int)

        summary_df.loc[row_index, "manual_exclude_fiber"] = True
        summary_df.loc[row_index, "fiber_exclude_reason"] = default_exclude_reason

        summary_df.loc[row_index, "manual_exclusion_toggle"] = "TRUE"
        summary_df.loc[row_index, "manual_exclusion_previous_state"] = "not_excluded"
        summary_df.loc[row_index, "manual_exclusion_toggle_timestamp"] = timestamp
        summary_df.loc[row_index, "manual_exclusion_toggle_note"] = "manual exclude toggle"

        print(f"Fiber {fiber_id}: NOT manually excluded -> manually excluded", flush=True)

    summary_df = recompute_qc_columns(summary_df)

    # Print updated status
    updated_row = summary_df.loc[row_index]

    print(
        f"  manual_exclude_fiber = {updated_row['manual_exclude_fiber']}",
        flush=True
    )

    if "fiber_qc_pass" in summary_df.columns:
        print(
            f"  fiber_qc_pass = {updated_row['fiber_qc_pass']}",
            flush=True
        )

    if "analysis_qc_pass" in summary_df.columns:
        print(
            f"  analysis_qc_pass = {updated_row['analysis_qc_pass']}",
            flush=True
        )

    return summary_df, exclude_df, True


# Main script

print("Starting manual fiber exclusion toggle script...", flush=True)

if not summary_csv_path.exists():
    raise FileNotFoundError(f"Could not find: {summary_csv_path}")

summary_df = pd.read_csv(summary_csv_path)
summary_df = ensure_summary_columns(summary_df)

exclude_df = load_exclude_fibers(exclude_fibers_path)

make_backup(summary_csv_path, "manual_fiber_exclusion_toggle")
make_backup(exclude_fibers_path, "manual_fiber_exclusion_toggle")

print("")
print("Enter fiber IDs to toggle manual exclusion.")
print("You can enter one ID at a time:")
print("  1068")
print("Or several IDs separated by commas or spaces:")
print("  1068, 1204, 1450")
print("Press Enter with no number to save and quit.")
print("")

toggled_ids = []

while True:
    user_input = input("Fiber ID(s) to toggle manual exclusion: ").strip()

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
        summary_df, exclude_df, changed = toggle_manual_exclusion(
            summary_df,
            exclude_df,
            fiber_id
        )

        if changed:
            toggled_ids.append(fiber_id)

    # Save after each batch
    exclude_df = exclude_df.sort_values("fiber_id").reset_index(drop=True)
    exclude_df.to_csv(exclude_fibers_path, index=False)

    summary_df.to_csv(summary_csv_path, index=False)

    print(f"Saved updated exclusion file: {exclude_fibers_path}", flush=True)
    print(f"Saved updated summary CSV: {summary_csv_path}", flush=True)
    print("")

print("Done.", flush=True)

if len(toggled_ids) > 0:
    print(f"Toggled manual exclusion for {len(toggled_ids)} fiber entries:", flush=True)
    print(toggled_ids, flush=True)
else:
    print("No fibers were toggled.", flush=True)