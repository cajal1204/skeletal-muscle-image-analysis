import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil


# User input

csv_path = Path("image02_fiber_summary_with_dystrophin.csv")


# Helper functions

def as_bool_value(x):
    """
    Convert common TRUE/FALSE values to Python bool.
    """
    if pd.isna(x):
        return None

    if isinstance(x, bool):
        return x

    x_str = str(x).strip().lower()

    if x_str in ["true", "1", "yes", "y", "t", "positive"]:
        return True

    if x_str in ["false", "0", "no", "n", "f", "negative"]:
        return False

    return None


def bool_to_text(x):
    """
    Store booleans as simple TRUE/FALSE text for CSV compatibility.
    """
    if x is True:
        return "TRUE"
    if x is False:
        return "FALSE"
    return ""


def make_backup(path):
    """
    Make a timestamped backup before modifying the CSV.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(
        f"{path.stem}_backup_before_dystrophin_toggle_{timestamp}{path.suffix}"
    )

    shutil.copy2(path, backup_path)
    print(f"Backup saved as: {backup_path}", flush=True)


def ensure_override_columns(df):
    """
    Add manual override columns if missing.
    Force them to object/string-compatible columns so pandas does not complain.
    """
    if "dystrophin_manual_override" not in df.columns:
        df["dystrophin_manual_override"] = "FALSE"

    if "dystrophin_manual_override_previous_positive" not in df.columns:
        df["dystrophin_manual_override_previous_positive"] = ""

    if "dystrophin_manual_override_previous_class" not in df.columns:
        df["dystrophin_manual_override_previous_class"] = ""

    if "dystrophin_manual_override_timestamp" not in df.columns:
        df["dystrophin_manual_override_timestamp"] = ""

    if "dystrophin_manual_override_note" not in df.columns:
        df["dystrophin_manual_override_note"] = ""

    # Force these columns to generic object type so mixed values do not crash.
    override_cols = [
        "dystrophin_manual_override",
        "dystrophin_manual_override_previous_positive",
        "dystrophin_manual_override_previous_class",
        "dystrophin_manual_override_timestamp",
        "dystrophin_manual_override_note",
    ]

    for col in override_cols:
        df[col] = df[col].astype("object")

    return df


def toggle_fiber_status(df, fiber_id):
    """
    Toggle dystrophin status for one fiber.
    Positive becomes negative.
    Negative becomes positive.
    """

    if "fiber_id" not in df.columns:
        raise ValueError("CSV must contain a column named 'fiber_id'.")

    if "dystrophin_positive" not in df.columns:
        raise ValueError("CSV must contain a column named 'dystrophin_positive'.")

    if "dystrophin_class" not in df.columns:
        print("Warning: dystrophin_class column not found. Creating it.", flush=True)
        df["dystrophin_class"] = ""

    df = ensure_override_columns(df)

    mask = df["fiber_id"].astype(int) == int(fiber_id)

    if mask.sum() == 0:
        print(f"Fiber {fiber_id} was not found.", flush=True)
        return df

    if mask.sum() > 1:
        raise ValueError(f"Fiber ID {fiber_id} appears more than once in the file.")

    row_index = df.index[mask][0]

    old_value = as_bool_value(df.loc[row_index, "dystrophin_positive"])

    if old_value is None:
        print(
            f"Fiber {fiber_id} has unclear dystrophin_positive value: "
            f"{df.loc[row_index, 'dystrophin_positive']}"
        )
        response = input("Treat this as negative and toggle to positive? y/n: ").strip().lower()

        if response not in ["y", "yes"]:
            print("No change made.", flush=True)
            return df

        old_value = False

    new_value = not old_value

    old_class = df.loc[row_index, "dystrophin_class"]

    if pd.isna(old_class):
        old_class = ""

    new_class = "positive" if new_value else "negative"

    # Update final dystrophin status
    df.loc[row_index, "dystrophin_positive"] = new_value
    df.loc[row_index, "dystrophin_class"] = new_class

    # Optional: if your CSV has analysis_qc_pass, do not change it.
    # This script only toggles biological dystrophin status, not QC status.

    # Record manual override metadata as text
    df.loc[row_index, "dystrophin_manual_override"] = "TRUE"
    df.loc[row_index, "dystrophin_manual_override_previous_positive"] = bool_to_text(old_value)
    df.loc[row_index, "dystrophin_manual_override_previous_class"] = str(old_class)
    df.loc[row_index, "dystrophin_manual_override_timestamp"] = datetime.now().isoformat(timespec="seconds")
    df.loc[row_index, "dystrophin_manual_override_note"] = "manual toggle"

    print(
        f"Fiber {fiber_id}: "
        f"dystrophin_positive {old_value} -> {new_value}; "
        f"dystrophin_class {old_class} -> {new_class}",
        flush=True
    )

    return df


# Main script

print("Starting dystrophin status toggle script...", flush=True)

if not csv_path.exists():
    raise FileNotFoundError(f"Could not find: {csv_path}")

df = pd.read_csv(csv_path)

make_backup(csv_path)

print("")
print("Enter fiber IDs to toggle dystrophin status.")
print("You can enter one ID at a time:")
print("  1068")
print("Or several IDs separated by commas or spaces:")
print("  1068, 1204, 1450")
print("Press Enter with no number to save and quit.")
print("")

toggled_ids = []

while True:
    user_input = input("Fiber ID(s) to toggle: ").strip()

    # Blank input exits the loop
    if user_input == "":
        break

    # Allow comma-separated or space-separated entries
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
        df = toggle_fiber_status(df, fiber_id)
        toggled_ids.append(fiber_id)

    # Save after each batch of entries
    df.to_csv(csv_path, index=False)
    print(f"Saved updated CSV: {csv_path}", flush=True)
    print("")

print("Done.", flush=True)

if len(toggled_ids) > 0:
    print(f"Toggled {len(toggled_ids)} fiber entries:", flush=True)
    print(toggled_ids, flush=True)
else:
    print("No fibers were toggled.", flush=True)