from pathlib import Path
from datetime import datetime
import shutil

import numpy as np
import pandas as pd


# ============================================================
# 1. DIRECTORIES
# ============================================================

# Original data: multiple instrument-download folders
old_dir = Path(
    "/Users/maywang/Library/CloudStorage/"
    "OneDrive-DalhousieUniversity/Thesis/IMS/git_code/"
    "SIMBA/data/2024"
)

# New data: one merged TXT file per variable
new_dir = Path(
    "/Users/maywang/Library/CloudStorage/"
    "OneDrive-DalhousieUniversity/Thesis/IMS/HeatBudget/"
    "heat_budget_refactor_v2/2024/SIMBA"
)

new_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. VARIABLES TO MERGE
# ============================================================

variables = [
    "TEMPDATA",
    "DELDATA0",
    "DELDATA1",
    "DELDATA4",
]


# ============================================================
# 3. FUNCTION TO MERGE ORIGINAL TXT FILES
# ============================================================

def merge_simba_files(variable):

    # Find files across all original download folders
    files = sorted(
        old_dir.glob(f"*/{variable}*")
    )

    if not files:
        raise FileNotFoundError(
            f"No original files found for {variable}"
        )

    print(f"\n{'=' * 55}")
    print(f"Merging {variable}")
    print("=" * 55)

    # Store one original text row per unique timestamp
    records = {}

    total_rows = 0
    duplicate_rows = 0

    for file in files:

        # Read data using the original SIMBA approach
        df = pd.read_fwf(
            file,
            header=None,
        )

        # Read the actual text lines so original
        # measurement precision is preserved
        with open(file, "rb") as f:
            lines = [
                line
                for line in f
                if line.strip()
            ]

        if len(lines) != len(df):
            raise ValueError(
                f"Number of text lines does not match "
                f"parsed rows in {file}"
            )

        # Reconstruct timestamps as in the original loader
        times = pd.to_datetime(
            df.iloc[:, 1].astype(str)
            + " "
            + df.iloc[:, 2].astype(str)
        )

        total_rows += len(df)

        for i, timestamp in enumerate(times):

            # Original unmodified row
            line = lines[i]

            # SIMBA measurement columns
            measurements = df.iloc[i, 8:].to_numpy(
                dtype=float
            )

            if timestamp in records:

                old_line, old_measurements = records[timestamp]

                # Check that overlapping downloads contain
                # the same measurements
                identical = np.array_equal(
                    measurements,
                    old_measurements,
                    equal_nan=True,
                )

                if not identical:
                    raise ValueError(
                        f"Conflicting measurements in {variable} "
                        f"at {timestamp}.\n"
                        f"Conflicting file: {file}\n"
                        "Merge stopped. Check the source data."
                    )

                duplicate_rows += 1

            else:

                records[timestamp] = (
                    line,
                    measurements,
                )

        print(
            f"{file.parent.name}/{file.name}: "
            f"{len(df)} profiles"
        )

    # --------------------------------------------------------
    # Sort observations chronologically
    # --------------------------------------------------------

    sorted_times = sorted(records)

    output_file = new_dir / f"{variable}.TXT"

    temporary_file = new_dir / f"{variable}.TXT.tmp"

    # Write original text rows in chronological order
    with open(temporary_file, "wb") as f:

        for timestamp in sorted_times:

            line, _ = records[timestamp]

            f.write(
                line.rstrip(b"\r\n") + b"\n"
            )

    # --------------------------------------------------------
    # Verify the merged file BEFORE replacing anything
    # --------------------------------------------------------

    check = pd.read_fwf(
        temporary_file,
        header=None,
    )

    check_times = pd.to_datetime(
        check.iloc[:, 1].astype(str)
        + " "
        + check.iloc[:, 2].astype(str)
    )

    expected_times = pd.DatetimeIndex(sorted_times)

    if len(check) != len(sorted_times):
        raise ValueError(
            f"Verification failed for {variable}: "
            "incorrect number of profiles."
        )

    if not np.array_equal(
        check_times.to_numpy(),
        expected_times.to_numpy(),
    ):
        raise ValueError(
            f"Verification failed for {variable}: "
            "timestamps do not match."
        )

    # Verify measurement values as well
    expected_measurements = np.stack([
        records[t][1]
        for t in sorted_times
    ])

    written_measurements = check.iloc[:, 8:].to_numpy(
        dtype=float
    )

    if not np.array_equal(
        expected_measurements,
        written_measurements,
        equal_nan=True,
    ):
        raise ValueError(
            f"Verification failed for {variable}: "
            "measurement values do not match."
        )

    # --------------------------------------------------------
    # Back up existing merged file and replace it
    # --------------------------------------------------------

    if output_file.exists():

        shutil.copy2(
            output_file,
            backup_dir / output_file.name,
        )

    temporary_file.replace(output_file)

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\nMerge successful!")

    print("Total input profiles:", total_rows)
    print("Duplicate profiles removed:", duplicate_rows)
    print("Final profiles:", len(sorted_times))

    print("First timestamp:", sorted_times[0])
    print("Last timestamp:", sorted_times[-1])

    print("Saved to:", output_file)


# ============================================================
# 4. CREATE BACKUP DIRECTORY
# ============================================================

backup_dir = (
    new_dir
    / f"backup_before_merge_{datetime.now():%Y%m%d_%H%M%S}"
)

backup_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 5. MERGE ALL FOUR VARIABLES
# ============================================================

for variable in variables:

    merge_simba_files(variable)


print("\nAll four SIMBA files successfully merged.")
print("Backup directory:", backup_dir)