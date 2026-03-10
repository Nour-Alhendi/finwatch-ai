# =====================================================
# Data Corruption Injector
# =====================================================

import pandas as pd
from pathlib import Path
from scenarios import SCENARIOS

INPUT_DIR  = Path("data/raw/raw_clean")
OUTPUT_DIR = Path("data/raw/raw_corrupted")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------
# Function: corrupt_file
# Purpose : Apply all scenarios to a single parquet file
#           and save the result to OUTPUT_DIR
# -----------------------------------------------------
def corrupt_file(file_path):

    # load the clean data
    df = pd.read_parquet(file_path)

    # add anomaly_type column — None means the row is clean
    df["data_quality_alert"] = None

    # apply each injection function with its rate
    for inject_fn, rate in SCENARIOS.items():
        df = inject_fn(df, rate)

    # save corrupted file to output directory
    output_path = OUTPUT_DIR / file_path.name
    df.to_parquet(output_path)

    print(f"Corrupted: {file_path.name} → saved to {output_path}")


# -----------------------------------------------------
# Function: run_injector
# Purpose : Loop over all parquet files and corrupt them
# -----------------------------------------------------
def run_injector():
    files = list(INPUT_DIR.glob("*.parquet"))

    if not files:
        print(f"No parquet files found in {INPUT_DIR}")
        return

    print(f"\nFound {len(files)} files. Starting corruption...\n")

    for file in files:
        corrupt_file(file)

    print(f"\nDone. Corrupted files saved to {OUTPUT_DIR}")


# -----------------------------------------------------
# Entry Point
# -----------------------------------------------------
if __name__ == "__main__":
    run_injector()
