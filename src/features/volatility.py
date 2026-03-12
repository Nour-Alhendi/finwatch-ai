import pandas as pd
from pathlib import Path

INPUT_DIR  = Path("data/features")
OUTPUT_DIR = Path("data/features/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# calculate volatility
def volatility(file_path):
    df = pd.read_parquet(file_path)
    df["volatility"] = df["returns"].rolling(20).std()
    return df

# loops overall files and save results
def run_volatility():
    for file in INPUT_DIR.glob("*.parquet"):
        df = volatility(file)
        df.to_parquet(OUTPUT_DIR/file.name)
        print(f"Saved: {file.name}")

# Entry Point
if __name__ == "__main__":
    run_volatility()