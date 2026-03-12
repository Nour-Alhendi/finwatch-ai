import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# calculate correlation
def correlation(file_path):
    df = pd.read_parquet(file_path)
    market = pd.read_parquet(INPUT_DIR / "^SPX.parquet")
    df["corr_spx"] = df["returns"].rolling(60).corr(market["returns"])
    return df

# loop over all files and saves results
def run_correlation():
    for file in INPUT_DIR.glob("*.parquet"):
        if file.name == "^SPX.parquet":
            continue
        df = correlation(file)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")

# Entry Point
if __name__ == "__main__":
    run_correlation()