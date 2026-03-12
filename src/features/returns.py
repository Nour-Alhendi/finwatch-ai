import pandas as pd
from pathlib import Path

INPUT_DIR  = Path("data/raw/raw_clean")
OUTPUT_DIR = Path("data/features/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# calculates daily % change in Close price
def returns(file_path):
    df = pd.read_parquet(file_path)
    df["returns"] = df["Close"].pct_change()
    return df


# loops over all files and saves results
def run_returns():
    for file in INPUT_DIR.glob("*.parquet"):
        df = returns(file)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")
    


# Entry Point
if __name__ == "__main__":
    run_returns()