import pandas as pd
from pathlib import Path

INPUT_DIR  = Path("data/features")
OUTPUT_DIR = Path("data/detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# calculate Z-Score and Anomly flag
def zscore(file_path):
    df = pd.read_parquet(file_path)
    df["z_score"] = (df["returns"] - df["rolling_mean"]) / df["rolling_std"]
    df["z_anomaly"] = df["z_score"].abs() > 3
    df["rolling_mean_60"] = df["returns"].rolling(60).mean()
    df["rolling_std_60"] = df["returns"].rolling(60).std()
    df["z_score_60"] = (df["returns"] - df["rolling_mean_60"]) / df["rolling_std_60"]
    df["z_anomaly_60"] = df["z_score_60"].abs() > 2
    return df

# loop over all files and saves results 
def run_zscore():
    for file in INPUT_DIR.glob("*.parquet"):
        df = zscore(file)
        df.to_parquet(OUTPUT_DIR/file.name)
        print(f"Saved {file.name}")

# Entry point 
if __name__ == "__main__":
    run_zscore()