# ––– state_context: How extreme is the current market environment? –––
# Volatility Regime: is the market calm or turbulent right now?
# Volume Analysis: is today's trading volume unusual?

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/detection")
OUTPUT_DIR = Path("data/context")

# Volatility regime
def detect_vol_regime():
    df = pd.read_parquet(INPUT_DIR / "^SPX.parquet")
    df = df.set_index("Date")
    df["vol_regime"] = pd.cut(
        df["volatility"],
        bins=[0, 0.005, 0.015, float("inf")],
        labels=["low", "moderate", "high"]
    )
    return df[["vol_regime"]]

# Volume Analysis
def detect_volume_state():
    for file in OUTPUT_DIR.glob("*.parquet"):
        df = pd.read_parquet(file)
        df["volume_ma20"] = df["Volume"].rolling(20).mean()
        df["volume_zscore"] = (df["Volume"] - df["volume_ma20"]) / df["Volume"].rolling(20).std()
        df["is_high_volume"] = df["volume_zscore"] > 2
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")

# Main function
def run_state_context():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vol_regime_df = detect_vol_regime()
    for file in OUTPUT_DIR.glob("*.parquet"):
        df = pd.read_parquet(file)
        df = df.set_index("Date").join(vol_regime_df, how="left").reset_index()
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")
    detect_volume_state()

if __name__ == "__main__":
    run_state_context()
