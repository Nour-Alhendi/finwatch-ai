import pandas as pd
from pathlib import Path
import numpy as np

INPUT_DIR = Path("data/detection")
OUTPUT_DIR = Path("data/context")

# Detect the current market regime using MA50 &  MA200 (Trend Following)
def detect_regime():
    df = pd.read_parquet(INPUT_DIR / "^SPX.parquet")
    df = df.set_index("Date")
    df["ma200"] = df["Close"].rolling(window=200).mean()
    df["ma50"] = df["Close"].rolling(window=50).mean()

    conditions = [
        (df["Close"] > df["ma200"] * 1.01) & (df["ma50"] > df["ma200"]),
        (df["Close"] < df["ma200"] * 0.99) & (df["ma50"] < df["ma200"]),
        (df["Close"] < df["ma200"] * 0.99) & (df["ma50"] > df["ma200"]),
        (df["Close"] > df["ma200"] * 1.01) & (df["ma50"] < df["ma200"])
    ]
    choices = ["bull", "bear", "transition_down", "transition_up"]

    df["regime"] = np.select(conditions, choices, default="sideways")
    df.loc[df["ma200"].isna(), "regime"] = "unknown"

    return df[["regime", "ma200", "ma50"]]

# loop over all diles and saves results
def run_market_trend():
      OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
      regime_df = detect_regime()
      for file in OUTPUT_DIR.glob("*.parquet"):
            df = pd.read_parquet(file)
            df = df.set_index("Date").join(regime_df, how="left").reset_index()
            df.to_parquet(OUTPUT_DIR / file.name)
            print(f"Saved: {file.name}")

# Entry point
if __name__ == "__main__":
        run_market_trend()