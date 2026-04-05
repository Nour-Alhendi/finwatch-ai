# Isolation Forest anomaly detection — group-aware training.
# One model per group, trained on all stocks in that group combined (calm periods only).
# Groups are defined by volatility behavior (Stable vs Volatile) within each sector.
# Columns: if_anomaly

import sys
import joblib
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.features import IF_FEATURES

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)

INPUT_DIR      = Path("data/detection")
OUTPUT_DIR     = Path("data/detection")
TRAIN_DATA_END = pd.Timestamp("2026-03-01")  # training data cutoff
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "Technology-Stable":   {"tickers": ["AAPL", "MSFT", "GOOG", "INTC"],      "calm_q": 0.75, "percentile": 3},
    "Technology-Volatile": {"tickers": ["NVDA", "AMD"],                                "calm_q": 0.65, "percentile": 4},
    "Semiconductors":      {"tickers": ["AVGO", "QCOM", "MU"],                        "calm_q": 0.70, "percentile": 3},
    "AI-Stable":           {"tickers": ["CRM", "SNOW"],                               "calm_q": 0.70, "percentile": 3},
    "AI-Volatile":         {"tickers": ["PLTR", "META"],                              "calm_q": 0.65, "percentile": 4},
    "Cybersecurity":       {"tickers": ["PANW", "CRWD", "NET"],                       "calm_q": 0.65, "percentile": 4},
    "Consumer-Stable":     {"tickers": ["NKE", "MCD", "SBUX"],                        "calm_q": 0.75, "percentile": 3},
    "Consumer-Volatile":   {"tickers": ["TSLA", "AMZN"],                              "calm_q": 0.70, "percentile": 3},
    "Financials":          {"tickers": ["JPM", "BAC", "GS", "MS", "BLK", "V", "MA"], "calm_q": 0.70, "percentile": 3},
    "Healthcare":          {"tickers": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "AMGN"], "calm_q": 0.75, "percentile": 3},
    "Consumer Staples":    {"tickers": ["PG", "KO", "COST", "WMT", "CL"],             "calm_q": 0.75, "percentile": 3},
    "Energy":              {"tickers": ["XOM", "CVX", "COP", "SLB", "EOG"],           "calm_q": 0.75, "percentile": 3},
    "Industrials":         {"tickers": ["CAT", "HON", "BA", "GE", "RTX"],             "calm_q": 0.70, "percentile": 3},
    "Green Energy":        {"tickers": ["ENPH", "NEE", "FSLR"],                       "calm_q": 0.75, "percentile": 3},
    "Crypto-Volatile":     {"tickers": ["BITF", "IREN", "MARA", "RIOT", "COIN"],      "calm_q": 0.60, "percentile": 5},
    "SmallCap-Volatile":   {"tickers": ["BE", "PLUG", "RCAT", "KRKNF", "QNC", "RZLV", "NBIS", "AI"], "calm_q": 0.55, "percentile": 5},
}


def run_isolation_forest():
    for group_name, params in GROUPS.items():
        tickers = params["tickers"]
        calm_q = params["calm_q"]
        perc = params["percentile"]

        # Load all stocks in the group
        frames = []
        for ticker in tickers:
            file = INPUT_DIR / f"{ticker}.parquet"
            if not file.exists():
                print(f"  Skipping {ticker}: file not found")
                continue
            df = pd.read_parquet(file)
            df["_is_train"] = (df["Date"] >= df["Date"].max() - pd.DateOffset(years=4)) & (df["Date"] < TRAIN_DATA_END)
            df = df.dropna(subset=IF_FEATURES).reset_index(drop=True)  # type: ignore
            if df.empty:
                print(f"  Skipping {ticker}: no rows after dropna (missing features)")
                continue
            df["_ticker"] = ticker
            frames.append(df)

        if not frames:
            print(f"Skipping group {group_name}: no data")
            continue

        # Load model from disk if already trained — otherwise train and save
        model_path = MODELS_DIR / f"if_{group_name}.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"[{group_name}] model loaded from disk")
        else:
            group_df = pd.concat(frames, ignore_index=True)
            normal_mask = group_df["volatility"] < group_df["volatility"].quantile(calm_q)
            X_normal = group_df[normal_mask & group_df["_is_train"]][IF_FEATURES].values
            model = IsolationForest(n_estimators=100, random_state=42)
            model.fit(X_normal)
            joblib.dump(model, model_path)
            print(f"[{group_name}] trained on {X_normal.shape[0]} calm rows, saved to disk")

        # Predict for each stock individually
        for df in frames:
            if df.empty:
                continue
            ticker = df["_ticker"].iloc[0]
            df = df.drop(columns=["_ticker"])
            df = df.drop(columns=["_is_train"])

            scores = model.score_samples(df[IF_FEATURES].values)
            threshold = np.percentile(scores, perc)
            df["if_anomaly"] = scores < threshold

            df.to_parquet(OUTPUT_DIR / f"{ticker}.parquet")
            print(f"  Saved: {ticker}.parquet  ({df['if_anomaly'].sum()} anomalies)")


if __name__ == "__main__":
    run_isolation_forest()
