# Combines all anomaly signals into a single score and combined_anomaly flag.
# Signals: z_anomaly (20d), z_anomaly_60 (60d), if_anomaly, ae_anomaly
# Columns: anomaly_score, combined_anomaly
#
# Threshold selection (Option 1):
# The combined_anomaly threshold is loaded from backtesting results
# (data/backtesting/anomaly_precision.parquet) if available.
# The threshold with the best F1 score is selected automatically —
# balancing precision (few false alarms) and recall (catching real risks).
# Falls back to 0.30 if backtesting has not been run yet.

import pandas as pd
from pathlib import Path
from typing import Optional

INPUT_DIR  = Path("data/detection")
OUTPUT_DIR = Path("data/detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FALLBACK_THRESHOLD = 0.30


def _load_best_threshold() -> float:
    """
    Read anomaly_precision.parquet and return the anomaly_score_weighted
    threshold with the highest F1 score.
    Falls back to FALLBACK_THRESHOLD if the file does not exist yet.
    """
    p = Path("data/backtesting/anomaly_precision.parquet")
    if not p.exists():
        return FALLBACK_THRESHOLD

    prec = pd.read_parquet(p)
    # Use only the validation split for threshold selection — never the holdout
    candidates = prec[
        (prec["detector"] == "anomaly_score_weighted") &
        (prec.get("split", pd.Series(["validation"] * len(prec))) == "validation")
    ]
    if candidates.empty:
        # Fallback: file exists but has no split column (old format)
        candidates = prec[prec["detector"] == "anomaly_score_weighted"]
    if candidates.empty:
        return FALLBACK_THRESHOLD

    best_row = candidates.loc[candidates["f1"].idxmax()]
    threshold = float(best_row["threshold"]) if "threshold" in best_row.index else float(best_row["signal"].split(">=")[1].strip().split()[0])
    print(
        f"  [combine] threshold from backtesting: {threshold:.2f}"
        f"  (F1={best_row['f1']:.3f}  precision={best_row['precision']:.3f}"
        f"  recall={best_row['recall']:.3f})"
    )
    return threshold


def combine(file_path, threshold: Optional[float] = None):
    df = pd.read_parquet(file_path)

    # Weighted anomaly score — ML models (IF, AE) weighted higher than pure statistics
    # Z-score: 0.20 each (rule-based, fast but simple)
    # Isolation Forest: 0.30 (ML, trained on calm periods)
    # LSTM Autoencoder: 0.30 (deep learning, sequence-aware)
    z_short = df["z_anomaly"].fillna(False).astype(float)
    z_long  = df["z_anomaly_60"].fillna(False).astype(float)
    iso     = df["if_anomaly"].fillna(False).astype(float) if "if_anomaly" in df.columns else pd.Series(0.0, index=df.index)
    ae      = df["ae_anomaly"].fillna(False).astype(float) if "ae_anomaly" in df.columns else pd.Series(0.0, index=df.index)

    df["anomaly_score_weighted"] = (
        0.20 * z_short +
        0.20 * z_long  +
        0.30 * iso     +
        0.30 * ae
    ).round(4)

    # Keep integer score (0-4) for backward compatibility
    df["anomaly_score"] = (z_short + z_long + iso + ae).astype(int)

    # combined_anomaly threshold — data-driven if backtesting results exist
    if threshold is None:
        threshold = _load_best_threshold()

    df["combined_anomaly"] = df["anomaly_score_weighted"] >= threshold
    df["market_anomaly"]   = df["is_market_wide"] & df["combined_anomaly"]
    df["sector_anomaly"]   = df["is_sector_wide"] & df["combined_anomaly"]

    return df


def run_combine():
    threshold = _load_best_threshold()
    for file in INPUT_DIR.glob("*.parquet"):
        if file.stem == "^SPX":
            continue
        df = combine(file, threshold=threshold)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")


if __name__ == "__main__":
    run_combine()
