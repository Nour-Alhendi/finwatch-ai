"""
FinWatch AI — Sentiment History Collector
==========================================
Runs daily to build a historical sentiment dataset.
In 3-6 months this data can be used as a training feature for XGBoost.

Output: data/sentiment/history.parquet
Columns: date, ticker, avg_sentiment (-1 to +1), n_articles

Usage:
    python -m src.ingestion.sentiment_collector          # all tickers
    python -m src.ingestion.sentiment_collector AAPL MSFT  # specific tickers
"""

import sys
import time
import logging
import pandas as pd
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data/sentiment/history.parquet"

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _tickers_from_detection() -> list:
    detection_dir = ROOT / "data/detection"
    return sorted(
        f.stem for f in detection_dir.glob("*.parquet")
        if not f.stem.startswith("^")
    )


def collect(tickers: list = None) -> pd.DataFrame:
    """Fetch headlines + VADER scores for each ticker. Returns DataFrame."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not os.environ.get("FINNHUB_API_KEY"):
        logging.error(
            "FINNHUB_API_KEY not set — no news can be fetched.\n"
            "Export it first:  export FINNHUB_API_KEY=your_key_here"
        )
        return pd.DataFrame()


    if tickers is None:
        tickers = _tickers_from_detection()

    sys.path.insert(0, str(ROOT / "src"))
    from explainability.finbert import fetch_news, aggregate_scores

    today = date.today().isoformat()
    rows  = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{total}] {ticker}...", end=" ", flush=True)
        headlines, _, timestamps = fetch_news(ticker, limit=5, retries=2)
        time.sleep(1.2)

        if not headlines:
            print("no news")
            continue

        scores = aggregate_scores(headlines, timestamps)
        rows.append({
            "date":                 today,
            "ticker":               ticker,
            "vader_score":          scores["vader_score"],
            "finbert_score":        scores["finbert_score"],
            "avg_sentiment":        scores["news_sentiment_score"],
            "n_articles":           scores["n_articles"],
        })
        print(f"vader={scores['vader_score']:+.3f}  finbert={scores['finbert_score']:+.3f}  combined={scores['news_sentiment_score']:+.3f}")

    return pd.DataFrame(rows)


def save(df: pd.DataFrame) -> Path:
    """Append today's records to history.parquet (deduplicates same-day entries)."""
    if df.empty:
        logging.warning("Nothing to save — empty DataFrame.")
        return OUT_PATH

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = df["date"].iloc[0]

    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        # Remove any existing rows for today before appending (idempotent)
        existing = existing[existing["date"] != today]
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    combined.to_parquet(OUT_PATH, index=False)
    return OUT_PATH


if __name__ == "__main__":
    tickers = sys.argv[1:] or None
    label = ", ".join(tickers) if tickers else "all tickers"
    print(f"Collecting sentiment for {label}...")

    df = collect(tickers)
    if not df.empty:
        out = save(df)
        print(f"\nSaved {len(df)} records → {out}")
    else:
        print("No records collected.")
