"""
FinWatch AI — Options / Put-Call Ratio Collector
==================================================
Downloads options chain data via yfinance and computes Put/Call ratio.

Why Put/Call ratio matters:
  Options traders bet with real money.
  High put/call ratio → market is buying protection → fear signal.
  Low put/call ratio  → market is buying calls      → greed signal.

  put_call_ratio > 1.0  = more puts than calls  → bearish sentiment
  put_call_ratio < 0.7  = more calls than puts  → bullish sentiment
  put_call_ratio ~ 0.85 = historical average

Output: data/fundamental/options.parquet
Columns:
  ticker, date,
  put_volume, call_volume, put_call_ratio,
  options_fear  (1 if put_call_ratio > 1.0, else 0)

Usage:
  python -m src.ingestion.options_collector
"""

import time
import logging
import pandas as pd
import yfinance as yf
from typing import Optional
from pathlib import Path
from datetime import date

ROOT     = Path(__file__).resolve().parents[2]
OUT_DIR  = ROOT / "data/fundamental"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "options.parquet"

logging.basicConfig(level=logging.INFO, format="%(message)s")

FEAR_THRESHOLD = 1.0   # put/call ratio above this = fear


def _get_tickers() -> list:
    detection_dir = ROOT / "data/detection"
    return sorted(f.stem for f in detection_dir.glob("*.parquet") if not f.stem.startswith("^"))


def _fetch_put_call(ticker: str) -> Optional[dict]:
    """
    Fetch nearest-expiry options chain via yfinance.
    Returns put volume, call volume, and put/call ratio.
    """
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None

        # Use the nearest expiry for the most current market sentiment
        chain = t.option_chain(expirations[0])

        put_vol  = chain.puts["volume"].fillna(0).sum()
        call_vol = chain.calls["volume"].fillna(0).sum()

        if call_vol == 0:
            return None

        ratio = round(put_vol / call_vol, 4)
        return {
            "put_volume":    int(put_vol),
            "call_volume":   int(call_vol),
            "put_call_ratio": ratio,
            "options_fear":  1 if ratio > FEAR_THRESHOLD else 0,
        }
    except Exception as e:
        logging.warning(f"  options fetch failed: {e}")
        return None


def collect(tickers: list = None) -> pd.DataFrame:
    if tickers is None:
        tickers = _get_tickers()

    rows  = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{total}] {ticker}...", end=" ", flush=True)

        result = _fetch_put_call(ticker)
        time.sleep(0.3)

        if result is None:
            print("no options data")
            continue

        rows.append({
            "ticker":          ticker,
            "date":            date.today().isoformat(),
            **result,
        })

        fear_label = "FEAR" if result["options_fear"] else "normal"
        print(
            f"puts={result['put_volume']:,}  calls={result['call_volume']:,}  "
            f"ratio={result['put_call_ratio']:.3f}  [{fear_label}]"
        )

    return pd.DataFrame(rows)


def save(df: pd.DataFrame) -> Path:
    if df.empty:
        logging.warning("Nothing to save.")
        return OUT_PATH
    today = df["date"].iloc[0]
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        existing = existing[existing["date"] != today]
        df = pd.concat([existing, df], ignore_index=True)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df.to_parquet(OUT_PATH, index=False)
    return OUT_PATH


if __name__ == "__main__":
    print("Collecting options / put-call ratio data...")
    df = collect()
    if not df.empty:
        out = save(df)
        print(f"\nSaved {len(df)} records → {out}")
    else:
        print("No records collected.")
