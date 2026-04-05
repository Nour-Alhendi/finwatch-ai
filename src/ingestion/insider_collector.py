"""
FinWatch AI — Insider Trading Collector
=========================================
Downloads insider buy/sell transactions from Finnhub.

Why insider trading matters:
  Insiders (CEOs, CFOs, board members) know more than anyone.
  When insiders sell heavily → bearish signal.
  When insiders buy heavily  → bullish signal (they use their own money).

Output: data/fundamental/insider.parquet
Columns:
  ticker, date,
  buys_30d, sells_30d, net_shares_30d,
  insider_sentiment  (-1 = heavy selling, 0 = neutral, +1 = heavy buying)

Usage:
  python -m src.ingestion.insider_collector
"""

import os
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

ROOT     = Path(__file__).resolve().parents[2]
OUT_DIR  = ROOT / "data/fundamental"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "insider.parquet"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
BASE_URL        = "https://finnhub.io/api/v1"
LOOKBACK_DAYS   = 30

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _get_tickers() -> list:
    detection_dir = ROOT / "data/detection"
    return sorted(f.stem for f in detection_dir.glob("*.parquet") if not f.stem.startswith("^"))


def _fetch_insider(ticker: str) -> list:
    """Fetch insider transactions from Finnhub (last LOOKBACK_DAYS days)."""
    if not FINNHUB_API_KEY:
        return []
    start = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    end   = date.today().isoformat()
    try:
        r = requests.get(
            f"{BASE_URL}/stock/insider-transactions",
            params={"symbol": ticker, "from": start, "to": end, "token": FINNHUB_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("data", [])
    except Exception as e:
        logging.warning(f"  [{ticker}] insider fetch failed: {e}")
        return []


def _insider_sentiment(net_shares: int, total_shares: int) -> float:
    """
    Normalize net insider activity to -1 / 0 / +1 range.
    net_shares > 0 = net buying, < 0 = net selling.
    """
    if total_shares == 0:
        return 0.0
    ratio = net_shares / total_shares
    return round(max(-1.0, min(1.0, ratio * 10)), 4)


def collect(tickers: list = None) -> pd.DataFrame:
    if not FINNHUB_API_KEY:
        logging.error("FINNHUB_API_KEY not set.")
        return pd.DataFrame()

    if tickers is None:
        tickers = _get_tickers()

    rows  = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{total}] {ticker}...", end=" ", flush=True)

        transactions = _fetch_insider(ticker)
        time.sleep(0.6)

        buys_count  = 0
        sells_count = 0
        buy_shares  = 0
        sell_shares = 0

        for tx in transactions:
            change = tx.get("change", 0) or 0
            if change > 0:
                buys_count += 1
                buy_shares += change
            elif change < 0:
                sells_count += 1
                sell_shares += abs(change)

        net_shares  = buy_shares - sell_shares
        total_shares = buy_shares + sell_shares
        sentiment   = _insider_sentiment(net_shares, total_shares)

        rows.append({
            "ticker":            ticker,
            "date":              date.today().isoformat(),
            "buys_30d":          buys_count,
            "sells_30d":         sells_count,
            "buy_shares_30d":    buy_shares,
            "sell_shares_30d":   sell_shares,
            "net_shares_30d":    net_shares,
            "insider_sentiment": sentiment,
        })

        label = "buying" if sentiment > 0.1 else ("selling" if sentiment < -0.1 else "neutral")
        print(f"buys={buys_count}  sells={sells_count}  sentiment={sentiment:+.3f} ({label})")

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
    print("Collecting insider trading data from Finnhub...")
    df = collect()
    if not df.empty:
        out = save(df)
        print(f"\nSaved {len(df)} records → {out}")
    else:
        print("No records collected.")
