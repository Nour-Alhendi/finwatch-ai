"""
FinWatch AI — Earnings Collector
==================================
Downloads upcoming and historical earnings dates from Finnhub.

Why earnings matter:
  Stocks behave differently in the days before and after earnings.
  Knowing "earnings is in 3 days" is a strong context signal for the models.

Output: data/fundamental/earnings.parquet
Columns:
  ticker, date, eps_actual, eps_estimate, eps_surprise_pct,
  revenue_actual, revenue_estimate, days_to_next_earnings, is_earnings_week

Usage:
  python -m src.ingestion.earnings_collector
"""

import os
import time
import logging
import requests
import pandas as pd
from typing import Optional
from pathlib import Path
from datetime import date, timedelta

ROOT     = Path(__file__).resolve().parents[2]
OUT_DIR  = ROOT / "data/fundamental"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "earnings.parquet"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
BASE_URL        = "https://finnhub.io/api/v1"

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _get_tickers() -> list:
    detection_dir = ROOT / "data/detection"
    return sorted(f.stem for f in detection_dir.glob("*.parquet") if not f.stem.startswith("^"))


def _fetch_earnings_history(ticker: str, lookback_years: int = 3) -> list:
    """Fetch past earnings results (EPS actual vs estimate)."""
    if not FINNHUB_API_KEY:
        return []
    start = (date.today() - timedelta(days=365 * lookback_years)).isoformat()
    end   = date.today().isoformat()
    try:
        r = requests.get(
            f"{BASE_URL}/stock/earnings",
            params={"symbol": ticker, "token": FINNHUB_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            return []
        return data
    except Exception as e:
        logging.warning(f"  [{ticker}] earnings history failed: {e}")
        return []


def _fetch_next_earnings(ticker: str) -> Optional[dict]:
    """Fetch the next scheduled earnings date from the calendar endpoint."""
    if not FINNHUB_API_KEY:
        return None
    today = date.today()
    end   = (today + timedelta(days=90)).isoformat()
    try:
        r = requests.get(
            f"{BASE_URL}/calendar/earnings",
            params={
                "symbol": ticker,
                "from":   today.isoformat(),
                "to":     end,
                "token":  FINNHUB_API_KEY,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        events = data.get("earningsCalendar", [])
        if events:
            return events[0]   # nearest upcoming
        return None
    except Exception as e:
        logging.warning(f"  [{ticker}] next earnings failed: {e}")
        return None


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

        # Historical earnings (past EPS surprises)
        history = _fetch_earnings_history(ticker)
        time.sleep(0.5)

        # Next scheduled earnings date
        next_event = _fetch_next_earnings(ticker)
        time.sleep(0.5)

        # Days to next earnings
        days_to_next = None
        is_earnings_week = False
        if next_event and next_event.get("date"):
            try:
                earn_date    = date.fromisoformat(next_event["date"])
                days_to_next = (earn_date - date.today()).days
                is_earnings_week = 0 <= days_to_next <= 7
            except Exception:
                pass

        # Most recent past earnings
        if history:
            last = history[0]
            eps_actual   = last.get("actual")
            eps_estimate = last.get("estimate")
            eps_surprise = (
                round((eps_actual - eps_estimate) / abs(eps_estimate) * 100, 2)
                if eps_actual is not None and eps_estimate and eps_estimate != 0
                else None
            )
            rev_actual   = last.get("revenueActual")
            rev_estimate = last.get("revenueEstimate")
        else:
            eps_actual = eps_estimate = eps_surprise = None
            rev_actual = rev_estimate = None

        rows.append({
            "ticker":               ticker,
            "date":                 date.today().isoformat(),
            "eps_actual":           eps_actual,
            "eps_estimate":         eps_estimate,
            "eps_surprise_pct":     eps_surprise,
            "revenue_actual":       rev_actual,
            "revenue_estimate":     rev_estimate,
            "days_to_next_earnings": days_to_next,
            "is_earnings_week":     is_earnings_week,
        })

        status = f"next earnings in {days_to_next}d" if days_to_next is not None else "no upcoming date"
        surprise = f"  EPS surprise={eps_surprise:+.1f}%" if eps_surprise is not None else ""
        print(f"{status}{surprise}")

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
    print("Collecting earnings data from Finnhub...")
    df = collect()
    if not df.empty:
        out = save(df)
        print(f"\nSaved {len(df)} records → {out}")
    else:
        print("No records collected.")
