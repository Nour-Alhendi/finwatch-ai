"""
FinWatch AI — Valuation Collector
===================================
Downloads fundamental valuation metrics from Finnhub.

Why valuation matters:
  RSI oversold at P/E=8 (cheap) is a real entry signal.
  RSI oversold at P/E=80 (expensive) is a falling knife.
  Without valuation, "low risk + oversold" looks the same in both cases.

Metrics collected:
  pe_ratio       - Trailing P/E (price / TTM earnings)
  pe_forward     - Forward P/E (price / next year expected earnings)
  pb_ratio       - Price-to-Book ratio
  revenue_growth - Revenue growth YoY (TTM), as decimal (0.12 = 12%)

Output: data/fundamental/valuation.parquet

Usage:
  python -m src.ingestion.valuation_collector
"""

import os
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import date

ROOT     = Path(__file__).resolve().parents[2]
OUT_DIR  = ROOT / "data/fundamental"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "valuation.parquet"

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


def _fetch_metrics(ticker: str) -> dict:
    if not FINNHUB_API_KEY:
        return {}
    try:
        r = requests.get(
            f"{BASE_URL}/stock/metric",
            params={"symbol": ticker, "metric": "all", "token": FINNHUB_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        m = r.json().get("metric", {})

        # Trailing P/E — try multiple field names Finnhub uses
        pe = m.get("peBasicExclExtraTTM") or m.get("peTTM") or m.get("peNormalizedAnnual")

        # Forward P/E — not always available
        pe_forward = m.get("forwardPE") or m.get("peForward")

        # Price-to-Book
        pb = m.get("pbAnnual") or m.get("pbQuarterly")

        # Revenue growth YoY (TTM) — decimal, e.g. 0.12 = +12%
        rev_growth = m.get("revenueGrowthTTMYoy") or m.get("revenueGrowthAnnual")

        def _safe(v):
            try:
                f = float(v)
                return None if (f != f) else round(f, 4)   # nan check
            except (TypeError, ValueError):
                return None

        return {
            "pe_ratio":      _safe(pe),
            "pe_forward":    _safe(pe_forward),
            "pb_ratio":      _safe(pb),
            "revenue_growth": _safe(rev_growth),
        }
    except Exception as e:
        logging.warning(f"  [{ticker}] metrics failed: {e}")
        return {}


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

        m = _fetch_metrics(ticker)
        time.sleep(0.5)

        if not m:
            print("no data")
            continue

        rows.append({
            "ticker":         ticker,
            "date":           date.today().isoformat(),
            "pe_ratio":       m.get("pe_ratio"),
            "pe_forward":     m.get("pe_forward"),
            "pb_ratio":       m.get("pb_ratio"),
            "revenue_growth": m.get("revenue_growth"),
        })

        pe_str  = f"P/E={m['pe_ratio']:.1f}"      if m.get("pe_ratio")      is not None else "P/E=N/A"
        pf_str  = f"  fwdPE={m['pe_forward']:.1f}" if m.get("pe_forward")   is not None else ""
        pb_str  = f"  P/B={m['pb_ratio']:.2f}"     if m.get("pb_ratio")     is not None else ""
        rg_str  = f"  rev={m['revenue_growth']:+.1%}" if m.get("revenue_growth") is not None else ""
        print(f"{pe_str}{pf_str}{pb_str}{rg_str}")

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
    print("Collecting valuation metrics from Finnhub...")
    df = collect()
    if not df.empty:
        out = save(df)
        print(f"\nSaved {len(df)} records → {out}")
    else:
        print("No records collected.")
