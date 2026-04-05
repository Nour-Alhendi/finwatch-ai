import pandas as pd
import yaml
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]


def run():
    with open(ROOT / "config/assets.yaml", "r") as f:
        config = yaml.safe_load(f)

    assets     = config["assets"]
    references = config["references"]

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=365 * 10)

    output_dir = ROOT / "data/raw/raw_clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = ROOT / "data/raw/references"
    ref_dir.mkdir(parents=True, exist_ok=True)

    def _download(ticker: str, out_dir: Path):
        print(f"Downloading {ticker}...", end=" ")
        try:
            df = yf.download(ticker, start=start_date, end=end_date,
                             auto_adjust=True, progress=False)
            if df.empty:
                print("no data")
                return
            df = df.reset_index()
            # yfinance MultiIndex columns → flatten
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if col[1] == "" else col[0] for col in df.columns]
            df.to_parquet(out_dir / f"{ticker}.parquet")
            print(f"saved ({len(df)} rows, latest: {df['Date'].max().date()})")
        except Exception as e:
            print(f"failed: {e}")

    for asset in references:
        _download(asset["ticker"], ref_dir)

    for asset in assets:
        _download(asset["ticker"], output_dir)

    print("Download finished.")


if __name__ == "__main__":
    run()
