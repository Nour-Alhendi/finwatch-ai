import pandas as pd
import yaml
from pathlib import Path

with open ("config/assets.yaml", "r") as f:
    config = yaml.safe_load(f)
assets = config["assets"]
sector_etfs = config["sector_etfs"]

INPUT_DIR = Path("data/detection")
OUTPUT_DIR = Path("data/context")
REF_DIR = Path("data/raw/references")
TICKERS = [a["ticker"] for a in assets if a["sector"] != "Index"]

# Load S&P 500 daily returns from detection data (already processed in Layer 4)
def load_spx():
    df = pd.read_parquet(INPUT_DIR / "^SPX.parquet")
    df = df.set_index("Date")
    return df[["returns"]].rename(columns={"returns": "spx_return"})

# Load all sector ETFs from reference data (downloaded in Layer 1)
def load_etf_data():
    etf_data = {}
    for etf in set(sector_etfs.values()):
        df = pd.read_parquet(REF_DIR / f"{etf}.parquet")
        df = df.set_index("Date")
        df = df[["Close"]].rename(columns={"Close": f"{etf}_close"})
        etf_data[etf] = df
    return etf_data

# Load SPX volatility as market fear proxy (rolling std of S&P 500 returns)
def load_volatility():
    df = pd.read_parquet(INPUT_DIR / "^SPX.parquet")
    df = df.set_index("Date")
    return df[["volatility"]]

# Add market context to each stock: is the anomaly market-wide or stock-specific?
def compare(df, spx, volatility, etf_data, ticker):
    sector = next(a["sector"] for a in assets if a["ticker"] == ticker)
    etf_name = sector_etfs[sector]
    etf_df = etf_data[etf_name]
    df = df.set_index("Date")
    df = df.join(spx, how="left")
    df = df.join(volatility, how="left", rsuffix="_spx")
    df = df.join(etf_df, how="left")

    # Calculate ETF return for sector context
    etf_col = f"{etf_name}_close"
    df["etf_return"] = df[etf_col].pct_change()
    df["sector_volatility"] = df["etf_return"].rolling(window=20).std()


    # True if market also moved strongly that day (systemic, not stock-specific)
    df["is_market_wide"] = (df["spx_return"].abs() > 0.02) & df["combined_anomaly"]

    # True if sector also moved strongly that day (sector-wide, not stock-specific)
    df["is_sector_wide"] = (df["etf_return"].abs() > 0.025) & df["combined_anomaly"]

    # How much did the stock move beyond the market?
    df["excess_return"] = df["returns"] - df["spx_return"]

    # Volatility regime: low / moderate / high market turbulence
    df["volatility_regime"] = pd.cut(
        df["volatility_spx"],
        bins=[0, 0.005, 0.015, float("inf")],
        labels=["low", "moderate", "high"]
    )
    return df.reset_index()

# Main function: run index comparison for all stocks and save results
def run_index_comparison():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spx = load_spx()
    volatility = load_volatility()
    etf_data = load_etf_data()
    for ticker in TICKERS:
        file = INPUT_DIR / f"{ticker}.parquet"
        if not file.exists():
            print(f"Skipping {ticker}: file not found")
            continue
        df = pd.read_parquet(file)
        df = compare(df, spx, volatility, etf_data, ticker)
        df.to_parquet(OUTPUT_DIR / f"{ticker}.parquet")
        print(f"Saved: {ticker}.parquet")

if __name__ == "__main__":
    run_index_comparison()
