"""
FinWatch AI — Drawdown Probability Model
=========================================

Question: "What is the probability of a drawdown > 5% in the next 20 days?"

Why this is good:
  - Binary classification (clean labels, ~36% positive rate)
  - Directly answers a useful investment question
  - Measurable, backtestable, interpretable
  - Does NOT try to predict direction (near-impossible with OHLCV)

Output:
  p_drawdown  : float 0-1   — probability of >5% drawdown in 20 days
  drawdown_risk: "high" / "low"  (threshold: p_drawdown >= 0.45)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

ROOT      = Path(__file__).resolve().parents[3]
DATA_DIR  = ROOT / "data/detection"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgboost_drawdown.pkl"

HORIZON          = 20     # days forward
THRESHOLD        = 0.05   # 5% drawdown = event
TRAIN_DATA_END   = pd.Timestamp("2026-03-01")  # training + test data cutoff

FEATURES = [
    # Anomaly signals — weighted score is more informative than raw integer
    'anomaly_score', 'anomaly_score_weighted',
    'z_score', 'z_score_60',
    'ae_error', 'ae_anomaly', 'if_anomaly',
    # Returns & volatility
    'returns', 'volatility', 'vol_5', 'vol_20', 'vol_change', 'volatility_ratio',
    # Momentum
    'momentum_5', 'momentum_10', 'trend_strength', 'rsi',
    'return_lag_1', 'return_lag_2', 'return_lag_3',
    # Drawdown history
    'max_drawdown_30d',
    # Volume — rising/falling volume confirms price moves
    'volume_zscore', 'is_high_volume', 'volume_trend',
    # OBV divergence — price up but OBV down = bearish signal
    'obv_signal',
    # Market & sector context
    'spx_return', 'etf_return', 'relative_return', 'excess_return',
    'sector_relative',
    'is_market_wide', 'is_sector_wide',
    'beta', 'regime_encoded',
    # Stock-specific MA position (above/below own 200-day MA — key structural signal)
    'price_vs_ma200_stock', 'price_vs_ma50_stock',
    # Rolling stats
    'rolling_mean', 'rolling_std', 'rolling_std_60',
    # Tail risk
    'var_95', 'es_95', 'es_ratio',
    # VIX context
    'vix_level', 'vix_change', 'vix_high',
]


def _add_vix(data: pd.DataFrame) -> pd.DataFrame:
    try:
        from pandas_datareader import data as web
        vix = web.DataReader("VIXCLS", "fred", "2015-01-01", "2030-12-31").reset_index()
        vix.columns = ["Date", "vix_level"]
        vix["Date"] = pd.to_datetime(vix["Date"])
    except Exception:
        vix = pd.DataFrame({
            "Date": pd.date_range("2015-01-01", "2030-12-31"),
            "vix_level": 20.0,
        })
    vix["vix_change"] = vix["vix_level"].pct_change().fillna(0)
    vix["vix_high"]   = (vix["vix_level"] > 25).astype(int)
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    data = data.merge(vix[["Date", "vix_level", "vix_change", "vix_high"]],
                      on="Date", how="left")
    data["vix_level"]  = data["vix_level"].ffill().fillna(20)
    data["vix_change"] = data["vix_change"].ffill().fillna(0)
    data["vix_high"]   = data["vix_high"].ffill().fillna(0)
    return data


def _add_stock_ma_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add each stock's own 50-day and 200-day MA position.
    The existing ma50/ma200 columns are SPX MAs (market context), not stock MAs.
    price_vs_ma200_stock = (close / own_MA200) - 1
    """
    def _per_ticker(df):
        df = df.copy().sort_values("Date")
        close = df["Close"]
        ma50  = close.rolling(50,  min_periods=20).mean()
        ma200 = close.rolling(200, min_periods=50).mean()
        df["price_vs_ma50_stock"]  = (close / ma50  - 1).where(ma50  > 0, 0.0)
        df["price_vs_ma200_stock"] = (close / ma200 - 1).where(ma200 > 0, 0.0)
        return df
    return data.groupby("ticker", group_keys=False).apply(_per_ticker)


def load_data() -> pd.DataFrame:
    dfs = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        if f.stem.startswith("^"):
            continue
        df = pd.read_parquet(f)
        df["ticker"] = f.stem
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["ticker", "Date"]).reset_index(drop=True)
    data = _add_vix(data)
    data = _add_stock_ma_features(data)
    return data


def generate_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Label = 1 if max drawdown in next HORIZON days exceeds THRESHOLD."""
    def _label_ticker(df):
        df = df.copy().sort_values("Date")
        closes = df["Close"].values
        labels = []
        for i in range(len(closes)):
            if i + HORIZON < len(closes):
                window = closes[i + 1: i + HORIZON + 1]
                dd = (window.min() - closes[i]) / closes[i]
                labels.append(1 if dd <= -THRESHOLD else 0)
            else:
                labels.append(np.nan)
        df["drawdown_event"] = labels
        return df

    data = data.groupby("ticker", group_keys=False).apply(_label_ticker)
    return data.dropna(subset=["drawdown_event"])


def _prep_X(df: pd.DataFrame) -> pd.DataFrame:
    avail = [f for f in FEATURES if f in df.columns]
    return df[avail].apply(
        lambda c: pd.to_numeric(c, errors="coerce")
    ).fillna(0)


def train(data: pd.DataFrame) -> XGBClassifier:
    train_mask = data["Date"] < pd.Timestamp("2024-01-01")
    test_mask  = (data["Date"] >= pd.Timestamp("2024-01-01")) & (data["Date"] < TRAIN_DATA_END)
    X_all = _prep_X(data)
    y_all = data["drawdown_event"].astype(int)

    X_train, y_train = X_all[train_mask], y_all[train_mask]

    # 10% validation split for early stopping
    split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    pos_rate = y_tr.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate   # handle class imbalance

    model = XGBClassifier(
        n_estimators=1200,
        max_depth=5,
        learning_rate=0.015,
        subsample=0.75,
        colsample_bytree=0.65,
        colsample_bylevel=0.8,
        min_child_weight=15,
        gamma=0.2,
        reg_alpha=0.3,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)

    joblib.dump(model, MODEL_PATH)

    # Evaluate on test set (2024-01-01 → 2026-03-01)
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]

    probs  = model.predict_proba(X_test)[:, 1]
    preds  = (probs >= 0.45).astype(int)
    auc    = roc_auc_score(y_test, probs)

    print(f"  Best round: {model.best_iteration}  |  Test AUC: {auc:.4f}")
    print()
    print(classification_report(y_test, preds,
                                 target_names=["no_drawdown", "drawdown>5%"],
                                 digits=2))
    print(f"  → Model saved: {MODEL_PATH}")
    return model


def predict(data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict drawdown probability for the latest row of each ticker.
    Returns DataFrame with: ticker, p_drawdown, drawdown_risk
    """
    model  = joblib.load(MODEL_PATH)
    latest = data.groupby("ticker").last().reset_index()
    X      = _prep_X(latest)
    # Align columns to training feature order
    avail  = [f for f in FEATURES if f in X.columns]
    probs  = model.predict_proba(X[avail])[:, 1]

    return pd.DataFrame({
        "ticker":        latest["ticker"].values,
        "p_drawdown":    probs.round(4),
        "drawdown_risk": ["high" if p >= 0.45 else "low" for p in probs],
    }).sort_values("p_drawdown", ascending=False).reset_index(drop=True)


def run():
    print("Loading data...")
    data = load_data()
    print("Generating labels (>5% drawdown in 20 days)...")
    data = generate_labels(data)
    dist = data["drawdown_event"].value_counts(normalize=True)
    print(f"  Label distribution: drawdown={dist[1]:.1%}  no_drawdown={dist[0]:.1%}")
    print("Training...")
    train(data)
    print("\nPredictions (latest):")
    results = predict(data)
    print(results.head(15).to_string(index=False))


if __name__ == "__main__":
    run()
