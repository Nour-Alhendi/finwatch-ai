import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

ROOT      = Path(__file__).resolve().parents[3]
DATA_DIR  = ROOT / "data/detection"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD      = 0.5
HORIZON        = 5
TRAIN_DATA_END = pd.Timestamp("2026-03-01")  # training + test data cutoff

FEATURES = [
    # Core Anomaly
    'anomaly_score', 'combined_anomaly', 'z_score', 'z_anomaly',
    'if_anomaly', 'ae_error', 'ae_anomaly',
    # Returns & Volatility
    'returns', 'volatility', 'vol_5', 'vol_20', 'vol_change',
    'rolling_mean', 'rolling_std',
    # Momentum & Trend
    'momentum_5', 'momentum_10', 'trend_strength', 'rsi',
    'return_lag_1', 'return_lag_2', 'return_lag_3',
    # Volume
    'volume_zscore', 'is_high_volume',
    # Market Context
    'spx_return', 'etf_return', 'excess_return', 'relative_return', 'sector_relative',
    'is_market_wide', 'is_sector_wide', 'volatility_ratio',
    # Regime
    'regime_encoded', 'beta',
    # Extended Z-Score + Risk
    'z_score_60', 'z_anomaly_60', 'var_95', 'es_95', 'es_ratio',
    # Directional / Market Fear
    'dist_to_52w_high', 'dist_to_52w_low', 'macd_signal',
    'vix_level', 'vix_change', 'vix_high',
]


def _add_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker: distance to 52w high/low and MACD signal line."""
    df = df.copy().sort_values("Date")
    closes = df["Close"].values
    rolling_high = pd.Series(closes).rolling(252, min_periods=20).max().values
    rolling_low  = pd.Series(closes).rolling(252, min_periods=20).min().values
    df["dist_to_52w_high"] = (closes / rolling_high) - 1
    df["dist_to_52w_low"]  = (closes / rolling_low)  - 1
    ema12  = pd.Series(closes).ewm(span=12).mean()
    ema26  = pd.Series(closes).ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["macd_signal"] = (macd - signal).values
    return df


def _add_vix_features(data: pd.DataFrame) -> pd.DataFrame:
    """Fetch VIX from FRED and merge. Falls back to neutral 20."""
    try:
        from pandas_datareader import data as web
        vix_raw = web.DataReader("VIXCLS", "fred", "2015-01-01", "2030-12-31")
        vix_raw = vix_raw.reset_index()
        vix_raw.columns = ["Date", "vix_level"]
        vix_raw["Date"] = pd.to_datetime(vix_raw["Date"])
    except Exception as e:
        print(f"  VIX download failed ({e}) — using fallback 20")
        vix_raw = pd.DataFrame({"Date": pd.date_range("2015-01-01", "2030-12-31"),
                                 "vix_level": 20.0})
    vix_raw["vix_change"] = vix_raw["vix_level"].pct_change().fillna(0)
    vix_raw["vix_high"]   = (vix_raw["vix_level"] > 25).astype(int)

    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    data = data.merge(
        vix_raw[["Date", "vix_level", "vix_change", "vix_high"]],
        on="Date", how="left"
    )
    data["vix_level"]  = data["vix_level"].ffill().fillna(20)
    data["vix_change"] = data["vix_change"].ffill().fillna(0)
    data["vix_high"]   = data["vix_high"].ffill().fillna(0)
    return data


def load_data() -> pd.DataFrame:
    all_dfs = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        if f.stem.startswith("^"):
            continue
        df = pd.read_parquet(f)
        df["ticker"] = f.stem
        all_dfs.append(df)
    data = pd.concat(all_dfs, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["ticker", "Date"]).reset_index(drop=True)
    data = data.groupby("ticker", group_keys=False).apply(_add_direction_features)
    data = _add_vix_features(data)
    return data


def generate_labels(data: pd.DataFrame) -> pd.DataFrame:
    """high/low risk labels via per-ticker 85th-percentile future drawdown."""
    def _drawdown(df):
        df = df.copy().sort_values("Date")
        closes = df["Close"].values
        future_drawdowns = []
        for i in range(len(closes)):
            window   = closes[i:i + HORIZON]
            drawdown = abs((window.min() - closes[i]) / closes[i])
            future_drawdowns.append(drawdown)
        df["future_max_drawdown"] = future_drawdowns
        return df

    data = data.groupby("ticker", group_keys=False).apply(_drawdown)
    data = data.dropna(subset=["future_max_drawdown"])

    # Threshold from training data only — applied to ALL data (no lookahead)
    # 80th percentile gives ~20% high / ~80% low — more signal than 85th
    train_mask    = data["Date"] < pd.Timestamp("2024-01-01")
    var_by_ticker = data.loc[train_mask].groupby("ticker")["future_max_drawdown"].quantile(0.80)

    data["_thresh"] = data["ticker"].map(var_by_ticker)
    data["risk_level"] = (data["future_max_drawdown"] >= data["_thresh"]).map(
        {True: "high", False: "low"}
    )
    data = data.drop(columns=["_thresh"])
    return data


def train(data: pd.DataFrame):
    """Train XGBoost risk model with early stopping + regularization."""
    X = data[FEATURES].apply(lambda col: pd.to_numeric(col, errors="coerce")).fillna(0)
    y = data["risk_level"].astype(str)

    # Force order: low=0, high=1 so XGBoost treats "high" as positive class
    le = LabelEncoder()
    le.fit(["low", "high"])
    y_enc = le.transform(y)

    train_mask = data["Date"] < pd.Timestamp("2024-01-01")
    X_train, y_train = X[train_mask], y_enc[train_mask]

    # Validation split from training data (last 10%)
    val_split        = int(len(X_train) * 0.9)
    X_tr, X_val      = X_train.iloc[:val_split], X_train.iloc[val_split:]
    y_tr, y_val      = y_train[:val_split], y_train[val_split:]

    # class 0 = "low" (majority ~80%), class 1 = "high" (minority ~20%)
    n_low  = (y_tr == 0).sum()
    n_high = (y_tr == 1).sum()
    spw = n_low / n_high  # ~4.0 — upweights "high" (the minority positive class)
    print(f"  Labels — low: {n_low}  high: {n_high}  scale_pos_weight: {spw:.2f}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        eval_metric="logloss",
        scale_pos_weight=spw,
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
    print(f"  Best round: {model.best_iteration}  |  Best logloss: {model.best_score:.5f}")

    # ── Holdout evaluation (2024+, never seen during training)
    holdout_mask = data["Date"] >= pd.Timestamp("2024-01-01")
    X_hold = X[holdout_mask]
    y_hold = y_enc[holdout_mask]
    if len(X_hold) > 0:
        y_pred  = model.predict(X_hold)
        y_proba = model.predict_proba(X_hold)[:, 1]
        print("\n  ── Holdout Evaluation (2024+) ──")
        print(classification_report(y_hold, y_pred, target_names=["low", "high"], digits=3))
        try:
            auc = roc_auc_score(y_hold, y_proba)
            print(f"  ROC-AUC: {auc:.4f}  (0.5=random, 0.7+=good, 0.8+=great)")
        except Exception:
            pass

    joblib.dump(model, MODEL_DIR / "xgboost_risk.pkl")
    joblib.dump(le,    MODEL_DIR / "risk_label_encoder.pkl")
    print("  Model saved.")
    return model, le


def predict(data: pd.DataFrame) -> pd.DataFrame:
    """Predict risk level for the latest row of each ticker."""
    model = joblib.load(MODEL_DIR / "xgboost_risk.pkl")
    le    = joblib.load(MODEL_DIR / "risk_label_encoder.pkl")

    latest = data.groupby("ticker").last().reset_index()
    X_latest = latest[FEATURES].apply(lambda col: pd.to_numeric(col, errors="coerce")).fillna(0)

    probs  = model.predict_proba(X_latest)
    p_high = probs[:, list(le.classes_).index("high")]
    pred_labels = np.where(p_high >= THRESHOLD, "high", "low")

    return pd.DataFrame({
        "ticker":        latest["ticker"].values,
        "risk_level":    pred_labels,
        "p_low":         probs[:, list(le.classes_).index("low")],
        "p_high":        p_high,
        "anomaly_score": latest["anomaly_score"].values,
    }).sort_values("p_high", ascending=False)


def run():
    print("Loading data...")
    data = load_data()
    data = data[data["Date"] < TRAIN_DATA_END].copy()

    print("Generating labels...")
    data = generate_labels(data)

    print("Training model...")
    train(data)

    print("Predicting...")
    results = predict(data)
    print(results.to_string(index=False))


if __name__ == "__main__":
    run()
