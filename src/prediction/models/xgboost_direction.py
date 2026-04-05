import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, accuracy_score

ROOT      = Path(__file__).resolve().parents[3]
DATA_DIR  = ROOT / "data/detection"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HORIZON        = 10     # 10-day forward return — best balance of signal vs noise
TRAIN_DATA_END = pd.Timestamp("2026-03-01")  # training + test data cutoff
UP_THRESHOLD   = 0.025  # +2.5% over 10 days = clearly bullish
DOWN_THRESHOLD = -0.025 # -2.5% over 10 days = clearly bearish

FEATURES = [
    'anomaly_score', 'z_score', 'z_score_60',
    'ae_error', 'ae_anomaly',
    'returns', 'volatility', 'vol_5', 'vol_20', 'vol_change',
    'rolling_mean', 'rolling_std', 'rolling_std_60',
    'momentum_5', 'momentum_10', 'momentum_20', 'trend_strength', 'rsi',
    'return_lag_1', 'return_lag_2', 'return_lag_3',
    'volume_zscore', 'is_high_volume', 'volume_trend',
    'spx_return', 'etf_return', 'relative_return', 'sector_relative',
    'is_market_wide', 'is_sector_wide', 'volatility_ratio',
    'regime_encoded', 'beta', 'spx_volatility',
    'var_95', 'es_95', 'es_ratio',
    # Directional features
    'dist_to_52w_high', 'dist_to_52w_low', 'macd_signal',
    # MA position — key for 20-day prediction
    'price_vs_ma50', 'price_vs_ma200', 'ma50_vs_ma200',
    # Market context
    'vix_level', 'vix_change', 'vix_high',
]


def _add_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker: 52w high/low, MACD, MA position, 20-day momentum."""
    df = df.copy().sort_values("Date")
    closes = pd.Series(df["Close"].values)

    # 52-week high/low distance
    rolling_high = closes.rolling(252, min_periods=20).max().values
    rolling_low  = closes.rolling(252, min_periods=20).min().values
    df["dist_to_52w_high"] = (closes.values / rolling_high) - 1
    df["dist_to_52w_low"]  = (closes.values / rolling_low)  - 1

    # MACD signal line
    ema12  = closes.ewm(span=12).mean()
    ema26  = closes.ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["macd_signal"] = (macd - signal).values

    # 20-day momentum
    df["momentum_20"] = closes.pct_change(20).values

    # Price vs MA50 / MA200 (uses pre-computed ma50/ma200 from detection layer)
    if "ma50" in df.columns and "ma200" in df.columns:
        ma50  = df["ma50"].values
        ma200 = df["ma200"].values
        df["price_vs_ma50"]  = np.where(ma50  > 0, closes.values / ma50  - 1, 0.0)
        df["price_vs_ma200"] = np.where(ma200 > 0, closes.values / ma200 - 1, 0.0)
        df["ma50_vs_ma200"]  = np.where(ma200 > 0, ma50 / ma200 - 1, 0.0)
    else:
        df["price_vs_ma50"]  = 0.0
        df["price_vs_ma200"] = 0.0
        df["ma50_vs_ma200"]  = 0.0

    # Ensure volume_trend and sector_relative exist
    if "volume_trend" not in df.columns:
        df["volume_trend"] = 0.0
    if "sector_relative" not in df.columns:
        df["sector_relative"] = 0.0
    if "spx_volatility" not in df.columns:
        df["spx_volatility"] = 0.0

    return df


def _add_vix_features(data: pd.DataFrame) -> pd.DataFrame:
    """Fetch VIX from FRED and merge onto data. Falls back to neutral value 20."""
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

    # Add directional features per ticker
    data = data.groupby("ticker", group_keys=False).apply(_add_direction_features)
    # Add VIX market-context features
    data = _add_vix_features(data)
    return data


def generate_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker percentile labels on 10-day forward return.
    Thresholds (65th/35th percentile) computed from training data only — no lookahead.
    Gives ~35% up / ~30% stable / ~35% down per ticker regardless of volatility.
    """
    def _future_return(df):
        df = df.copy().sort_values("Date")
        closes = df["Close"].values
        future_returns = []
        for i in range(len(closes)):
            if i + HORIZON < len(closes):
                ret = (closes[i + HORIZON] - closes[i]) / closes[i]
            else:
                ret = np.nan
            future_returns.append(ret)
        df["future_return"] = future_returns
        return df

    data = data.groupby("ticker", group_keys=False).apply(_future_return)
    data = data.dropna(subset=["future_return"])

    # Compute thresholds from training period only (no lookahead bias)
    train_mask = data["Date"] < pd.Timestamp("2024-01-01")
    up_thresh   = data.loc[train_mask].groupby("ticker")["future_return"].quantile(0.65)
    down_thresh = data.loc[train_mask].groupby("ticker")["future_return"].quantile(0.35)

    data["_up"]   = data["ticker"].map(up_thresh)
    data["_down"] = data["ticker"].map(down_thresh)

    def _label(row):
        if row["future_return"] >= row["_up"]:
            return "up"
        elif row["future_return"] <= row["_down"]:
            return "down"
        return "stable"

    data["direction"] = data.apply(_label, axis=1)
    data = data.drop(columns=["_up", "_down"])
    return data


def train(data: pd.DataFrame):
    """Train XGBoost direction model with early stopping + regularization."""
    X = data[FEATURES].apply(lambda col: pd.to_numeric(col, errors="coerce")).fillna(0)
    y = data["direction"].astype(str)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    train_mask = data["Date"] < pd.Timestamp("2024-01-01")
    X_train, y_train = X[train_mask], y_enc[train_mask]

    # Use last 10% of training data as validation for early stopping
    val_split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    model = XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.75,
        colsample_bytree=0.7,
        min_child_weight=10,
        gamma=0.2,
        reg_alpha=0.3,
        reg_lambda=3.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
    )
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=0,
    )
    print(f"  Best round: {model.best_iteration}  |  Best logloss: {model.best_score:.5f}")

    # ── Holdout evaluation (2024+, never seen during training)
    holdout_mask = data["Date"] >= pd.Timestamp("2024-01-01")
    X_hold = X[holdout_mask]
    y_hold = y_enc[holdout_mask]
    if len(X_hold) > 0:
        y_pred = model.predict(X_hold)
        acc    = accuracy_score(y_hold, y_pred)
        print(f"\n  ── Holdout Evaluation (2024+) ──")
        print(f"  Accuracy: {acc:.3f}  (baseline random = 0.333)")
        print(classification_report(y_hold, y_pred, target_names=le.classes_, digits=3))

    joblib.dump(model, MODEL_DIR / "xgboost_direction.pkl")
    joblib.dump(le,    MODEL_DIR / "direction_label_encoder.pkl")
    print("  Model saved.")
    return model, le


def predict(data: pd.DataFrame) -> pd.DataFrame:
    """Predict direction for the latest row of each ticker."""
    model = joblib.load(MODEL_DIR / "xgboost_direction.pkl")
    le    = joblib.load(MODEL_DIR / "direction_label_encoder.pkl")

    latest = data.groupby("ticker").last().reset_index()
    X_latest = latest[FEATURES].apply(lambda col: pd.to_numeric(col, errors="coerce")).fillna(0)

    probs   = model.predict_proba(X_latest)
    classes = list(le.classes_)
    p_up    = probs[:, classes.index("up")]
    p_stable = probs[:, classes.index("stable")]
    p_down  = probs[:, classes.index("down")]

    def _classify(u, s, d):
        if u >= 0.38:
            return "up"
        elif d >= 0.38:
            return "down"
        return "stable"

    pred_labels = np.array([_classify(u, s, d) for u, s, d in zip(p_up, p_stable, p_down)])

    return pd.DataFrame({
        "ticker":    latest["ticker"].values,
        "direction": pred_labels,
        "p_up":      p_up,
        "p_stable":  p_stable,
        "p_down":    p_down,
    }).sort_values("p_down", ascending=False)


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
