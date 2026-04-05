"""
FinWatch AI — Meta-Model (Layer 6.5)
======================================
A second-level XGBoost trained on backtest results.

Idea (stacking):
  Layer 1: Drawdown Probability Model  →  p_drawdown
  Layer 2: Anomaly Detection           →  anomaly_score_weighted
  Layer 3: Technical signals           →  rsi, momentum, es_ratio, ...
  Layer 4: Decision Engine             →  severity, caution_flag, ...
  ──────────────────────────────────────────────────────────────────
  Meta-Model: learns WHICH combinations of the above actually led
              to a real drawdown in 12 walk-forward windows.

Output:
  p_drawdown_meta : float 0-1   — refined drawdown probability
  meta_confidence : float 0-1   — precision estimate from the meta-model

Model is saved to: models/meta_model.pkl
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT      = Path(__file__).resolve().parents[3]
DATA_DIR  = ROOT / "data/detection"
BT_DIR    = ROOT / "data/backtesting"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "meta_model.pkl"

# Features available from DecisionOutput (always present)
BASE_FEATURES = [
    "p_drawdown",
    "anomaly_score_weighted",
    "anomaly_score",
]

# Features from detection join (richer, but require the join)
DETECTION_FEATURES = [
    "rsi",
    "momentum_5",
    "momentum_10",
    "es_ratio",
    "volatility",
    "excess_return",
    "max_drawdown_30d",
    "volume_zscore",
    "vol_change",
    "trend_strength",
    "z_score",
    "z_score_60",
    "ae_error",
    "vix_level",
    "vix_change",
    "vix_high",
]

# Encoded from DecisionOutput fields
ENCODED_FEATURES = [
    "drawdown_risk_enc",    # 1 = high, 0 = low
    "caution_flag_enc",     # 1 = True, 0 = False
    "momentum_signal_enc",  # ordinal: negative=-1, neutral=0, pullback=0.5, bounce=0.5, positive=1
]

ALL_FEATURES = BASE_FEATURES + DETECTION_FEATURES + ENCODED_FEATURES


def _load_detection_lookup() -> pd.DataFrame:
    """Load all detection data into a ticker+date lookup for joining."""
    dfs = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        if f.stem.startswith("^"):
            continue
        df = pd.read_parquet(f)
        df["ticker"] = f.stem
        df["Date"] = pd.to_datetime(df["Date"])
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data.sort_values(["ticker", "Date"])


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
    vix["vix_change"] = vix["vix_level"].pct_change(fill_method=None).fillna(0)
    vix["vix_high"]   = (vix["vix_level"] > 25).astype(int)
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    data = data.merge(vix[["Date", "vix_level", "vix_change", "vix_high"]], on="Date", how="left")
    data["vix_level"]  = data["vix_level"].ffill().fillna(20)
    data["vix_change"] = data["vix_change"].ffill().fillna(0)
    data["vix_high"]   = data["vix_high"].ffill().fillna(0)
    return data


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["drawdown_risk_enc"] = (df["drawdown_risk"] == "high").astype(int)
    df["caution_flag_enc"]  = df["caution_flag"].astype(int)
    mom_map = {"negative": -1.0, "neutral": 0.0,
               "pullback": 0.5,  "bounce": 0.5, "positive": 1.0}
    df["momentum_signal_enc"] = df["momentum_signal"].map(mom_map).fillna(0.0)
    return df


def load_training_data() -> pd.DataFrame:
    """
    Load backtest results and enrich with raw detection features via join.
    Returns DataFrame ready for training.
    """
    bt = pd.read_parquet(BT_DIR / "backtest_results.parquet")
    bt["date"] = pd.to_datetime(bt["date"])

    # Load detection lookup
    print("  Loading detection data for feature join...")
    det = _load_detection_lookup()
    det = _add_vix(det)

    bt_cols  = set(bt.columns)
    det_join = [c for c in DETECTION_FEATURES if c in det.columns and c not in bt_cols]
    keep     = ["ticker", "Date"] + det_join
    det_slim = det[keep].rename(columns={"Date": "date"})

    merged = bt.merge(det_slim, on=["ticker", "date"], how="left")
    merged = _encode_features(merged)

    print(f"  Merged: {len(merged)} rows, {merged['actual_drawdown_event'].mean():.1%} drawdown rate")
    return merged


def _prep_X(df: pd.DataFrame) -> pd.DataFrame:
    avail = [f for f in ALL_FEATURES if f in df.columns]
    return df[avail].fillna(0).astype(float)


def train() -> XGBClassifier:
    print("Loading training data...")
    data = load_training_data()

    X = _prep_X(data)
    y = data["actual_drawdown_event"].astype(int)

    print(f"  Features used: {X.columns.tolist()}")
    print(f"  Samples: {len(X)}  |  Positive rate: {y.mean():.1%}")

    pos_rate = y.mean()
    spw = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,           # shallow — small dataset, avoid overfitting
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=0.3,
        reg_lambda=2.0,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validate first to check for overfitting
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"\n  5-fold CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    # Full fit
    model.fit(X, y, verbose=0)
    train_probs = model.predict_proba(X)[:, 1]
    train_auc   = roc_auc_score(y, train_probs)
    print(f"  Train AUC: {train_auc:.4f}  (CV is the real score)")

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=X.columns)
    print("\n  Top features:")
    print(imp.sort_values(ascending=False).head(10).to_string())

    joblib.dump({"model": model, "features": X.columns.tolist()}, MODEL_PATH)
    print(f"\n  Saved: {MODEL_PATH}")
    return model


def predict_single(record: dict, det_row=None) -> dict:
    """
    Predict refined drawdown probability for a single record.

    Args:
        record   : dict from Decision Engine (DecisionOutput fields)
        det_row  : optional dict with raw detection features (rsi, momentum_5, etc.)

    Returns:
        {"p_drawdown_meta": float, "meta_confidence": float}
    """
    if not MODEL_PATH.exists():
        return {"p_drawdown_meta": record.get("p_drawdown", 0.35), "meta_confidence": 0.0}

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]

    row = {}
    row.update(record)
    if det_row:
        row.update(det_row)

    # Encode categorical fields
    row["drawdown_risk_enc"] = 1 if row.get("drawdown_risk") == "high" else 0
    row["caution_flag_enc"]  = int(bool(row.get("caution_flag", False)))
    mom_map = {"negative": -1.0, "neutral": 0.0,
               "pullback": 0.5,  "bounce": 0.5, "positive": 1.0}
    row["momentum_signal_enc"] = mom_map.get(row.get("momentum_signal", "neutral"), 0.0)

    X = pd.DataFrame([{f: row.get(f, 0.0) for f in features}])
    prob = float(model.predict_proba(X)[:, 1][0])
    return {"p_drawdown_meta": round(prob, 4), "meta_confidence": round(prob, 2)}


def predict_batch(decisions_df: pd.DataFrame, detection_df=None) -> pd.DataFrame:
    """
    Add p_drawdown_meta and meta_confidence columns to a decisions DataFrame.
    decisions_df: output from decision pipeline (decisions.parquet)
    detection_df: latest detection signals (optional, for richer features)
    """
    if not MODEL_PATH.exists():
        decisions_df["p_drawdown_meta"]  = decisions_df["p_drawdown"]
        decisions_df["meta_confidence"]  = 0.0
        return decisions_df

    saved    = joblib.load(MODEL_PATH)
    model    = saved["model"]
    features = saved["features"]

    merged = decisions_df.copy()
    if detection_df is not None:
        # Only join detection features not already present in decisions_df
        existing = set(merged.columns)
        det_cols = [c for c in DETECTION_FEATURES if c in detection_df.columns and c not in existing]
        if det_cols:
            merged = merged.merge(detection_df[["ticker"] + det_cols], on="ticker", how="left")

    merged = _encode_features(merged)

    X = pd.DataFrame([{f: merged.iloc[i].get(f, 0.0) for f in features}
                      for i in range(len(merged))]).fillna(0).astype(float)

    probs = model.predict_proba(X)[:, 1]
    decisions_df = decisions_df.copy()
    decisions_df["p_drawdown_meta"] = probs.round(4)
    decisions_df["meta_confidence"] = probs.round(2)
    return decisions_df


if __name__ == "__main__":
    train()
