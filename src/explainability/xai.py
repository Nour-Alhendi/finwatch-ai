"""
FinWatch AI — Layer 7A: XAI (SHAP)
====================================
Loads XGBoost Drawdown model, computes SHAP values for the latest
row per ticker, and returns (latest_df, shap_matrix).

shap_matrix shape: (n_tickers, n_features)
Values: SHAP for the drawdown class (positive = pushes toward drawdown risk)
  positive → feature increases drawdown probability
  negative → feature reduces drawdown probability
"""

import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"


def compute(data: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Returns (latest_df, shap_matrix).

    latest_df   — one row per ticker (most recent)
    shap_matrix — shape (n_tickers, n_features), SHAP for drawdown probability
    """
    model = joblib.load(MODEL_DIR / "xgboost_drawdown.pkl")

    latest = data.groupby("ticker").last().reset_index()

    X = latest[features].apply(lambda c: pd.to_numeric(c, errors="coerce")).fillna(0)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Binary XGBoost: shap_values is 2D (n_samples, n_features)
    # For 3D output (older XGBoost), take index 1 (positive/drawdown class)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    return latest, sv


# Features that describe market/regime context — used as background, not as the primary driver.
# Excluding these from driver selection forces the narrative to show the stock-specific signal.
CONTEXT_FEATURES = {"regime_encoded"}


def top3(shap_row: np.ndarray, features: list[str]) -> list[tuple[str, float]]:
    """
    Returns top-3 features by |SHAP| as [(feature_name, shap_val), ...].
    The first element is always the stock-specific driver (regime_encoded excluded).
    Elements 2-3 are the next highest by |SHAP| regardless of feature type.
    """
    all_pairs = sorted(
        zip(features, shap_row),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    # Driver: highest |SHAP| excluding context features
    driver = next(
        (p for p in all_pairs if p[0] not in CONTEXT_FEATURES),
        all_pairs[0],  # fallback: use top feature if all are context features
    )

    # Top-3 for display: driver first, then next 2 by |SHAP| (may include regime)
    rest = [p for p in all_pairs if p[0] != driver[0]][:2]
    return [driver] + rest


def format_top3(top3_list: list[tuple[str, float]]) -> str:
    parts = []
    for name, val in top3_list:
        sign = "+" if val >= 0 else ""
        parts.append(f"{name}({sign}{val:.3f})")
    return ", ".join(parts)
