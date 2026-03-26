# FinWatch AI — System Architecture

**AI-Driven Financial Anomaly Detection and Monitoring System**

---

## Overview

An 8-layer modular pipeline that monitors 55 stocks and 10 sector ETFs, detects anomalies, classifies risk, and explains every decision in plain English.

---

## Layer 1 — Data Ingestion

- **Source:** Stooq API (historical, 10 years, daily)
- **Universe:** 55 stocks (Tech, Finance, Health, Energy, Industrials, Green Energy) + 10 reference ETFs (SPX, XLK, XLF, XLV, XLP, XLE, XLY, XLI, BOTZ, ICLN)
- **Format:** OHLCV (Open, High, Low, Close, Volume)
- **Output:** `data/raw/raw_clean/` — one parquet file per ticker

---

## Layer 2 — Data Quality Checks

| Check | What it catches |
|-------|----------------|
| Schema validation | Wrong column types or missing columns |
| Missing values | NaN, -999, zero-price rows |
| Duplicates | Exact copies or date conflicts |
| OHLC violations | High < Low, Open > High, Close < Low |
| Gap detection | Missing trading days |

- **Output:** Validated parquet files

---

## Layer 3 — Feature Engineering

### 3A · Basic Features
- `returns`, `volatility`
- `rolling_mean`, `rolling_std` (20-day window)
- `beta`, `corr_spx`
- `rsi`
- `max_drawdown_30d` — rolling 30-day peak-to-trough, no lookahead

### 3B · Context Features
- `spx_return`, `etf_return`
- `is_market_wide`, `is_sector_wide` — pure market signal, no anomaly label
- `excess_return`, `relative_return`, `sector_relative`
- `regime` (bull / bear / transition), `ma50`, `ma200`, `regime_encoded`
- `vol_regime`, `volume_ma20`, `volume_zscore`, `is_high_volume`

### 3C · Advanced Features
- `return_lag_1/2/3`, `momentum_5/10`
- `vol_5`, `vol_20`, `vol_change`
- `trend_strength`
- `volatility_ratio` (vs SPX)
- `volume_trend`

---

## Layer 4 — Anomaly Detection

### Detection Methods

| Method | Details |
|--------|---------|
| **Statistical** | Z-Score ±3σ, 20-day + 60-day window |
| **ML** | Isolation Forest — group-aware, percentile threshold per group |
| **Deep Learning** | Dual LSTM Autoencoder — `low_vol_model` (calm periods) + `high_vol_model` (volatile periods), regime-specific thresholds from train data only (no lookahead) |

### Combined Score

- `anomaly_score` (0–4): z + z_60 + isolation_forest + autoencoder
- `combined_anomaly` (bool)
- `market_anomaly` = `is_market_wide` & `combined_anomaly`
- `sector_anomaly` = `is_sector_wide` & `combined_anomaly`

### Severity
`normal` → `watch` → `warning` → `critical`

- **Output:** `data/detection/` — one parquet per ticker

---

## Layer 5 — Risk & Direction Prediction

### 5A · Prediction Features (`src/prediction/features/`)

**Expected Shortfall** (`expected_shortfall.py`)
- Rolling 252-day VaR (95%) + ES per ticker
- ES captures tail risk independently of volatility — same volatility can mean very different danger levels
- Columns added: `var_95`, `es_95`, `es_ratio` (ES / VaR)

**OBV Signal** (`obv_signal.py`)
- `obv_signal = volume_zscore × returns`
- Positive → high volume + positive return = institutional buying
- Negative → high volume + negative return = panic selling / distribution
- Used in: Decision Layer (sanity check) + Narrative Engine (confirmation)

### 5B · XGBoost Risk Classifier (`src/prediction/models/xgboost_risk.py`)

- **Task:** Binary classification — `high` risk vs `low` risk
- **Features:** 29 (anomaly signals, price/returns, momentum, context, volume, ES)
- **Labels:** Forward-looking 5-day max drawdown vs 85th percentile per ticker (no lookahead)
- **Class imbalance:** `scale_pos_weight`
- **Output:** `risk_level`, `p_high`, `p_low` per ticker

### 5C · XGBoost Direction Classifier (`src/prediction/models/xgboost_direction.py`)

- **Task:** Multiclass — `up` / `stable` / `down`
- **Features:** Same 29 features as 5B
- **Labels:** Forward-looking 5-day return vs 75th / 25th percentile per ticker (no lookahead)
- **Class imbalance:** Balanced sample weights
- **Output:** `direction`, `p_up`, `p_stable`, `p_down` per ticker

- **Saved models:** `models/xgboost_risk.pkl`, `models/xgboost_direction.pkl`

---

## Layer 6 — Guardrails + Decision Engine

**Input:** `risk_level` × `direction` × `anomaly_score` × `es_ratio` × `max_drawdown_30d`

### Priority Order

| Priority | Condition | Outcome |
|----------|-----------|---------|
| 1 | `max_drawdown_30d ≤ −5%` | **CRITICAL** (hard override) |
| 2 | `anomaly_score = 0` + `risk = high` | **REVIEW** (models contradict each other) |
| 3 | `risk = high` + `es_ratio ≥ 1.5` | **WARNING** (tail risk override) |
| 4a | `risk = high` + `p_high ≥ 0.60` + `down` (confirmed) | **CRITICAL** |
| 4b | `risk = high` + `p_high ≥ 0.60` + `down` (weak) | **WARNING** |
| 4c | `risk = high` + `p_high ≥ 0.60` + `stable` | **WARNING** |
| 4d | `risk = high` + `p_high ≥ 0.60` + `up` | **WATCH** *(Dead Cat Bounce flag)* |
| 5 | `risk = high` + `p_high < 0.60` | **WATCH** (low confidence catch) |
| 6a | `risk = low` + `up` + rising momentum | **POSITIVE_MOMENTUM** |
| 6b | `risk = low` + `down` (confirmed) | **WATCH** |
| 7 | Default | **NORMAL** |

### Outputs

- **Severity:** `CRITICAL` / `WARNING` / `WATCH` / `NORMAL` / `POSITIVE_MOMENTUM` / `REVIEW`
- **Action:** `ESCALATE` / `MONITOR` / `OBSERVE` / `NONE` / `FLAG`
- **Confidence:** derived from `anomaly_score`
- **Context:** `market_wide` / `sector_wide` / `idiosyncratic`
- **Output:** `data/decisions/decisions.parquet`

---

## Layer 7 — XAI + Narrative Engine + Reporting

### 7A · SHAP Explainability (`src/explainability/xai.py`)
- TreeExplainer on XGBoost Risk model
- Top-3 SHAP drivers per ticker (context features excluded to show stock-specific signals)
- **Output:** `data/explanations/explanations.parquet`

### 7B · Narrative Engine (`src/explainability/narrative_engine.py`)

**Inputs:** Decision (Layer 6) + SHAP Top-3 + OBV Signal

| Pattern | Condition |
|---------|-----------|
| `signals_aligned_bearish` | CRITICAL + OBV negative + driver pushes risk up |
| `signals_aligned_bullish` | NORMAL/WATCH + OBV positive + driver pulls risk down |
| `conflict_dead_cat_bounce` | CRITICAL/WARNING + OBV strongly positive |
| `blind_spot_review` | NORMAL/WATCH + OBV strongly negative |
| `mixed` | No dominant pattern |

### 7C · LLM Narrator (`src/explainability/llm_narrator.py`)
- Input: structured Narrative Engine output
- Output: plain-English summary per ticker (grounded — no hallucination possible)

### 7D · FinBERT (`src/explainability/finbert.py`)
- Financial sentiment analysis on news headlines
- Output: `positive` / `neutral` / `negative` sentiment score per ticker

### 7E · Report (`src/explainability/report.py`)
- Per ticker: Severity + SHAP Top-3 + OBV + LLM Summary + News Sentiment
- **Output:** `data/explanations/llm_news_enriched.parquet`

---

## Layer 8 — Logging & Audit (`src/reporting/`)

### 8A · Anomaly Log (`anomaly_log.py`)
- Appends every pipeline run with a UUID + timestamp
- Columns: `run_id`, `timestamp`, `ticker`, `date`, `severity`, `action`, `confidence`, `context`
- **Output:** `data/logs/anomaly_log.parquet`

### 8B · Daily Report (`daily_report.py`)
- Management-level summary per run
- Sections: severity breakdown, ESCALATE tier, MONITOR tier, POSITIVE_MOMENTUM tier
- **Output:** `data/reports/daily_summary.txt`

---

## Stress Testing (`src/stress_testing/`)

11 data corruption scenarios injected at controlled rates to validate the quality pipeline:

| Scenario | Rate |
|----------|------|
| Missing values (NaN / -999) | 2% |
| Price spikes (×10, ×0.01, +1000) | 1% |
| Zero values (clustered blocks) | 1% |
| Duplicate rows | 1% |
| Wrong dates (±30d, ±365d, weekends) | 0.5% |
| Stale prices (frozen feed, 3–5 rows) | 1% |
| OHLC violations (High < Low, etc.) | 0.5% |
| Zero volume | 1% |
| Extreme overnight gaps (−80–90%) | 0.3% |
| Negative volume | 0.5% |
| Timestamp conflicts (same date, diff Close) | 0.5% |

- **Input:** `data/raw/raw_clean/`
- **Output:** `data/raw/raw_corrupted/` + `data_quality_alert` column
