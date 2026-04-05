# FinWatch AI — Project State Document

> Last updated: 2026-03-29
> This file is the single source of truth for onboarding AI assistants.
> Show this file at the start of any new conversation to restore full context.

---

## What is this system?

**FinWatch AI** is a personal AI-powered stock monitoring and risk management system.

It does NOT predict stock prices. It detects anomalies and risk, then produces:
- Risk severity labels: `CRITICAL / WARNING / WATCH / NORMAL / POSITIVE_SIGNAL / REVIEW`
- Trading signals: `ENTRY / HOLD / EXIT / NEUTRAL`
- Human-readable explanations (via Groq LLM)

Run via: `python src/pipeline.py`
Dashboard via: `streamlit run finwatch/app.py`

---

## Architecture — 8-Layer Pipeline

```
Layer 0  — Data Ingestion          src/ingestion/download_historical.py
Layer 1  — Data Quality            src/quality/quality_pipeline.py
Layer 2  — Feature Engineering     src/features/feature_pipeline.py
Layer 3  — Anomaly Detection       src/detection/detection_pipeline.py
Layer 4  — Prediction Models       src/prediction/prediction_pipeline.py
Layer 5  — Fundamental Collectors  src/ingestion/{earnings,insider,options,valuation,sentiment}_collector.py
Layer 6  — Decision Engine         src/decision/decision_engine.py + decision_pipeline.py
Layer 7  — Explainability          src/explainability/explainability_pipeline.py
Layer 8  — Reporting + Dashboard   src/reporting/ + finwatch/app.py
```

Entry point: `src/pipeline.py`

---

## Detection Layer (Layer 3) — The Reliable Core

Four complementary anomaly detectors, combined into a **weighted anomaly score (0–1)**:

| Model | File | Weight | What it detects |
|---|---|---|---|
| LSTM Autoencoder | `src/detection/lstm_autoencoder.py` | 0.30 | Sequence anomalies (30 models, one per sector/volatility bucket) |
| Isolation Forest | `src/detection/isolation_forest.py` | 0.30 | Multivariate outliers (14 models, one per sector) |
| Z-Score | `src/detection/statistical.py` | 0.20 | Return distribution outliers |
| Sector Z-Score | `src/detection/statistical.py` | 0.20 | Stock vs sector peers |

`anomaly_score_weighted` = weighted sum, 0–1 continuous.
`anomaly_score` = integer 0–4 (legacy, still used for backcompat).

Saved to: `data/detection/{ticker}.parquet`

---

## Prediction Layer (Layer 4)

### Drawdown Probability Model
- **File**: `src/prediction/models/drawdown_probability.py`
- **Type**: XGBoost binary classifier
- **Target**: P(max drawdown > 5% in next 20 days)
- **Performance**: AUC ~0.64 (validated on holdout 2024+)
- **Model file**: `models/xgboost_drawdown.pkl`

### Meta-Model (Stacking Layer)
- **File**: `src/prediction/models/meta_model.py`
- **Type**: Logistic Regression stacking
- **Purpose**: Combines p_drawdown + anomaly signals + VIX for a refined `p_drawdown_meta`
- **Model file**: `models/meta_model.pkl`
- **Trained on**: backtest results (walk-forward)

### XGBoost Direction Model (NOT USED IN PRODUCTION)
- **File**: `src/prediction/models/xgboost_direction.py`
- **Accuracy**: ~40% (barely above 33% random baseline for 3 classes)
- **Status**: DISABLED — not called from decision pipeline
- **Reason**: EMH makes direction prediction near-impossible from technical data alone
- **Model file**: `models/xgboost_direction.pkl` (kept for research only)

### XGBoost Risk Model (DEPRECATED)
- **File**: `src/prediction/models/xgboost_risk.py`
- **Status**: Replaced by drawdown probability model
- **Model file**: `models/xgboost_risk.pkl` (kept for backward compat)

---

## Fundamental Data Collectors (Layer 5)

All collectors save to `data/fundamental/` as parquet files.
They run **after** the decision pipeline (Step 7 in pipeline.py).
This means: **the first run has no fundamentals; subsequent runs use yesterday's data (1-day lag). This is known and accepted.**

| Collector | File | Output | What it collects |
|---|---|---|---|
| Earnings | `src/ingestion/earnings_collector.py` | `data/fundamental/earnings.parquet` | `days_to_next_earnings` |
| Insider | `src/ingestion/insider_collector.py` | `data/fundamental/insider.parquet` | `insider_sentiment` (-1 to +1) |
| Options | `src/ingestion/options_collector.py` | `data/fundamental/options.parquet` | `put_call_ratio`, `options_fear` |
| Valuation | `src/ingestion/valuation_collector.py` | `data/fundamental/valuation.parquet` | `pe_ratio`, `pe_forward`, `pb_ratio`, `revenue_growth` |
| Sentiment | `src/ingestion/sentiment_collector.py` | `data/fundamental/sentiment.parquet` | historical news for future training |

**Valuation collector** was added in the last session. Uses Finnhub `/stock/metric?symbol=X&metric=all`.
Tries multiple Finnhub field names: `peBasicExclExtraTTM`, `peTTM`, `pbAnnual`, `revenueGrowthTTMYoy`, `forwardPE`, etc.

---

## Decision Engine (Layer 6)

### Files
- `src/decision/decision_engine.py` — core logic (`decide()` function)
- `src/decision/decision_pipeline.py` — loads all data sources, calls engine, saves output

### AnomalyInput dataclass fields
```python
# Core
ticker, date, p_drawdown, drawdown_risk, anomaly_score, anomaly_score_weighted
market_anomaly, sector_anomaly
# Technical
rsi, momentum_5, momentum_10, drawdown, obv_signal, volatility, excess_return, es_ratio, vix_level
# Sentiment
vader_score, finbert_score, news_sentiment_score
# Fundamental
days_to_next_earnings, insider_sentiment, put_call_ratio, options_fear
# Valuation (added last session)
pe_ratio, pe_forward, pb_ratio, revenue_growth
```

### DecisionOutput dataclass fields
```python
ticker, date, severity, action, confidence, context
p_drawdown, anomaly_score, anomaly_score_weighted, drawdown_risk
momentum_signal, caution_flag, override_reason, summary, sentiment_note
trading_signal  # ENTRY / HOLD / EXIT / NEUTRAL  — added last session
```

### Severity logic (in order)
1. `CRITICAL` — p_drawdown ≥ p_crit AND anomaly_w ≥ 0.30
2. `CRITICAL` — actual 30d drawdown ≤ -15%
3. `WARNING` — p_drawdown ≥ p_warn OR anomaly_w ≥ 0.30
4. `WARNING` — actual drawdown ≤ -8%
5. `WATCH` — anomaly_w ≥ 0.20 OR p_drawdown ≥ p_watch
6. `POSITIVE_SIGNAL` — p_dd < 30% AND RSI < 70 AND momentum positive/pullback AND anomaly_w < 0.20
7. `NORMAL` — none of the above
8. `REVIEW` — conflicting: high anomaly but low p_drawdown and stock outperforming

VIX-aware thresholds: at low VIX (< 20), thresholds are raised to reduce false positives.

### Valuation logic (added last session)
- `pe_ratio < 0` → block ENTRY ("negative earnings"), downgrade POSITIVE_SIGNAL → NORMAL
- `pe_ratio > 50` → block ENTRY ("overvalued"), downgrade POSITIVE_SIGNAL → NORMAL
- `pe_ratio < 15 OR pb_ratio < 1.5` → strengthen POSITIVE_SIGNAL with "cheap" note
- `revenue_growth < -0.05` → add "revenue declining" to context
- `pe_forward > pe_ratio * 1.2` on POSITIVE_SIGNAL → add "earnings contraction expected"

### Trading Signal logic (added last session)
```python
CRITICAL / WARNING  → EXIT
WATCH               → HOLD
POSITIVE_SIGNAL     → ENTRY if (RSI < 45 AND pe_ratio < 35 AND revenue_growth >= -0.02)
                      else HOLD
NORMAL              → HOLD
REVIEW              → NEUTRAL
```

### Output
Saved to: `data/decisions/decisions.parquet`

---

## Backtesting

**File**: `src/backtesting/backtest.py`

Walk-forward setup:
- Train: 4-year rolling window
- Test: 6-month windows
- No lookahead bias

Outputs:
- `data/backtesting/backtest_results.parquet` — 494 rows, per-signal with actual outcomes
- `data/backtesting/signal_precision.parquet` — precision per severity → used as confidence base
- `data/backtesting/anomaly_precision.parquet` — precision per anomaly detector bucket
- `data/backtesting/summary.txt` — human-readable

**Backtest validation results** (derived from historical 494 decisions):

| Signal | n | Avg 20d Return | Drawdown Rate |
|---|---|---|---|
| ENTRY | 12 | +3.37% | 17% |
| HOLD | 50 | +2.94% | 26% |
| EXIT | 432 | +3.79% | 36% |

**Interpretation**: Risk ordering is correct. EXIT → more drawdowns. ENTRY → fewest drawdowns.
Caveat: n=12 for ENTRY is small. New valuation logic not yet reflected in backtest.

---

## Explainability Layer (Layer 7)

- `src/explainability/finbert.py` — fetches news from Finnhub, computes VADER + Groq sentiment
- `src/explainability/llm_narrator.py` — generates natural language narrative per ticker (Groq)
- `src/explainability/explainability_pipeline.py` — orchestrates

Note: FinBERT was replaced with VADER + Groq in a previous session.
`finbert_score` in the output = Groq contextual score (misleading name, kept for backcompat).

---

## Dashboard

- **Main app**: `finwatch/app.py` — Streamlit, dark theme
- **Pages**:
  - Overview / Alert feed
  - Portfolio page: `finwatch/ui/portfolio_page.py` (added recently)
  - Components: `finwatch/ui/components.py`
- **Data loader**: `finwatch/data/loader.py`

**Dashboard now displays `trading_signal` (ENTRY/HOLD/EXIT) in:**
- Stock view: prominent badge in "AI Risk Analysis" column with color/glow
- Strategy box: redesigned as "AI Trading Signal" panel with ML signal + context factors
- Sidebar watchlist: signal arrow (▲/▼/◆/—) per ticker
- SPX market overview: "AI Trading Signals" summary (ENTRY/EXIT/HOLD counts + tickers)
- Landing page hero: "Entry Signals" count in stats
- Portfolio page: "AI Signal" stat shows trading_signal prominently

---

## Data Flow

```
data/raw/{ticker}.parquet           ← download_historical
data/processed/{ticker}.parquet     ← feature_pipeline
data/detection/{ticker}.parquet     ← detection_pipeline
data/predictions/{ticker}.parquet   ← prediction_pipeline
data/decisions/decisions.parquet    ← decision_pipeline
data/fundamental/                   ← collectors (1-day lag)
data/backtesting/                   ← backtest
```

---

## Models Directory

```
models/xgboost_drawdown.pkl         ← Drawdown Probability (ACTIVE)
models/meta_model.pkl               ← Meta-Model stacking (ACTIVE)
models/ae_{sector}_{vol}.keras      ← LSTM Autoencoders per bucket (ACTIVE)
models/if_{sector}.pkl              ← Isolation Forests per sector (ACTIVE)
models/xgboost_direction.pkl        ← Direction model (NOT USED)
models/xgboost_risk.pkl             ← Old risk model (DEPRECATED)
models/risk_label_encoder.pkl       ← For old risk model (DEPRECATED)
```

---

## Environment / Config

- **API keys needed**: `FINNHUB_API_KEY`, `GROQ_API_KEY` (in `.env`)
- **Config**: `config/assets.yaml` — ticker list + sector assignments
- **Requirements**: `requirements.txt`

---

## Known Issues / Technical Debt

1. **Fundamentals run after decisions** (pipeline.py Step 7 after Step 6):
   Fundamentals from the previous day are used in today's decisions. First-ever run has no fundamentals. Accepted design tradeoff.

2. **Trading signal not shown in dashboard**:
   `trading_signal` column exists in decisions.parquet but finwatch/app.py doesn't display it yet.

3. **XGBoost direction model trained but not used**:
   The model exists at `models/xgboost_direction.pkl` but is not called anywhere in the production pipeline. Only useful for research/comparison.

4. **n=12 ENTRY signals in backtest**:
   POSITIVE_SIGNAL is rare. ENTRY is rarer — requires above_ma200 + regime_ok + valuation_ok + revenue_ok. Backtest not yet rerun with new regime/MA-aware logic.

5. **lstm_inference.py missing**:
   Required for live demo (model.save() + standalone inference script). Not yet created.

6. **Groq daily token limit (100k TPD)**:
   LLM narrations (Layer 7) degrade to template fallbacks when daily quota exhausted. Not a code error — resets daily. Consider batching or caching narratives across runs.

---

## What Was Done in the Last Two Sessions

### Session N-1 (Valuation + Trading Signals)
1. Created `src/ingestion/valuation_collector.py` — new file, fetches P/E, P/B, Revenue Growth from Finnhub
2. Edited `src/decision/decision_engine.py`:
   - Added 4 valuation fields to `AnomalyInput` dataclass
   - Added `trading_signal` to `DecisionOutput`
   - Added valuation logic (overvalued/cheap/declining revenue gates on ENTRY)
   - Added trading signal logic (ENTRY/HOLD/EXIT/NEUTRAL)
3. Edited `src/decision/decision_pipeline.py`:
   - Added `_load_valuation_signals()` function
   - Added step "3c/4" to load valuation before building records
   - Passes valuation fields into records dict
4. Edited `src/pipeline.py`:
   - Added valuation collector import
   - Added valuation collection call in Step 7
5. Validated backtest: EXIT signals correctly correlate with higher drawdown rates

### Session N (MA/Regime Context + AUC Improvement + Bug Fixes)
1. **Fixed SPX MA bug**: `ma50`/`ma200` in detection parquets are S&P 500 MAs (not stock MAs).
   - Added `_compute_stock_ma_positions()` in `decision_pipeline.py` — computes per-stock MA50/MA200 from raw Close prices
   - Added `_ma_position_at(ticker, as_of_date, data)` in `backtest.py` for walk-forward correctness

2. **Enhanced `AnomalyInput` dataclass** with 5 new fields:
   - `price_vs_ma200`, `price_vs_ma50`, `regime`, `volume_trend`, `trend_strength`

3. **Improved decision logic in `decision_engine.py`**:
   - Regime-aware: bear market → upgrades WATCH→WARNING; blocks ENTRY
   - Overbought EXIT: RSI > 75 + >15% above MA200 + positive momentum → EXIT (profit-taking signal)
   - WARNING→EXIT only if p_dd ≥ 0.40 or anomaly_w ≥ 0.35 (reduces false EXIT signals)
   - ENTRY requires: above_ma200 + regime_ok + valuation_ok + revenue_ok (not just RSI)

4. **Improved drawdown model AUC 0.64 → 0.6671**:
   - Added stock-specific MA features (`price_vs_ma50_stock`, `price_vs_ma200_stock`) via `_add_stock_ma_features()`
   - Added missing features: `anomaly_score_weighted`, `volume_trend`, `obv_signal`, `sector_relative`
   - Tuned hyperparameters: `n_estimators=1200`, `max_depth=5`, `lr=0.015`, `min_child_weight=15`, `early_stopping_rounds=50`

5. **Fixed `_prep_X()` dtype bug** — replaced `select_dtypes(include="bool")` with `pd.to_numeric(errors="coerce")` to handle object-dtype columns (`if_anomaly`, `ae_anomaly`)

6. **Fixed `KeyError: 'ticker'`** in `_load_fundamental_signals()` — empty DataFrames from missing parquets

7. **Fixed SHAP crash** in `src/explainability/xai.py` — same `pd.to_numeric` fix for object dtype columns

8. **Full pipeline verified clean** — Layers 0–7 run without code errors (Groq LLM narrations gracefully degrade when daily quota hit)

---

## Decision Engine — AnomalyInput Fields (Updated)

```python
# Core
ticker, date, p_drawdown, drawdown_risk, anomaly_score, anomaly_score_weighted
market_anomaly, sector_anomaly
# Technical
rsi, momentum_5, momentum_10, drawdown, obv_signal, volatility, excess_return, es_ratio, vix_level
# MA / Regime context (added session N)
price_vs_ma200, price_vs_ma50   # stock-specific, not SPX
regime                           # "bull" / "bear" / "transition_down" / "transition_up" / "unknown"
volume_trend, trend_strength
# Sentiment
vader_score, finbert_score, news_sentiment_score
# Fundamental
days_to_next_earnings, insider_sentiment, put_call_ratio, options_fear
# Valuation
pe_ratio, pe_forward, pb_ratio, revenue_growth
```

---

## What Still Needs To Be Done

- [x] Show `trading_signal` (ENTRY/HOLD/EXIT) in the Streamlit dashboard — DONE
- [ ] Create `lstm_inference.py` for live demo presentation
- [ ] Rerun backtest with new valuation-aware + regime-aware trading_signal logic to get clean numbers
- [ ] Possibly add `AVOID` signal for P/E < 0 stocks (negative earnings = don't enter under any condition)
- [ ] Fix dashboard to show when valuation data was last updated (staleness warning)
