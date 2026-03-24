# FinWatch AI
––––––––––––––

Product Name:    FinWatch AI
GitHub Repo:     finwatch-ai
Description:     AI-Driven Financial Anomaly Detection
                 and Monitoring System



# LAYER 1 – DATA INGESTION
├── Stooq API (historical, 10 years, daily)
├── 55 Stocks, OHLCV (Tech, Finance, Health, Energy, Industrials, Green Energy)
├── 10 Reference ETFs (SPX, XLK, XLF, XLV, XLP, XLE, XLY, XLI, BOTZ, ICLN)
└── Output: raw_clean parquet files (data/raw/raw_clean/)

# LAYER 2 – DATA QUALITY CHECKS
├── Schema validation
├── Missing values
├── Duplicates
├── OHLC violations
└── Output: validated parquet files

# LAYER 3 – FEATURE ENGINEERING
│
├── 3A: Basic Features
│   ├── returns, volatility
│   ├── rolling_mean, rolling_std (20d)
│   ├── beta, corr_spx
│   └── rsi
│
├── 3B: Context Features
│   ├── spx_return, etf_return
│   ├── is_market_wide, is_sector_wide   (pure market signal, no anomaly)
│   ├── excess_return, relative_return, sector_relative
│   ├── regime (bull/bear/transition), ma50, ma200, regime_encoded
│   └── vol_regime, volume_ma20, volume_zscore, is_high_volume
│
└── 3C: Advanced Features
    ├── return_lag_1/2/3, momentum_5/10
    ├── vol_5, vol_20, vol_change
    ├── trend_strength
    ├── volatility_ratio (vs SPX)
    └── volume_trend

# LAYER 4 – ANOMALY DETECTION
│
├── Statistical:    Z-Score (±3σ, 20d + 60d window)
├── ML:             Isolation Forest (group-aware, percentile threshold per group)
├── Deep Learning:  Dual LSTM Autoencoder
│   ├── low_vol_model  → trained on calm periods per group
│   ├── high_vol_model → trained on volatile periods per group
│   └── regime-specific thresholds from train data only (no lookahead)
├── Combine:
│   ├── anomaly_score (0–4): z + z_60 + if + ae
│   ├── combined_anomaly (bool)
│   ├── market_anomaly  = is_market_wide  & combined_anomaly
│   └── sector_anomaly  = is_sector_wide  & combined_anomaly
├── Severity:       normal / watch / warning / critical
└── Output: data/detection/ parquet files

# LAYER 5 – RISK & DIRECTION PREDICTION
│
├── 5A: Expected Shortfall
│   ├── Rolling 252-day VaR (95%) + ES per ticker
│   ├── ES captures tail risk — not the same as volatility:
│   │   └── same volatility, very different ES → different danger level
│   ├── Columns: var_95, es_95, es_ratio (ES/VaR)
│   └── Output: written to detection parquets + data/risk/ snapshot
│
├── 5B: XGBoost Risk Classifier
│   ├── 38 features: anomaly signals, price/returns, context, volume, ES (var_95, es_95, es_ratio)
│   ├── Labels (forward-looking, 5 days) — VaR-based, per ticker, no lookahead:
│   │   ├── low  → max drawdown < 85th percentile
│   │   └── high → max drawdown ≥ 85th percentile
│   ├── Class imbalance handled via scale_pos_weight
│   └── Output: risk_level + p_low, p_high per ticker
│
├── 5C: XGBoost Direction Classifier
│   ├── 38 features: same as 5B
│   ├── Labels (forward-looking, 5 days) — percentile-based, per ticker, no lookahead:
│   │   ├── up     → future_return ≥ 75th percentile
│   │   ├── down   → future_return ≤ 25th percentile
│   │   └── stable → in between
│   ├── Class imbalance handled via sample_weight="balanced"
│   └── Output: direction + p_up, p_stable, p_down per ticker
│
└── Output: models/xgboost_risk.pkl + models/xgboost_direction.pkl

# LAYER 6 – GUARDRAILS + DECISION ENGINE
├── Input: risk_level (high/low) × direction (up/stable/down)
├── Decision matrix:
│   ├── high risk + down   → CRITICAL (escalate immediately)
│   ├── high risk + stable → WARNING  (monitor closely)
│   ├── high risk + up     → WATCH    (possible recovery)
│   ├── low risk  + down   → WATCH    (minor concern)
│   └── low risk  + stable/up → NORMAL
├── Actions: FIX / KEEP / ESCALATE per ticker
└── Output: final decision + severity per ticker

# LAYER 7 – REPORTING + XAI
├── Automatic report per ticker
├── SHAP explanation (why this risk level?)
├── Anomaly context (market-wide vs stock-specific)
└── Output: PDF/HTML report

# LAYER 8 – LOGGING & AUDIT
├── All decisions logged
├── Timestamp + model + reason
└── Output: audit trail
