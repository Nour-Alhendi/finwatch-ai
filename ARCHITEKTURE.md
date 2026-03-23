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
│   ├── rolling_mean, rolling_std (20d + 60d)
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

# LAYER 5 – RISK PREDICTION
├── XGBoost Classifier
│   ├── Features: anomaly_score, returns, volatility, rsi,
│   │             relative_return, trend_strength, momentum_5,
│   │             vol_change, volume_zscore
│   ├── Labels (forward-looking, 5 days):
│   │   ├── low    → max drawdown < 1%
│   │   ├── medium → max drawdown 1–3%
│   │   └── high   → max drawdown > 3%
│   └── Output: risk_level + p_low, p_medium, p_high per ticker
└── Output: models/xgboost_risk.pkl

# LAYER 6 – GUARDRAILS + DECISION ENGINE
├── FIX:       auto-correct data issues
├── KEEP:      flag but keep
├── ESCALATE:  alert human
└── Output: autonomous decision per ticker

# LAYER 7 – REPORTING + XAI
├── Automatic report per ticker
├── SHAP explanation (why this risk level?)
├── Anomaly context (market-wide vs stock-specific)
└── Output: PDF/HTML report

# LAYER 8 – LOGGING & AUDIT
├── All decisions logged
├── Timestamp + model + reason
└── Output: audit trail
