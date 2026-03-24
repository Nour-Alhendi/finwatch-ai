## Project Status

This project is currently a **Work in Progress (WIP)**.
Submission deadline: March 30, 2026 — WBS Coding School Final Project.

---

## Overview

An AI-powered monitoring system for financial time-series data that automatically detects market anomalies, classifies their risk level, and explains why they happened.

The system moves beyond simple anomaly detection — it understands **context**, makes **predictions**, and supports **data-driven decisions**.

---

## Problem Statement

In risk management, you always want to have a plan and know when anomalies could occur — and know how to deal with them quickly.

You need to understand:
- Why did this happen?
- Was it expected?
- How dangerous is it?
- What decision should I make?

Traditional anomaly detection systems only flag unusual values but do not provide context or decision support. This project addresses that gap.

---

## System Architecture

The system follows an 8-layer modular pipeline:

| Layer | Name | Description |
|-------|------|-------------|
| 1 | Data Ingestion | Stooq API, 55 stocks, 10 years daily OHLCV |
| 2 | Data Quality Checks | Missing values, duplicates, gaps, schema validation |
| 3 | Feature Engineering | Basic (returns, volatility, RSI) + Context (regime, ETFs) + Advanced (momentum, lags) |
| 4 | Anomaly Detection | Z-Score + Isolation Forest + Dual LSTM Autoencoder — combined score + severity |
| 5 | Risk & Direction Prediction | XGBoost Risk (VaR-based, high/low) + XGBoost Direction (up/stable/down) |
| 6 | Guardrails + Decision Engine | Rule-based decisions per ticker (fix / keep / escalate) |
| 7 | Reporting + XAI | SHAP explainability, anomaly context, alerts |
| 8 | Logging & Audit | Full audit trail of all decisions |

---

## Data Sources

- **Stooq** — historical daily OHLCV data (primary source)
- **55 stocks** across 9 sectors (Technology, AI & Robotics, Financials, Healthcare, Consumer Staples, Energy, Consumer Discretionary, Industrials, Green Energy)
- **10 Sector ETFs** — XLK, BOTZ, XLF, XLV, XLP, XLE, XLY, XLI, ICLN, ^SPX
- **Reference Index** — ^SPX (S&P 500)

---

## Key Features

- Multi-asset monitoring (55 stocks, 10 sector ETFs)
- Three-layer anomaly detection (Statistical + ML + Deep Learning)
- Severity classification (normal / watch / warning / critical)
- Market context (market-wide vs sector-specific vs stock-specific anomaly)
- Market regime detection (Bull / Bear / Transition)
- Risk classification with XGBoost (VaR-based labels, high/low)
- Direction prediction with XGBoost (up/stable/down, 5-day horizon)
- Expected Shortfall (VaR 95% + ES tail risk per ticker)
- Explainability layer (XAI + SHAP)
- Full audit logging

---

## What This Project Is NOT

- Not a trading bot
- Not an investment advisory system
- Not a high-frequency trading engine

It is a **monitoring and decision-support framework**.

---

## Project Structure

```
ai-monitoring-system/
├── config/
│   └── assets.yaml          # Stock tickers, sectors, ETF mappings
├── data/
│   ├── raw/                 # Downloaded OHLCV data
│   ├── features/            # Engineered features
│   ├── detection/           # Anomaly detection results
│   ├── context/             # Contextual validation results
│   └── risk/                # ES snapshots for Layer 6 (Decision Engine)
├── src/
│   ├── ingestion/           # Layer 1: Data download
│   ├── quality/             # Layer 2: Quality checks
│   ├── features/            # Layer 3: Feature engineering
│   ├── detection/           # Layer 4: Anomaly detection
│   ├── prediction/          # Layer 5: Risk & Direction prediction
│   │   ├── expected_shortfall.py    # VaR + ES tail risk (writes to detection parquets)
│   │   ├── prediction_pipeline.py   # Orchestrates ES → Risk → Direction
│   │   ├── xgboost_risk.py          # High/low risk classifier
│   │   └── xgboost_direction.py     # up/stable/down direction classifier
│   ├── decision/            # Layer 6: Guardrails + Decision engine
│   ├── reporting/           # Layer 7: XAI + Reporting (WIP)
│   └── pipeline.py          # Main entry point
├── notebooks/               # Exploration and analysis
└── ARCHITECTURE.md          # Detailed architecture documentation
```

---

## Future Improvements

- FinBERT news sentiment analysis
- Real-time / intraday data (yfinance 1h)
- Streamlit dashboard
- Performance benchmarking
- Expanded asset universe
