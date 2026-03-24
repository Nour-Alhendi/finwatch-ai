# Model Notes — FinWatch AI
> Personal notes for job interviews and project presentations

---

## XGBoost Risk Classifier (Layer 5)

### Goal
Predict whether a stock is at **high or low risk** of a significant price drop in the next 5 trading days.

### Label Design (the hard part)
Instead of using arbitrary thresholds, I used **Value at Risk (VaR)** — a concept from risk management:

- For each ticker, I calculated the **future max drawdown** over the next 5 days
- I then computed the **85th percentile** of that drawdown using **training data only** (no lookahead bias)
- Any day where the drawdown exceeded that threshold was labeled `high`, everything else `low`
- Result: ~15% high, ~85% low — a realistic and economically meaningful split

This means "high risk" = the stock is about to experience a drawdown **worse than 85% of historical days**.

### Class Imbalance Problem
Since only 15% of labels are `high`, the model would normally just predict `low` for everything and still get 85% accuracy. This is called **class imbalance**.

Fix: `scale_pos_weight = neg / pos`
- This tells XGBoost how much rarer the positive class ("high") is
- It forces the model to take high-risk predictions seriously
- Without it: recall for "high" was ~1%. With it: ~48% at threshold 0.5

### Key Parameters
| Parameter | Value | Why |
|---|---|---|
| `n_estimators` | 300 | Enough trees for a complex financial signal |
| `max_depth` | 5 | Prevents overfitting on noisy financial data |
| `learning_rate` | 0.05 | Slow and careful — more robust than 0.1+ |
| `subsample` | 0.8 | Each tree sees 80% of rows — reduces variance |
| `colsample_bytree` | 0.8 | Each tree sees 80% of features — reduces correlation |
| `scale_pos_weight` | ~5.7 | Corrects class imbalance (neg/pos ratio) |
| `eval_metric` | logloss | Standard for probability calibration |

### Train / Test Split
- **Train:** pre-2024 (historical data)
- **Test:** 2024 onwards (out-of-sample)
- No data leakage: VaR thresholds computed on training data only

### Features (35 total)
Grouped into:
- **Anomaly signals:** z_score, isolation forest, LSTM autoencoder reconstruction error
- **Price/returns:** volatility, momentum, RSI, trend strength
- **Context:** market-wide vs stock-specific signal, sector relative return, regime
- **Volume:** volume z-score, high volume flag

### Threshold Tuning
The model outputs a probability `p_high`. The decision threshold controls the tradeoff:

| Threshold | Recall (high) | Interpretation |
|---|---|---|
| 0.3 | ~97% | Catches almost all risk — many false alarms |
| 0.4 | ~79% | Good recall, fewer false alarms |
| 0.5 | ~48% | Conservative — only flags clear high-risk cases |

I chose **0.5** for the production system as a balance between precision and recall.

### Why XGBoost?
- Handles mixed feature types (bool, float, int) well
- Robust to noisy financial data
- Fast training and inference
- Interpretable with SHAP values (planned for Layer 7)
- Industry standard for tabular financial data

---

## LSTM Autoencoder (Layer 4 — Anomaly Detection)

### Goal
Detect anomalies by learning what "normal" price behavior looks like, then flagging deviations.

### Architecture
- Encoder: LSTM → Dense (bottleneck/latent space)
- Decoder: RepeatVector → LSTM → Dense (reconstruction)
- Input: sequences of 20 days, 10 features
- Output: reconstruction error per day

### Dual Model Design
Each sector group has **two separate models**:
- `low_vol_model` — trained on calm market periods
- `high_vol_model` — trained on volatile market periods

This prevents the model from flagging normal volatility in volatile stocks as anomalies.

### Group-Aware Training
Stocks are grouped by sector AND volatility behavior (Stable vs Volatile):
- e.g., "Technology-Stable": AAPL, MSFT, GOOG
- e.g., "Technology-Volatile": NVDA, AMD

One model pair per group → 12 groups × 2 models = **24 LSTM models total**

### Anomaly Threshold
Computed from training data only (no lookahead):
- Threshold = top `percentile`% of reconstruction errors on training sequences
- Each group has its own percentile (3–5%) tuned to its volatility profile

---

## Isolation Forest (Layer 4 — Anomaly Detection)

### Goal
Detect anomalies using an unsupervised ML approach — no labels needed.

### How it works
- Randomly splits feature space with decision trees
- Points that are easy to isolate (short path length) = anomalies
- Points that require many splits to isolate = normal

### Group-Aware Training
- One model per sector group (12 models total)
- Trained only on **calm periods** (low volatility rows) — learns what "normal" looks like
- Predicts on all data, flags the bottom `percentile`% of anomaly scores

### Key Design Decision
No `contamination` parameter — instead use a **percentile threshold** on the score distribution. This gives more control and avoids assuming a fixed anomaly rate.

---

## Combined Anomaly Score (Layer 4)

Four signals combined into a single integer score (0–4):
- Z-Score 20-day window
- Z-Score 60-day window
- Isolation Forest
- LSTM Autoencoder

`anomaly_score > 0` → `combined_anomaly = True`

Severity mapping:
- 0 → normal
- 1 → watch
- 2–3 → warning
- 4 → critical
