# LSTM Autoencoder for anomaly detection — dual model per group (low/high volatility regime).
# Trains separate models for calm and volatile periods.
# Columns: ae_error, ae_anomaly

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.features import LSTM_AE_FEATURES

INPUT_DIR  = Path("data/detection")
OUTPUT_DIR = Path("data/detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 20
LSTM_UNITS      = 128
LATENT_DIM      = 16
DROPOUT         = 0.1
SMOOTH_WINDOW   = 5
N_FEATURES      = len(LSTM_AE_FEATURES)

GROUPS = {
    "Technology-Stable":   {"tickers": ["AAPL", "MSFT", "GOOG"],             "calm_q": 0.65, "percentile": 3},
    "Technology-Volatile": {"tickers": ["NVDA", "AMD"],                       "calm_q": 0.65, "percentile": 3},
    "AI-Stable":           {"tickers": ["CRM", "SNOW"],                       "calm_q": 0.65, "percentile": 5},
    "AI-Volatile":         {"tickers": ["PLTR", "META", "AI"],                "calm_q": 0.75, "percentile": 3},
    "Consumer-Stable":     {"tickers": ["NKE", "MCD", "SBUX"],                "calm_q": 0.65, "percentile": 5},
    "Consumer-Volatile":   {"tickers": ["TSLA", "AMZN"],                      "calm_q": 0.70, "percentile": 3},
    "Financials":          {"tickers": ["JPM", "BAC", "GS", "MS", "BLK"],     "calm_q": 0.70, "percentile": 4},
    "Healthcare":          {"tickers": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],  "calm_q": 0.75, "percentile": 3},
    "Consumer Staples":    {"tickers": ["PG", "KO", "COST", "WMT", "CL"],     "calm_q": 0.70, "percentile": 3},
    "Energy":              {"tickers": ["XOM", "CVX", "COP", "SLB", "EOG"],   "calm_q": 0.70, "percentile": 3},
    "Industrials":         {"tickers": ["CAT", "HON", "BA", "GE", "RTX"],     "calm_q": 0.75, "percentile": 3},
    "Green Energy":        {"tickers": ["BE", "ENPH", "PLUG", "NEE", "FSLR"], "calm_q": 0.75, "percentile": 3},
}


def build_model():
    inputs = Input(shape=(SEQUENCE_LENGTH, N_FEATURES))
    x = LSTM(LSTM_UNITS, activation="relu", return_sequences=False)(inputs)
    x = Dropout(DROPOUT)(x)
    x = Dense(LATENT_DIM, activation="relu")(x)
    x = RepeatVector(SEQUENCE_LENGTH)(x)
    x = LSTM(LSTM_UNITS, activation="relu", return_sequences=True)(x)
    x = Dropout(DROPOUT)(x)
    output = TimeDistributed(Dense(N_FEATURES))(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="mse")
    return model


def build_sequences(data):
    return np.array([data[i-SEQUENCE_LENGTH:i] for i in range(SEQUENCE_LENGTH, len(data))])


def get_errors(model, X):
    X_pred = model.predict(X, verbose=0)
    errors = np.mean(np.abs(X - X_pred), axis=(1, 2))
    return pd.Series(errors).ewm(span=SMOOTH_WINDOW).mean().values


def run_autoencoder():
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    for group_name, params in GROUPS.items():
        tickers  = params["tickers"]
        calm_q   = params["calm_q"]
        perc     = params["percentile"]

        # Load data per ticker
        frames = []
        for ticker in tickers:
            file = INPUT_DIR / f"{ticker}.parquet"
            if not file.exists():
                print(f"  Skipping {ticker}: file not found")
                continue
            df = pd.read_parquet(file)
            df = df.dropna(subset=LSTM_AE_FEATURES).reset_index(drop=True)
            split_date = pd.Timestamp("2024-01-01")
            df["_is_train"] = df["Date"] < split_date
            df["_ticker"] = ticker
            frames.append(df)

        if not frames:
            print(f"Skipping group {group_name}: no data")
            continue

        # Split into low/high vol regime per ticker and scale separately
        low_frames  = []
        high_frames = []

        # Threshold and scaler only from training data (no lookahead)
        for df in frames:
            train_df = df[df["_is_train"]]
            vol_threshold = train_df["volatility"].quantile(calm_q)
            df["_is_low_vol"]    = df["volatility"] <= vol_threshold
            df["_vol_threshold"] = vol_threshold

            scaler_low  = MinMaxScaler()
            scaler_high = MinMaxScaler()

            low_df_train  = df[df["_is_low_vol"] &  df["_is_train"]].copy().reset_index(drop=True)
            high_df_train = df[~df["_is_low_vol"] & df["_is_train"]].copy().reset_index(drop=True)

            if len(low_df_train) > SEQUENCE_LENGTH:
                scaler_low.fit(low_df_train[LSTM_AE_FEATURES])
                scaler_high.fit(high_df_train[LSTM_AE_FEATURES])
                low_df_train[LSTM_AE_FEATURES]  = scaler_low.transform(low_df_train[LSTM_AE_FEATURES])
                high_df_train[LSTM_AE_FEATURES] = scaler_high.transform(high_df_train[LSTM_AE_FEATURES])
                low_frames.append((df, low_df_train, high_df_train, scaler_low, scaler_high))

        if not low_frames:
            print(f"Skipping group {group_name}: not enough low-vol data")
            continue

        # Build training sequences
        X_train_low  = []
        X_train_high = []

        for (df, low_df_train, high_df_train, scaler_low, scaler_high) in low_frames:
            if len(low_df_train) > SEQUENCE_LENGTH:
                X_train_low.append(build_sequences(low_df_train[LSTM_AE_FEATURES].values))
            if len(high_df_train) > SEQUENCE_LENGTH:
                X_train_high.append(build_sequences(high_df_train[LSTM_AE_FEATURES].values))

        X_train_low  = np.concatenate(X_train_low)  if X_train_low  else None
        X_train_high = np.concatenate(X_train_high) if X_train_high else None

        # Train dual models
        model_low = build_model()
        if X_train_low is not None:
            model_low.fit(X_train_low, X_train_low,
                          epochs=30, batch_size=32,
                          validation_split=0.1,
                          callbacks=[early_stop], verbose=0)
            print(f"[{group_name}] low_vol_model trained on {X_train_low.shape[0]} sequences")

        model_high = build_model()
        if X_train_high is not None:
            model_high.fit(X_train_high, X_train_high,
                           epochs=30, batch_size=32,
                           validation_split=0.1,
                           callbacks=[early_stop], verbose=0)
            print(f"[{group_name}] high_vol_model trained on {X_train_high.shape[0]} sequences")

        # Predict and save per ticker
        for (df, low_df_train, high_df_train, scaler_low, scaler_high) in low_frames:
            ticker        = df["_ticker"].iloc[0]
            vol_threshold = df["_vol_threshold"].iloc[0]

            orig_df = pd.read_parquet(INPUT_DIR / f"{ticker}.parquet")
            feat_df = orig_df.dropna(subset=LSTM_AE_FEATURES).reset_index(drop=True)
            orig_df["ae_error"]   = np.nan
            orig_df["ae_anomaly"] = False
            feat_df["_is_low_vol"] = feat_df["volatility"] <= vol_threshold

            # Low vol regime prediction
            low_idx  = feat_df[feat_df["_is_low_vol"]].index
            high_idx = feat_df[~feat_df["_is_low_vol"]].index

            if len(low_idx) > SEQUENCE_LENGTH:
                low_scaled  = scaler_low.transform(feat_df.loc[low_idx, LSTM_AE_FEATURES])
                X_low_all   = build_sequences(low_scaled)
                errors_low  = get_errors(model_low, X_low_all)
                train_low_idx = feat_df[feat_df["_is_low_vol"] & (feat_df["Date"] < split_date)].index
                train_low_errors = get_errors(model_low, build_sequences(scaler_low.transform(feat_df.loc[train_low_idx, LSTM_AE_FEATURES])))
                threshold_low = np.percentile(train_low_errors, 100 - perc)
                target_dates  = feat_df.loc[low_idx[SEQUENCE_LENGTH:SEQUENCE_LENGTH + len(errors_low)], "Date"].values
                mask = orig_df["Date"].isin(target_dates)
                orig_df.loc[mask, "ae_error"]   = errors_low
                orig_df.loc[mask, "ae_anomaly"] = errors_low > threshold_low

            # High vol regime prediction
            if len(high_idx) > SEQUENCE_LENGTH:
                high_scaled  = scaler_high.transform(feat_df.loc[high_idx, LSTM_AE_FEATURES])
                X_high_all   = build_sequences(high_scaled)
                errors_high  = get_errors(model_high, X_high_all)
                train_high_idx = feat_df[~feat_df["_is_low_vol"] & (feat_df["Date"] < split_date)].index
                train_high_errors = get_errors(model_high, build_sequences(scaler_high.transform(feat_df.loc[train_high_idx, LSTM_AE_FEATURES])))
                threshold_high = np.percentile(train_high_errors, 100 - perc)
                target_dates  = feat_df.loc[high_idx[SEQUENCE_LENGTH:SEQUENCE_LENGTH + len(errors_high)], "Date"].values
                mask = orig_df["Date"].isin(target_dates)
                orig_df.loc[mask, "ae_error"]   = errors_high
                orig_df.loc[mask, "ae_anomaly"] = errors_high > threshold_high

            orig_df.to_parquet(OUTPUT_DIR / f"{ticker}.parquet")
            print(f"  Saved: {ticker}.parquet  ({orig_df['ae_anomaly'].sum()} anomalies)")


if __name__ == "__main__":
    run_autoencoder()
