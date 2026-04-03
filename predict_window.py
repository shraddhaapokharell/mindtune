"""
MindTune EEG Brain State Classifier — Prediction Script
========================================================
Runs the trained model on any new EEG CSV file and outputs
predicted brain states (focus / relax / sudoku) per window.

Usage:
    python predict.py --input new_eeg_data.csv
    python predict.py --input new_eeg_data.csv --output results.csv

Requirements:
    - ./model/mindtune_model.pkl
    - ./model/mindtune_scaler.pkl
    - ./model/mindtune_label_encoder.pkl
    - ./model/mindtune_feature_names.pkl
    - ./model/mindtune_window_config.pkl

Input CSV must contain these columns (same as training data):
    Timestamp, Delta, Theta, Low_Alpha, High_Alpha,
    Low_Beta, High_Beta, Low_Gamma, Mid_Gamma

Optional columns (ignored if present):
    Person_ID, Mind_State, Attention, Meditation, Signal_Quality
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import joblib

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_DIR = "./model"
BAND_COLS = ["Delta", "Theta", "Low_Alpha", "High_Alpha",
             "Low_Beta", "High_Beta", "Low_Gamma", "Mid_Gamma"]
# ──────────────────────────────────────────────────────────────────────────────


def load_artifacts():
    required = {
        "model":         "mindtune_model.pkl",
        "scaler":        "mindtune_scaler.pkl",
        "label_encoder": "mindtune_label_encoder.pkl",
        "feature_names": "mindtune_feature_names.pkl",
        "window_config": "mindtune_window_config.pkl",
    }
    arts = {}
    for key, fname in required.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"[ERROR] Missing model file: {path}")
            print(f"        Please run train_model.py first.")
            sys.exit(1)
        arts[key] = joblib.load(path)
    return arts


def engineer_row_features(df):
    """Same per-row feature engineering as used during training."""
    d = df[BAND_COLS].copy().astype(float)

    d["Total_Power"] = d[BAND_COLS].sum(axis=1)
    safe_total = d["Total_Power"].replace(0, np.nan)

    for col in BAND_COLS:
        d[f"Rel_{col}"] = d[col] / safe_total

    safe_alpha = (d["Low_Alpha"] + d["High_Alpha"]).replace(0, np.nan)
    safe_beta  = (d["Low_Beta"]  + d["High_Beta"]).replace(0, np.nan)
    slow = d["Delta"] + d["Theta"]
    fast = d["Low_Alpha"] + d["High_Alpha"] + d["Low_Beta"] + d["High_Beta"]

    d["Theta_Alpha_Ratio"] = d["Theta"] / safe_alpha
    d["Beta_Alpha_Ratio"]  = safe_beta  / safe_alpha
    d["Slow_Fast_Ratio"]   = slow / fast.replace(0, np.nan)
    d["Gamma_Ratio"]       = (d["Low_Gamma"] + d["Mid_Gamma"]) / safe_total
    d["Alpha_Asymmetry"]   = d["High_Alpha"] / d["Low_Alpha"].replace(0, np.nan)

    for col in BAND_COLS:
        d[f"Log_{col}"] = np.log1p(d[col])

    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.fillna(0, inplace=True)
    return d


def apply_sliding_window(feature_df, window_size, step_size):
    """
    Apply sliding window and return X (windows x features) and
    the start/end row index of each window for output alignment.
    """
    values = feature_df.values
    n_rows, n_feats = values.shape
    t = np.arange(window_size, dtype=float)

    X_wins, window_spans = [], []

    for start in range(0, n_rows - window_size + 1, step_size):
        window = values[start : start + window_size]
        agg = []
        for f in range(n_feats):
            col = window[:, f]
            agg += [col.mean(), col.std(), col.min(), col.max(),
                    np.polyfit(t, col, 1)[0]]
        X_wins.append(agg)
        window_spans.append((start, start + window_size - 1))

    return np.array(X_wins), window_spans


def validate_input(df):
    missing = [c for c in BAND_COLS if c not in df.columns]
    if missing:
        print(f"[ERROR] Input CSV is missing required columns: {missing}")
        print(f"        Required: {BAND_COLS}")
        sys.exit(1)


def predict(input_path, output_path=None):
    # ── Load artifacts ──────────────────────────────────────────────────────
    print(f"\nLoading model from {MODEL_DIR}/...")
    arts          = load_artifacts()
    model         = arts["model"]
    scaler        = arts["scaler"]
    le            = arts["label_encoder"]
    feat_names    = arts["feature_names"]
    window_config = arts["window_config"]

    WINDOW_SIZE = window_config["window_size"]
    STEP_SIZE   = window_config["step_size"]
    print(f"  Window size : {WINDOW_SIZE} rows  |  Step size: {STEP_SIZE} rows")

    # ── Load input ──────────────────────────────────────────────────────────
    print(f"Reading input: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df)}")

    df.drop(columns=["Attention", "Meditation", "Signal_Quality"],
            errors="ignore", inplace=True)
    validate_input(df)

    if len(df) < WINDOW_SIZE:
        print(f"[ERROR] Input has only {len(df)} rows but window size is {WINDOW_SIZE}.")
        print(f"        Need at least {WINDOW_SIZE} rows to make a prediction.")
        sys.exit(1)

    # ── Feature engineering + windowing ─────────────────────────────────────
    print("Engineering features and applying sliding window...")
    feat_df = engineer_row_features(df)
    X_wins, spans = apply_sliding_window(feat_df, WINDOW_SIZE, STEP_SIZE)
    print(f"  {len(df)} rows -> {len(X_wins)} windows")

    # ── Scale & predict ─────────────────────────────────────────────────────
    X_scaled       = scaler.transform(X_wins)
    y_pred         = model.predict(X_scaled)
    y_proba        = model.predict_proba(X_scaled)
    predicted_labels = le.inverse_transform(y_pred)

    # ── Build output DataFrame (one row per window) ─────────────────────────
    # Attach the timestamp of the window's first row for reference
    timestamps = df["Timestamp"].values if "Timestamp" in df.columns else None

    results = []
    for i, (start, end) in enumerate(spans):
        row = {
            "Window":           i + 1,
            "Row_Start":        start,
            "Row_End":          end,
            "Predicted_State":  predicted_labels[i],
            "Confidence_%":     round(y_proba[i].max() * 100, 1),
        }
        if timestamps is not None:
            row["Timestamp_Start"] = timestamps[start]
            row["Timestamp_End"]   = timestamps[end]
        for j, cls in enumerate(le.classes_):
            row[f"Prob_{cls}_%"] = round(y_proba[i, j] * 100, 1)
        results.append(row)

    result_df = pd.DataFrame(results)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n── Prediction Summary ──────────────────────────────")
    counts = pd.Series(predicted_labels).value_counts()
    total  = len(predicted_labels)
    for state, count in counts.items():
        bar = "█" * int(count / total * 30)
        print(f"  {state:<12} {count:>3} windows  ({count/total*100:5.1f}%)  {bar}")
    print(f"\n  Average confidence : {result_df['Confidence_%'].mean():.1f}%")
    print(f"  Total windows      : {total}  (from {len(df)} input rows)")

    # ── Save ─────────────────────────────────────────────────────────────────
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_predictions.csv"

    result_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    print("────────────────────────────────────────────────────\n")
    return result_df


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindTune EEG Brain State Predictor")
    parser.add_argument("--input",  "-i", required=True, help="Path to input EEG CSV file")
    parser.add_argument("--output", "-o", default=None,  help="Path to save predictions CSV")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    predict(args.input, args.output)
