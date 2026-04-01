"""
MindTune EEG Brain State Classifier — Prediction Script
========================================================
Runs the trained model on any new EEG CSV file and outputs
predicted brain states (focus / relax / sudoku) per row.

Usage:
    python predict.py --input new_eeg_data.csv
    python predict.py --input new_eeg_data.csv --output results.csv

Requirements:
    - ./model/mindtune_model.pkl
    - ./model/mindtune_scaler.pkl
    - ./model/mindtune_label_encoder.pkl
    - ./model/mindtune_feature_names.pkl

Input CSV must contain these columns (same as training data):
    Timestamp, Delta, Theta, Low_Alpha, High_Alpha,
    Low_Beta, High_Beta, Low_Gamma, Mid_Gamma

Optional columns (ignored if present):
    Person_ID, Mind_State, Attention, Meditation, Signal_Quality
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_DIR = "./model"
BAND_COLS = ["Delta", "Theta", "Low_Alpha", "High_Alpha",
             "Low_Beta", "High_Beta", "Low_Gamma", "Mid_Gamma"]
# ──────────────────────────────────────────────────────────────────────────────


def load_model_artifacts():
    """Load all saved model artifacts from ./model/"""
    required = {
        "model":         "mindtune_model.pkl",
        "scaler":        "mindtune_scaler.pkl",
        "label_encoder": "mindtune_label_encoder.pkl",
        "feature_names": "mindtune_feature_names.pkl",
    }
    artifacts = {}
    for key, filename in required.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"[ERROR] Missing model file: {path}")
            print(f"        Please run train_model.py first.")
            sys.exit(1)
        artifacts[key] = joblib.load(path)
    return artifacts


def engineer_features(df):
    """
    Build the same 30 features used during training.
    Input df must contain BAND_COLS.
    """
    d = df[BAND_COLS].copy().astype(float)

    # Total power
    d["Total_Power"] = d[BAND_COLS].sum(axis=1)
    safe_total = d["Total_Power"].replace(0, np.nan)

    # Relative band power
    for col in BAND_COLS:
        d[f"Rel_{col}"] = d[col] / safe_total

    # Neuroscience ratios
    safe_alpha = (d["Low_Alpha"] + d["High_Alpha"]).replace(0, np.nan)
    safe_beta  = (d["Low_Beta"]  + d["High_Beta"]).replace(0, np.nan)
    slow       = d["Delta"] + d["Theta"]
    fast       = (d["Low_Alpha"] + d["High_Alpha"] +
                  d["Low_Beta"]  + d["High_Beta"])

    d["Theta_Alpha_Ratio"] = d["Theta"] / safe_alpha
    d["Beta_Alpha_Ratio"]  = safe_beta  / safe_alpha
    d["Slow_Fast_Ratio"]   = slow       / fast.replace(0, np.nan)
    d["Gamma_Ratio"]       = (d["Low_Gamma"] + d["Mid_Gamma"]) / safe_total
    d["Alpha_Asymmetry"]   = d["High_Alpha"] / d["Low_Alpha"].replace(0, np.nan)

    # Log-transformed raw bands
    for col in BAND_COLS:
        d[f"Log_{col}"] = np.log1p(d[col])

    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.fillna(0, inplace=True)

    return d


def validate_input(df):
    """Check that required columns are present."""
    missing = [c for c in BAND_COLS if c not in df.columns]
    if missing:
        print(f"[ERROR] Input CSV is missing required columns: {missing}")
        print(f"        Required: {BAND_COLS}")
        sys.exit(1)


def predict(input_path, output_path=None):
    # ── Load artifacts ──────────────────────────────────────────────────────
    print(f"\nLoading model from {MODEL_DIR}/...")
    arts = load_model_artifacts()
    model    = arts["model"]
    scaler   = arts["scaler"]
    le       = arts["label_encoder"]
    feat_names = arts["feature_names"]

    # ── Load input data ─────────────────────────────────────────────────────
    print(f"Reading input: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df)}")

    # Drop columns we don't need (won't cause errors if absent)
    df.drop(columns=["Attention", "Meditation", "Signal_Quality"],
            errors="ignore", inplace=True)

    validate_input(df)

    # ── Feature engineering ─────────────────────────────────────────────────
    print("Engineering features...")
    features = engineer_features(df)

    # Ensure column order matches training exactly
    missing_feats = [f for f in feat_names if f not in features.columns]
    if missing_feats:
        print(f"[ERROR] Feature mismatch after engineering: {missing_feats}")
        sys.exit(1)

    X = features[feat_names].values

    # ── Scale ───────────────────────────────────────────────────────────────
    X_scaled = scaler.transform(X)

    # ── Predict ─────────────────────────────────────────────────────────────
    print("Running predictions...")
    y_pred        = model.predict(X_scaled)
    y_proba       = model.predict_proba(X_scaled)
    predicted_labels = le.inverse_transform(y_pred)

    # ── Build output DataFrame ──────────────────────────────────────────────
    result = df.copy()
    result["Predicted_State"]      = predicted_labels
    result["Confidence_%"]         = (y_proba.max(axis=1) * 100).round(1)

    # Per-class probabilities
    for i, cls in enumerate(le.classes_):
        result[f"Prob_{cls}_%"] = (y_proba[:, i] * 100).round(1)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n── Prediction Summary ──────────────────────────────")
    counts = pd.Series(predicted_labels).value_counts()
    total  = len(predicted_labels)
    for state, count in counts.items():
        bar = "█" * int(count / total * 30)
        print(f"  {state:<12} {count:>4} rows  ({count/total*100:5.1f}%)  {bar}")

    avg_conf = result["Confidence_%"].mean()
    print(f"\n  Average confidence: {avg_conf:.1f}%")
    print(f"  Total rows predicted: {total}")

    # ── Save ─────────────────────────────────────────────────────────────────
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_predictions.csv"

    result.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    print("────────────────────────────────────────────────────\n")

    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MindTune EEG Brain State Predictor"
    )
    parser.add_argument(
        "--input",  "-i", required=True,
        help="Path to input EEG CSV file"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to save predictions CSV (default: <input>_predictions.csv)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    predict(args.input, args.output)
