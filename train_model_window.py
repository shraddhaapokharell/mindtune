"""
MindTune EEG Brain State Classifier — Training Script
======================================================
Trains a tuned Random Forest classifier to classify Focus / Relaxation / Sudoku
states from EEG brainwave data.

Windowing approach:
  Each CSV is processed independently (no mixing across sessions).
  A sliding window is applied per file to capture how brainwave patterns
  evolve over time. Each window produces one training sample by computing
  statistical aggregates (mean, std, min, max, slope) across the window,
  plus the existing neuroscience ratio features.

Usage:
    python train_model.py

Outputs saved to ./model/:
    mindtune_model.pkl          — trained Random Forest model
    mindtune_scaler.pkl         — fitted StandardScaler
    mindtune_label_encoder.pkl  — LabelEncoder (focus/relax/sudoku -> 0/1/2)
    mindtune_feature_names.pkl  — ordered feature name list
    training_report.txt         — full evaluation report
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR     = "./cleaned_data"
MODEL_DIR    = "./model"
BAND_COLS    = ["Delta", "Theta", "Low_Alpha", "High_Alpha",
                "Low_Beta", "High_Beta", "Low_Gamma", "Mid_Gamma"]
WINDOW_SIZE  = 10    # number of consecutive rows per window (~10 seconds of EEG)
STEP_SIZE    = 1     # slide window forward by 1 row (row 1-10, then 2-11, then 3-12...)
RANDOM_STATE = 42
N_SPLITS     = 5
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
lines = []

def log(msg=""):
    print(msg)
    lines.append(str(msg))


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: per-row feature engineering (ratios, log transforms, relative power)
# Applied before windowing so window stats capture engineered signals too.
# ══════════════════════════════════════════════════════════════════════════════
def engineer_row_features(df):
    """
    Compute per-row features from raw band columns.
    Returns a DataFrame of engineered features (no label/meta columns).
    """
    d = df[BAND_COLS].copy().astype(float)

    # Total power
    d["Total_Power"] = d[BAND_COLS].sum(axis=1)
    safe_total = d["Total_Power"].replace(0, np.nan)

    # Relative band power — normalises amplitude differences across sessions
    for col in BAND_COLS:
        d[f"Rel_{col}"] = d[col] / safe_total

    # Neuroscience-backed ratios
    safe_alpha = (d["Low_Alpha"] + d["High_Alpha"]).replace(0, np.nan)
    safe_beta  = (d["Low_Beta"]  + d["High_Beta"]).replace(0, np.nan)
    slow = d["Delta"] + d["Theta"]
    fast = d["Low_Alpha"] + d["High_Alpha"] + d["Low_Beta"] + d["High_Beta"]

    d["Theta_Alpha_Ratio"] = d["Theta"] / safe_alpha       # rises during relaxation
    d["Beta_Alpha_Ratio"]  = safe_beta  / safe_alpha       # rises during focus
    d["Slow_Fast_Ratio"]   = slow / fast.replace(0, np.nan)
    d["Gamma_Ratio"]       = (d["Low_Gamma"] + d["Mid_Gamma"]) / safe_total
    d["Alpha_Asymmetry"]   = d["High_Alpha"] / d["Low_Alpha"].replace(0, np.nan)

    # Log-transform raw bands (EEG power is log-normally distributed)
    for col in BAND_COLS:
        d[f"Log_{col}"] = np.log1p(d[col])

    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.fillna(0, inplace=True)
    return d


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: sliding window aggregation
# Each window -> one sample with mean/std/min/max of each feature + slope.
# Slope captures the *trend* direction within the window (rising/falling band).
# ══════════════════════════════════════════════════════════════════════════════
def apply_sliding_window(feature_df, label, window_size, step_size):
    """
    Slide a window over feature_df rows.
    Returns X (np.ndarray of window samples) and y (list of labels).
    """
    values = feature_df.values
    n_rows, n_feats = values.shape
    feat_names_local = feature_df.columns.tolist()

    X_wins, y_wins = [], []
    t = np.arange(window_size, dtype=float)  # time index for slope

    for start in range(0, n_rows - window_size + 1, step_size):
        window = values[start : start + window_size]   # shape (window_size, n_feats)

        agg = []
        for f in range(n_feats):
            col = window[:, f]
            agg.append(col.mean())
            agg.append(col.std())
            agg.append(col.min())
            agg.append(col.max())
            # Linear slope via least-squares (trend direction)
            slope = np.polyfit(t, col, 1)[0]
            agg.append(slope)

        X_wins.append(agg)
        y_wins.append(label)

    # Build window feature names (only once, reuse across calls)
    win_feat_names = []
    for fname in feat_names_local:
        for stat in ["mean", "std", "min", "max", "slope"]:
            win_feat_names.append(f"{fname}_{stat}")

    return np.array(X_wins), y_wins, win_feat_names


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & WINDOW EACH CSV INDEPENDENTLY
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 65)
log("  MindTune EEG Classifier — Training (with Windowing)")
log("=" * 65)
log(f"\n  Window size : {WINDOW_SIZE} rows")
log(f"  Step size   : {STEP_SIZE} rows  ({int((1 - STEP_SIZE/WINDOW_SIZE)*100)}% overlap)")

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "mindtune_full_eeg_data_*.csv")))
if not csv_files:
    raise FileNotFoundError(f"No CSVs found in {DATA_DIR}/")

all_X, all_y = [], []
win_feat_names = None

for f in csv_files:
    df = pd.read_csv(f)
    label = df["Mind_State"].iloc[0]          # all rows in file share same label

    # Per-row feature engineering first
    feat_df = engineer_row_features(df)

    # Sliding window over this file's rows
    X_win, y_win, wfn = apply_sliding_window(feat_df, label, WINDOW_SIZE, STEP_SIZE)

    if win_feat_names is None:
        win_feat_names = wfn                  # capture feature names from first file

    all_X.append(X_win)
    all_y.extend(y_win)

    log(f"  {os.path.basename(f):<45}  {len(df):>4} rows  ->  {len(X_win):>3} windows")

X = np.vstack(all_X)
log(f"\n  Total windows : {len(all_y)}")
log(f"  Features/window: {X.shape[1]}  "
    f"({len(win_feat_names.count('_') for _ in [0])})")   # just display count

# Count per class
from collections import Counter
counts = Counter(all_y)
log(f"  Windows per class: { {k: counts[k] for k in sorted(counts)} }")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ENCODE LABELS
# ══════════════════════════════════════════════════════════════════════════════
le = LabelEncoder()
y  = le.fit_transform(all_y)
log(f"\n[2] Labels encoded: {dict(enumerate(le.classes_))}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BALANCE CLASSES
# ══════════════════════════════════════════════════════════════════════════════
log("\n[3] Balancing classes via oversampling...")

X_df, y_s = pd.DataFrame(X), pd.Series(y)
max_count  = y_s.value_counts().max()
Xp, yp     = [], []
for cls in sorted(y_s.unique()):
    Xc, yc = X_df[y_s == cls], y_s[y_s == cls]
    Xr, yr = resample(Xc, yc, replace=True, n_samples=max_count, random_state=RANDOM_STATE)
    Xp.append(Xr); yp.append(yr)

X_bal = pd.concat(Xp).values
y_bal = pd.concat(yp).values
log(f"  Balanced: {len(y_bal)} total windows ({max_count} per class)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SCALE
# ══════════════════════════════════════════════════════════════════════════════
scaler        = StandardScaler()
X_scaled      = scaler.fit_transform(X_bal)
X_orig_scaled = scaler.transform(X)


# ══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATE RANDOM FOREST (baseline)
# ══════════════════════════════════════════════════════════════════════════════
log("\n[4] Cross-validating Random Forest baseline (5-fold, real distribution)...")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=2,
                             max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
scores = cross_val_score(rf, X_orig_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)
log(f"  {'Random Forest':<22}  {scores.mean():.4f} +/- {scores.std():.4f}   folds: {scores.round(3).tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
log("\n[5] Tuning Random Forest via GridSearchCV...")

param_grid = {
    "n_estimators":     [200, 300, 500],
    "max_depth":        [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", "log2"],
}
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0
)
grid_rf.fit(X_orig_scaled, y)
log(f"  Best params : {grid_rf.best_params_}")
log(f"  Best CV acc : {grid_rf.best_score_:.4f}")
best_rf = grid_rf.best_estimator_


# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAIN FINAL MODEL ON FULL BALANCED DATA
# ══════════════════════════════════════════════════════════════════════════════
log("\n[6] Training final Random Forest on balanced windowed data...")

best_rf.fit(X_scaled, y_bal)
final_scores = cross_val_score(best_rf, X_orig_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)
log(f"  Final RF CV  {final_scores.mean():.4f} +/- {final_scores.std():.4f}   folds: {final_scores.round(3).tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
log("\n[7] Final evaluation on original (unbalanced) windowed data:")
y_pred = best_rf.predict(X_orig_scaled)
log("\n" + classification_report(y, y_pred, target_names=le.classes_))
log("Confusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y, y_pred)
log(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

log("\n[8] Top 15 features by importance:")
feat_imp = best_rf.feature_importances_
top_idx  = np.argsort(feat_imp)[::-1][:15]
for rank, i in enumerate(top_idx, 1):
    log(f"  {rank:>2}. {win_feat_names[i]:<35} {feat_imp[i]:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE
# ══════════════════════════════════════════════════════════════════════════════
log("\n[9] Saving artifacts...")

# Also save window config so predict.py uses the same settings
window_config = {"window_size": WINDOW_SIZE, "step_size": STEP_SIZE}

joblib.dump(best_rf,       os.path.join(MODEL_DIR, "mindtune_model.pkl"))
joblib.dump(scaler,        os.path.join(MODEL_DIR, "mindtune_scaler.pkl"))
joblib.dump(le,            os.path.join(MODEL_DIR, "mindtune_label_encoder.pkl"))
joblib.dump(win_feat_names,os.path.join(MODEL_DIR, "mindtune_feature_names.pkl"))
joblib.dump(window_config, os.path.join(MODEL_DIR, "mindtune_window_config.pkl"))

with open(os.path.join(MODEL_DIR, "training_report.txt"), "w") as f:
    f.write("\n".join(lines))

log("  Saved: model/mindtune_model.pkl")
log("  Saved: model/mindtune_scaler.pkl")
log("  Saved: model/mindtune_label_encoder.pkl")
log("  Saved: model/mindtune_feature_names.pkl")
log("  Saved: model/mindtune_window_config.pkl")
log("  Saved: model/training_report.txt")
log("\n  Training complete!")
