"""
MindTune EEG Brain State Classifier — Training Script
======================================================
Trains a tuned Random Forest classifier to classify Focus / Relaxation / Sudoku
states from EEG brainwave data.

Usage:
    python train_model.py

Outputs saved to ./model/:
    mindtune_model.pkl          — trained Random Forest model
    mindtune_scaler.pkl         — fitted StandardScaler
    mindtune_label_encoder.pkl  — LabelEncoder (focus/relax/sudoku → 0/1/2)
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
RANDOM_STATE = 42
N_SPLITS     = 5
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
lines = []

def log(msg=""):
    print(msg)
    lines.append(str(msg))



# 1. LOAD DATA

log("=" * 65)
log("  MindTune EEG Classifier — Training")
log("=" * 65)

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "mindtune_full_eeg_data_*.csv")))
if not csv_files:
    raise FileNotFoundError(f"No CSVs found in {DATA_DIR}/")

frames = []
for f in csv_files:
    df = pd.read_csv(f)
    frames.append(df)
    log(f"  Loaded  {os.path.basename(f):<45} {len(df):>4} rows")

data = pd.concat(frames, ignore_index=True)
log(f"\n  Total rows : {len(data)}")
log(f"  Class counts:\n{data['Mind_State'].value_counts().to_string()}")



# 2. FEATURE ENGINEERING

log("\n[1] Engineering features...")

def engineer_features(df):
    d = df[BAND_COLS].copy().astype(float)

    # Total power
    d["Total_Power"] = d[BAND_COLS].sum(axis=1)
    safe_total = d["Total_Power"].replace(0, np.nan)

    # Relative band power (normalises amplitude differences across sessions)
    for col in BAND_COLS:
        d[f"Rel_{col}"] = d[col] / safe_total

    # Neuroscience-backed ratios
    safe_alpha = (d["Low_Alpha"] + d["High_Alpha"]).replace(0, np.nan)
    safe_beta  = (d["Low_Beta"]  + d["High_Beta"]).replace(0, np.nan)
    slow = d["Delta"] + d["Theta"]
    fast = d["Low_Alpha"] + d["High_Alpha"] + d["Low_Beta"] + d["High_Beta"]

    d["Theta_Alpha_Ratio"] = d["Theta"] / safe_alpha
    d["Beta_Alpha_Ratio"]  = safe_beta  / safe_alpha
    d["Slow_Fast_Ratio"]   = slow / fast.replace(0, np.nan)
    d["Gamma_Ratio"]       = (d["Low_Gamma"] + d["Mid_Gamma"]) / safe_total
    d["Alpha_Asymmetry"]   = d["High_Alpha"] / d["Low_Alpha"].replace(0, np.nan)

    # Log-transform raw bands (EEG power is log-normally distributed)
    for col in BAND_COLS:
        d[f"Log_{col}"] = np.log1p(d[col])

    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.fillna(0, inplace=True)
    return d

features   = engineer_features(data)
feat_names = features.columns.tolist()
log(f"  Total features: {len(feat_names)}")



# 3. ENCODE LABELS

le = LabelEncoder()
y  = le.fit_transform(data["Mind_State"])
X  = features.values
log(f"\n[2] Labels encoded: {dict(enumerate(le.classes_))}")



# 4. BALANCE CLASSES

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
log(f"  Balanced: {len(y_bal)} total samples ({max_count} per class)")



# 5. SCALE

scaler        = StandardScaler()
X_scaled      = scaler.fit_transform(X_bal)
X_orig_scaled = scaler.transform(X)



# 6. CROSS-VALIDATE RANDOM FOREST (baseline)

log("\n[4] Cross-validating Random Forest baseline (5-fold, real distribution)...")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=2,
                             max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
scores = cross_val_score(rf, X_orig_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)
log(f"  {'Random Forest':<22}  {scores.mean():.4f} +/- {scores.std():.4f}   folds: {scores.round(3).tolist()}")



# 7. HYPERPARAMETER TUNING

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



# 8. TRAIN FINAL MODEL ON FULL BALANCED DATA

log("\n[6] Training final Random Forest on balanced data...")

best_rf.fit(X_scaled, y_bal)
final_scores = cross_val_score(best_rf, X_orig_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)
log(f"  Final RF CV  {final_scores.mean():.4f} +/- {final_scores.std():.4f}   folds: {final_scores.round(3).tolist()}")



# 9. FINAL EVALUATION

log("\n[7] Final evaluation on original (unbalanced) data:")
y_pred = best_rf.predict(X_orig_scaled)
log("\n" + classification_report(y, y_pred, target_names=le.classes_))
log("Confusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y, y_pred)
log(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

log("\n[8] Top 15 features by importance:")
feat_imp = best_rf.feature_importances_
top_idx  = np.argsort(feat_imp)[::-1][:15]
for rank, i in enumerate(top_idx, 1):
    log(f"  {rank:>2}. {feat_names[i]:<28} {feat_imp[i]:.4f}")



# 10. SAVE

log("\n[9] Saving artifacts...")
joblib.dump(best_rf,    os.path.join(MODEL_DIR, "mindtune_model.pkl"))
joblib.dump(scaler,     os.path.join(MODEL_DIR, "mindtune_scaler.pkl"))
joblib.dump(le,         os.path.join(MODEL_DIR, "mindtune_label_encoder.pkl"))
joblib.dump(feat_names, os.path.join(MODEL_DIR, "mindtune_feature_names.pkl"))

with open(os.path.join(MODEL_DIR, "training_report.txt"), "w") as f:
    f.write("\n".join(lines))

log("  Saved: model/mindtune_model.pkl")
log("  Saved: model/mindtune_scaler.pkl")
log("  Saved: model/mindtune_label_encoder.pkl")
log("  Saved: model/mindtune_feature_names.pkl")
log("  Saved: model/training_report.txt")
log("\n  Training complete!")
