"""
EEG Data Cleaning Script for MindTune
Processes all mindtune_full_eeg_data_[person_id]_[state].csv files
from training_data/ and saves cleaned versions to cleaned_data/
"""

import pandas as pd
import os
import re

TRAINING_DIR = "training_data"
CLEANED_DIR  = "cleaned_data"

os.makedirs(CLEANED_DIR, exist_ok=True)

# Columns to drop
DROP_COLS = ["Attention", "Meditation", "Signal_Quality"]

# Final column order
OUTPUT_COLS = [
    "Person_ID", "Mind_State", "Timestamp",
    "Delta", "Theta", "Low_Alpha", "High_Alpha",
    "Low_Beta", "High_Beta", "Low_Gamma", "Mid_Gamma"
]

files = [f for f in os.listdir(TRAINING_DIR) if f.endswith(".csv")]

if not files:
    print("No CSV files found in training_data/")
    exit(0)

summary = []

for filename in sorted(files):
    # Parse person_id and state from filename
    match = re.match(r"mindtune_full_eeg_data_(\d+)_(\w+)\.csv", filename)
    if not match:
        print(f"  [SKIP] Filename doesn't match expected pattern: {filename}")
        continue

    person_id = int(match.group(1))
    mind_state = match.group(2).lower()   # focus / relaxation / sudoku

    filepath = os.path.join(TRAINING_DIR, filename)
    df = pd.read_csv(filepath)

    # Drop unwanted columns (ignore if already missing)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Parse and clean timestamp — keep as seconds-resolution datetime string
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Timestamp"] = df["Timestamp"].dt.floor("s")          # strip sub-seconds
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Drop rows where timestamp couldn't be parsed
    bad_ts = df["Timestamp"].isna().sum()
    if bad_ts:
        print(f"  [WARN] {bad_ts} rows with unparseable timestamps dropped in {filename}")
    df.dropna(subset=["Timestamp"], inplace=True)

    # Add identifier columns at the front
    df.insert(0, "Person_ID",   person_id)
    df.insert(1, "Mind_State",  mind_state)

    # Reorder / keep only expected columns (in case source has extras)
    existing_out_cols = [c for c in OUTPUT_COLS if c in df.columns]
    df = df[existing_out_cols]

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Save
    out_path = os.path.join(CLEANED_DIR, filename)
    df.to_csv(out_path, index=False)

    summary.append({
        "file": filename,
        "person_id": person_id,
        "mind_state": mind_state,
        "rows": len(df),
        "columns": len(df.columns)
    })
    print(f"  [OK] {filename}  →  {len(df)} rows, {len(df.columns)} cols")

print(f"\nDone. {len(summary)} file(s) cleaned → {CLEANED_DIR}/")
print("\nSummary:")
for s in summary:
    print(f"  Person {s['person_id']:>3} | {s['mind_state']:<12} | {s['rows']:>4} rows | {s['columns']} cols")
