import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================

file = "mindtune_full_eeg_data_5_focus.csv"   # <-- replace with your CSV path

df = pd.read_csv(file)

# Clean column names
df.columns = df.columns.str.strip()

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
df = df.dropna(subset=["Timestamp"])

# Convert to time in seconds (relative)
df["Time_sec"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()

# =========================
# SELECT SIGNAL COLUMNS
# =========================

exclude_cols = ["Timestamp", "Time_sec", "Label"]
signal_cols = [c for c in df.columns if c not in exclude_cols]

# =========================
# OPTIONAL: NORMALIZATION
# =========================

for col in signal_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    if max_val - min_val != 0:
        df[col] = (df[col] - min_val) / (max_val - min_val)

# =========================
# PLOT (ALL SIGNALS)
# =========================

plt.figure(figsize=(12, 6))

for col in signal_cols:
    plt.plot(df["Time_sec"], df[col], label=col)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (normalized)")
plt.title("EEG Signals vs Time")
plt.legend(ncol=2, fontsize=8)
plt.grid()

plt.show()