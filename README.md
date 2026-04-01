# MindTune EEG Brain State Classifier

A machine learning pipeline that classifies EEG brainwave data into three mental states:
**Focus**, **Relaxation**, and **Sudoku** (attention-demanding task).

---

## Project Structure

```
mindtune/
│
├── README.md                  ← You are here
│
├── train_model.py             ← Train and save the model
├── predict.py                 ← Run predictions on new CSV files
├── clean_eeg_data.py          ← Clean raw EEG CSVs before training
│
├── training_data/             ← Raw cleaned CSVs used for training
│   ├── mindtune_full_eeg_data_1_focus.csv
│   ├── mindtune_full_eeg_data_1_relax.csv
│   ├── mindtune_full_eeg_data_1_sudoku.csv
│   ├── mindtune_full_eeg_data_2_focus.csv
│   ├── mindtune_full_eeg_data_2_relax.csv
│   └── mindtune_full_eeg_data_2_sudoku.csv
│
├── cleaned_data/              ← Output of clean_eeg_data.py
│
└── model/                     ← Saved model artifacts (output of train_model.py)
    ├── mindtune_model.pkl
    ├── mindtune_scaler.pkl
    ├── mindtune_label_encoder.pkl
    └── mindtune_feature_names.pkl
```

---

## Requirements

```
Python 3.8+
scikit-learn
pandas
numpy
joblib
```

Install with:
```bash
pip install scikit-learn pandas numpy joblib
```

---

## Step 1 — Clean Raw EEG Data

If you have new raw CSV files from your device, run the cleaner first:

```bash
python clean_eeg_data.py
```

- Input:  `training_data/mindtune_full_eeg_data_[id]_[state].csv`
- Output: `cleaned_data/mindtune_full_eeg_data_[id]_[state].csv`

The cleaner removes `Attention`, `Meditation`, and `Signal_Quality` columns,
adds `Person_ID` and `Mind_State` columns, and normalizes timestamps.

---

## Step 2 — Train the Model

```bash
python train_model.py
```

This will:
1. Load all CSVs from `training_data/`
2. Engineer 30 features (relative band power, neuroscience ratios, log transforms)
3. Balance classes via oversampling
4. Cross-validate Random Forest, Gradient Boosting, and SVM
5. Tune hyperparameters via GridSearchCV
6. Train a soft-voting ensemble
7. Save all artifacts to `model/`

**Reported CV accuracy: ~60%** (3-class problem, 1,462 training samples)

---

## Step 3 — Predict on New Data

```bash
python predict.py --input path/to/new_eeg_data.csv
python predict.py --input path/to/new_eeg_data.csv --output results.csv
```

### Input CSV requirements

Must contain these columns:
```
Timestamp, Delta, Theta, Low_Alpha, High_Alpha,
Low_Beta, High_Beta, Low_Gamma, Mid_Gamma
```

Optional columns (ignored automatically):
```
Person_ID, Mind_State, Attention, Meditation, Signal_Quality
```

### Output

The prediction script appends these columns to your input CSV:
| Column | Description |
|---|---|
| `Predicted_State` | focus / relax / sudoku |
| `Confidence_%` | model confidence (0–100%) |
| `Prob_focus_%` | probability of focus state |
| `Prob_relax_%` | probability of relax state |
| `Prob_sudoku_%` | probability of sudoku state |

---

## Model Details

| Component | Details |
|---|---|
| Algorithm | Soft-voting ensemble |
| Models | Random Forest (w=3) + Gradient Boosting (w=2) + SVM RBF (w=2) |
| Features | 30 (relative power, log bands, Theta/Alpha, Beta/Alpha, Slow/Fast ratio, Gamma ratio) |
| Balancing | Majority-class oversampling |
| Scaling | StandardScaler |
| CV | 5-fold Stratified KFold |
| CV Accuracy | ~60% (honest generalization estimate) |

**Top predictive features:** Relative Beta power, Gamma ratio, Beta/Alpha ratio —
consistent with neuroscience (beta waves are primary markers of focused vs. relaxed states).

---

## Improving Accuracy

- **More data** — even 300–500 additional rows per class would help significantly
- **More people** — currently trained on 2 subjects; adding more improves generalization
- **Per-person models** — if you only need to classify one person, retrain on their data alone
- **Longer recording sessions** — more temporal context per state

---

## Adding New Training Data

1. Place new raw CSVs in `training_data/` following the naming convention:
   `mindtune_full_eeg_data_[person_id]_[state].csv`
   where state is one of: `focus`, `relax`, `sudoku`
2. Run `python train_model.py` — it will automatically pick up all files

---

*Built with scikit-learn · Data from MindTune EEG device*
