"""
Gold XAUUSD - Stage 2 Probability Calibrator
=============================================
Trains a logistic regression on OOF + 2026 predictions to estimate
P(model direction is correct | prediction confidence + macro regime)

Inputs  : cv_predictions_oof.csv, test_2026_results.csv,
          dataset_train_val.csv, dataset_test.csv
Outputs : calibrator.pkl, calibrator_report.csv
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
OOF_FILE      = os.path.join(BASE_DIR, "cv_predictions_oof.csv")
TEST_FILE     = os.path.join(BASE_DIR, "test_2026_results.csv")
TV_FILE       = os.path.join(BASE_DIR, "dataset_train_val.csv")
TEST_FEAT     = os.path.join(BASE_DIR, "dataset_test.csv")
OUT_MODEL     = os.path.join(BASE_DIR, "calibrator.pkl")
OUT_REPORT    = os.path.join(BASE_DIR, "calibrator_report.csv")

PRED_Z_WINDOW = 252

# all numeric — Bull_Trend replaced categorical Market_State
CALIB_FEATURES = ["oof_prediction", "pred_z", "abs_pred_z", "Macro_Fast", "Bull_Trend"]
N_FOLDS        = 5


# ── data loading ──────────────────────────────────────────────────────────────
def load_predictions():
    oof = pd.read_csv(OOF_FILE, index_col=0, parse_dates=True)
    oof = oof.dropna(subset=["oof_prediction"])
    oof = oof.rename(columns={"actual": "actual_return"})

    preds = oof[["actual_return", "oof_prediction"]]

    if os.path.exists(TEST_FILE):
        test = pd.read_csv(TEST_FILE, index_col=0, parse_dates=True)
        test = test.rename(columns={"predicted_return": "oof_prediction"})
        preds = pd.concat([preds, test[["actual_return", "oof_prediction"]]])
        print(f"oof rows       : {len(oof)}  test rows appended: {len(test)}")
    else:
        print(f"oof rows       : {len(oof)}  (no test file found)")

    preds = preds.sort_index()
    preds.index.name = "Date"
    print(f"combined range : {preds.index[0].date()} -> {preds.index[-1].date()}")
    return preds


def load_features():
    tv   = pd.read_csv(TV_FILE,   index_col=0, parse_dates=True)
    test = pd.read_csv(TEST_FEAT, index_col=0, parse_dates=True)
    feats = pd.concat([tv, test]).sort_index()
    feats.index.name = "Date"

    available = [c for c in ["Macro_Fast", "Bull_Trend"] if c in feats.columns]
    missing   = set(["Macro_Fast", "Bull_Trend"]) - set(available)
    if missing:
        raise ValueError(f"required calibrator features missing from dataset: {missing}")
    if "Market_State" in feats.columns:
        raise ValueError("Market_State still in dataset — rebuild with build_dataset.py first")

    print(f"features loaded: {available}")
    return feats[available]


def compute_pred_z(df):
    # strictly causal: rolling window uses only past predictions
    roll         = df["oof_prediction"].shift(1).rolling(PRED_Z_WINDOW)
    df["pred_z"] = (df["oof_prediction"] - roll.mean()) / roll.std()
    df["abs_pred_z"] = df["pred_z"].abs()
    return df


def build_dataset():
    preds = load_predictions()
    feats = load_features()
    df    = preds.join(feats, how="inner")
    df    = compute_pred_z(df)

    # binary target: 1 if model got direction right
    df["is_correct"] = (
        np.sign(df["oof_prediction"]) == np.sign(df["actual_return"])
    ).astype(int)

    df = df.dropna(subset=CALIB_FEATURES + ["is_correct"])

    print(f"calibrator rows: {len(df)}  ({df.index[0].date()} -> {df.index[-1].date()})")
    print(f"overall hit rate: {df['is_correct'].mean():.4f}  "
          f"(correct={df['is_correct'].sum()}  wrong={( df['is_correct']==0).sum()})")
    return df


# ── model ─────────────────────────────────────────────────────────────────────
def build_pipeline():
    return Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ])


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(y_true, y_prob, label=""):
    auc   = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    acc   = float(np.mean(y_true == (y_prob >= 0.5).astype(int)))
    print(f"  {label:<28} auc={auc:.4f}  brier={brier:.4f}  acc={acc:.4f}")
    return dict(auc=auc, brier=brier, accuracy=acc)


def reliability_table(y_true, y_prob, n_bins=10):
    bins   = np.linspace(0, 1, n_bins + 1)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(n_bins)]
    bucket = pd.cut(y_prob, bins=bins, labels=labels, include_lowest=True)
    tbl = (pd.DataFrame({"bucket": bucket, "actual": y_true, "prob": y_prob})
             .groupby("bucket", observed=True)
             .agg(count=("actual","size"),
                  actual_hit_rate=("actual","mean"),
                  mean_predicted_prob=("prob","mean"))
             .reset_index())
    return tbl


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("calibrator training started")
    df = build_dataset()
    X  = df[CALIB_FEATURES]
    y  = df["is_correct"]

    # ── walk-forward CV ───────────────────────────────────────────────────────
    print(f"\n{N_FOLDS}-fold walk-forward validation")
    tscv      = TimeSeriesSplit(n_splits=N_FOLDS)
    oof_probs = np.full(len(df), np.nan)
    fold_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        pipe = build_pipeline()
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        probs = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        oof_probs[val_idx] = probs
        m = evaluate(y.iloc[val_idx].values, probs, label=f"fold {fold}")
        fold_aucs.append(m["auc"])

    print(f"\n  cv mean auc : {np.mean(fold_aucs):.4f}  std={np.std(fold_aucs):.4f}")

    # ── OOF reliability ───────────────────────────────────────────────────────
    valid = ~np.isnan(oof_probs)
    print(f"\noof reliability table ({valid.sum()} rows)")
    rel = reliability_table(y.values[valid], oof_probs[valid])
    print(rel.to_string(index=False))
    evaluate(y.values[valid], oof_probs[valid], label="oof overall")

    # ── signal quality at different thresholds ────────────────────────────────
    print(f"\nsignal quality by prob threshold")
    for thresh in [0.50, 0.52, 0.55, 0.58, 0.60]:
        mask  = oof_probs[valid] >= thresh
        n     = mask.sum()
        if n > 0:
            hit = y.values[valid][mask].mean()
            print(f"  prob >= {thresh:.2f}  n={n:4d}  actual_hit={hit:.4f}")

    # ── train final calibrator on full dataset ────────────────────────────────
    print("\ntraining final calibrator on full dataset ...")
    final_pipe = build_pipeline()
    final_pipe.fit(X, y)

    with open(OUT_MODEL, "wb") as f:
        pickle.dump(final_pipe, f)
    print(f"saved calibrator.pkl")

    # ── save full report ──────────────────────────────────────────────────────
    report = df[["actual_return", "oof_prediction", "pred_z",
                 "abs_pred_z", "Macro_Fast", "Bull_Trend", "is_correct"]].copy()
    report["calibrated_prob"] = final_pipe.predict_proba(X)[:, 1]
    report.to_csv(OUT_REPORT)
    print(f"saved calibrator_report.csv  ({len(report)} rows)")

    # ── final evaluation ──────────────────────────────────────────────────────
    print(f"\nfinal in-sample reliability table")
    final_probs = report["calibrated_prob"].values
    rel_final   = reliability_table(y.values, final_probs)
    print(rel_final.to_string(index=False))
    evaluate(y.values, final_probs, label="full dataset")

    print("\ncalibrator training complete")


if __name__ == "__main__":
    main()