"""
Gold XAUUSD - Stage 2 Probability Calibrator
=============================================
Predicts P(Stage 1 model direction is correct) given market regime + signal strength.

Inputs  : cv_predictions_oof.csv, test_2026_results.csv,
          dataset_train_val.csv, dataset_test.csv
Outputs : calibrator.pkl, calibrator_report.csv, final_2026_calibrated_signals.csv
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

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OOF_FILE   = os.path.join(BASE_DIR, "cv_predictions_oof.csv")
TEST_FILE  = os.path.join(BASE_DIR, "test_2026_results.csv")
TV_FILE    = os.path.join(BASE_DIR, "dataset_train_val.csv")
TEST_FEAT  = os.path.join(BASE_DIR, "dataset_test.csv")
OUT_MODEL  = os.path.join(BASE_DIR, "calibrator.pkl")
OUT_REPORT = os.path.join(BASE_DIR, "calibrator_report.csv")
OUT_2026   = os.path.join(BASE_DIR, "final_2026_calibrated_signals.csv")

PRED_Z_WINDOW = 252
FEATURE_COLS  = ["Bull_Trend", "Macro_Fast", "BB_PctB", "Price_Over_EMA200", "MACD_Signal_Norm"]
CALIB_FEATS   = ["prediction_value", "abs_prediction", "Bull_Trend",
                 "Macro_Fast", "BB_PctB", "Price_Over_EMA200"]
N_FOLDS       = 5
AUC_TARGET    = 0.56


# ── data assembly ─────────────────────────────────────────────────────────────
def load_oof():
    df = pd.read_csv(OOF_FILE, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df = df.rename(columns={"actual": "actual_return",
                             "oof_prediction": "prediction_value"})
    # warmup purge: drop rows where model had no lookback data
    before = len(df)
    df = df.dropna(subset=["prediction_value"])
    print(f"oof warmup purge : dropped {before - len(df)} rows, kept {len(df)}")
    return df[["actual_return", "prediction_value"]]


def load_test():
    if not os.path.exists(TEST_FILE):
        print("no test_2026_results.csv found, skipping")
        return pd.DataFrame()
    df = pd.read_csv(TEST_FILE, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df = df.rename(columns={"predicted_return": "prediction_value"})
    return df[["actual_return", "prediction_value"]]


def load_features():
    tv   = pd.read_csv(TV_FILE,   index_col=0, parse_dates=True)
    test = pd.read_csv(TEST_FEAT, index_col=0, parse_dates=True)
    feats = pd.concat([tv, test]).sort_index()
    feats.index.name = "Date"

    missing = set(FEATURE_COLS) - set(feats.columns)
    if missing:
        raise ValueError(f"features missing from dataset: {missing}")

    return feats[FEATURE_COLS]


def build_dataset():
    oof   = load_oof()
    test  = load_test()
    feats = load_features()

    # stack oof + 2026
    if not test.empty:
        preds = pd.concat([oof, test])
        print(f"stacked oof+test : {len(oof)} + {len(test)} = {len(preds)} rows")
    else:
        preds = oof

    preds = preds.sort_index()

    # inner join with features
    df = preds.join(feats, how="inner")
    print(f"after feature join : {len(df)} rows  "
          f"({df.index[0].date()} -> {df.index[-1].date()})")

    # holiday filter: drop zero-return rows (non-trading days / stale fills)
    before = len(df)
    df = df[df["actual_return"] != 0.0]
    print(f"holiday filter     : dropped {before - len(df)} zero-return rows")

    # signal strength features
    df["abs_prediction"] = df["prediction_value"].abs()

    # causal pred_z for reference (not used as calib feature but saved in report)
    roll     = df["prediction_value"].shift(1).rolling(PRED_Z_WINDOW)
    roll_std = roll.std().replace(0, np.nan)
    df["pred_z"]     = (df["prediction_value"] - roll.mean()) / roll_std
    df["pred_z"]     = df["pred_z"].replace([np.inf, -np.inf], np.nan)
    df["abs_pred_z"] = df["pred_z"].abs()

    # binary target
    df["is_correct"] = (
        np.sign(df["prediction_value"]) == np.sign(df["actual_return"])
    ).astype(int)

    # final dropna on calibrator features
    before = len(df)
    df = df.dropna(subset=CALIB_FEATS + ["is_correct"])
    print(f"dropna on features : dropped {before - len(df)} rows, kept {len(df)}")

    hit = df["is_correct"].mean()
    print(f"overall hit rate   : {hit:.4f}  "
          f"(correct={df['is_correct'].sum()}  wrong={(df['is_correct']==0).sum()})")

    # warn if hit rate too low for calibrator to find signal
    if hit < 0.48 or hit > 0.52:
        print(f"hit rate is outside 48-52% — calibrator has something to work with")
    else:
        print(f"hit rate near 50% — calibrator signal may be weak")

    return df


# ── model ─────────────────────────────────────────────────────────────────────
def build_pipeline():
    return Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", LogisticRegression(
            # class_weight=None,      # RISKY: Don't balance; let the model bias toward the majority
            C=10.0,                   # RISKY: High C = less regularization. Trust the data more.
            max_iter=2000,
            random_state=42,
        )),
    ])


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(y_true, y_prob, label=""):
    auc   = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    acc   = float(np.mean(y_true == (y_prob >= 0.5).astype(int)))
    flag  = "OK" if auc >= AUC_TARGET else "BELOW TARGET"
    print(f"  {label:<30} auc={auc:.4f} [{flag}]  brier={brier:.4f}  acc={acc:.4f}")
    return dict(auc=auc, brier=brier, accuracy=acc)


def reliability_table(y_true, y_prob, n_bins=10):
    bins   = np.linspace(0, 1, n_bins + 1)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(n_bins)]
    bucket = pd.cut(y_prob, bins=bins, labels=labels, include_lowest=True)
    tbl = (pd.DataFrame({"bucket": bucket, "actual": y_true, "prob": y_prob})
             .groupby("bucket", observed=True)
             .agg(count=("actual", "size"),
                  actual_hit_rate=("actual", "mean"),
                  mean_predicted_prob=("prob", "mean"))
             .reset_index())
    return tbl


def signal_quality(y_true, y_prob):
    print(f"\nsignal quality by probability threshold")
    print(f"  {'threshold':<12} {'n_signals':>10} {'pct_of_days':>12} {'actual_hit':>12}")
    for thresh in [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60]:
        mask = y_prob >= thresh
        n    = mask.sum()
        if n > 0:
            hit  = y_true[mask].mean()
            pct  = n / len(y_true)
            flag = " <-- edge" if hit > 0.53 else ""
            print(f"  {thresh:<12.2f} {n:>10}  {pct:>11.1%}  {hit:>11.4f}{flag}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("calibrator training started")
    df = build_dataset()

    X = df[CALIB_FEATS]
    y = df["is_correct"]

    # ── walk-forward CV ───────────────────────────────────────────────────────
    print(f"\n{N_FOLDS}-fold walk-forward validation")
    tscv      = TimeSeriesSplit(n_splits=N_FOLDS)
    oof_probs = np.full(len(df), np.nan)
    fold_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        val_start   = df.index[val_idx[0]].date()
        val_end     = df.index[val_idx[-1]].date()

        pipe = build_pipeline()
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs

        m = evaluate(y_val.values, probs,
                     label=f"fold {fold} ({val_start}->{val_end})")
        fold_aucs.append(m["auc"])

    mean_auc = np.mean(fold_aucs)
    print(f"\n  cv mean auc : {mean_auc:.4f}  std={np.std(fold_aucs):.4f}")
    if mean_auc < AUC_TARGET:
        print(f"  WARNING: mean auc {mean_auc:.4f} below target {AUC_TARGET}")
        print(f"  the base model signal may be too weak for reliable calibration")
        print(f"  consider retraining base model with min_data_in_leaf=30")

    # ── OOF reliability ───────────────────────────────────────────────────────
    valid = ~np.isnan(oof_probs)
    print(f"\noof reliability table ({valid.sum()} rows)")
    rel = reliability_table(y.values[valid], oof_probs[valid])
    print(rel.to_string(index=False))
    evaluate(y.values[valid], oof_probs[valid], label="oof overall")
    signal_quality(y.values[valid], oof_probs[valid])

    # ── train final calibrator on full dataset ────────────────────────────────
    print("\ntraining final calibrator on full dataset ...")
    final_pipe = build_pipeline()
    final_pipe.fit(X, y)

    with open(OUT_MODEL, "wb") as f:
        pickle.dump(final_pipe, f)
    print(f"saved calibrator.pkl")

    # ── full report ───────────────────────────────────────────────────────────
    report = df[["actual_return", "prediction_value", "abs_prediction",
                 "pred_z", "abs_pred_z", "Bull_Trend", "Macro_Fast",
                 "BB_PctB", "Price_Over_EMA200", "is_correct"]].copy()
    report["win_probability"] = final_pipe.predict_proba(X)[:, 1]
    report.to_csv(OUT_REPORT)
    print(f"saved calibrator_report.csv  ({len(report)} rows)")

    # ── 2026 calibrated signals ───────────────────────────────────────────────
    mask_2026 = df.index.year >= 2026
    if mask_2026.sum() > 0:
        df_2026   = df[mask_2026].copy()
        X_2026    = X[mask_2026]
        probs2026 = final_pipe.predict_proba(X_2026)[:, 1]

        out2026 = pd.DataFrame({
            "actual_return":    df_2026["actual_return"].values,
            "prediction_value": df_2026["prediction_value"].values,
            "abs_prediction":   df_2026["abs_prediction"].values,
            "Bull_Trend":       df_2026["Bull_Trend"].values,
            "Macro_Fast":       df_2026["Macro_Fast"].values,
            "win_probability":  probs2026,
            "signal":           np.where(probs2026 >= 0.55, "BUY",
                                np.where(probs2026 <= 0.45, "SELL", "NO SIGNAL")),
            "is_correct":       df_2026["is_correct"].values,
        }, index=df_2026.index)

        out2026.to_csv(OUT_2026)
        print(f"saved final_2026_calibrated_signals.csv  ({len(out2026)} rows)")

        print(f"\n2026 calibrated signals summary")
        print(f"  total days        : {len(out2026)}")
        print(f"  BUY signals       : {(out2026['signal']=='BUY').sum()}")
        print(f"  NO SIGNAL days    : {(out2026['signal']=='NO SIGNAL').sum()}")
        buy_hit = out2026[out2026['signal']=='BUY']['is_correct'].mean()
        print(f"  BUY hit rate      : {buy_hit:.4f}" if (out2026['signal']=='BUY').sum() > 0 else "  BUY hit rate      : n/a")

        print(f"\n2026 in-sample reliability")
        rel2026 = reliability_table(df_2026["is_correct"].values, probs2026)
        print(rel2026[rel2026["count"] > 0].to_string(index=False))
    else:
        print("no 2026 rows found in dataset")

    # ── final in-sample evaluation ────────────────────────────────────────────
    print(f"\nfinal in-sample reliability (full dataset)")
    final_probs = report["win_probability"].values
    rel_final   = reliability_table(y.values, final_probs)
    print(rel_final[rel_final["count"] > 0].to_string(index=False))
    evaluate(y.values, final_probs, label="full dataset in-sample")

    print("\ncalibrator training complete")


if __name__ == "__main__":
    main()