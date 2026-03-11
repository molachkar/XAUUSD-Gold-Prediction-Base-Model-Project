"""
Gold XAUUSD - LightGBM V3 Training Pipeline
============================================
Walk-forward 5-fold CV on 2004-2025, holdout test 2026+
Outputs: model, OOF predictions, feature importance, test results, drift report
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV  = os.path.join(BASE_DIR, "dataset_train_val.csv")
TEST_CSV   = os.path.join(BASE_DIR, "dataset_test.csv")
TARGET_COL = "target_log_return"

LGBM_PARAMS = dict(
    n_estimators     = 30000,
    learning_rate    = 0.01,
    num_leaves       = 40,   # Increased from 31
    min_data_in_leaf = 30,   # Reduced from 50 (Crucial Fix)
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq     = 1,
    reg_alpha        = 1.0,
    reg_lambda       = 1.0,
    n_jobs           = -1,
    random_state     = 42,
    verbose          = -1,
)
EARLY_STOP = 500
N_FOLDS    = 5


def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))

def ic(y, yhat):
    return float(pd.Series(y).corr(pd.Series(yhat)))

def hit_rate(y, yhat):
    return float(np.mean(np.sign(y) == np.sign(yhat)))

def metrics(y, yhat, label=""):
    r, i, h = rmse(y, yhat), ic(y, yhat), hit_rate(y, yhat)
    print(f"  {label:<22} rmse={r:.6f}  ic={i:+.4f}  hit={h:.4f}")
    return dict(rmse=r, ic=i, hit_rate=h)


def drift_report(train_df, feat_cols):
    print("\ndrift check (2004-2010 vs 2024-2025)")
    early = train_df[train_df.index.year <= 2010][feat_cols]
    late  = train_df[train_df.index.year >= 2024][feat_cols]
    drift = pd.DataFrame({
        "mean_2004_2010": early.mean(),
        "mean_2024_2025": late.mean(),
        "abs_delta":      (late.mean() - early.mean()).abs(),
        "pct_delta":      ((late.mean() - early.mean()) / (early.mean().abs() + 1e-9) * 100).round(1),
    }).sort_values("abs_delta", ascending=False)
    print(drift.to_string())
    drift.to_csv(os.path.join(BASE_DIR, "drift_report.csv"))
    print("saved drift_report.csv")
    return drift


def main():
    print(f"training pipeline started  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nloading data ...")
    train_df = pd.read_csv(TRAIN_CSV, index_col=0, parse_dates=True)
    test_df  = pd.read_csv(TEST_CSV,  index_col=0, parse_dates=True)

    feat_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_train   = train_df[feat_cols]
    y_train   = train_df[TARGET_COL]
    X_test    = test_df[feat_cols]
    y_test    = test_df[TARGET_COL]

    print(f"train rows   : {len(train_df)}  ({train_df.index[0].date()} -> {train_df.index[-1].date()})")
    print(f"test rows    : {len(test_df)}  ({test_df.index[0].date()} -> {test_df.index[-1].date()})")
    print(f"features     : {len(feat_cols)}  {feat_cols}")

    # verify expected features are present
    required = {"Bull_Trend", "Macro_Fast"}
    missing  = required - set(feat_cols)
    banned   = {"Market_State"} & set(feat_cols)
    if missing:
        raise ValueError(f"missing required features: {missing}")
    if banned:
        raise ValueError(f"banned features still present: {banned} — rebuild dataset first")

    # ── walk-forward CV ───────────────────────────────────────────────────────
    print(f"\n{N_FOLDS}-fold walk-forward cross-validation")
    tscv           = TimeSeriesSplit(n_splits=N_FOLDS)
    oof_preds      = np.full(len(train_df), np.nan)
    fold_metrics   = []
    importance_dfs = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        val_start = X_val.index[0].date()
        val_end   = X_val.index[-1].date()
        print(f"\nfold {fold}  train={len(X_tr)}  val={len(X_val)}  ({val_start} -> {val_end})")

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        best_iter        = model.best_iteration_
        val_pred         = model.predict(X_val, num_iteration=best_iter)
        oof_preds[val_idx] = val_pred

        m = metrics(y_val.values, val_pred, label=f"fold {fold}")
        m.update(dict(fold=fold, best_iter=best_iter,
                      val_start=str(val_start), val_end=str(val_end)))
        fold_metrics.append(m)

        importance_dfs.append(pd.DataFrame({
            "feature": feat_cols,
            "gain":    model.booster_.feature_importance(importance_type="gain"),
            "fold":    fold,
        }))

    # ── OOF summary ──────────────────────────────────────────────────────────
    valid_mask = ~np.isnan(oof_preds)
    print(f"\noof summary ({valid_mask.sum()} rows)")
    metrics(y_train.values[valid_mask], oof_preds[valid_mask], label="oof overall")

    oof_df = pd.DataFrame({
        "date":           train_df.index,
        "actual":         y_train.values,
        "oof_prediction": oof_preds,
    }).set_index("date")
    oof_df.to_csv(os.path.join(BASE_DIR, "cv_predictions_oof.csv"))
    print(f"saved cv_predictions_oof.csv  ({len(oof_df)} rows)")

    # ── feature importance ────────────────────────────────────────────────────
    print("\nfeature importance (mean gain across folds)")
    imp_all    = pd.concat(importance_dfs)
    imp_stable = (imp_all.groupby("feature")["gain"]
                         .agg(["mean", "std", "min", "max"])
                         .rename(columns={"mean": "mean_gain", "std": "std_gain"})
                         .sort_values("mean_gain", ascending=False))
    imp_stable["cv_stability"] = 1 - (imp_stable["std_gain"] /
                                      (imp_stable["mean_gain"].abs() + 1e-9))
    print(imp_stable.to_string())
    imp_stable.to_csv(os.path.join(BASE_DIR, "cv_importance_stable.csv"))
    print("saved cv_importance_stable.csv")

    # ── fold metrics summary ──────────────────────────────────────────────────
    print("\nfold metrics summary")
    fm_df = pd.DataFrame(fold_metrics)
    print(fm_df[["fold", "val_start", "val_end", "rmse", "ic",
                 "hit_rate", "best_iter"]].to_string(index=False))
    fm_df.to_csv(os.path.join(BASE_DIR, "fold_metrics.csv"), index=False)

    # ── check fold 1 & 2 best_iter ───────────────────────────────────────────
    early_iters = [m["best_iter"] for m in fold_metrics[:2]]
    if any(i < 100 for i in early_iters):
        print(f"\nWARNING: fold 1/2 best_iter still low {early_iters}")
        print("early folds may still be underfitting — consider min_data_in_leaf=30")
    else:
        print(f"\nfold 1/2 best_iter OK: {early_iters}")

    # ── retrain final model on full train set ─────────────────────────────────
    print("\nretraining final model on full training set ...")
    # exclude broken folds (iter < 100) from final iteration estimate
    # folds that stopped at 1-12 carry no signal and distort the median downward
    valid_iters = [m["best_iter"] for m in fold_metrics if m["best_iter"] >= 100]
    broken      = [m["best_iter"] for m in fold_metrics if m["best_iter"] < 100]
    if broken:
        print(f"excluding broken fold iters from median: {broken}")
    final_iter = int(np.median(valid_iters)) if valid_iters else int(np.median([m["best_iter"] for m in fold_metrics]))
    print(f"final model iterations: {final_iter}  (from valid folds: {valid_iters})")

    final_params = {**LGBM_PARAMS, "n_estimators": final_iter}
    final_model  = lgb.LGBMRegressor(**final_params)
    final_model.fit(X_train, y_train,
                    callbacks=[lgb.log_evaluation(period=-1)])

    with open(os.path.join(BASE_DIR, "cv_best_fold_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)
    print("saved cv_best_fold_model.pkl")

    # ── 2026 test evaluation ──────────────────────────────────────────────────
    if len(test_df) > 0:
        print(f"\n2026 test evaluation  ({len(test_df)} rows)")
        test_pred = final_model.predict(X_test)
        metrics(y_test.values, test_pred, label="2026 holdout")

        oof_valid = oof_df["oof_prediction"].dropna()
        test_results = pd.DataFrame({
            "actual_return":    y_test.values,
            "predicted_return": test_pred,
            "pred_z_score":     (test_pred - oof_valid.mean()) / oof_valid.std(),
        }, index=test_df.index)
        test_results.to_csv(os.path.join(BASE_DIR, "test_2026_results.csv"))
        print(f"saved test_2026_results.csv  ({len(test_results)} rows)")
    else:
        print("\nno 2026 test data available yet")

    drift_report(train_df, feat_cols)

    print(f"\npipeline complete  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()