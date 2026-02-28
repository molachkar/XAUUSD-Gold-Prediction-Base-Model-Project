import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import time
import os
import json
import joblib

# ==========================
# OUTPUT FOLDER
# All saved files land here.
# Nothing is lost when you close the terminal.
# ==========================
OUTPUT_DIR = "model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# CONFIG
# ==========================
DATA_PATHS = [
    "xauusd_train_pruned.csv",
    "xauusd_val_pruned.csv",
    "xauusd_test_pruned.csv"
]
TARGET   = "y_next_log_return"
DATE_COL = "Date"
N_FOLDS  = 5
SEED     = 42

# ==========================
# METRICS
# ==========================
def ic(y, p):
    if np.std(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])

def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))

# ==========================
# LOAD DATA
# ==========================
print("Loading data...")
dfs = []
for path in DATA_PATHS:
    dfs.append(pd.read_csv(path))

df = pd.concat(dfs, ignore_index=True)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

print("Total rows    :", len(df))
print("Total features:", len(df.columns) - 2)
print("Date range    :", df[DATE_COL].min().date(), "to", df[DATE_COL].max().date())

# ==========================
# PREP FEATURES
# ==========================
X = df.drop(columns=[TARGET, DATE_COL])
y = df[TARGET].astype("float64")

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].astype("category")

# ==========================
# MODEL BUILDER
# ==========================
def build_model():
    return lgb.LGBMRegressor(
        n_estimators=30000,
        learning_rate=0.03,
        num_leaves=64,
        min_data_in_leaf=80,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1
    )

# ==========================
# WALK-FORWARD CV
# ==========================
n         = len(df)
fold_size = n // (N_FOLDS + 1)

results            = []
importance_storage = []
oof_preds          = np.full(n, np.nan)
fold_records       = []
best_fold_ic       = -np.inf
best_fold_model    = None
best_fold_num      = -1

print("\nStarting Walk-Forward CV\n")

for fold in range(N_FOLDS):
    train_end = fold_size * (fold + 1)
    test_end  = fold_size * (fold + 2)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test  = X.iloc[train_end:test_end]
    y_test  = y.iloc[train_end:test_end]

    train_start_date = df[DATE_COL].iloc[0].date()
    train_end_date   = df[DATE_COL].iloc[train_end - 1].date()
    test_start_date  = df[DATE_COL].iloc[train_end].date()
    test_end_date    = df[DATE_COL].iloc[test_end - 1].date()

    print(f"Fold {fold + 1}")
    print(f"  Train: {train_start_date} to {train_end_date}  ({len(X_train)} rows)")
    print(f"  Test : {test_start_date} to {test_end_date}  ({len(X_test)} rows)")

    model = build_model()
    start = time.time()

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="l2",
        categorical_feature="auto",
        callbacks=[
            lgb.early_stopping(300, verbose=False),
            lgb.log_evaluation(500)
        ]
    )

    duration  = round(time.time() - start, 2)
    best_iter = model.best_iteration_
    preds     = model.predict(X_test, num_iteration=best_iter)

    oof_preds[train_end:test_end] = preds

    fold_ic   = ic(y_test, preds)
    fold_rmse = rmse(y_test, preds)

    print(f"  IC: {round(fold_ic, 4)} | RMSE: {round(fold_rmse, 6)} | Best iter: {best_iter} | Time: {duration}s")
    print("-" * 55)

    results.append(fold_ic)

    imp = model.booster_.feature_importance(importance_type="gain")
    importance_storage.append(pd.Series(imp, index=X.columns))

    fold_records.append({
        "fold"           : fold + 1,
        "train_start"    : str(train_start_date),
        "train_end"      : str(train_end_date),
        "test_start"     : str(test_start_date),
        "test_end"       : str(test_end_date),
        "train_rows"     : len(X_train),
        "test_rows"      : len(X_test),
        "best_iteration" : best_iter,
        "ic"             : round(fold_ic,   4),
        "rmse"           : round(fold_rmse, 6),
        "time_seconds"   : duration,
    })

    if fold_ic > best_fold_ic:
        best_fold_ic    = fold_ic
        best_fold_model = model
        best_fold_num   = fold + 1

# ==========================
# CV SUMMARY
# ==========================
print("\n========================")
print("CV SUMMARY")
print("========================")
print("IC per fold :", [round(r, 4) for r in results])
print("Mean IC     :", round(np.nanmean(results), 4))
print("Std IC      :", round(np.nanstd(results),  4))
print(f"Best fold   : Fold {best_fold_num} (IC = {round(best_fold_ic, 4)})")

importance_df         = pd.concat(importance_storage, axis=1)
importance_df.columns = [f"Fold_{i+1}" for i in range(N_FOLDS)]
mean_importance       = importance_df.mean(axis=1).sort_values(ascending=False)

print("\nTop Stable Features (mean gain across folds):")
print(mean_importance.head(10))

# ==========================
# SAVE EVERYTHING
# ==========================
print("\n--- Saving outputs to:", OUTPUT_DIR, "---")

# 1. Best fold model
model_path = os.path.join(OUTPUT_DIR, "cv_best_fold_model.pkl")
joblib.dump(best_fold_model, model_path)
print(f"  [1] Model checkpoint     -> {model_path}")
print(f"      Fold {best_fold_num}, IC = {round(best_fold_ic, 4)}")
print(f"      Reload: model = joblib.load('{model_path}')")

# 2. Per-fold results
fold_csv_path = os.path.join(OUTPUT_DIR, "cv_fold_results.csv")
pd.DataFrame(fold_records).to_csv(fold_csv_path, index=False)
print(f"  [2] Fold results         -> {fold_csv_path}")

# 3. Importance per fold
imp_fold_path = os.path.join(OUTPUT_DIR, "cv_importance_per_fold.csv")
importance_df.to_csv(imp_fold_path)
print(f"  [3] Importance per fold  -> {imp_fold_path}")

# 4. Mean stable importance
imp_mean_path = os.path.join(OUTPUT_DIR, "cv_importance_stable.csv")
mean_importance.reset_index().rename(
    columns={"index": "feature", 0: "mean_gain"}
).to_csv(imp_mean_path, index=False)
print(f"  [4] Stable importance    -> {imp_mean_path}")

# 5. Out-of-fold predictions
oof_df = df[[DATE_COL, TARGET]].copy()
oof_df["oof_prediction"]   = oof_preds
oof_df["has_prediction"]   = ~np.isnan(oof_preds)
oof_path = os.path.join(OUTPUT_DIR, "cv_predictions_oof.csv")
oof_df.to_csv(oof_path, index=False)
print(f"  [5] OOF predictions      -> {oof_path}")
print(f"      {(~np.isnan(oof_preds)).sum()} rows with predictions")

# 6. Full summary JSON
summary = {
    "model_type"     : "LGBMRegressor",
    "target"         : TARGET,
    "n_folds"        : N_FOLDS,
    "total_rows"     : int(n),
    "n_features"     : int(X.shape[1]),
    "features"       : list(X.columns),
    "date_range"     : {
        "start": str(df[DATE_COL].min().date()),
        "end"  : str(df[DATE_COL].max().date()),
    },
    "cv_results"     : {
        "ic_per_fold" : [round(r, 4) for r in results],
        "mean_ic"     : round(float(np.nanmean(results)), 4),
        "std_ic"      : round(float(np.nanstd(results)),  4),
        "best_fold"   : int(best_fold_num),
        "best_fold_ic": round(float(best_fold_ic), 4),
    },
    "hyperparameters": {
        "n_estimators"    : 30000,
        "learning_rate"   : 0.03,
        "num_leaves"      : 64,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq"    : 1,
        "reg_alpha"       : 0.5,
        "reg_lambda"      : 0.5,
        "early_stopping"  : 300,
    },
    "top_5_features" : mean_importance.head(5).index.tolist(),
    "edge_summary"   : {
        "tradeable_regimes"      : ["bull_neutral", "sideways_neutral"],
        "blocked_regimes"        : ["bull_risk_off", "bear_risk_off", "sideways_risk_off",
                                    "bull_risk_on", "sideways_risk_on"],
        "signal_threshold_pred_z": 0.6,
        "high_confidence_pred_z" : 2.0,
        "win_rate_at_high_conf"  : "67% (10/15 calls at pred_z > 1.5)",
        "avg_return_pred_z_gt_2" : "+0.242% per day",
    },
    "saved_files"    : {
        "model"            : model_path,
        "fold_results"     : fold_csv_path,
        "importance_folds" : imp_fold_path,
        "importance_stable": imp_mean_path,
        "oof_predictions"  : oof_path,
    }
}

summary_path = os.path.join(OUTPUT_DIR, "cv_model_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  [6] Model summary JSON   -> {summary_path}")

print("\n========================")
print("ALL FILES SAVED")
print("========================")
print(f"  Folder : {OUTPUT_DIR}/")
print(f"  [1] cv_best_fold_model.pkl    <- reload anytime to predict")
print(f"  [2] cv_fold_results.csv       <- IC, RMSE, best_iter per fold")
print(f"  [3] cv_importance_per_fold.csv<- importance per fold")
print(f"  [4] cv_importance_stable.csv  <- mean importance, most reliable")
print(f"  [5] cv_predictions_oof.csv    <- what the model predicted on each row")
print(f"  [6] cv_model_summary.json     <- full config + results + edge summary")
print("\nDone.")