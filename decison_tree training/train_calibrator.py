# train_calibrator.py
import os
import sys
import json
# train_calibrator.py
import os, sys, json, pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier

# ---------------- CONFIG ----------------
INPUT_CSV = "minimal oof.csv"

OUT_TRAIN = "train_calib.csv"
OUT_VAL   = "val_calib.csv"
OUT_TEST  = "test_calib.csv"

MODEL_OUT  = "calibrator.pkl"
REPORT_OUT = "train_report.json"

DATE_COL   = "Date"
TARGET_COL = "target_up"

# small robust feature set (edit only if you know why)
NUM_COLS = ["oof_prediction", "pred_z", "abs_pred_z", "Macro_Fast"]
CAT_COLS = ["Market_State"]

# time split (edit if needed)
TRAIN_END  = "2020-12-31"
VAL_START  = "2021-01-01"
VAL_END    = "2022-12-31"
TEST_START = "2023-01-01"

# threshold search (on VAL only)
P_LONG_GRID  = np.arange(0.55, 0.76, 0.01)   # LONG if p_up >= p_long
P_SHORT_GRID = np.arange(0.45, 0.24, -0.01)  # SHORT if p_up <= p_short
MAX_TRADE_RATE = 0.30                         # enforce "not always trade"

COST_BPS = 5  # cost per position change (enter/exit/flip), adjust to your reality
# ----------------------------------------


def require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

def save_split(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def exists_and_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def policy_backtest(df: pd.DataFrame, p_up: np.ndarray, p_long: float, p_short: float, cost_bps: float):
    """
    1-day holding policy:
      LONG  if p_up >= p_long
      SHORT if p_up <= p_short
      else FLAT ("WAIT")

    Returns hit_rate ONLY on traded days.
    """
    yret = df["y_next_log_return"].to_numpy(dtype=float)

    pos = np.zeros_like(p_up, dtype=float)
    pos[p_up >= p_long] = 1.0
    pos[p_up <= p_short] = -1.0

    traded = pos != 0
    n_trades = int(traded.sum())
    trade_rate = float(traded.mean())

    if n_trades == 0:
        return None

    # cost when changing position (enter/exit/flip)
    turn = (np.abs(np.diff(pos, prepend=0.0)) > 0).astype(float)
    cost = turn * (cost_bps / 1e4)

    pnl = pos * yret - cost

    # hit rate only on traded days
    hit_rate = float((pos[traded] * yret[traded] > 0).mean())

    return {
        "p_long": float(p_long),
        "p_short": float(p_short),
        "trades": n_trades,
        "trade_rate": trade_rate,
        "mean_pnl_all_days": float(pnl.mean()),
        "mean_pnl_traded_days": float(pnl[traded].mean()),
        "sum_pnl": float(pnl.sum()),
        "hit_rate_traded_days": hit_rate,
    }


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: input file not found: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    require_cols(df, [DATE_COL, TARGET_COL, "y_next_log_return"] + NUM_COLS + CAT_COLS)

    # time splits
    train = df[df[DATE_COL] <= pd.to_datetime(TRAIN_END)].copy()
    val   = df[(df[DATE_COL] >= pd.to_datetime(VAL_START)) & (df[DATE_COL] <= pd.to_datetime(VAL_END))].copy()
    test  = df[df[DATE_COL] >= pd.to_datetime(TEST_START)].copy()

    # save splits
    save_split(train, OUT_TRAIN)
    save_split(val, OUT_VAL)
    save_split(test, OUT_TEST)

    if not (exists_and_nonempty(OUT_TRAIN) and exists_and_nonempty(OUT_VAL) and exists_and_nonempty(OUT_TEST)):
        print("ERROR: split files were not created correctly. Aborting.")
        sys.exit(2)

    # preprocessing
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ],
        remainder="drop",
    )

    # classifier (strong, but regularized)
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=1500,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=7,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, y_train = train[NUM_COLS + CAT_COLS], train[TARGET_COL].astype(int).to_numpy()
    X_val, y_val     = val[NUM_COLS + CAT_COLS],   val[TARGET_COL].astype(int).to_numpy()
    X_test, y_test   = test[NUM_COLS + CAT_COLS],  test[TARGET_COL].astype(int).to_numpy()

    pipe.fit(X_train, y_train)

    p_train = pipe.predict_proba(X_train)[:, 1]
    p_val   = pipe.predict_proba(X_val)[:, 1]
    p_test  = pipe.predict_proba(X_test)[:, 1]

    report = {
        "splits": {
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "train_range": [str(train[DATE_COL].min().date()), str(train[DATE_COL].max().date())] if len(train) else None,
            "val_range": [str(val[DATE_COL].min().date()), str(val[DATE_COL].max().date())] if len(val) else None,
            "test_range": [str(test[DATE_COL].min().date()), str(test[DATE_COL].max().date())] if len(test) else None,
        },
        "features": {"num": NUM_COLS, "cat": CAT_COLS},
        "model": "HistGradientBoostingClassifier + OneHot Market_State",
        "metrics": {
            "train_auc": float(roc_auc_score(y_train, p_train)) if len(np.unique(y_train)) > 1 else None,
            "val_auc": float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else None,
            "test_auc": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else None,
            "train_logloss": float(log_loss(y_train, p_train, labels=[0,1])),
            "val_logloss": float(log_loss(y_val, p_val, labels=[0,1])),
            "test_logloss": float(log_loss(y_test, p_test, labels=[0,1])),
        },
        "policy_search": {
            "cost_bps": COST_BPS,
            "max_trade_rate": MAX_TRADE_RATE,
            "best_on_val": None,
            "test_with_best": None,
        }
    }

    # threshold search on validation
    best = None
    for p_long in P_LONG_GRID:
        for p_short in P_SHORT_GRID:
            if p_short >= p_long:
                continue
            res = policy_backtest(val, p_val, p_long, p_short, COST_BPS)
            if res is None:
                continue
            if res["trade_rate"] > MAX_TRADE_RATE:
                continue
            if (best is None) or (res["mean_pnl_all_days"] > best["mean_pnl_all_days"]):
                best = res

    report["policy_search"]["best_on_val"] = best
    if best is not None:
        report["policy_search"]["test_with_best"] = policy_backtest(test, p_test, best["p_long"], best["p_short"], COST_BPS)

    # save
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(pipe, f)

    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)

    print("OK")
    print(f"Saved splits: {OUT_TRAIN}, {OUT_VAL}, {OUT_TEST}")
    print(f"Saved model: {MODEL_OUT}")
    print(f"Saved report: {REPORT_OUT}")
    print("Val best policy:", report["policy_search"]["best_on_val"])
    print("Test policy:", report["policy_search"]["test_with_best"])


if __name__ == "__main__":
    main()