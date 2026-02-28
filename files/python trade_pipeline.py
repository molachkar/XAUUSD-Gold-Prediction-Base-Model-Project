import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# PATHS
# ============================================================
TRAIN_PATH       = "xauusd_train_pruned.csv"
VAL_PATH         = "xauusd_val_pruned.csv"
TEST_PATH        = "xauusd_test_pruned.csv"
DATE_COL         = "Date"
TARGET_COL       = "y_next_log_return"
MARKET_STATE_COL = "Market_State"

# ============================================================
# STRATEGY PARAMETERS
# ============================================================
Z_WINDOW      = 252   # days of history to normalize predictions against
VOL_WINDOW    = 20    # days of history to estimate volatility
Z_THRESHOLD   = 0.6  # signal threshold
CONFIRM_DAYS  = 2    # signal must hold same direction N days before entry
HOLD_DAYS     = 3    # max days to hold a trade before forced exit
                     # model predicts next-day return, not a week
                     # holding longer just bleeds transaction costs
SMOOTH_ALPHA  = 0.6  # position ramp speed
MAX_LEVERAGE  = 1.0
COST_PER_TURN = 0.0005
SEED          = 42

# ============================================================
# REGIME IC WEIGHTS  (measured on val set)
# ============================================================
REGIME_IC = {
    "bear_risk_on"      : 0.17,
    "bear_neutral"      : 0.18,
    "bull_neutral"      : 0.12,
    "sideways_neutral"  : 0.12,
    "bull_risk_on"      : 0.0,
    "bull_risk_off"     : 0.0,
    "bear_risk_off"     : 0.0,
    "sideways_risk_off" : 0.0,
    "sideways_risk_on"  : 0.0,
}
MAX_REGIME_IC = max(v for v in REGIME_IC.values() if v > 0)

# ============================================================
# HELPERS
# ============================================================
def ic(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0: return np.nan
    return float(np.corrcoef(y, p)[0, 1])

def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))

def sharpe(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    if r.size < 2 or np.std(r) == 0: return np.nan
    return float((np.mean(r) / np.std(r)) * np.sqrt(252))

def max_drawdown(eq):
    eq = np.asarray(eq, float)
    return float(np.min(eq / np.maximum.accumulate(eq) - 1.0))

def win_rate(r):
    r = np.asarray(r, float); r = r[~np.isnan(r) & (r != 0)]
    return float((r > 0).mean()) if len(r) > 0 else np.nan

def profit_factor(r):
    r = np.asarray(r, float); r = r[~np.isnan(r) & (r != 0)]
    losses = abs(r[r < 0].sum())
    return float(r[r > 0].sum() / losses) if losses > 0 else np.inf

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def row(label, value, indent=4):
    print(f"{' '*indent}{label:<32} {value}")

def make_xy(df):
    X = df.drop(columns=[DATE_COL, TARGET_COL], errors="ignore").copy()
    y = df[TARGET_COL].astype("float64").copy()
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category")
        elif not np.issubdtype(X[c].dtype, np.number) and str(X[c].dtype) != "category":
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    X.drop(columns=[c for c in X.columns if c.startswith("y_")], inplace=True, errors="ignore")
    return X, y

def align_categories(X_ref, X_other, cat_cols):
    for c in cat_cols:
        X_other[c] = X_other[c].astype("category").cat.set_categories(X_ref[c].cat.categories)
    return X_other

# ============================================================
# LOAD
# ============================================================
section("LOADING DATA")
train = pd.read_csv(TRAIN_PATH)
val   = pd.read_csv(VAL_PATH)
test  = pd.read_csv(TEST_PATH)

for df in (train, val, test):
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="raise")
    df.sort_values(DATE_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

dev = pd.concat([train, val], ignore_index=True).sort_values(DATE_COL).reset_index(drop=True)

X_train, y_train = make_xy(train)
X_val,   y_val   = make_xy(val)
X_dev,   y_dev   = make_xy(dev)
X_test,  y_test  = make_xy(test)

cat_cols = [c for c in X_dev.columns if str(X_dev[c].dtype) == "category"]
X_val    = align_categories(X_train, X_val,  cat_cols)
X_test   = align_categories(X_dev,   X_test, cat_cols)

row("Train rows",    f"{len(train)}   ({train[DATE_COL].min().date()} to {train[DATE_COL].max().date()})")
row("Val rows",      f"{len(val)}    ({val[DATE_COL].min().date()} to {val[DATE_COL].max().date()})")
row("Test rows",     f"{len(test)}    ({test[DATE_COL].min().date()} to {test[DATE_COL].max().date()})")
row("Features",      X_dev.shape[1])

# ============================================================
# STEP 1 — FIND OPTIMAL n_estimators
# ============================================================
section("STEP 1 — FINDING OPTIMAL n_estimators")
print("    Train-only model with early stopping on val...")

probe = lgb.LGBMRegressor(
    n_estimators=50000, learning_rate=0.02, num_leaves=128,
    min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, lambda_l1=1.0, lambda_l2=1.0, max_bin=255,
    random_state=SEED, n_jobs=-1, objective="regression", verbosity=-1,
)
probe.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    categorical_feature=cat_cols if cat_cols else "auto",
    callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(500)],
)

best_iter          = probe.best_iteration_
final_n_estimators = int(best_iter * 1.1)

row("Best iteration (early stop)",  best_iter)
row("Final n_estimators (+10%)",    final_n_estimators)
row("Probe val IC (sanity check)",  round(ic(y_val, probe.predict(X_val)), 4))

# ============================================================
# STEP 2 — TRAIN FINAL MODEL
# ============================================================
section("STEP 2 — TRAINING FINAL MODEL (train + val)")
print(f"    n_estimators={final_n_estimators}  lr=0.02  num_leaves=128")

model = lgb.LGBMRegressor(
    n_estimators=final_n_estimators, learning_rate=0.02, num_leaves=128,
    min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, lambda_l1=1.0, lambda_l2=1.0, max_bin=255,
    random_state=SEED, n_jobs=-1, objective="regression", verbosity=-1,
)
model.fit(X_dev, y_dev, categorical_feature=cat_cols if cat_cols else "auto")
print("    Training complete.")

# ============================================================
# STEP 3 — TEST PREDICTION METRICS
# ============================================================
section("STEP 3 — TEST PREDICTION METRICS")
pred_test = model.predict(X_test)
test_ic   = ic(y_test, pred_test)

row("IC   (want > 0.05)",   round(test_ic,                              4))
row("RMSE",                 round(rmse(y_test, pred_test),              6))
row("MAE",                  round(float(mean_absolute_error(y_test, pred_test)), 6))
row("IC verdict",           "OK — edge exists" if test_ic > 0.05 else "WEAK — marginal edge")

# ============================================================
# STEP 4 — BUILD SIGNALS
# ============================================================
section("STEP 4 — BUILDING SIGNALS")

X_all = align_categories(X_dev, pd.concat([X_dev, X_test], ignore_index=True), cat_cols)
pred_all = model.predict(X_all)

market_state_all = pd.concat(
    [dev[MARKET_STATE_COL], test[MARKET_STATE_COL]], ignore_index=True
) if MARKET_STATE_COL in dev.columns else pd.Series(["unknown"] * len(X_all))

sig_all = pd.DataFrame({
    DATE_COL         : pd.concat([dev[DATE_COL], test[DATE_COL]], ignore_index=True),
    "pred"           : pred_all,
    TARGET_COL       : pd.concat([y_dev, y_test],                 ignore_index=True),
    MARKET_STATE_COL : market_state_all,
})

# Z-SCORE: is today's prediction unusually strong vs last 252 days?
pred_s = sig_all["pred"].shift(1)
mu     = pred_s.rolling(Z_WINDOW, min_periods=50).mean()
sd     = pred_s.rolling(Z_WINDOW, min_periods=50).std(ddof=0)
sig_all["pred_z"] = (sig_all["pred"] - mu) / sd.replace(0, np.nan)

# REGIME WEIGHT: scale by how well the model worked in this regime
sig_all["regime_weight"] = (
    sig_all[MARKET_STATE_COL].map(REGIME_IC).fillna(0.0) / MAX_REGIME_IC
)

# WEIGHTED SIGNAL: confident model + trustworthy regime = trade
sig_all["pred_z_weighted"] = sig_all["pred_z"] * sig_all["regime_weight"]

# RAW SIGNAL: crosses threshold?
sig_all["raw_signal"] = 0
sig_all.loc[sig_all["pred_z_weighted"] >  Z_THRESHOLD, "raw_signal"] =  1
sig_all.loc[sig_all["pred_z_weighted"] < -Z_THRESHOLD, "raw_signal"] = -1

# CONFIRMATION: must hold same direction for CONFIRM_DAYS in a row
# day 1 fires → no trade yet. day 2 same direction → confirmed entry.
confirmed = np.zeros(len(sig_all), dtype=int)
raw = sig_all["raw_signal"].values
for i in range(CONFIRM_DAYS - 1, len(sig_all)):
    window = raw[i - CONFIRM_DAYS + 1 : i + 1]
    if   np.all(window ==  1): confirmed[i] =  1
    elif np.all(window == -1): confirmed[i] = -1
sig_all["signal"] = confirmed

# HOLD DAYS: force exit after HOLD_DAYS in same trade.
# The model predicts next-day return — holding longer than a few days
# just bleeds transaction costs on a signal that has already expired.
# After a forced exit we also require the signal to drop back to 0
# before allowing re-entry — prevents immediately jumping back in on
# the same stale signal cluster.
signal_with_hold = confirmed.copy()
days_held   = 0
in_trade    = False
trade_dir   = 0
forced_exit = False  # True after a hold-limit exit; wait for flat before re-entry

for i in range(len(signal_with_hold)):
    s = confirmed[i]

    if forced_exit:
        # block re-entry until signal goes flat
        signal_with_hold[i] = 0
        if s == 0:
            forced_exit = False
        continue

    if not in_trade:
        if s != 0:
            in_trade  = True
            trade_dir = s
            days_held = 1
    else:
        if days_held >= HOLD_DAYS:
            # max hold reached — flat and wait for fresh signal
            signal_with_hold[i] = 0
            in_trade    = False
            trade_dir   = 0
            days_held   = 0
            forced_exit = True
        elif s == trade_dir:
            days_held += 1
        else:
            # signal flipped or dropped — exit naturally
            signal_with_hold[i] = 0
            in_trade  = False
            trade_dir = 0
            days_held = 0

sig_all["signal"] = signal_with_hold

# POSITION SIZING: inverse vol-scaled + exponential smoothing
sig_all["vol"] = sig_all[TARGET_COL].shift(1).rolling(VOL_WINDOW, min_periods=10).std(ddof=0)
eps = 1e-12
sig_all["pos_raw"] = (sig_all["signal"] / (sig_all["vol"] + eps)).clip(-MAX_LEVERAGE, MAX_LEVERAGE).fillna(0.0)

pos = np.zeros(len(sig_all), dtype=float)
for i in range(1, len(sig_all)):
    pos[i] = (1.0 - SMOOTH_ALPHA) * pos[i-1] + SMOOTH_ALPHA * sig_all["pos_raw"].iloc[i]
sig_all["position"] = pos

n_raw       = int((sig_all["raw_signal"] != 0).sum())
n_confirmed = int((confirmed             != 0).sum())
n_held      = int((sig_all["signal"]     != 0).sum())
row("Raw signals (pre-confirmation)",        n_raw)
row(f"Confirmed signals ({CONFIRM_DAYS}-day filter)",  n_confirmed)
row(f"Active trade days ({HOLD_DAYS}-day hold limit)", n_held)
row("Days cut by hold limit",                n_confirmed - n_held)

# ============================================================
# STEP 5 — STRATEGY METRICS
# ============================================================
section("STEP 5 — STRATEGY METRICS (test period only)")

sig = sig_all.iloc[len(dev):].copy().reset_index(drop=True)
sig["position_lag"] = sig["position"].shift(1).fillna(0.0)
sig["turnover"]     = (sig["position"] - sig["position"].shift(1)).abs().fillna(0.0)
sig["cost"]         = sig["turnover"] * COST_PER_TURN
sig["strategy_ret"] = sig["position_lag"] * sig[TARGET_COL] - sig["cost"]
sig["equity"]       = (1.0 + sig["strategy_ret"].fillna(0.0)).cumprod()

active  = sig[sig["signal"] != 0]
n_long  = int((sig["signal"] ==  1).sum())
n_short = int((sig["signal"] == -1).sum())

sp  = sharpe(sig["strategy_ret"])
mdd = max_drawdown(sig["equity"])
feq = float(sig["equity"].iloc[-1])
wr  = win_rate(active["strategy_ret"])      if len(active) > 0 else np.nan
pf  = profit_factor(active["strategy_ret"]) if len(active) > 0 else np.nan
td  = float((sig["signal"] != 0).mean() * 100)

row("Sharpe        (want > 0.8)",   round(sp,  3))
row("Max Drawdown  (want > -0.15)", round(mdd, 3))
row("Final Equity  (start = 1.0)",  round(feq, 4))
row("Win Rate      (want > 0.45)",  round(wr,  3) if not np.isnan(wr) else "n/a")
row("Profit Factor (want > 1.2)",   round(pf,  3) if not np.isnan(pf) else "n/a")
row("Trade days %",                 f"{round(td, 2)}%")
row("Avg Turnover per day",         round(float(sig['turnover'].mean()), 4))
row("Long signals",                 n_long)
row("Short signals",                n_short)
row("Total active trade days",      len(active))

print()
if sp > 0.8 and (wr > 0.45 or np.isnan(wr)) and (pf > 1.2 or np.isnan(pf)):
    verdict = "PROFITABLE — passes all targets"
elif sp > 0 and (pf > 1.0 or np.isnan(pf)):
    verdict = "MARGINALLY POSITIVE — needs work before live trading"
else:
    verdict = "NOT PROFITABLE — do not trade live at these settings"
row("VERDICT", verdict)

# ============================================================
# STEP 6 — REGIME BREAKDOWN
# ============================================================
section("STEP 6 — REGIME BREAKDOWN (test, active trades only)")
print(f"    {'Regime':<25} {'N':>4}  {'IC':>7}  {'WinRate':>7}  {'AvgRet':>9}  {'ProfitFactor':>12}")
print(f"    {'-'*25} {'-'*4}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*12}")
for state, grp in sig.groupby(MARKET_STATE_COL):
    ag = grp[grp["signal"] != 0]
    if len(ag) < 2: continue
    print(f"    {state:<25} {len(ag):>4}  "
          f"{ic(ag[TARGET_COL], ag['pred']):>7.4f}  "
          f"{win_rate(ag['strategy_ret']):>7.3f}  "
          f"{ag['strategy_ret'].mean():>9.5f}  "
          f"{profit_factor(ag['strategy_ret']):>12.3f}")

# ============================================================
# STEP 7 — FEATURE IMPORTANCE
# ============================================================
section("STEP 7 — FEATURE IMPORTANCE (gain)")
fi          = pd.Series(model.feature_importances_, index=X_dev.columns).sort_values(ascending=False)
total_gain  = fi.sum()
print(f"    {'Feature':<30} {'Gain':>8}   {'Share':>7}")
print(f"    {'-'*30} {'-'*8}   {'-'*7}")
for feat, gain in fi.head(10).items():
    bar = "█" * int(gain / total_gain * 40)
    print(f"    {feat:<30} {gain:>8.0f}   {gain/total_gain*100:>6.1f}%  {bar}")

# ============================================================
# STEP 8 — TRADE LOG
# ============================================================
section("STEP 8 — EVERY CONFIRMED TRADE (test period)")
print(f"    {'Date':<12} {'Regime':<22} {'pred_z':>7}  {'Wt':>5}  {'Day':>4}  {'Dir':<6}  {'GoldRet':>8}  {'StratRet':>9}  {'W/L':>4}")
print(f"    {'-'*12} {'-'*22} {'-'*7}  {'-'*5}  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*4}")

# reconstruct hold-day counter for display
day_counter = 0
prev_in_trade = False
hold_day_col = []
for _, r in sig.iterrows():
    if r["signal"] != 0:
        if not prev_in_trade:
            day_counter = 1
        else:
            day_counter += 1
        prev_in_trade = True
    else:
        day_counter = 0
        prev_in_trade = False
    hold_day_col.append(day_counter)
sig["hold_day"] = hold_day_col

for _, r in sig[sig["signal"] != 0].iterrows():
    d  = "LONG " if r["signal"] == 1 else "SHORT"
    wl = "WIN " if r["strategy_ret"] > 0 else "LOSS"
    print(f"    {str(r[DATE_COL])[:10]:<12} "
          f"{r[MARKET_STATE_COL]:<22} "
          f"{r['pred_z']:>7.3f}  "
          f"{r['regime_weight']:>5.3f}  "
          f"{int(r['hold_day']):>4}  "
          f"{d:<6}  "
          f"{r[TARGET_COL]:>8.4f}  "
          f"{r['strategy_ret']:>9.5f}  "
          f"{wl:>4}")

# ============================================================
# STEP 9 — SAVE
# ============================================================
section("STEP 9 — SAVING OUTPUT")
out = sig[[DATE_COL, MARKET_STATE_COL, "pred_z", "regime_weight",
           "pred_z_weighted", "raw_signal", "signal", "hold_day",
           "position", TARGET_COL, "strategy_ret", "equity"]].copy()
out.to_csv("signals_for_metatrader.csv", index=False)
row("File saved",  "signals_for_metatrader.csv")
row("Rows",        len(out))
row("Columns",     "Date, Market_State, pred_z, regime_weight,")
row("",            "pred_z_weighted, raw_signal, signal, hold_day,")
row("",            "position, actual_return, strategy_ret, equity")
print("\n    Done.")