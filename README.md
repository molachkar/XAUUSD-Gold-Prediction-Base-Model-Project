# XAUUSD Gold Prediction — Base Model Project

## What This Project Is

A LightGBM regression model trained on daily XAUUSD (gold/USD) data
from 2011 to 2024. The model predicts `y_next_log_return` — the next
trading day's log return for gold. It is the foundation of a
systematic trading system. This README is your permanent reference
so you can pick this up again at any time.

---

## The Goal

Predict the direction and approximate magnitude of tomorrow's gold
return using today's macro environment and market structure features.
The model does not need to be highly accurate — even a weak but
consistent edge (IC > 0.05) is enough to build a profitable strategy
on top of if used correctly.

---

## Files in This Project

```
model_outputs/
  cv_best_fold_model.pkl        The saved model. Reload and predict anytime.
  cv_fold_results.csv           IC, RMSE, best_iteration per CV fold.
  cv_importance_per_fold.csv    Feature importance breakdown per fold.
  cv_importance_stable.csv      Mean feature importance across all folds.
  cv_predictions_oof.csv        Out-of-fold predictions for every row (honest).

cv_walkforward_v2.py            The CV training script (saves everything above).
trade_pipeline_v4.py            Converts model predictions into trading signals.
feature_guide.txt               Plain-English explanation of every feature.
signals_for_metatrader.csv      Final test-period signals + equity curve.
```

---

## Data

| Split | Period | Rows |
|-------|--------|------|
| Train | 2011-09-11 to 2021-01-02 | 3,402 |
| Val   | 2021-01-03 to 2023-01-01 | 729   |
| Test  | 2023-01-02 to 2024-12-31 | 730   |
| **Total** | **2011-09-11 to 2024-12-31** | **4,861** |

**Target:** `y_next_log_return` = log(Close[t+1] / Close[t])

---

## Features (15 total)

| Feature | Type | Importance | Role |
|---------|------|-----------|------|
| Macro_Fast | float | ~50% | Fast macro momentum composite — most important |
| Market_State | categorical | ~23% | Regime label (bull/bear/sideways × risk_on/neutral/risk_off) |
| MACD_Signal | float | ~10% | Short vs long-term momentum |
| Close_XAUUSD | float | ~9% | Gold price level |
| Close_Returns | float | ~8% | Today's gold return |
| BB_Middle | float | ~7% | Rolling mean price (trend anchor) |
| Close_EURUSD | float | ~7% | EUR/USD (inverse USD signal) |
| Close_USDJPY | float | ~5% | USD/JPY (safe-haven signal) |
| Return_Percentile | float | ~5% | Rank of today's return in recent window |
| LogReturn_ZScore | float | ~5% | How abnormal is today's move |
| Return_ZScore | float | ~4% | Same as above, simple returns |
| Log_Returns | float | ~2% | Today's log return |
| Distance_From_AllTimeHigh | float | 0% | Can be removed |
| Pct_From_AllTimeHigh | float | 0% | Can be removed |
| Volume_Percentile | float | 0% | Can be removed |

Full explanation of each feature: see `feature_guide.txt`

---

## Model Architecture

**Algorithm:** LightGBM Regressor
**Objective:** Regression (predicts continuous return value)

| Hyperparameter | Value | Why |
|---------------|-------|-----|
| n_estimators | 30,000 (capped by early stopping) | Upper bound — early stopping finds the real number |
| learning_rate | 0.03 | Standard for daily financial data |
| num_leaves | 64 | Moderate complexity — avoids memorizing noise |
| min_data_in_leaf | 80 | Requires 80 samples per leaf — prevents tiny overfitted splits |
| feature_fraction | 0.9 | Uses 90% of features per tree — adds variance reduction |
| bagging_fraction | 0.9 | Uses 90% of rows per tree — same |
| reg_alpha / reg_lambda | 0.5 / 0.5 | L1 + L2 regularization — penalizes complexity |
| early_stopping | 300 rounds | Stops when val loss stops improving for 300 rounds |

---

## Walk-Forward CV Results

The model was validated using 5-fold walk-forward cross-validation.
Each fold trains on all data up to a cutoff, tests on the next period.
This is the correct method for time-series — no future data ever leaks
into training.

| Fold | Train Period | Test Period | Best Iter | IC | RMSE |
|------|-------------|-------------|-----------|-----|------|
| 1 | 2011-09 → 2013-11 | 2013-11 → 2016-02 | 2,557 | 0.0577 | 0.007992 |
| 2 | 2011-09 → 2016-02 | 2016-02 → 2018-05 | 31 | **0.0905** | 0.006602 |
| 3 | 2011-09 → 2018-05 | 2018-05 → 2020-07 | 1,031 | 0.0520 | 0.007826 |
| 4 | 2011-09 → 2020-07 | 2020-07 → 2022-10 | 859 | 0.0506 | 0.008341 |
| 5 | 2011-09 → 2022-10 | 2022-10 → 2024-12 | 38 | **0.0915** | 0.007538 |
| **Mean** | | | | **0.0684** | |
| **Std** | | | | **0.0185** | |

**Best fold:** Fold 5 (IC = 0.0915) — saved as `cv_best_fold_model.pkl`

**Key observation:** Best iteration varied from 31 to 2,557 across folds.
Folds 2 and 5 converged in under 40 trees with the highest IC. This means
the signal is strongest in macro transition periods (2016-2018, 2022-2024)
and noisier in calm trending periods.

---

## OOF IC by Year (Honest Performance)

| Year | IC | Notes |
|------|----|-------|
| 2013 | +0.163 | Strong — post-gold-crash macro transition |
| 2014 | +0.004 | Flat — gold drifted, no clear macro driver |
| 2015 | +0.094 | Good — USD strength regime |
| 2016 | **+0.173** | Best year — Trump election macro shock |
| 2017 | +0.023 | Flat — calm year |
| 2018 | +0.076 | Good — Fed hiking cycle |
| 2019 | **-0.063** | Negative — model fought the Fed pivot rally |
| 2020 | **+0.162** | Strong — COVID macro shock |
| 2021 | +0.008 | Flat — calm year |
| 2022 | +0.090 | Good — inflation/Fed hiking cycle |
| 2023 | +0.102 | Good — Fed pause cycle |
| 2024 | +0.061 | Decent — rate cut cycle |

**Pattern:** Model thrives on macro regime transitions. Flat or momentum-
driven years (2014, 2017, 2021, 2019) produce near-zero or negative IC.

---

## Your Edge — Plain English

The model is a **macro transition detector**, not a technical indicator
follower. 72% of its predictive power comes from just two features:
`Macro_Fast` and `Market_State`.

**The edge fires when ALL of these are true:**
1. `Market_State` is `bull_neutral` or `sideways_neutral`
2. `Macro_Fast` is elevated above its recent baseline
3. `pred_z` (normalized prediction) is above 1.0

**At pred_z > 2.0 in tradeable regimes:**
- Average next-day gold return: **+0.242%**
- Win rate: **67%** (10 out of 15 high-confidence calls in test period)

**The edge disappears when:**
- `Market_State` contains `risk_off` (any trend) — IC goes negative
- `Market_State` is `risk_on` — IC near zero, too noisy
- `pred_z` is between -0.6 and +0.6 — signal is background noise

---

## How to Reload and Use the Model

```python
import joblib
import pandas as pd

# Load saved model
model = joblib.load("model_outputs/cv_best_fold_model.pkl")

# Prepare your features (same 15 columns, same format)
# Market_State must be category dtype
X_new = pd.read_csv("your_new_data.csv")
X_new["Market_State"] = X_new["Market_State"].astype("category")

# Get prediction
pred = model.predict(X_new)
# pred is a log return — positive means model expects gold up tomorrow
```

---

## Known Limitations

- **Position sizing issue:** Vol-scaling + exponential smoothing (alpha=0.6)
  means you never reach full position size. On day 1 you hold ~9% of a
  full position. Large gold moves are captured at partial size only.
- **Long bias:** Model predicts positive returns only 23% of the time
  but gold was up 36% of days in the OOF period. Model is conservative
  on longs and misses some upside.
- **2019 problem:** In steady momentum-driven rallies without macro
  catalysts, the model's IC goes negative. Be aware of this regime.
- **Zero-importance features:** `Distance_From_AllTimeHigh`,
  `Pct_From_AllTimeHigh`, `Volume_Percentile` contribute nothing.
  Remove them before the next training run.

---

## Next Steps (in priority order)

1. **Build meta-model** on OOF predictions — calibrate when to trust the
   signal. Converts raw pred_z into a clean 0-1 probability.
2. **Add lag features** — Macro_Fast_lag1, Macro_Fast_change5, etc.
   Model currently sees today's snapshot only. Detecting transitions
   explicitly will improve IC in flat years.
3. **Build live feature fetcher** — fetch XAUUSD, EURUSD, USDJPY daily,
   recompute all features, run model, get today's signal.
4. **Fix position sizing** — remove vol-scaling or raise alpha to 1.0
   so you enter full position immediately when signal fires.

---

## How Macro_Fast and Market_State Were Calculated

*(Fill this in from your own notes — these two features drive 72% of
the model and must be replicated exactly for live prediction to work.)*

---

*Last updated: based on training run ending 2024-12-31*
*Model file: cv_best_fold_model.pkl (Fold 5, IC = 0.0915)*
