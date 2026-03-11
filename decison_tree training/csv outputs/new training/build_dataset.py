import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from datetime import date

warnings.filterwarnings("ignore")

FRED_API_KEY  = "219d0c44b2e3b4a8b690c3f69b91a5bb"
BUFFER_START  = "2003-01-01"
TARGET_START  = "2004-01-01"
TRAIN_END     = "2025-12-31"
TEST_START    = "2026-01-01"
TODAY         = date.today().strftime("%Y-%m-%d")
OUTPUT_DIR    = os.path.dirname(os.path.abspath(__file__))
MACRO_SERIES  = ["DFII10", "DFII5", "DGS2", "FEDFUNDS"]


def fetch_single(ticker, start, end, retries=3, chunk_years=5):
    """download in yearly chunks with retries to avoid timeout on long ranges."""
    import time
    from datetime import datetime

    chunks = []
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)

    while s < e:
        chunk_end = min(s + pd.DateOffset(years=chunk_years), e)
        for attempt in range(retries):
            try:
                df = yf.download(ticker,
                                 start=s.strftime("%Y-%m-%d"),
                                 end=chunk_end.strftime("%Y-%m-%d"),
                                 auto_adjust=False, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                if not df.empty:
                    chunks.append(df)
                break
            except Exception as ex:
                print(f"  {ticker} {s.date()}->{chunk_end.date()} attempt {attempt+1} failed: {ex}")
                time.sleep(2)
        s = chunk_end

    if not chunks:
        return pd.DataFrame()
    result = pd.concat(chunks)
    result = result[~result.index.duplicated(keep="first")].sort_index()
    return result


def fetch_fred(start, end):
    """
    Try FRED API first. If network fails, fall back to local CSV files.
    Download CSVs manually from:
      https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10
      https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII5
      https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2
      https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS
    Save them in the same folder as this script.
    """
    import time
    series = {}
    for s in MACRO_SERIES:
        # try API first
        try:
            fred = Fred(api_key=FRED_API_KEY)
            data = fred.get_series(s, start, end)
            series[s] = data
            print(f"  {s} fetched from API ({len(data)} rows)")
            continue
        except Exception:
            pass
        # fall back to local CSV
        local = os.path.join(OUTPUT_DIR, f"{s}.csv")
        if os.path.exists(local):
            df = pd.read_csv(local, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            col = df.columns[0]
            data = df[col].replace(".", np.nan).astype(float)
            data = data[(data.index >= start) & (data.index <= end)]
            series[s] = data
            print(f"  {s} loaded from local CSV ({len(data)} rows)")
        else:
            raise FileNotFoundError(
                f"FRED API unreachable and no local file found for {s}. "
                f"Download from: https://fred.stlouisfed.org/graph/fredgraph.csv?id={s} "
                f"and save as {local}"
            )
    macro = pd.DataFrame(series)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    return macro


def fetch_all(start, end):
    print("fetching gold ...")
    gold = fetch_single("GC=F", start, end)
    if gold.empty:
        raise ValueError("GC=F returned no data — try start=2003-01-01 or later")

    print("fetching eurusd ...")
    eur = fetch_single("EURUSD=X", start, end)

    print("fetching usdjpy ...")
    jpy = fetch_single("JPY=X", start, end)

    print("fetching fred macro ...")
    macro = fetch_fred(start, end)

    # build on gold trading day index
    prices = pd.DataFrame({
        "Close_XAUUSD":  gold["Close"],
        "Volume_XAUUSD": gold["Volume"],
        "Close_EURUSD":  eur["Close"],
        "Close_USDJPY":  jpy["Close"],
    })

    # reindex all series to a complete calendar of weekdays
    # this catches any missing trading days in the raw feed
    full_idx = pd.date_range(start=prices.index.min(),
                             end=prices.index.max(), freq="B")
    prices = prices.reindex(full_idx)
    macro  = macro.reindex(full_idx)

    df = prices.join(macro, how="left")

    # forward fill everything — handles FRED lags, market holidays,
    # missing settlement days, and any yfinance gaps
    df = df.ffill()

    # backward fill for any leading NaNs at the very start
    df = df.bfill()

    df.dropna(subset=["Close_XAUUSD"], inplace=True)
    df.index.name = "Date"

    print(f"raw rows after fill: {len(df)}  ({df.index[0].date()} -> {df.index[-1].date()})")
    return df


def engineer(df):
    print("engineering features ...")
    out  = pd.DataFrame(index=df.index)
    gold = df["Close_XAUUSD"]

    # returns
    out["Close_Returns"]  = gold.pct_change()
    out["Log_Returns"]    = np.log(gold / gold.shift(1))
    out["EURUSD_Returns"] = df["Close_EURUSD"].pct_change()
    out["USDJPY_Returns"] = df["Close_USDJPY"].pct_change()

    # bollinger %B  (0-1, scale-invariant)
    sma20 = gold.rolling(20).mean()
    std20 = gold.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["BB_PctB"] = (gold - lower) / (upper - lower)

    # price / ema ratios  (always near 1.0, scale-invariant)
    out["Price_Over_EMA50"]  = gold / gold.ewm(span=50,  adjust=False).mean()
    out["Price_Over_EMA200"] = gold / gold.ewm(span=200, adjust=False).mean()

    # macd signal normalized by price
    macd = (gold.ewm(span=12, adjust=False).mean()
           - gold.ewm(span=26, adjust=False).mean())
    out["MACD_Signal_Norm"] = macd.ewm(span=9, adjust=False).mean() / gold

    # 20-day z-scores
    r20 = out["Log_Returns"].rolling(20)
    out["LogReturn_ZScore"] = (out["Log_Returns"] - r20.mean()) / r20.std()

    c20 = out["Close_Returns"].rolling(20)
    out["Return_ZScore"] = (out["Close_Returns"] - c20.mean()) / c20.std()

    # 100-day percentile ranks
    out["Return_Percentile"] = out["Close_Returns"].rolling(100).rank(pct=True)
    vol_clean = df["Volume_XAUUSD"].replace(0, np.nan).ffill()
    out["Volume_Percentile"] = vol_clean.rolling(100).rank(pct=True)

    # pct from all-time-high (causal)
    ath = gold.expanding().max()
    out["Pct_From_AllTimeHigh"] = (ath - gold) / ath

    # bull_trend: continuous pct gap between ema50 and ema200
    # positive = bull, negative = bear, near-zero = consolidation
    # replaces binary Market_State which went dead in high-vol bull regimes
    ema50   = gold.ewm(span=50,  adjust=False).mean()
    ema200  = gold.ewm(span=200, adjust=False).mean()
    out["Bull_Trend"] = (ema50 - ema200) / ema200

    # macro_fast: 8-component causal z-score
    z_cols = []
    for col in MACRO_SERIES:
        df[f"{col}_delta"] = df[col].diff()
        for feat in [col, f"{col}_delta"]:
            roll = df[feat].shift(1).rolling(252)
            out[f"{feat}_z"] = (df[feat] - roll.mean()) / roll.std()
            z_cols.append(f"{feat}_z")
    out["Macro_Fast"] = (out[z_cols].mean(axis=1)
                          .replace([np.inf, -np.inf], np.nan)
                          .ffill()
                          .bfill()
                          .clip(-5, 5))
    out.drop(columns=z_cols, inplace=True)

    return out


def make_labels(df_raw, features):
    features["target_log_return"] = np.log(
        df_raw["Close_XAUUSD"].shift(-1) / df_raw["Close_XAUUSD"]
    )
    return features


def split(df):
    df = df[df.index >= TARGET_START].copy()
    df.dropna(subset=["target_log_return"], inplace=True)
    train_val = df[df.index <= TRAIN_END]
    test      = df[df.index >= TEST_START]
    return train_val, test


def main():
    print("gold dataset builder starting")

    raw      = fetch_all(BUFFER_START, TODAY)
    features = engineer(raw.copy())
    full     = make_labels(raw, features)
    train_val, test = split(full)

    feat_cols = [c for c in train_val.columns if c != "target_log_return"]
    nan_tv    = train_val.isnull().sum().sum()
    nan_test  = test.isnull().sum().sum()

    print(f"features         : {len(feat_cols)}")
    print(f"feature names    : {feat_cols}")
    print(f"train/val rows   : {len(train_val)}  ({train_val.index[0].date()} -> {train_val.index[-1].date()})")
    if len(test):
        print(f"test rows        : {len(test)}  ({test.index[0].date()} -> {test.index[-1].date()})")
    else:
        print(f"test rows        : 0  (no data after {TEST_START})")
    print(f"nan in train/val : {nan_tv}")
    print(f"nan in test      : {nan_test}")

    tv_path   = os.path.join(OUTPUT_DIR, "dataset_train_val.csv")
    test_path = os.path.join(OUTPUT_DIR, "dataset_test.csv")
    train_val.to_csv(tv_path)
    test.to_csv(test_path)

    print(f"saved dataset_train_val.csv  ({len(train_val)} rows)")
    print(f"saved dataset_test.csv  ({len(test)} rows)")

    # hard verification — crashes immediately if any required feature is missing
    # or contains NaN/inf in either output file, so downstream training never
    # runs on broken data again
    REQUIRED = {"Bull_Trend", "Macro_Fast", "BB_PctB", "Price_Over_EMA50",
                "Price_Over_EMA200", "MACD_Signal_Norm", "target_log_return"}
    for name, out_df in [("train_val", train_val), ("test", test)]:
        missing = REQUIRED - set(out_df.columns)
        if missing:
            raise RuntimeError(f"VERIFICATION FAILED: {name} missing columns: {missing}")
        nan_count = out_df[list(REQUIRED)].isnull().sum().sum()
        inf_count = np.isinf(out_df.select_dtypes(include=np.number)).sum().sum()
        if nan_count > 0:
            bad = out_df[list(REQUIRED)].isnull().sum()
            raise RuntimeError(f"VERIFICATION FAILED: {name} has {nan_count} NaN values:\n{bad[bad>0]}")
        if inf_count > 0:
            raise RuntimeError(f"VERIFICATION FAILED: {name} has {inf_count} inf values")
        print(f"verified {name}: all required features present, zero NaN, zero inf")

    print("done")


if __name__ == "__main__":
    main()