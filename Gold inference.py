"""
XAUUSD Gold Intelligence — Streamlit Inference App
===================================================
Stage 1 : LightGBM base model  (cv_best_fold_model.pkl)
Stage 2 : Logistic calibrator  (calibrator.pkl)

Calibrator expects exactly 6 features in this order:
  prediction_value, abs_prediction, Bull_Trend, Macro_Fast, BB_PctB, Price_Over_EMA200
"""

import os, pickle, warnings, time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
from fredapi import Fred

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
DAYS_BACK        = 520          # fetch window — enough for all rolling windows
PROB_THRESHOLD   = 0.45       # calibrated from OOF signal quality table
Z_THRESHOLD      = 0.6     # minimum |pred_z| to emit a signal
PRED_Z_LOOKBACK  = 252
FRED_API_KEY     = "219d0c44b2e3b4a8b690c3f69b91a5bb"
MACRO_SERIES     = ["DFII10", "DFII5", "DGS2", "FEDFUNDS"]
ARTEFACT_DIR     = os.path.dirname(os.path.abspath(__file__))

# exact feature order the base model was trained on
BASE_FEATURES = [
    "Close_Returns", "Log_Returns", "EURUSD_Returns", "USDJPY_Returns",
    "BB_PctB", "Price_Over_EMA50", "Price_Over_EMA200", "MACD_Signal_Norm",
    "LogReturn_ZScore", "Return_ZScore", "Return_Percentile", "Volume_Percentile",
    "Pct_From_AllTimeHigh", "Bull_Trend", "Macro_Fast",
]

# exact 6-feature order calibrator.pkl was fitted on (verified from pkl inspection)
CALIB_FEATURES = [
    "prediction_value", "abs_prediction",
    "Bull_Trend", "Macro_Fast", "BB_PctB", "Price_Over_EMA200",
]


# ── data fetching ─────────────────────────────────────────────────────────────
def _download(ticker, start, end, retries=3):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if not df.empty:
                return df
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return pd.DataFrame()


def fetch_fred_local(start, end):
    """Try FRED API, fall back to local CSV files."""
    series = {}
    fred_obj = Fred(api_key=FRED_API_KEY)
    for s in MACRO_SERIES:
        try:
            data = fred_obj.get_series(s, start, end)
            series[s] = data
        except Exception:
            local = os.path.join(ARTEFACT_DIR, f"{s}.csv")
            if os.path.exists(local):
                df = pd.read_csv(local, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                col = df.columns[0]
                data = df[col].replace(".", np.nan).astype(float)
                data = data[(data.index >= str(start)) & (data.index <= str(end))]
                series[s] = data
            else:
                raise FileNotFoundError(
                    f"FRED API failed and no local {s}.csv found. "
                    f"Download from https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
                )
    macro = pd.DataFrame(series)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    return macro


def fetch_data(start, end):
    gold = _download("GC=F",     start, end)
    eur  = _download("EURUSD=X", start, end)
    jpy  = _download("JPY=X",    start, end)

    prices = pd.DataFrame({
        "Close_XAUUSD":  gold["Close"],
        "Volume_XAUUSD": gold["Volume"],
        "Close_EURUSD":  eur["Close"],
        "Close_USDJPY":  jpy["Close"],
    })

    macro = fetch_fred_local(start, end)

    full_idx = pd.date_range(start=prices.index.min(),
                             end=prices.index.max(), freq="B")
    prices = prices.reindex(full_idx)
    macro  = macro.reindex(full_idx)

    df = prices.join(macro, how="left").ffill().bfill()
    df.dropna(subset=["Close_XAUUSD"], inplace=True)
    df.index.name = "Date"
    return df


# ── feature engineering ───────────────────────────────────────────────────────
def engineer_features(df):
    out  = pd.DataFrame(index=df.index)
    gold = df["Close_XAUUSD"]

    out["Close_Returns"]  = gold.pct_change()
    out["Log_Returns"]    = np.log(gold / gold.shift(1))
    out["EURUSD_Returns"] = df["Close_EURUSD"].pct_change()
    out["USDJPY_Returns"] = df["Close_USDJPY"].pct_change()

    sma20 = gold.rolling(20).mean()
    std20 = gold.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["BB_PctB"] = (gold - lower) / (upper - lower)

    ema50  = gold.ewm(span=50,  adjust=False).mean()
    ema200 = gold.ewm(span=200, adjust=False).mean()
    out["Price_Over_EMA50"]  = gold / ema50
    out["Price_Over_EMA200"] = gold / ema200
    out["Bull_Trend"]        = (ema50 - ema200) / ema200

    macd = (gold.ewm(span=12, adjust=False).mean()
           - gold.ewm(span=26, adjust=False).mean())
    out["MACD_Signal_Norm"] = macd.ewm(span=9, adjust=False).mean() / gold

    r20 = out["Log_Returns"].rolling(20)
    out["LogReturn_ZScore"] = (out["Log_Returns"] - r20.mean()) / r20.std()
    c20 = out["Close_Returns"].rolling(20)
    out["Return_ZScore"] = (out["Close_Returns"] - c20.mean()) / c20.std()

    out["Return_Percentile"] = out["Close_Returns"].rolling(100).rank(pct=True)
    vol = df["Volume_XAUUSD"].replace(0, np.nan).ffill()
    out["Volume_Percentile"] = vol.rolling(100).rank(pct=True)

    ath = gold.expanding().max()
    out["Pct_From_AllTimeHigh"] = (ath - gold) / ath

    # Macro_Fast: 8-component causal z-score, clipped at [-5, 5]
    z_cols = []
    for col in MACRO_SERIES:
        df[f"{col}_delta"] = df[col].diff()
        for feat in [col, f"{col}_delta"]:
            roll = df[feat].shift(1).rolling(252)
            out[f"{feat}_z"] = (df[feat] - roll.mean()) / roll.std()
            z_cols.append(f"{feat}_z")
    out["Macro_Fast"] = (out[z_cols].mean(axis=1)
                         .replace([np.inf, -np.inf], np.nan)
                         .ffill().bfill().clip(-5, 5))
    out.drop(columns=z_cols, inplace=True)

    # attach raw price for display
    out["Close_XAUUSD"] = gold

    return out.dropna(subset=BASE_FEATURES)


# ── inference ─────────────────────────────────────────────────────────────────
def load_artefacts():
    def _load(name):
        path = os.path.join(ARTEFACT_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found in {ARTEFACT_DIR}")
        with open(path, "rb") as f:
            return pickle.load(f)

    base_model  = _load("cv_best_fold_model.pkl")
    calibrator  = _load("calibrator.pkl")
    oof_history = pd.read_csv(
        os.path.join(ARTEFACT_DIR, "cv_predictions_oof.csv"),
        index_col=0, parse_dates=True
    )
    return base_model, calibrator, oof_history


def run_inference(feat_df, base_model, calibrator, oof_history):
    today      = feat_df.iloc[[-1]].copy()
    today_date = today.index[0]

    # stage 1
    pred_val   = float(base_model.predict(today[BASE_FEATURES].values)[0])
    abs_pred   = abs(pred_val)

    # pred_z from OOF history
    hist     = oof_history["oof_prediction"].dropna().tail(PRED_Z_LOOKBACK)
    h_std    = hist.std()
    pred_z   = float((pred_val - hist.mean()) / h_std) if h_std > 0 else 0.0

    # stage 2 — exactly 6 features in the fitted order
    bull_trend  = float(today["Bull_Trend"].iloc[0])
    macro_fast  = float(today["Macro_Fast"].iloc[0])
    bb_pctb     = float(today["BB_PctB"].iloc[0])
    ema200      = float(today["Price_Over_EMA200"].iloc[0])
    close       = float(today["Close_XAUUSD"].iloc[0])

    calib_input = pd.DataFrame(
        [[pred_val, abs_pred, bull_trend, macro_fast, bb_pctb, ema200]],
        columns=CALIB_FEATURES,
    )
    prob = float(calibrator.predict_proba(calib_input)[0][1])

    signal = "NO SIGNAL"
    if prob >= PROB_THRESHOLD and abs(pred_z) >= Z_THRESHOLD:
        signal = "BUY" if pred_val > 0 else "SELL"

    # recent 20-day price data for mini chart
    recent = feat_df["Close_XAUUSD"].tail(20)

    return {
        "date":        today_date,
        "signal":      signal,
        "pred_val":    pred_val,
        "abs_pred":    abs_pred,
        "pred_z":      pred_z,
        "abs_pred_z":  abs(pred_z),
        "prob":        prob,
        "bull_trend":  bull_trend,
        "macro_fast":  macro_fast,
        "bb_pctb":     bb_pctb,
        "ema200":      ema200,
        "close":       close,
        "recent":      recent,
        "feat_df":     feat_df,
    }


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

:root {
    --bg:       #05070a;
    --surf:     #0c0f14;
    --surf2:    #111520;
    --border:   #1c2030;
    --border2:  #252a38;
    --text:     #e2e8f0;
    --muted:    #3d4a5c;
    --muted2:   #5a6a80;
    --accent:   #f5c842;
    --accent2:  #e6a800;
    --buy:      #10d988;
    --buy2:     #0aab68;
    --sell:     #ff4d6a;
    --sell2:    #cc2a45;
    --mono:     'JetBrains Mono', monospace;
    --sans:     'Space Grotesk', sans-serif;
    --glow-buy:  0 0 24px rgba(16,217,136,0.18);
    --glow-sell: 0 0 24px rgba(255,77,106,0.18);
    --glow-gold: 0 0 24px rgba(245,200,66,0.15);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
section.main,
.main .block-container { background: var(--bg) !important; color: var(--text) !important; font-family: var(--sans) !important; }
body { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 2.4rem 1.6rem 6rem !important; max-width: 900px !important; }
* { box-sizing: border-box; }

/* ── header ── */
.app-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 0 1.4rem; border-bottom: 1px solid var(--border);
    margin-bottom: 2.2rem;
}
.app-logo {
    display: flex; align-items: center; gap: 0.75rem;
}
.logo-hex {
    width: 34px; height: 34px; background: var(--accent);
    clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 0.6rem; font-weight: 700; color: #000;
    flex-shrink: 0;
}
.logo-text { font-family: var(--mono); font-size: 0.78rem; font-weight: 600; letter-spacing: 0.2em; color: var(--accent); text-transform: uppercase; }
.logo-sub  { font-family: var(--mono); font-size: 0.56rem; color: var(--muted2); letter-spacing: 0.12em; margin-top: 2px; }
.header-right { text-align: right; }
.header-ts { font-family: var(--mono); font-size: 0.58rem; color: var(--muted2); letter-spacing: 0.08em; }
.header-status { font-family: var(--mono); font-size: 0.56rem; color: var(--buy); letter-spacing: 0.12em; margin-top: 3px; }

/* ── run button ── */
div.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #000 !important; border: none !important; border-radius: 2px !important;
    font-family: var(--mono) !important; font-size: 0.72rem !important;
    font-weight: 700 !important; letter-spacing: 0.2em !important;
    text-transform: uppercase !important; padding: 0.85rem 2rem !important;
    width: 100% !important; transition: all 0.15s !important;
    box-shadow: 0 0 20px rgba(245,200,66,0.25) !important;
}
div.stButton > button:hover { opacity: 0.88 !important; box-shadow: 0 0 32px rgba(245,200,66,0.4) !important; }

/* ── signal banner ── */
.sig-banner {
    position: relative; border-radius: 3px; overflow: hidden;
    margin: 1.8rem 0; padding: 2rem 2rem 2rem 2.4rem;
    display: flex; align-items: center; justify-content: space-between;
    background: var(--surf);
}
.sig-banner::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
}
.sig-banner::after {
    content: ''; position: absolute; inset: 0; opacity: 0.04;
    background: radial-gradient(ellipse at left center, currentColor 0%, transparent 70%);
    pointer-events: none;
}
.sb-buy  { border: 1px solid rgba(16,217,136,0.3); box-shadow: var(--glow-buy); }
.sb-buy::before  { background: var(--buy); }
.sb-sell { border: 1px solid rgba(255,77,106,0.3); box-shadow: var(--glow-sell); }
.sb-sell::before { background: var(--sell); }
.sb-none { border: 1px solid var(--border2); }
.sb-none::before { background: var(--muted); }

.sig-label { font-family: var(--mono); font-size: 2.6rem; font-weight: 700; letter-spacing: 0.04em; line-height: 1; }
.sb-buy  .sig-label { color: var(--buy);  text-shadow: 0 0 30px rgba(16,217,136,0.5); }
.sb-sell .sig-label { color: var(--sell); text-shadow: 0 0 30px rgba(255,77,106,0.5); }
.sb-none .sig-label { color: var(--muted2); }
.sig-sub { font-family: var(--mono); font-size: 0.6rem; color: var(--muted2); letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.5rem; }
.sig-right { text-align: right; }
.sig-prob { font-family: var(--mono); font-size: 1.4rem; font-weight: 600; }
.sb-buy .sig-prob  { color: var(--buy);  }
.sb-sell .sig-prob { color: var(--sell); }
.sb-none .sig-prob { color: var(--muted2); }
.sig-prob-lbl { font-family: var(--mono); font-size: 0.56rem; color: var(--muted2); letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.3rem; }
.sig-reason { font-family: var(--mono); font-size: 0.6rem; color: var(--muted2); line-height: 1.9; margin-top: 0.5rem; }

/* ── section label ── */
.section-label {
    font-family: var(--mono); font-size: 0.56rem; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--muted2);
    padding: 0 0 0.6rem; border-bottom: 1px solid var(--border);
    margin: 1.8rem 0 1rem;
}

/* ── kpi strip ── */
.kpi-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 2px; overflow: hidden; margin-bottom: 1px; }
.kpi-strip-2 { grid-template-columns: repeat(2, 1fr); }
.kpi-cell { background: var(--surf); padding: 1.1rem 1rem; }
.kpi-lbl  { font-family: var(--mono); font-size: 0.52rem; color: var(--muted2); letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.4rem; }
.kpi-val  { font-family: var(--mono); font-size: 1.1rem; font-weight: 600; color: var(--text); }
.kpi-val.gold  { color: var(--accent); }
.kpi-val.bull  { color: var(--buy);   }
.kpi-val.bear  { color: var(--sell);  }
.kpi-val.muted { color: var(--muted2); }
.kpi-sub  { font-family: var(--mono); font-size: 0.52rem; color: var(--muted); margin-top: 0.25rem; }

/* ── mini sparkline ── */
.spark-wrap { background: var(--surf); border: 1px solid var(--border); border-radius: 2px; padding: 1.1rem 1.2rem; margin-bottom: 1px; }
.spark-hdr  { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem; }
.spark-title { font-family: var(--mono); font-size: 0.56rem; letter-spacing: 0.16em; color: var(--muted2); text-transform: uppercase; }
.spark-price { font-family: var(--mono); font-size: 0.78rem; font-weight: 600; color: var(--accent); }
svg.spark { width: 100%; height: 56px; display: block; }

/* ── data table ── */
.dtbl { background: var(--surf); border: 1px solid var(--border); border-radius: 2px; overflow: hidden; }
.dtbl-hdr { display: grid; font-family: var(--mono); font-size: 0.52rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted2); padding: 0.6rem 1.2rem; border-bottom: 1px solid var(--border); background: var(--bg); }
.dtbl-row { display: flex; justify-content: space-between; align-items: center; padding: 0.55rem 1.2rem; border-bottom: 1px solid var(--border); }
.dtbl-row:last-child { border-bottom: none; }
.dtbl-row:hover { background: var(--surf2); }
.dtbl-k { font-family: var(--mono); font-size: 0.62rem; color: var(--muted2); letter-spacing: 0.08em; }
.dtbl-v { font-family: var(--mono); font-size: 0.68rem; font-weight: 500; color: var(--text); }
.dtbl-v.buy  { color: var(--buy);   }
.dtbl-v.sell { color: var(--sell);  }
.dtbl-v.gold { color: var(--accent); }
.dtbl-v.muted{ color: var(--muted2); }

/* ── news card ── */
.news-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 0.8rem; }
.news-card { background: var(--surf); border: 1px solid var(--border); border-radius: 2px; overflow: hidden; cursor: pointer; transition: border-color 0.15s; text-decoration: none; display: block; }
.news-card:hover { border-color: var(--border2); }
.news-img { width: 100%; height: 110px; object-fit: cover; display: block; background: var(--surf2); }
.news-body { padding: 0.75rem 0.9rem; }
.news-src  { font-family: var(--mono); font-size: 0.5rem; letter-spacing: 0.14em; color: var(--accent); text-transform: uppercase; margin-bottom: 0.35rem; }
.news-title{ font-family: var(--sans); font-size: 0.72rem; font-weight: 500; color: var(--text); line-height: 1.45; }
.news-age  { font-family: var(--mono); font-size: 0.5rem; color: var(--muted2); margin-top: 0.35rem; }

/* ── model card ── */
.model-row { display: flex; gap: 1px; background: var(--border); margin-bottom: 1px; }
.model-cell { background: var(--surf); padding: 0.9rem 1.1rem; flex: 1; }
.model-lbl { font-family: var(--mono); font-size: 0.5rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted2); margin-bottom: 0.3rem; }
.model-val { font-family: var(--mono); font-size: 0.78rem; font-weight: 500; color: var(--text); }

/* ── idle ── */
.idle { font-family: var(--mono); font-size: 0.68rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; padding: 4rem 0; border: 1px dashed var(--border2); border-radius: 2px; margin-top: 1.6rem; background: var(--surf); }
.idle-sub { font-size: 0.56rem; color: var(--muted); margin-top: 0.5rem; letter-spacing: 0.08em; }

/* ── spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stStatusWidget"] { display: none !important; }
</style>
"""


# ── sparkline SVG ─────────────────────────────────────────────────────────────
def make_sparkline(prices: pd.Series, color: str) -> str:
    vals = prices.dropna().values.astype(float)
    if len(vals) < 2:
        return ""
    mn, mx = vals.min(), vals.max()
    rng = mx - mn if mx != mn else 1
    W, H, PAD = 400, 56, 4
    pts = []
    for i, v in enumerate(vals):
        x = PAD + (i / (len(vals) - 1)) * (W - 2 * PAD)
        y = PAD + (1 - (v - mn) / rng) * (H - 2 * PAD)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    # area fill
    fill_pts = f"{PAD},{H-PAD} " + poly + f" {W-PAD},{H-PAD}"
    up = vals[-1] >= vals[0]
    c  = color
    return (
        f'<svg class="spark" viewBox="0 0 {W} {H}" preserveAspectRatio="none">'
        f'<defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{c}" stop-opacity="0.25"/>'
        f'<stop offset="100%" stop-color="{c}" stop-opacity="0"/>'
        f'</linearGradient></defs>'
        f'<polygon points="{fill_pts}" fill="url(#sg)"/>'
        f'<polyline points="{poly}" fill="none" stroke="{c}" stroke-width="1.5" stroke-linejoin="round"/>'
        f'</svg>'
    )


# ── news fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_news():
    try:
        ticker = yf.Ticker("GC=F")
        news   = ticker.news or []
        items  = []
        for n in news[:6]:
            title = n.get("title", "")
            url   = n.get("link", "#")
            src   = n.get("publisher", "")
            age_s = n.get("providerPublishTime", 0)
            img   = ""
            if "thumbnail" in n and n["thumbnail"]:
                try:
                    img = n["thumbnail"]["resolutions"][0]["url"]
                except Exception:
                    img = ""
            if age_s:
                dt  = datetime.fromtimestamp(age_s)
                hrs = (datetime.now() - dt).seconds // 86400
                age = f"{hrs}h ago" if hrs < 24 else dt.strftime("%b %d")
            else:
                age = ""
            items.append({"title": title, "url": url, "src": src, "age": age, "img": img})
        return items
    except Exception:
        return []


# ── helpers ───────────────────────────────────────────────────────────────────
def _tbl_row(k, v, cls=""):
    return (f'<div class="dtbl-row">'
            f'<span class="dtbl-k">{k}</span>'
            f'<span class="dtbl-v {cls}">{v}</span>'
            f'</div>')


def _kpi(lbl, val, cls="", sub=""):
    s = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi-cell">'
            f'<div class="kpi-lbl">{lbl}</div>'
            f'<div class="kpi-val {cls}">{val}</div>'
            f'{s}</div>')


def _section(label):
    return f'<div class="section-label">{label}</div>'


# ── main app ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="XAUUSD Intelligence",
        page_icon="⬡",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    now = datetime.now()
    st.markdown(f"""
    <div class="app-header">
      <div class="app-logo">
        <div class="logo-hex">XAU</div>
        <div>
          <div class="logo-text">XAUUSD &nbsp; Intelligence</div>
          <div class="logo-sub">ML Signal System &nbsp;·&nbsp; Stage 1 + Stage 2</div>
        </div>
      </div>
      <div class="header-right">
        <div class="header-ts">{now.strftime('%Y-%m-%d &nbsp; %H:%M:%S UTC')}</div>
        <div class="header-status">● LIVE</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    run = st.button("▶  GENERATE TODAY'S SIGNAL", use_container_width=True)

    if not run:
        st.markdown("""
        <div class="idle">
            Press ▶ to generate today's signal
            <div class="idle-sub">Fetches live market data · Runs LightGBM + Calibrator · Outputs probability-weighted signal</div>
        </div>
        """, unsafe_allow_html=True)

        # show news even on idle
        st.markdown(_section("LATEST GOLD NEWS"), unsafe_allow_html=True)
        _render_news()
        return

    # ── run inference ─────────────────────────────────────────────────────────
    try:
        end   = now
        start = end - timedelta(days=DAYS_BACK)

        with st.spinner("Fetching market data  (GC=F · EURUSD · USDJPY · FRED) ..."):
            raw = fetch_data(start, end)

        with st.spinner("Engineering 15 features ..."):
            feat_df = engineer_features(raw)

        with st.spinner("Loading model artefacts ..."):
            base_model, calibrator, oof_history = load_artefacts()

        with st.spinner("Running Stage 1 + Stage 2 inference ..."):
            r = run_inference(feat_df, base_model, calibrator, oof_history)

    except FileNotFoundError as e:
        st.error(f"Missing artefact: {e}")
        return
    except Exception as e:
        st.exception(e)
        return

    sig   = r["signal"]
    sc    = {"BUY": "sb-buy", "SELL": "sb-sell", "NO SIGNAL": "sb-none"}[sig]
    vc    = {"BUY": "buy",    "SELL": "sell",     "NO SIGNAL": "muted"}[sig]
    bt    = r["bull_trend"]
    regime_label = "BULL" if bt > 0.02 else "BEAR" if bt < -0.02 else "NEUTRAL"
    regime_cls   = "bull" if bt > 0.02 else "bear" if bt < -0.02 else "muted"

    # reason for NO SIGNAL
    reason_html = ""
    if sig == "NO SIGNAL":
        reasons = []
        if abs(r["pred_z"]) < Z_THRESHOLD:
            reasons.append(f"|z| {r['pred_z']:.2f} &lt; {Z_THRESHOLD} (low conviction)")
        if r["prob"] < PROB_THRESHOLD:
            reasons.append(f"prob {r['prob']:.4f} &lt; {PROB_THRESHOLD} (below threshold)")
        reason_html = "<br>".join(reasons)

    # ── signal banner ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sig-banner {sc}">
      <div>
        <div class="sig-label">{sig}</div>
        <div class="sig-sub">XAUUSD · {r['date'].strftime('%Y-%m-%d')} · Daily Close</div>
        <div class="sig-reason">{reason_html}</div>
      </div>
      <div class="sig-right">
        <div class="sig-prob">{r['prob']:.1%}</div>
        <div class="sig-prob-lbl">Win Probability</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── sparkline ─────────────────────────────────────────────────────────────
    spark_color = "#10d988" if r["recent"].iloc[-1] >= r["recent"].iloc[0] else "#ff4d6a"
    chg_pct     = (r["recent"].iloc[-1] / r["recent"].iloc[0] - 1) * 100
    chg_cls     = "bull" if chg_pct >= 0 else "bear"
    spark_svg   = make_sparkline(r["recent"], spark_color)

    st.markdown(f"""
    <div class="spark-wrap">
      <div class="spark-hdr">
        <span class="spark-title">XAU/USD · 20-Day Price</span>
        <span class="spark-price">
          ${r['close']:,.2f}
          <span style="font-size:0.62rem; color:var(--{'buy' if chg_pct>=0 else 'sell'})">
            &nbsp;{'+' if chg_pct>=0 else ''}{chg_pct:.2f}%
          </span>
        </span>
      </div>
      {spark_svg}
    </div>
    """, unsafe_allow_html=True)

    # ── KPI strip row 1 ───────────────────────────────────────────────────────
    macro_lbl = "TIGHT" if r["macro_fast"] > 0 else "EASY"
    macro_cls = "sell" if r["macro_fast"] > 0.5 else "bull" if r["macro_fast"] < -0.5 else ""
    pct_from_ath = r["feat_df"]["Pct_From_AllTimeHigh"].iloc[-1]

    st.markdown(f"""
    <div class="kpi-strip">
      {_kpi("XAU/USD Close",   f"${r['close']:,.2f}",              "gold")}
      {_kpi("Market Regime",   f"{regime_label} {bt:+.3f}",        regime_cls)}
      {_kpi("Win Probability", f"{r['prob']:.1%}",                 vc)}
      {_kpi("Macro Pressure",  f"{macro_lbl} ({r['macro_fast']:+.2f})", macro_cls)}
    </div>
    <div class="kpi-strip">
      {_kpi("BB %B",           f"{r['bb_pctb']:.3f}",              "gold" if r['bb_pctb'] > 0.8 else "")}
      {_kpi("EMA200 Ratio",    f"{r['ema200']:.4f}",               "bull" if r['ema200'] > 1 else "bear")}
      {_kpi("Pred Z-Score",    f"{r['pred_z']:+.3f}",              "bull" if r['pred_z']>0 else "bear")}
      {_kpi("% From ATH",      f"{pct_from_ath:.2%}",              "muted")}
    </div>
    """, unsafe_allow_html=True)

    # ── full output table ─────────────────────────────────────────────────────
    st.markdown(_section("MODEL OUTPUT — FULL DETAIL"), unsafe_allow_html=True)

    rows = [
        ("DATE",              r["date"].strftime("%Y-%m-%d"),      ""),
        ("CLOSE PRICE",       f"${r['close']:,.2f}",               "gold"),
        ("SIGNAL",            sig,                                  vc),
        ("WIN PROBABILITY",   f"{r['prob']:.4f}  ({r['prob']:.1%})", vc),
        ("BASE PREDICTION",   f"{r['pred_val']:+.8f}",             ""),
        ("ABS PREDICTION",    f"{r['abs_pred']:.8f}",              ""),
        ("PRED Z-SCORE",      f"{r['pred_z']:+.4f}",               ""),
        ("|Z| CONVICTION",    f"{r['abs_pred_z']:.4f}",            "bull" if r['abs_pred_z'] >= Z_THRESHOLD else "muted"),
        ("BULL TREND",        f"{r['bull_trend']:+.4f}  ({regime_label})", regime_cls),
        ("MACRO FAST",        f"{r['macro_fast']:+.4f}  ({macro_lbl})",    macro_cls),
        ("BB PCTB",           f"{r['bb_pctb']:.4f}",              ""),
        ("PRICE / EMA200",    f"{r['ema200']:.4f}",               "bull" if r['ema200']>1 else "bear"),
        ("PROB THRESHOLD",    f"{PROB_THRESHOLD}",                 "muted"),
        ("Z THRESHOLD",       f"{Z_THRESHOLD}",                    "muted"),
    ]
    rows_html = "".join(_tbl_row(k, v, c) for k, v, c in rows)
    st.markdown(f'<div class="dtbl">{rows_html}</div>', unsafe_allow_html=True)

    # ── recent feature snapshot (last 5 rows) ─────────────────────────────────
    st.markdown(_section("LIVE FEATURE SNAPSHOT — LAST 5 TRADING DAYS"), unsafe_allow_html=True)
    snap_cols = ["Close_XAUUSD", "Bull_Trend", "Macro_Fast", "BB_PctB",
                 "Price_Over_EMA200", "Return_ZScore", "Volume_Percentile"]
    snap = r["feat_df"][snap_cols].tail(5).copy()
    snap["Close_XAUUSD"] = snap["Close_XAUUSD"].map("${:,.2f}".format)
    snap.index = snap.index.strftime("%Y-%m-%d")
    snap_html = snap.to_html(classes="", border=0)

    # style the dataframe table to match theme
    st.markdown(f"""
    <div class="dtbl" style="overflow-x:auto;">
      <style>
        .dtbl table {{ width:100%; border-collapse:collapse; font-family:var(--mono); font-size:0.6rem; }}
        .dtbl th {{ padding:0.55rem 0.8rem; border-bottom:1px solid var(--border); color:var(--muted2); letter-spacing:0.1em; text-align:left; background:var(--bg); font-weight:400; }}
        .dtbl td {{ padding:0.5rem 0.8rem; border-bottom:1px solid var(--border); color:var(--text); }}
        .dtbl tr:last-child td {{ border-bottom:none; }}
        .dtbl tr:hover td {{ background:var(--surf2); }}
      </style>
      {snap_html}
    </div>
    """, unsafe_allow_html=True)

    # ── model info ────────────────────────────────────────────────────────────
    st.markdown(_section("MODEL ARTEFACTS"), unsafe_allow_html=True)
    st.markdown(f"""
    <div class="model-row">
      <div class="model-cell"><div class="model-lbl">Stage 1</div><div class="model-val">LightGBM Regressor</div></div>
      <div class="model-cell"><div class="model-lbl">Features</div><div class="model-val">{len(BASE_FEATURES)} inputs</div></div>
      <div class="model-cell"><div class="model-lbl">Target</div><div class="model-val">Log Return (t+1)</div></div>
    </div>
    <div class="model-row">
      <div class="model-cell"><div class="model-lbl">Stage 2</div><div class="model-val">Logistic Regression</div></div>
      <div class="model-cell"><div class="model-lbl">Features</div><div class="model-val">{len(CALIB_FEATURES)} inputs</div></div>
      <div class="model-cell"><div class="model-lbl">Target</div><div class="model-val">P(direction correct)</div></div>
    </div>
    <div class="model-row">
      <div class="model-cell"><div class="model-lbl">Entry Gate</div><div class="model-val">Prob ≥ {PROB_THRESHOLD} AND |Z| ≥ {Z_THRESHOLD}</div></div>
      <div class="model-cell"><div class="model-lbl">OOF Window</div><div class="model-val">Last {PRED_Z_LOOKBACK} days</div></div>
      <div class="model-cell"><div class="model-lbl">Data Fetch</div><div class="model-val">{DAYS_BACK}-day window</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── news ──────────────────────────────────────────────────────────────────
    st.markdown(_section("LATEST GOLD NEWS"), unsafe_allow_html=True)
    _render_news()


def _render_news():
    news = fetch_news()
    if not news:
        st.markdown('<div class="idle" style="padding:1.5rem">No news available</div>',
                    unsafe_allow_html=True)
        return

    cards = ""
    for n in news[:6]:
        img_html = (f'<img class="news-img" src="{n["img"]}" '
                    f'onerror="this.style.display=\'none\'">'
                    if n["img"] else
                    '<div class="news-img" style="background:var(--surf2)"></div>')
        cards += (f'<a class="news-card" href="{n["url"]}" target="_blank">'
                  f'{img_html}'
                  f'<div class="news-body">'
                  f'<div class="news-src">{n["src"]}</div>'
                  f'<div class="news-title">{n["title"]}</div>'
                  f'<div class="news-age">{n["age"]}</div>'
                  f'</div></a>')

    st.markdown(f'<div class="news-grid">{cards}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()