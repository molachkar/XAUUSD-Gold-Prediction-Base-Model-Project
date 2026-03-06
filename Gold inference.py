import os
import pickle
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from fredapi import Fred

warnings.filterwarnings("ignore")

DAYS_BACK            = 500
PROB_THRESHOLD       = 0.52
CONVICTION_THRESHOLD = 1.0
PRED_Z_LOOKBACK      = 252
FRED_API_KEY         = "219d0c44b2e3b4a8b690c3f69b91a5bb"
MACRO_SERIES         = ['DFII10', 'DFII5', 'DGS2', 'FEDFUNDS']
ARTEFACT_DIR         = os.path.dirname(os.path.abspath(__file__))

BASE_FEATURES = [
    'Macro_Fast', 'Market_State', 'Close_XAUUSD', 'LogReturn_ZScore',
    'Close_Returns', 'Return_ZScore', 'Close_USDJPY', 'Log_Returns',
    'BB_Middle', 'Return_Percentile', 'MACD_Signal',
    'Distance_From_AllTimeHigh', 'Close_EURUSD', 'Pct_From_AllTimeHigh',
    'Volume_Percentile'
]

CALIB_FEATURES = ['oof_prediction', 'pred_z', 'abs_pred_z', 'Macro_Fast', 'Market_State']


def _single(ticker, start, end):
    """Download one ticker, flatten any MultiIndex, return clean DataFrame."""
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def fetch_data(start, end):
    gold  = _single('GC=F',     start, end)
    eur   = _single('EURUSD=X', start, end)
    jpy   = _single('JPY=X',    start, end)

    prices = pd.DataFrame({
        'Close_XAUUSD': gold['Close'],
        'Close_EURUSD': eur['Close'],
        'Close_USDJPY': jpy['Close'],
        'Volume':       gold['Volume'],
    })

    fred  = Fred(api_key=FRED_API_KEY)
    macro = pd.DataFrame(
        {s: fred.get_series(s, start, end) for s in MACRO_SERIES}
    )
    macro.index = pd.to_datetime(macro.index).tz_localize(None)

    df = prices.join(macro, how='left').sort_index()
    df = df.ffill().dropna(subset=['Close_XAUUSD'])
    return df


def engineer_features(df):
    z_cols = []
    for col in MACRO_SERIES:
        df[f"{col}_delta"] = df[col].diff()
        for feat in [col, f"{col}_delta"]:
            roll = df[feat].shift(1).rolling(252)
            df[f"{feat}_z"] = (df[feat] - roll.mean()) / roll.std()
            z_cols.append(f"{feat}_z")
    df['Macro_Fast'] = df[z_cols].mean(axis=1)

    df['Close_Returns'] = df['Close_XAUUSD'].pct_change()
    df['Log_Returns']   = np.log(df['Close_XAUUSD'] / df['Close_XAUUSD'].shift(1))

    ema50   = df['Close_XAUUSD'].ewm(span=50,  adjust=False).mean()
    ema200  = df['Close_XAUUSD'].ewm(span=200, adjust=False).mean()
    mkt_vol = df['Close_Returns'].shift(1).rolling(20).std()
    vol_med = mkt_vol.expanding().median()
    df['Market_State'] = 0
    df.loc[(ema50 > ema200) & (mkt_vol < vol_med), 'Market_State'] = 1
    df.loc[(ema50 < ema200) & (mkt_vol > vol_med), 'Market_State'] = -1

    r20 = df['Log_Returns'].rolling(20)
    df['LogReturn_ZScore'] = (df['Log_Returns'] - r20.mean()) / r20.std()

    c20 = df['Close_Returns'].rolling(20)
    df['Return_ZScore'] = (df['Close_Returns'] - c20.mean()) / c20.std()

    df['BB_Middle']          = df['Close_XAUUSD'].rolling(20).mean()
    df['Return_Percentile']  = df['Close_Returns'].rolling(100).rank(pct=True)
    df['Volume']             = df['Volume'].replace(0, np.nan).ffill()
    df['Volume_Percentile']  = df['Volume'].rolling(100).rank(pct=True)

    macd = (df['Close_XAUUSD'].ewm(span=12, adjust=False).mean()
          - df['Close_XAUUSD'].ewm(span=26, adjust=False).mean())
    df['MACD_Signal'] = macd.ewm(span=9, adjust=False).mean()

    ath = df['Close_XAUUSD'].expanding().max()
    df['Distance_From_AllTimeHigh'] = ath - df['Close_XAUUSD']
    df['Pct_From_AllTimeHigh']      = (df['Distance_From_AllTimeHigh'] / ath) * 100

    return df[BASE_FEATURES].dropna()


def load_artefacts():
    def load(name):
        with open(os.path.join(ARTEFACT_DIR, name), 'rb') as f:
            return pickle.load(f)
    base_model  = load("cv_best_fold_model.pkl")
    calibrator  = load("calibrator.pkl")
    oof_history = pd.read_csv(
        os.path.join(ARTEFACT_DIR, "cv_predictions_oof.csv"),
        index_col=0, parse_dates=True
    )
    return base_model, calibrator, oof_history


def run_inference(feature_df, base_model, calibrator, oof_history):
    today      = feature_df.iloc[[-1]].copy()
    today_date = today.index[0].date()

    oof_pred = float(base_model.predict(today[BASE_FEATURES].values)[0])

    pred_col   = next((c for c in oof_history.columns if 'pred' in c.lower()), oof_history.columns[0])
    hist       = oof_history[pred_col].dropna().tail(PRED_Z_LOOKBACK)
    pred_z     = float((oof_pred - hist.mean()) / hist.std()) if hist.std() != 0 else 0.0
    abs_pred_z = abs(pred_z)

    macro_fast       = float(today['Macro_Fast'].iloc[0])
    market_state     = int(today['Market_State'].iloc[0])
    market_state_str = str(market_state)   # calibrator OHE fitted on strings

    calib_input = pd.DataFrame(
        [[oof_pred, pred_z, abs_pred_z, macro_fast, market_state_str]],
        columns=CALIB_FEATURES
    )
    prob = float(calibrator.predict_proba(calib_input)[0][1])

    signal = "NO SIGNAL"
    if prob >= PROB_THRESHOLD and abs_pred_z >= CONVICTION_THRESHOLD:
        signal = "BUY" if oof_pred > 0 else "SELL"

    return {
        "date":           str(today_date),
        "signal":         signal,
        "oof_prediction": round(oof_pred, 6),
        "pred_z":         round(pred_z, 4),
        "abs_pred_z":     round(abs_pred_z, 4),
        "prob_success":   round(prob, 4),
        "macro_fast":     round(macro_fast, 4),
        "market_state":   market_state,
        "Close_XAUUSD":   round(float(today['Close_XAUUSD'].iloc[0]), 2),
    }


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #080808;
    --surf:    #101010;
    --border:  #1e1e1e;
    --b2:      #272727;
    --text:    #e6e6e6;
    --muted:   #4a4a4a;
    --accent:  #f0a500;
    --buy:     #00c47a;
    --sell:    #ff3636;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'IBM Plex Sans', sans-serif;
}

/* ── base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
section.main,
.main .block-container,
[class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

[data-testid="stAppViewContainer"] > .main {
    background-color: var(--bg) !important;
}

/* kill white flash on load */
body { background: var(--bg) !important; }

#MainMenu, footer, header { visibility: hidden !important; }
.block-container {
    padding: 2rem 1.4rem 5rem !important;
    max-width: 760px !important;
}

/* ── header ── */
.hdr {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.8rem;
}
.hdr-logo {
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    color: var(--accent);
    text-transform: uppercase;
}
.hdr-ts {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.06em;
}

/* ── button ── */
div.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 1px !important;
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.12s !important;
}
div.stButton > button:hover { opacity: 0.8 !important; }

/* ── signal card ── */
.sig-card {
    position: relative;
    border: 1px solid var(--border);
    padding: 1.8rem 1.4rem 1.8rem 1.7rem;
    margin: 1.4rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surf);
    overflow: hidden;
}
.sig-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
}
.sig-buy  { border-color: var(--buy); }
.sig-buy::before  { background: var(--buy); }
.sig-sell { border-color: var(--sell); }
.sig-sell::before { background: var(--sell); }
.sig-none { border-color: var(--b2); }
.sig-none::before { background: var(--muted); }

.sig-main { font-family: var(--mono); font-size: 1.9rem; font-weight: 600; letter-spacing: 0.08em; }
.sig-buy  .sig-main { color: var(--buy);  }
.sig-sell .sig-main { color: var(--sell); }
.sig-none .sig-main { color: var(--muted); }

.sig-meta {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.35rem;
}
.sig-why {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
    text-align: right;
    line-height: 1.8;
}

/* ── 2x2 grid ── */
.kpi-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    margin: 1.4rem 0;
}
.kpi-cell {
    background: var(--surf);
    padding: 1rem 1.1rem;
}
.kpi-lbl {
    font-family: var(--mono);
    font-size: 0.58rem;
    color: var(--muted);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.kpi-val {
    font-family: var(--mono);
    font-size: 1.2rem;
    font-weight: 500;
    color: var(--text);
}
.kpi-val.gold   { color: var(--accent); }
.kpi-val.bull   { color: var(--buy);    }
.kpi-val.bear   { color: var(--sell);   }

/* ── detail table ── */
.tbl {
    border: 1px solid var(--border);
    margin-top: 1.4rem;
    background: var(--surf);
}
.tbl-hdr {
    font-family: var(--mono);
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0.65rem 1.2rem;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
}
.tbl-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 1.2rem;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.7rem;
}
.tbl-row:last-child { border-bottom: none; }
.tbl-k { color: var(--muted); }
.tbl-v { color: var(--text); font-weight: 500; }
.tbl-v.buy  { color: var(--buy);   }
.tbl-v.sell { color: var(--sell);  }
.tbl-v.none { color: var(--muted); }

/* ── idle ── */
.idle {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-align: center;
    padding: 3.5rem 0;
    border: 1px dashed var(--b2);
    margin-top: 1.4rem;
    background: var(--surf);
}

/* ── spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
"""


def main():
    st.set_page_config(
        page_title="XAUUSD Signal",
        page_icon="⬡",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    st.markdown(CSS, unsafe_allow_html=True)

    now = datetime.now()
    st.markdown(f"""
    <div class="hdr">
        <div class="hdr-logo">⬡ &nbsp; XAU / USD &nbsp; Intelligence</div>
        <div class="hdr-ts">{now.strftime('%Y-%m-%d &nbsp; %H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶  RUN INFERENCE", use_container_width=True):
        try:
            with st.spinner("Fetching market data..."):
                end   = now
                start = end - timedelta(days=DAYS_BACK)
                df    = fetch_data(start, end)

            with st.spinner("Engineering features..."):
                features = engineer_features(df)

            with st.spinner("Loading models..."):
                base_model, calibrator, oof_history = load_artefacts()

            with st.spinner("Running inference..."):
                r = run_inference(features, base_model, calibrator, oof_history)

            signal  = r["signal"]
            sc      = {"BUY": "sig-buy", "SELL": "sig-sell", "NO SIGNAL": "sig-none"}[signal]
            label   = "BULL" if r["market_state"] == 1 else "BEAR" if r["market_state"] == -1 else "NEUTRAL"
            lc      = "bull" if r["market_state"] == 1 else "bear" if r["market_state"] == -1 else ""
            vc      = {"BUY": "buy", "SELL": "sell", "NO SIGNAL": "none"}[signal]

            why = ""
            if signal == "NO SIGNAL":
                parts = []
                if r["abs_pred_z"] < CONVICTION_THRESHOLD:
                    parts.append(f"|z| {r['abs_pred_z']:.2f} &lt; {CONVICTION_THRESHOLD}")
                if r["prob_success"] < PROB_THRESHOLD:
                    parts.append(f"prob {r['prob_success']:.4f} &lt; {PROB_THRESHOLD}")
                why = "<br>".join(parts)

            st.markdown(f"""
            <div class="sig-card {sc}">
                <div>
                    <div class="sig-main">{signal}</div>
                    <div class="sig-meta">XAUUSD &nbsp;·&nbsp; {r['date']} &nbsp;·&nbsp; Daily Close</div>
                </div>
                <div class="sig-why">{why}</div>
            </div>
            """, unsafe_allow_html=True)

            macro_lbl = "TIGHT" if r["macro_fast"] > 0 else "EASY"
            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-cell">
                    <div class="kpi-lbl">XAUUSD Close</div>
                    <div class="kpi-val gold">${r['Close_XAUUSD']:,.2f}</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-lbl">Market Regime</div>
                    <div class="kpi-val {lc}">{label}</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-lbl">Prob Success</div>
                    <div class="kpi-val">{r['prob_success']:.1%}</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-lbl">Macro Pressure</div>
                    <div class="kpi-val">{macro_lbl} <span style="font-size:.8rem;color:var(--muted)">({r['macro_fast']:+.3f})</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            rows = [
                ("DATE",            r["date"],                          ""),
                ("CLOSE PRICE",     f"${r['Close_XAUUSD']:,.2f}",       "gold"),
                ("BASE PREDICTION", f"{r['oof_prediction']:+.6f}",      ""),
                ("PRED Z-SCORE",    f"{r['pred_z']:+.4f}",              ""),
                ("ABS PRED Z",      f"{r['abs_pred_z']:.4f}",           ""),
                ("PROB SUCCESS",    f"{r['prob_success']:.4f}",         ""),
                ("MACRO FAST",      f"{r['macro_fast']:+.4f}",          ""),
                ("MARKET STATE",    f"{r['market_state']} ({label})",   ""),
                ("SIGNAL",          signal,                             vc),
            ]
            rows_html = "".join(
                f'<div class="tbl-row"><span class="tbl-k">{k}</span>'
                f'<span class="tbl-v {c}">{v}</span></div>'
                for k, v, c in rows
            )
            st.markdown(f"""
            <div class="tbl">
                <div class="tbl-hdr">Full Output</div>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)

        except FileNotFoundError as e:
            st.error(f"Missing artefact: {e}")
        except Exception as e:
            st.exception(e)
    else:
        st.markdown(
            '<div class="idle">Press ▶ RUN INFERENCE to generate today\'s signal</div>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()