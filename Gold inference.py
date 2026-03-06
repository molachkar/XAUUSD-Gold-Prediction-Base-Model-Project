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


def fetch_data(start, end):
    TICKER_MAP = {
        'GC=F':     'Close_XAUUSD',
        'EURUSD=X': 'Close_EURUSD',
        'JPY=X':    'Close_USDJPY',
        '^TNX':     'TNX_Raw',
    }
    raw = yf.download(list(TICKER_MAP.keys()), start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw['Close'].rename(columns=TICKER_MAP)
        volume   = raw['Volume']['GC=F'].rename('Volume')
    else:
        close_df = raw[['Close']].rename(columns={'Close': 'Close_XAUUSD'})
        volume   = raw['Volume'].rename('Volume')

    fred  = Fred(api_key=FRED_API_KEY)
    macro = pd.DataFrame({s: fred.get_series(s, start, end) for s in MACRO_SERIES})
    macro.index = pd.to_datetime(macro.index)

    df = close_df.join(macro, how='left').join(volume, how='left').sort_index()
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

    df['BB_Middle'] = df['Close_XAUUSD'].rolling(20).mean()
    df['Return_Percentile'] = df['Close_Returns'].rolling(100).rank(pct=True)
    df['Volume'] = df['Volume'].replace(0, np.nan).ffill()
    df['Volume_Percentile'] = df['Volume'].rolling(100).rank(pct=True)

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
    market_state_str = str(market_state)

    calib_input = pd.DataFrame(
        [[oof_pred, pred_z, abs_pred_z, macro_fast, market_state_str]],
        columns=CALIB_FEATURES
    )
    calib_input['Market_State'] = calib_input['Market_State'].astype(str)
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


DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #090909;
    --surface:   #111111;
    --border:    #222222;
    --border2:   #2a2a2a;
    --text:      #e8e8e8;
    --muted:     #555555;
    --accent:    #f0a500;
    --buy:       #00c97a;
    --sell:      #ff3b3b;
    --neutral:   #888888;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
}

/* ── global reset ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 780px !important; }

/* ── header bar ── */
.xau-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.xau-logo {
    font-family: var(--mono);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
}
.xau-timestamp {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}

/* ── run button ── */
div.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.15s ease !important;
}
div.stButton > button:hover { opacity: 0.85 !important; }

/* ── signal banner ── */
.sig-banner {
    border: 1px solid var(--border);
    padding: 2rem 1.5rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.sig-banner::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
}
.sig-buy   { border-color: var(--buy);  }
.sig-buy::before   { background: var(--buy);  }
.sig-sell  { border-color: var(--sell); }
.sig-sell::before  { background: var(--sell); }
.sig-none  { border-color: var(--border2); }
.sig-none::before  { background: var(--muted); }

.sig-label {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: 0.1em;
}
.sig-buy  .sig-label { color: var(--buy);  }
.sig-sell .sig-label { color: var(--sell); }
.sig-none .sig-label { color: var(--muted); }

.sig-sub {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.sig-reason {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--muted);
    text-align: right;
    line-height: 1.7;
}

/* ── stat grid ── */
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    margin: 1.5rem 0;
}
.stat-cell {
    background: var(--surface);
    padding: 1.1rem 1.2rem;
}
.stat-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.stat-value {
    font-family: var(--mono);
    font-size: 1.25rem;
    font-weight: 500;
    color: var(--text);
    letter-spacing: 0.02em;
}
.stat-value.bull { color: var(--buy);  }
.stat-value.bear { color: var(--sell); }
.stat-value.accent { color: var(--accent); }

/* ── detail table ── */
.detail-block {
    border: 1px solid var(--border);
    margin-top: 1.5rem;
}
.detail-header {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0.7rem 1.2rem;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
}
.detail-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.65rem 1.2rem;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.72rem;
}
.detail-row:last-child { border-bottom: none; }
.detail-key   { color: var(--muted); letter-spacing: 0.05em; }
.detail-val   { color: var(--text);  font-weight: 500; }
.detail-val.buy  { color: var(--buy);  }
.detail-val.sell { color: var(--sell); }
.detail-val.none { color: var(--muted); }

/* ── spinner / info / error ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
div[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 2px !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
}

/* ── idle state ── */
.idle-msg {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-align: center;
    padding: 3rem 0;
    border: 1px dashed var(--border2);
    margin-top: 1.5rem;
}
</style>
"""


def main():
    st.set_page_config(
        page_title="XAUUSD Signal",
        page_icon="⬡",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    st.markdown(DARK_CSS, unsafe_allow_html=True)

    now = datetime.now()
    st.markdown(f"""
    <div class="xau-header">
        <div>
            <div class="xau-logo">⬡ &nbsp; XAU / USD &nbsp; Intelligence</div>
        </div>
        <div class="xau-timestamp">{now.strftime("%Y-%m-%d &nbsp;&nbsp; %H:%M:%S UTC")}</div>
    </div>
    """, unsafe_allow_html=True)

    run = st.button("RUN INFERENCE", type="primary", use_container_width=True)

    if run:
        try:
            with st.spinner("Fetching market & macro data..."):
                end   = now
                start = end - timedelta(days=DAYS_BACK)
                df    = fetch_data(start, end)

            with st.spinner("Engineering features..."):
                features = engineer_features(df)

            with st.spinner("Loading models..."):
                base_model, calibrator, oof_history = load_artefacts()

            with st.spinner("Running inference..."):
                r = run_inference(features, base_model, calibrator, oof_history)

            signal = r["signal"]
            sig_cls = {"BUY": "sig-buy", "SELL": "sig-sell", "NO SIGNAL": "sig-none"}[signal]

            reason_html = ""
            if signal == "NO SIGNAL":
                lines = []
                if r["abs_pred_z"] < CONVICTION_THRESHOLD:
                    lines.append(f"|pred_z| {r['abs_pred_z']:.2f} &lt; {CONVICTION_THRESHOLD} threshold")
                if r["prob_success"] < PROB_THRESHOLD:
                    lines.append(f"prob {r['prob_success']:.4f} &lt; {PROB_THRESHOLD} threshold")
                reason_html = "<br>".join(lines)

            st.markdown(f"""
            <div class="sig-banner {sig_cls}">
                <div>
                    <div class="sig-label">{signal}</div>
                    <div class="sig-sub">XAUUSD · {r['date']} · Daily Close Signal</div>
                </div>
                <div class="sig-reason">{reason_html}</div>
            </div>
            """, unsafe_allow_html=True)

            label     = "BULL" if r["market_state"] == 1 else "BEAR" if r["market_state"] == -1 else "NEUTRAL"
            label_cls = "bull" if r["market_state"] == 1 else "bear" if r["market_state"] == -1 else ""
            prob_pct  = f"{r['prob_success']:.1%}"
            macro_sign = "TIGHT" if r["macro_fast"] > 0 else "EASY"

            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-cell">
                    <div class="stat-label">XAUUSD Close</div>
                    <div class="stat-value accent">${r['Close_XAUUSD']:,.2f}</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-label">Market Regime</div>
                    <div class="stat-value {label_cls}">{label}</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-label">Prob Success</div>
                    <div class="stat-value">{prob_pct}</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-label">Macro Pressure</div>
                    <div class="stat-value">{macro_sign} &nbsp;<span style="font-size:0.85rem;color:var(--muted)">({r['macro_fast']:+.3f})</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            sig_val_cls = {"BUY": "buy", "SELL": "sell", "NO SIGNAL": "none"}[signal]
            rows = [
                ("DATE",             r["date"]),
                ("CLOSE PRICE",      f"${r['Close_XAUUSD']:,.2f}"),
                ("BASE PREDICTION",  f"{r['oof_prediction']:+.6f}"),
                ("PRED Z-SCORE",     f"{r['pred_z']:+.4f}"),
                ("ABS PRED Z",       f"{r['abs_pred_z']:.4f}"),
                ("PROB SUCCESS",     f"{r['prob_success']:.4f}"),
                ("MACRO FAST",       f"{r['macro_fast']:+.4f}"),
                ("MARKET STATE",     f"{r['market_state']} ({label})"),
                ("SIGNAL",           signal),
            ]
            rows_html = ""
            for key, val in rows:
                vc = sig_val_cls if key == "SIGNAL" else ""
                rows_html += f"""
                <div class="detail-row">
                    <span class="detail-key">{key}</span>
                    <span class="detail-val {vc}">{val}</span>
                </div>"""

            st.markdown(f"""
            <div class="detail-block">
                <div class="detail-header">Full Output</div>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)

        except FileNotFoundError as e:
            st.error(f"Missing artefact: {e}")
        except Exception as e:
            st.exception(e)
    else:
        st.markdown("""
        <div class="idle-msg">
            Press RUN INFERENCE to generate today's signal
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()