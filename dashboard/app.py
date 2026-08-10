"""
app.py

Streamlit dashboard. Reads the JSON file that generate_daily_signals.py
produces and displays it -- this file does NOT call yfinance or run any
statistical tests itself, it only reads and displays already-computed
results. This keeps the dashboard fast and independent of whether Yahoo
Finance is reachable at the moment someone visits the page.
"""

import json
import os

import pandas as pd
import streamlit as st

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "latest_signals.json")

st.set_page_config(page_title="Pairs Trading Signals", layout="wide")
st.title("Statistical Arbitrage Pairs Trading — Live Paper Signals")
st.caption(
    "Paper trading only -- no real capital is involved. Pairs are re-screened "
    "using the same walk-forward-style methodology as the backtest: a trailing "
    "12-month formation window, Engle-Granger + Johansen cointegration tests "
    "with a Bonferroni correction, and an economic plausibility filter on beta."
)

if not os.path.exists(DATA_PATH):
    st.warning(
        "No signals file found yet. Run `python src/generate_daily_signals.py` "
        "locally, or wait for the daily GitHub Actions job to run."
    )
    st.stop()

with open(DATA_PATH) as f:
    data = json.load(f)

st.write(f"**Last updated:** {data['generated_at']}")
st.write(f"**Formation window:** {data['formation_window']['start']} to {data['formation_window']['end']}")

if not data["pairs"]:
    st.info(
        "No pairs currently pass the cointegration screen (Bonferroni-corrected "
        "Engle-Granger + Johansen, plus the beta plausibility filter). This is "
        "expected behavior, not an error -- see the backtest results in the README "
        "for how often this happens historically."
    )
    st.stop()

for pair in data["pairs"]:
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader(f"{pair['ticker_a']} / {pair['ticker_b']}")
            st.metric("Current z-score", pair["current_zscore"])

            position_label = {1: "LONG the spread", -1: "SHORT the spread", 0: "FLAT"}[pair["current_position"]]
            st.write(f"**Current signal:** {position_label}")
            st.caption(pair["current_reason"])

            st.write(f"Engle-Granger p-value: `{pair['eg_pvalue']}`")
            st.write(f"Hedge ratio (beta): `{pair['beta']}`")

        with col2:
            chart_df = pd.DataFrame({
                "date": pd.to_datetime(pair["recent_dates"]),
                "zscore": pair["recent_zscores"],
            }).set_index("date")
            st.line_chart(chart_df)
            st.caption(
                f"Formation-window spread stats (frozen, not re-estimated during "
                f"trading): mean={pair['formation_mean']}, std={pair['formation_std']}"
            )

st.divider()
st.caption(
    "Methodology, transaction cost assumptions, bias flags, and full backtest "
    "results are documented in the project README."
)
