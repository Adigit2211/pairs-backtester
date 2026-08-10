"""
generate_daily_signals.py

Job of this file: figure out "if we were trading live today, what would
the strategy be doing" and save that as a small JSON file for the
dashboard to display.

This is intentionally separate from the dashboard itself (dashboard/app.py)
so the dashboard never has to call yfinance directly -- it just reads
whatever this script last saved. This script is meant to be run once a day
(by the GitHub Actions workflow), not on every dashboard page-load.

Uses the exact same formation-window logic as the backtest (12 months of
trailing data, same cointegration screening, same guardrails) -- this is
not a different, hand-picked "demo" version of the strategy.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
from dateutil.relativedelta import relativedelta

from data_pipeline import fetch_price_data, TICKERS
from cointegration import screen_pairs
from strategy import compute_spread, formation_spread_stats, generate_signals

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "dashboard", "data", "latest_signals.json")


def compute_today_signals():
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    formation_start = today - relativedelta(months=12)

    # Pull a little extra history before the formation window too, so we
    # have enough trailing days to show a short recent z-score chart for
    # context, not just today's single number.
    fetch_start = formation_start - relativedelta(months=1)
    prices = fetch_price_data(tickers=TICKERS, start=fetch_start.strftime("%Y-%m-%d"))

    formation_prices = prices.loc[formation_start:today]
    selected_pairs = screen_pairs(formation_prices)

    results = []
    for _, row in selected_pairs.iterrows():
        ticker_a, ticker_b = row["ticker_a"], row["ticker_b"]
        alpha, beta, direction = row["alpha"], row["beta"], row["direction"]
        col_a, col_b = (ticker_a, ticker_b) if direction == "a_on_b" else (ticker_b, ticker_a)

        formation_mean, formation_std = formation_spread_stats(
            formation_prices[col_a], formation_prices[col_b], alpha, beta
        )

        # Look at the last ~40 trading days for a short recent history --
        # far short of a full trading window, just enough for a sparkline
        # and to know the current signal state.
        recent_prices = prices.loc[:today].tail(40)
        recent_spread = compute_spread(recent_prices[col_a], recent_prices[col_b], alpha, beta)
        signals = generate_signals(recent_spread, formation_mean, formation_std)

        latest = signals.iloc[-1]

        results.append({
            "ticker_a": col_a,
            "ticker_b": col_b,
            "eg_pvalue": round(row["eg_pvalue"], 6),
            "beta": round(beta, 4),
            "formation_mean": round(formation_mean, 2),
            "formation_std": round(formation_std, 2),
            "current_zscore": round(latest["zscore"], 2),
            "current_position": int(latest["position"]),
            "current_reason": latest["reason"],
            "recent_zscores": [round(z, 2) for z in signals["zscore"].tolist()],
            "recent_dates": [d.strftime("%Y-%m-%d") for d in signals.index],
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "formation_window": {
            "start": formation_start.strftime("%Y-%m-%d"),
            "end": today.strftime("%Y-%m-%d"),
        },
        "pairs": results,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(results)} pair(s) to {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    compute_today_signals()
