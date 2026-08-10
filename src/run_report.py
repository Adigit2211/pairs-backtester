"""
run_report.py

Job of this file: run the walk-forward backtest ONCE over the full price
history, then split the results by date into:
  - "in-sample" -- everything before the final holdout cutoff. This is the
    period we were implicitly looking at while designing/adjusting the
    strategy (e.g. when we added the cooldown period in Step 4).
  - "out-of-sample holdout" -- the final 12 months. Every strategy
    parameter (z-score thresholds, stop-loss, cooldown, cost assumption,
    Bonferroni correction, beta filter) was already fixed BEFORE this
    script ever looked at how it performs in this window.

WHY SPLIT THE OUTPUT OF ONE RUN, INSTEAD OF RUNNING TWICE?
The walk-forward engine itself already guarantees that every trading
window's decisions used only past (formation-window) data -- that
discipline doesn't care whether we look at the results afterward split by
date, or as two separate runs. Splitting the output is equivalent, and
simpler. What actually matters for avoiding bias is the promise that
follows: after seeing the holdout numbers below, we do NOT go back and
retune entry_z, exit_z, stop_z, cooldown_days, or cost_bps. If we did,
the "holdout" would silently become just another in-sample period.
"""

import pandas as pd
from dateutil.relativedelta import relativedelta

from data_pipeline import load_prices
from backtest_engine import run_walk_forward
from metrics import summarize, print_summary


def main():
    prices = load_prices()

    print("Running the full walk-forward backtest (one pass, ~1-2 minutes)...\n")
    daily_returns, pair_log, trade_log = run_walk_forward(prices)

    holdout_start = daily_returns.index.max() - relativedelta(months=12)
    print(f"Final holdout cutoff: {holdout_start.date()} onward is out-of-sample,")
    print(f"never used to choose any strategy parameter.\n")

    in_sample_returns = daily_returns.loc[:holdout_start]
    holdout_returns = daily_returns.loc[holdout_start:]

    in_sample_trades = trade_log[trade_log["trading_start"] < holdout_start.date()] if len(trade_log) else trade_log
    holdout_trades = trade_log[trade_log["trading_start"] >= holdout_start.date()] if len(trade_log) else trade_log

    in_sample_summary = summarize(in_sample_returns, in_sample_trades, label="IN-SAMPLE (walk-forward)")
    holdout_summary = summarize(holdout_returns, holdout_trades, label="OUT-OF-SAMPLE (final holdout)")

    print_summary(in_sample_summary)
    print_summary(holdout_summary)

    return in_sample_summary, holdout_summary, pair_log, trade_log


if __name__ == "__main__":
    main()
