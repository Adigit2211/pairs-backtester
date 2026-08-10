"""
backtest_engine.py

Walk-forward backtest engine. Ties together cointegration screening
(cointegration.py) and signal generation (strategy.py) into a single loop
that steps forward through time, never letting a trading window see
information from its own formation step or from the future.

WALK-FORWARD LOOP, IN PLAIN WORDS:
  1. Take a window of past data (the "formation window").
  2. Use ONLY that window to (a) screen for cointegrated pairs, (b) fit
     alpha/beta for each surviving pair, (c) compute each pair's spread
     mean/std.
  3. Trade the NEXT chunk of time (the "trading window") using the frozen
     numbers from step 2. No refitting mid-window.
  4. Slide forward by one trading window's length and repeat.
  5. Glue together every trading window's daily returns into one long
     return series -- that is the walk-forward result.

EXECUTION-TIMING ASSUMPTION (explicitly flagged):
  We only have daily CLOSE prices (no intraday open/high/low in our cached
  data). To avoid using the same day's close for both the trading decision
  and the trade execution (which would be a form of lookahead -- you can't
  actually trade at a price the instant you observe it), every position
  decided using day t's close is applied starting from day t+1's return.
  This is a simple, conservative proxy for realistic execution -- not a
  full intraday simulation.

HEDGE WEIGHTING ASSUMPTION (explicitly flagged):
  We convert the share-based hedge ratio (beta) into a dollar-neutral
  weight ONCE, using the trading window's first-day prices, and hold that
  weight fixed for the entire window (no continuous rebalancing). This is
  a standard simplification in pairs-trading backtests.
"""

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from cointegration import screen_pairs
from strategy import compute_spread, formation_spread_stats, generate_signals


def generate_walk_forward_windows(start_date, end_date, formation_months=12, trading_months=3, step_months=3):
    """
    Builds the list of (formation_start, formation_end, trading_start,
    trading_end) date ranges the walk-forward loop will step through.
    """
    windows = []
    formation_start = pd.Timestamp(start_date)

    while True:
        formation_end = formation_start + relativedelta(months=formation_months)
        trading_start = formation_end
        trading_end = trading_start + relativedelta(months=trading_months)

        if trading_end > pd.Timestamp(end_date):
            break

        windows.append({
            "formation_start": formation_start,
            "formation_end": formation_end,
            "trading_start": trading_start,
            "trading_end": trading_end,
        })

        formation_start = formation_start + relativedelta(months=step_months)

    return windows


def _pair_return_series(price_a, price_b, beta, position, cost_bps):
    """
    Converts a position series (+1/0/-1) for one pair into a daily
    PERCENTAGE RETURN series for a dollar-neutral long/short portfolio,
    including transaction costs charged whenever the position changes.
    """
    a0, b0 = price_a.iloc[0], price_b.iloc[0]
    weight_a = 1.0 / (1.0 + beta * b0 / a0)
    weight_b = 1.0 - weight_a

    ret_a = price_a.pct_change()
    ret_b = price_b.pct_change()

    # Signal decided using day t's close is applied to the t -> t+1 return.
    applied_position = position.shift(1).fillna(0)

    gross_return = applied_position * (weight_a * ret_a - weight_b * ret_b)

    # Cost charged whenever the position changes (entry, exit, or flip).
    position_change = position.diff().abs().fillna(abs(position.iloc[0]))
    cost = position_change * (cost_bps / 10000.0)

    return (gross_return - cost).fillna(0)


def run_walk_forward(
    prices: pd.DataFrame,
    formation_months=12,
    trading_months=3,
    step_months=3,
    entry_z=2.0,
    exit_z=0.5,
    stop_z=3.5,
    max_holding_days=30,
    cooldown_days=5,
    cost_bps=5,
    eg_pvalue_threshold=0.05,
):
    """
    Runs the full walk-forward loop over the given price history.

    Returns:
      daily_returns -- one Series of the combined portfolio's daily return,
                        spanning every trading window back to back.
      pair_log      -- a DataFrame recording which pairs were selected in
                        each window, for transparency/auditing.
    """
    start_date = prices.index.min()
    end_date = prices.index.max()
    windows = generate_walk_forward_windows(start_date, end_date, formation_months, trading_months, step_months)

    all_daily_returns = []
    pair_log = []

    for w in windows:
        formation_prices = prices.loc[w["formation_start"]:w["formation_end"]]
        trading_prices = prices.loc[w["trading_start"]:w["trading_end"]]

        if len(trading_prices) == 0:
            continue

        selected_pairs = screen_pairs(formation_prices, eg_pvalue_threshold=eg_pvalue_threshold)
        window_pair_returns = []

        for _, row in selected_pairs.iterrows():
            ticker_a, ticker_b = row["ticker_a"], row["ticker_b"]
            alpha, beta, direction = row["alpha"], row["beta"], row["direction"]
            col_a, col_b = (ticker_a, ticker_b) if direction == "a_on_b" else (ticker_b, ticker_a)

            if col_a not in trading_prices.columns or col_b not in trading_prices.columns:
                continue

            formation_mean, formation_std = formation_spread_stats(
                formation_prices[col_a], formation_prices[col_b], alpha, beta
            )
            trading_spread = compute_spread(trading_prices[col_a], trading_prices[col_b], alpha, beta)
            signals = generate_signals(
                trading_spread, formation_mean, formation_std,
                entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
                max_holding_days=max_holding_days, cooldown_days=cooldown_days,
            )
            pair_returns = _pair_return_series(
                trading_prices[col_a], trading_prices[col_b], beta, signals["position"], cost_bps
            )
            window_pair_returns.append(pair_returns)

            pair_log.append({
                "formation_start": w["formation_start"].date(),
                "trading_start": w["trading_start"].date(),
                "trading_end": w["trading_end"].date(),
                "ticker_a": col_a,
                "ticker_b": col_b,
                "beta": round(beta, 4),
                "num_trades": int((signals["position"].diff().abs() > 0).sum()),
            })

        if window_pair_returns:
            window_portfolio_return = pd.concat(window_pair_returns, axis=1).mean(axis=1)
        else:
            window_portfolio_return = pd.Series(0.0, index=trading_prices.index)

        all_daily_returns.append(window_portfolio_return)

    daily_returns = pd.concat(all_daily_returns).sort_index()
    daily_returns = daily_returns[~daily_returns.index.duplicated(keep="first")]

    return daily_returns, pd.DataFrame(pair_log)


if __name__ == "__main__":
    from data_pipeline import load_prices

    prices = load_prices()

    print("Running walk-forward backtest (screens cointegration fresh at")
    print("every quarterly step -- this can take a minute or two)...\n")

    daily_returns, pair_log = run_walk_forward(prices)

    print(f"Backtest covers {daily_returns.index.min().date()} to {daily_returns.index.max().date()}")
    print(f"({len(daily_returns)} trading days)\n")

    print("Pairs selected at each formation step:")
    print(pair_log.to_string(index=False) if len(pair_log) else "(none found in any window)")

    cumulative = (1 + daily_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    print(f"\nTotal return over full walk-forward period: {total_return:.2%}")
