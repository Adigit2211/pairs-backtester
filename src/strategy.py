"""
strategy.py

Job of this file: turn a pair's spread into actual position signals
(flat / long-the-spread / short-the-spread) using z-score entry, exit,
stop-loss, and time-stop rules.

KEY DISCIPLINE POINT: the mean and standard deviation used to compute the
z-score come ONLY from the formation window (the past), and are held FIXED
while trading. We never recompute them using data from the trading window
itself -- doing so would mean today's signal is partly informed by prices
that, in real trading, haven't happened yet.
"""

import numpy as np
import pandas as pd


def compute_spread(price_a: pd.Series, price_b: pd.Series, alpha: float, beta: float) -> pd.Series:
    """
    spread = actual price_a - predicted price_a (predicted from price_b
    using the alpha/beta we already estimated during formation).
    """
    return price_a - (alpha + beta * price_b)


def formation_spread_stats(price_a: pd.Series, price_b: pd.Series, alpha: float, beta: float):
    """
    Computes the mean and standard deviation of the spread, using ONLY the
    formation-window prices passed in here. These numbers get frozen and
    handed to generate_signals() for use during the trading window.
    """
    spread = compute_spread(price_a, price_b, alpha, beta)
    return spread.mean(), spread.std()


def generate_signals(
    trading_spread: pd.Series,
    formation_mean: float,
    formation_std: float,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_holding_days: int = 30,
    cooldown_days: int = 5,
) -> pd.DataFrame:
    """
    Walks day by day through the trading-window spread and decides a
    position for each day, using ONLY the frozen formation_mean/formation_std
    (never anything computed from trading_spread itself).

    Position convention:
       +1 = "long the spread"  -> long ticker_a, short beta*ticker_b
       -1 = "short the spread" -> short ticker_a, long beta*ticker_b
        0 = flat

    After a STOP-LOSS exit specifically (not a normal exit, not a time-stop),
    we enforce a cooldown of `cooldown_days` before allowing re-entry into
    this pair. Reasoning: a stop-loss firing means the spread moved further
    from normal than we expected -- that's a signal the relationship may be
    temporarily broken, so immediately re-entering (and possibly getting
    stopped out again right away) is not a risk we want to take blindly.

    Returns a DataFrame with the z-score, the position, and a plain-English
    reason for each day's decision (useful for debugging and for showing an
    interviewer exactly why a trade was entered/exited).
    """
    zscores = (trading_spread - formation_mean) / formation_std

    positions = []
    reasons = []
    current_position = 0
    days_in_position = 0
    cooldown_remaining = 0

    for date, z in zscores.items():
        reason = "hold flat"

        if current_position == 0:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                reason = f"cooling down ({cooldown_remaining} days left)"
            elif z >= entry_z:
                current_position = -1
                days_in_position = 0
                reason = f"enter short-spread (z={z:.2f} >= {entry_z})"
            elif z <= -entry_z:
                current_position = 1
                days_in_position = 0
                reason = f"enter long-spread (z={z:.2f} <= -{entry_z})"

        else:
            # In a trade -- check stop-loss, time-stop, then normal exit
            days_in_position += 1

            if abs(z) >= stop_z:
                reason = f"stop-loss exit (|z|={abs(z):.2f} >= {stop_z})"
                current_position = 0
                days_in_position = 0
                cooldown_remaining = cooldown_days
            elif days_in_position >= max_holding_days:
                reason = f"time-stop exit (held {days_in_position} days)"
                current_position = 0
                days_in_position = 0
            elif abs(z) <= exit_z:
                reason = f"normal exit (|z|={abs(z):.2f} <= {exit_z})"
                current_position = 0
                days_in_position = 0
            else:
                reason = f"holding position (z={z:.2f})"

        positions.append(current_position)
        reasons.append(reason)

    return pd.DataFrame({
        "spread": trading_spread,
        "zscore": zscores,
        "position": positions,
        "reason": reasons,
    })


if __name__ == "__main__":
    # Sanity check only, using the one pair that survived Step 3's screening:
    # ICICIBANK.NS / SBIN.NS. Formation window = 2015 (matches Step 3), and
    # we trade the NEXT period (early 2016) -- never the same period we
    # formed the relationship on.
    from data_pipeline import load_prices
    from cointegration import engle_granger_pair

    prices = load_prices()

    formation = prices.loc["2015-01-01":"2015-12-31"]
    trading = prices.loc["2016-01-01":"2016-06-30"]

    eg_result = engle_granger_pair(formation["ICICIBANK.NS"], formation["SBIN.NS"])
    alpha, beta = eg_result["alpha"], eg_result["beta"]
    print(f"Using formation-window alpha={alpha:.4f}, beta={beta:.4f} (direction={eg_result['winning_direction']})")

    # NOTE: engle_granger_pair picked a "winning direction" -- we need to
    # apply alpha/beta consistently with that direction when building the
    # spread for the trading window too.
    if eg_result["winning_direction"] == "a_on_b":
        price_a_form, price_b_form = formation["ICICIBANK.NS"], formation["SBIN.NS"]
        price_a_trade, price_b_trade = trading["ICICIBANK.NS"], trading["SBIN.NS"]
    else:
        price_a_form, price_b_form = formation["SBIN.NS"], formation["ICICIBANK.NS"]
        price_a_trade, price_b_trade = trading["SBIN.NS"], trading["ICICIBANK.NS"]

    formation_mean, formation_std = formation_spread_stats(price_a_form, price_b_form, alpha, beta)
    print(f"Formation-window spread mean={formation_mean:.2f}, std={formation_std:.2f}\n")

    trading_spread = compute_spread(price_a_trade, price_b_trade, alpha, beta)
    signals = generate_signals(trading_spread, formation_mean, formation_std)

    # Only print the days where something changed (entries/exits), not every
    # single day, so it's easy to read.
    changed = signals[signals["reason"].str.contains("enter|exit|cooling down \\(0")]
    print(f"Trading window: {trading_spread.index.min().date()} to {trading_spread.index.max().date()}")
    print(f"Entry/exit events:\n")
    print(changed[["zscore", "position", "reason"]])
