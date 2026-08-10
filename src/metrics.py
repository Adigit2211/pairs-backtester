"""
metrics.py

Job of this file: turn a daily return series (and a trade log) into the
standard performance numbers -- Sharpe ratio, max drawdown, win rate.

Nothing in this file makes any trading decision or looks at any price
data directly. It only does arithmetic on results that were already
produced by the (walk-forward, cost-aware) backtest engine.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio, assuming a 0% risk-free rate (a common,
    conservative simplification -- Indian T-bill yields are not zero, but
    using 0 avoids needing to source and align a risk-free rate series,
    and slightly UNDERSTATES the Sharpe ratio rather than overstating it).
    """
    if daily_returns.std() == 0 or len(daily_returns) == 0:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(periods_per_year)


def max_drawdown(daily_returns: pd.Series) -> float:
    """
    Worst peak-to-trough decline in the cumulative return curve.
    Returned as a negative number, e.g. -0.18 means an 18% drawdown.
    """
    cumulative = (1 + daily_returns).cumprod()
    running_peak = cumulative.cummax()
    drawdown = (cumulative - running_peak) / running_peak
    return drawdown.min()


def win_rate(trade_log: pd.DataFrame) -> float:
    """
    Fraction of individual trades (not days) that ended with a positive
    return. Returns None if there were no trades at all, so callers can
    display "n/a" instead of a misleading 0%.
    """
    if trade_log is None or len(trade_log) == 0:
        return None
    return (trade_log["trade_return"] > 0).mean()


def summarize(daily_returns: pd.Series, trade_log: pd.DataFrame, label: str = "") -> dict:
    """
    Bundles all three headline metrics together, plus a couple of
    supporting numbers (total return, number of trades) that give context
    for how much to trust the headline numbers.
    """
    total_return = (1 + daily_returns).prod() - 1
    wr = win_rate(trade_log)
    num_trades = 0 if trade_log is None else len(trade_log)

    result = {
        "label": label,
        "period_start": daily_returns.index.min(),
        "period_end": daily_returns.index.max(),
        "num_trading_days": len(daily_returns),
        "num_trades": num_trades,
        "total_return": total_return,
        "annualized_sharpe": sharpe_ratio(daily_returns),
        "max_drawdown": max_drawdown(daily_returns),
        "win_rate": wr,
    }
    return result


def print_summary(result: dict):
    wr_str = f"{result['win_rate']:.1%}" if result["win_rate"] is not None else "n/a (no trades)"
    print(f"--- {result['label']} ---")
    print(f"Period:            {result['period_start'].date()} to {result['period_end'].date()}")
    print(f"Trading days:      {result['num_trading_days']}")
    print(f"Number of trades:  {result['num_trades']}")
    print(f"Total return:      {result['total_return']:.2%}")
    print(f"Annualized Sharpe: {result['annualized_sharpe']:.2f}")
    print(f"Max drawdown:      {result['max_drawdown']:.2%}")
    print(f"Win rate:          {wr_str}")
    print()
