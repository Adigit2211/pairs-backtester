"""
data_pipeline.py

Job of this file: download daily price history for our banking-stock universe
from Yahoo Finance, and save it to a local cache so we don't hit the API
every time we run an experiment.

Nothing in this file makes any trading decision. It only fetches and stores
raw data. Keeping this separate from strategy logic means the data layer
can be tested/trusted on its own.
"""

import os
import pandas as pd
import yfinance as yf

# The banking universe we picked in the plan.
# NOTE (survivorship bias, flagged on purpose):
# these are today's listed tickers. Any bank that delisted, merged, or was
# restructured out of existence during our sample period (e.g. Yes Bank's
# 2020 reconstruction) is silently absent from this list. yfinance has no
# point-in-time constituent history, so this is a real, documented
# limitation of the project -- not something this script tries to hide.
TICKERS = [
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "SBIN.NS",
    "INDUSINDBK.NS",
    "BANKBARODA.NS",
    "PNB.NS",
    "FEDERALBNK.NS",
    "IDFCFIRSTB.NS",
    "AUBANK.NS",
    "BANDHANBNK.NS",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def fetch_price_data(tickers=TICKERS, start="2015-01-01", end=None):
    """
    Downloads daily adjusted close prices for the given tickers.

    We use "adjusted close" (auto_adjust=True) rather than raw close.
    Reason: raw close prices jump artificially on stock split / dividend
    dates, which would create fake, huge one-day "spread" moves that have
    nothing to do with the actual relationship between two stocks. Adjusted
    close removes that distortion. This is a standard backward-looking
    correction, not a forward-looking one, so it does not leak future info.

    Returns a single DataFrame: rows = dates, columns = tickers.
    """
    print(f"Downloading {len(tickers)} tickers from {start} to {end or 'today'}...")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # gives us split/dividend-adjusted prices directly
        progress=False,
    )

    # yfinance returns a multi-level column DataFrame when given multiple
    # tickers. We only want the "Close" (which, because auto_adjust=True,
    # is already the adjusted close) price for each ticker.
    prices = raw["Close"].copy()

    # Drop any ticker that came back completely empty (e.g. typo, delisted,
    # not available on Yahoo Finance) -- better to fail loudly here than
    # silently carry a column of all-NaN into the cointegration step.
    empty_cols = [col for col in prices.columns if prices[col].isna().all()]
    if empty_cols:
        print(f"WARNING: no data returned for {empty_cols}, dropping them.")
        prices = prices.drop(columns=empty_cols)

    return prices


def save_prices(prices: pd.DataFrame, filename="bank_prices.parquet"):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    prices.to_parquet(path)
    print(f"Saved {prices.shape[0]} rows x {prices.shape[1]} tickers to {path}")
    return path


def load_prices(filename="bank_prices.parquet"):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached data at {path}. Run fetch_price_data() and save_prices() first."
        )
    return pd.read_parquet(path)


if __name__ == "__main__":
    prices = fetch_price_data()
    save_prices(prices)
    print(prices.tail())
