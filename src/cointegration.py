"""
cointegration.py

Job of this file: given a table of stock prices, figure out which PAIRS of
stocks have a genuine, stable, mean-reverting relationship (cointegration),
as opposed to pairs whose charts just happen to look similar by chance.

IMPORTANT DISCIPLINE NOTE (read this before calling these functions):
Every function here should only ever be given a "formation window" slice of
data -- e.g. the last 12 months, as of some cutoff date. NEVER call these
functions on the full history and then trade over that same full history.
That would mean the pair-selection decision "knew about" price moves that,
in a live trading system, hadn't happened yet. The walk-forward engine
(built in a later step) is what enforces this properly -- this file just
provides the testing logic itself.
"""

import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def engle_granger_pair(price_a: pd.Series, price_b: pd.Series):
    """
    Runs the Engle-Granger two-step test in BOTH directions
    (A explained by B, and B explained by A), and returns whichever
    direction gives the stronger (lower p-value) result.

    Step 1: OLS regression to find the hedge ratio (beta).
    Step 2: ADF test on the regression residual (the "spread") to check
             if it's stationary (mean-reverting) or not.

    Returns a dict with both directions' results, plus which one "won".
    """
    results = {}

    for direction, (y, x) in [("a_on_b", (price_a, price_b)), ("b_on_a", (price_b, price_a))]:
        x_with_const = sm.add_constant(x)
        model = sm.OLS(y, x_with_const).fit()
        beta = model.params.iloc[1]
        alpha = model.params.iloc[0]
        spread = y - (alpha + beta * x)

        # adfuller returns (test_stat, p_value, ...) -- we only need the p-value
        adf_result = adfuller(spread, autolag="AIC")
        p_value = adf_result[1]

        results[direction] = {
            "alpha": alpha,
            "beta": beta,
            "adf_pvalue": p_value,
            "spread": spread,
        }

    # Pick whichever direction gave the more strongly stationary spread
    winner = min(results, key=lambda d: results[d]["adf_pvalue"])

    return {
        "winning_direction": winner,
        "adf_pvalue": results[winner]["adf_pvalue"],
        "beta": results[winner]["beta"],
        "alpha": results[winner]["alpha"],
        "spread": results[winner]["spread"],
        "both_directions": results,  # kept for transparency/auditing
    }


def johansen_confirms_cointegration(price_a: pd.Series, price_b: pd.Series, confidence="95%"):
    """
    Runs the Johansen trace test as an independent, symmetric cross-check.
    Returns True if the trace statistic exceeds the critical value at the
    chosen confidence level (meaning: reject "no cointegration").

    Johansen doesn't require picking a dependent variable, which is why we
    use it as a robustness check on top of Engle-Granger rather than relying
    on either test alone.
    """
    price_matrix = pd.concat([price_a, price_b], axis=1).values
    result = coint_johansen(price_matrix, det_order=0, k_ar_diff=1)

    confidence_col = {"90%": 0, "95%": 1, "99%": 2}[confidence]
    trace_stat = result.lr1[0]              # trace statistic for "at least 1 cointegrating relationship"
    critical_value = result.cvt[0, confidence_col]

    return trace_stat > critical_value


def screen_pairs(
    prices: pd.DataFrame,
    eg_pvalue_threshold=0.05,
    johansen_confidence="95%",
    use_bonferroni=True,
    min_abs_beta=0.1,
    require_positive_beta=True,
):
    """
    Tests every possible pair of tickers in `prices` and returns only the
    pairs that pass BOTH the Engle-Granger test AND the Johansen test,
    plus two extra guardrails explained below.

    `prices` should already be sliced down to a single formation window
    before this function is called -- this function has no idea what a
    "formation window" is, it just tests whatever data it's handed.

    GUARDRAIL 1 -- Bonferroni correction (use_bonferroni=True):
    We're running one statistical test per pair. Testing many pairs at once
    means some will look "significant" purely by chance (the more tests you
    run, the more 5%-level false positives you should expect). Bonferroni
    correction fixes this by dividing the p-value threshold by the number
    of tests run, e.g. 0.05 / 66 pairs =~ 0.00076 -- a much stricter bar.
    This trades away some real pairs (fewer false negatives avoided) in
    exchange for far fewer false positives, which is the safer trade for a
    trading strategy.

    GUARDRAIL 2 -- economic plausibility filter on beta:
    A statistically "significant" hedge ratio that is negative or very
    close to zero usually means the test found a coincidental pattern, not
    a real economic relationship -- two banks in the same sector should
    generally move in the SAME direction with a believable ratio. We
    reject any pair whose beta doesn't meet this basic sanity check, even
    if it passed both statistical tests.
    """
    tickers = prices.columns.tolist()
    num_tests = len(tickers) * (len(tickers) - 1) // 2
    effective_pvalue_threshold = (
        eg_pvalue_threshold / num_tests if use_bonferroni else eg_pvalue_threshold
    )

    candidates = []

    for ticker_a, ticker_b in itertools.combinations(tickers, 2):
        price_a = prices[ticker_a].dropna()
        price_b = prices[ticker_b].dropna()

        # Align dates in case of any missing-data mismatches
        aligned = pd.concat([price_a, price_b], axis=1, sort=True).dropna()
        if len(aligned) < 60:  # not enough overlapping history to trust the test
            continue

        eg_result = engle_granger_pair(aligned[ticker_a], aligned[ticker_b])

        if eg_result["adf_pvalue"] >= effective_pvalue_threshold:
            continue  # fails the (Bonferroni-corrected, if enabled) Engle-Granger bar

        beta = eg_result["beta"]
        if abs(beta) < min_abs_beta:
            continue  # too close to zero to be a believable real relationship
        if require_positive_beta and beta <= 0:
            continue  # same-sector banks should move together, not oppositely

        passes_johansen = johansen_confirms_cointegration(
            aligned[ticker_a], aligned[ticker_b], confidence=johansen_confidence
        )
        if not passes_johansen:
            continue

        candidates.append({
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "eg_pvalue": eg_result["adf_pvalue"],
            "eg_threshold_used": effective_pvalue_threshold,
            "beta": beta,
            "alpha": eg_result["alpha"],
            "direction": eg_result["winning_direction"],
        })

    return pd.DataFrame(candidates).sort_values("eg_pvalue").reset_index(drop=True)


if __name__ == "__main__":
    # Quick sanity check only -- NOT a real backtest formation window choice.
    # We deliberately use a SMALL, clearly-labeled slice of history here just
    # to confirm the code runs and produces sane-looking output. The actual
    # walk-forward engine (later step) is what decides real formation windows.
    from data_pipeline import load_prices

    prices = load_prices()
    sample_window = prices.loc["2015-01-01":"2015-12-31"]

    print(f"Sanity-check formation window: {sample_window.index.min()} to {sample_window.index.max()}")
    print(f"({len(sample_window)} trading days)\n")

    num_tests = len(prices.columns) * (len(prices.columns) - 1) // 2
    pairs = screen_pairs(sample_window)
    print(f"Tested {num_tests} possible pairs, using a Bonferroni-corrected")
    print(f"p-value threshold of 0.05 / {num_tests} = {0.05/num_tests:.6f}")
    print(f"plus a beta plausibility filter (positive, and not near-zero).\n")
    print(f"Found {len(pairs)} pairs that survive all the guardrails:\n")
    print(pairs)
