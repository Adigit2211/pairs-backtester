# Statistical Arbitrage Pairs Trading Backtester

A walk-forward-validated, cost-aware pairs trading backtester on Indian
banking stocks, built to be **auditable** by a skeptical reader — every
place this backtest could have leaked future information is explicitly
flagged below, rather than left for a reviewer to have to find.

**Headline result, stated upfront and honestly:** this strategy, in this
form, does **not** show a real edge after realistic transaction costs, in
either the in-sample walk-forward period or the untouched final holdout.
See [Results](#results) for the full numbers and why that is the expected,
credible outcome for this strategy family, not a sign of a bug.

---

## 1. Universe

12 NSE-listed Indian banking stocks, pulled via `yfinance`:
`HDFCBANK, ICICIBANK, KOTAKBANK, AXISBANK, SBIN, INDUSINDBK, BANKBARODA,
PNB, FEDERALBNK, IDFCFIRSTB, AUBANK, BANDHANBNK` (all `.NS`).

**Survivorship bias — explicitly unmitigated.** `yfinance` only exposes
tickers that exist *today*. There is no point-in-time constituent list
available through this data source, so any bank that delisted, merged, or
was restructured out of existence during the sample period (e.g. Yes
Bank's 2020 reconstruction) is silently absent from this universe. This is
a real limitation of the project, stated here rather than hidden.

Adjusted close prices are used throughout (`yfinance(auto_adjust=True)`),
to avoid fake price jumps on stock-split/dividend dates. This is a
standard backward-looking correction and does not introduce lookahead.

## 2. Cointegration methodology

**Primary test: Engle-Granger two-step**, chosen because for a two-asset
pair it gives a directly tradeable hedge ratio and is the standard first
pass in the pairs-trading literature. Its known weakness is **asymmetry**
(regressing A-on-B vs B-on-A can disagree) — we run both directions and
keep the stronger one, logging both for auditability
(`src/cointegration.py::engle_granger_pair`).

**Cross-check: Johansen trace test**, which is symmetric and doesn't
require picking a dependent variable. A pair is only accepted if **both**
tests agree.

**Multiple-testing correction.** Testing all 66 possible pairs at a raw
5% significance level would produce ~3.3 false positives from chance alone.
We apply a **Bonferroni correction**: the effective p-value threshold is
`0.05 / (number of pairs tested)`. An early uncorrected run found 17
"cointegrated" pairs; after correction, this fell to 1 pair for the same
window — several of the discarded pairs had implausible hedge ratios
(near-zero or negative), confirming they were likely statistical noise.

**Economic plausibility filter.** Independent of the statistical tests, we
reject any pair whose beta is negative or too close to zero
(`min_abs_beta`, `require_positive_beta` in `screen_pairs()`) — two banks
in the same sector should move in the same direction with a believable
ratio; a statistically "significant" result that fails this basic sanity
check is more likely spurious than a real relationship.

## 3. Strategy

- Spread: `price_A - (alpha + beta * price_B)`, alpha/beta fixed from the
  formation-window regression.
- Z-score: `(spread - formation_mean) / formation_std`, using mean/std
  computed **only** from the formation window and held fixed while trading.
- Entry: `|z| >= 2.0`. Exit: `|z| <= 0.5`.
- Stop-loss: `|z| >= 3.5` (relationship may have broken down — exit rather
  than hope).
- Time-stop: force-exit after 30 trading days in a position.
- Cooldown: after a stop-loss, wait 5 trading days before re-entering the
  same pair. Added after observing (see `git log`) that the strategy would
  otherwise repeatedly re-enter a relationship that had just shown it
  wasn't reverting normally.

## 4. Walk-forward backtest architecture

12-month formation window → 3-month trading window, stepped forward every
3 months, re-screening cointegration fresh at every step. See
`src/backtest_engine.py::generate_walk_forward_windows` and
`run_walk_forward`.

### Explicit leakage checklist

| # | Risk | How it's prevented |
|---|------|---------------------|
| 1 | Fitting alpha/beta/spread-stats on data that includes the trading window | Formation and trading windows are strictly non-overlapping; formation always precedes trading chronologically. |
| 2 | Same-bar signal + execution | Position decided using day *t*'s close is applied starting from day *t*→*t+1*'s return (`position.shift(1)` in `_pair_return_series`), not day *t*'s own return. |
| 3 | Re-estimating z-score stats mid-window using future data | `formation_mean`/`formation_std` are computed once from the formation window and passed as fixed constants into `generate_signals()`. |
| 4 | Cherry-picking the regression direction post hoc | Both Engle-Granger directions are computed and logged; the choice is by lowest p-value, an objective rule, not by which "looks better" afterward. |
| 5 | Retuning strategy parameters after seeing out-of-sample results | Not done. See [Results](#results) — the holdout numbers are reported as obtained, with no adjustment afterward. |
| 6 | Multiple-testing false positives in pair selection | Bonferroni-corrected p-value threshold (see Section 2). |
| 7 | Universe selection informed by hindsight ("only include banks I know did well") | Universe was fixed before any backtest was run, based on current index membership, not backtest performance. |

### Two explicitly flagged simplifications (not leakage, but worth knowing)

- **Execution lag proxy**: only daily close prices are available (no
  intraday data), so the 1-day lag described above stands in for realistic
  execution delay. It is not a full order-book/intraday simulation.
- **Static dollar-neutral hedge weighting**: the share-based hedge ratio is
  converted to a dollar-neutral weight once, using the trading window's
  first-day prices, and held fixed for that window (no continuous
  rebalancing). Standard simplification in pairs-trading backtests.

## 5. Transaction costs

5 bps per leg, charged whenever a position changes (entry, exit, or flip) —
`cost_bps` in `run_walk_forward`. No separate slippage model beyond the
execution-lag proxy above. Short-selling costs (securities lending /
delivery-short mechanics specific to NSE) are **not** modeled; this is a
simplification, explicitly flagged, not a hidden assumption.

## 6. Results

Full history: 2015–2026. Final 12 months (2025-07-01 onward) held out as
out-of-sample and never used to select any strategy parameter.

| Metric | In-sample (2016–2025) | Out-of-sample holdout (last 12mo) |
|---|---|---|
| Total return | -17.15% | -3.08% |
| Annualized Sharpe | -0.31 | -0.72 |
| Max drawdown | -26.91% | -5.43% |
| Win rate | 32.9% | 47.1% |
| Number of trades | 76 | 17 |

**Why this is a credible result, not a failed project.** Classic two-stock
statistical arbitrage on large, liquid, heavily-analyzed names — which is
exactly what NSE banking blue-chips are — is widely documented in the
academic literature to have decayed in profitability since the 2000s, as
more capital chased the same simple mean-reversion signal. A rigorous,
leakage-free backtest producing a modest loss is a far more credible
outcome than a suspiciously large profit from the same simple rules would
have been. Notably, the out-of-sample max drawdown (-5.4%) is much smaller
than in-sample (-26.9%), suggesting the risk controls (stop-loss,
time-stop, cooldown) did limit downside even in a losing period.

We did not retune any parameter after observing the holdout result. If
someone wants to explore further (broader universe, different sectors,
intraday execution data), that requires a **new** experiment with its own
fresh, never-seen holdout period — not a rerun against this one.

## 7. How to run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/data_pipeline.py      # downloads and caches price data
python src/run_report.py         # runs the full walk-forward backtest + report
```

## 8. Project structure

```
src/                  backtest engine (data, cointegration, strategy, backtest, metrics)
dashboard/             Streamlit app showing current pairs and live paper signals
.github/workflows/     daily scheduled job that refreshes signals
data/                  cached price data (gitignored)
```
