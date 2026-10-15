# forecasting-study

**Working title:** *When do simple forecasts win? A pre-specified backtest of baselines, statistical and ML
models with calibrated prediction intervals.*

**Status: Phase 1 of 2 (protocol and infrastructure). There are no results yet.** No forecasting model has been
fitted or scored on the real data. Everything below describes what has been committed so far and what is still to do.

## Question
On real monthly and daily business/economic series, when does a more complex model beat strong simple baselines,
and are its prediction intervals honest? Four hypotheses (H1-H4) with numeric decision rules are fixed in
[`docs/protocol.md`](docs/protocol.md), which was committed and tagged `protocol-v1` before any metric or harness
code that touches real data was written.

## Scope (honest)
* Data: 30 US monthly series from FRED (1992-01 to 2025-09) and 30 daily electricity-load series (2012-2014,
  aggregated from the Monash/UCI hourly data). Public data only; raw data is **not** redistributed here
  (see `data/MANIFEST.json` for URLs, hashes and licence notes; run `scripts/download_data.py` to fetch it).
* Methods (Phase 2): naive, seasonal naive, drift, seasonal mean, ETS, ARIMA, Theta, gradient boosting with lag
  features (global and per-series), an equal-weight combination, and split-conformal intervals.
* Not covered: deep learning, foundation models, exogenous regressors, hierarchical reconciliation, probabilistic
  models beyond the intervals above, any data outside the two sources.
* What a finished study could and could not show: it is a backtest on 60 series from two sources, not a general
  ranking of methods. The series are correlated with each other (e.g. seasonally adjusted/unadjusted twins), the
  test period contains the post-pandemic inflation shock, and a single run of 8 origins per series is a small sample.

## What exists in Phase 1
| Path | Content |
|---|---|
| `docs/protocol.md` | Full pre-specified protocol, tagged `protocol-v1`, with a deviations log |
| `data/series_list.json` | The pre-specified series list with the reason each was chosen |
| `scripts/download_data.py` | Downloads the data, writes `data/MANIFEST.json` (URL, date, SHA-256, rows, range, licence) |
| `scripts/plot_raw_sanity.py` | Raw sanity plots (`results/figures/raw_*.png`); no model is fitted |
| `src/forecasting_study/` | Metrics, rolling-origin splitter, four baselines, lag-feature builder, split-conformal code |
| `tests/` | Unit tests (metrics vs hand-computed values, split-leakage test, feature-leakage test, conformal coverage test on synthetic data) |
| `.github/workflows/tests.yml` | CI that runs the tests (configured, not yet run on GitHub) |

## Reproduce
```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -I scripts/download_data.py        # fetch data + write data/MANIFEST.json
.venv\Scripts\python -m pytest
```
FRED values can be revised, so a fresh download may differ from the hashes in the manifest; compare with
`python -I scripts/download_data.py --verify`.

## Phase 2 (not started)
Fit ETS, ARIMA, Theta and the gradient-boosted models, tune only on the validation window defined in the protocol,
run the confirmatory backtest once, regenerate all tables from one script, write `docs/report.md`.

## Licence and attribution
Code: not yet licensed (choose before publishing). Data: see `data/MANIFEST.json`. Electricity data: CC BY 4.0,
Godahewa et al. (2021), Monash Time Series Forecasting Repository, derived from Trindade (2015), UCI ML Repository.
FRED data: source agencies as listed in `data/series_list.json`, accessed via FRED.
