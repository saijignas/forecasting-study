# Protocol: when do simple forecasts win?

Version: `protocol-v1`. Written 2026-10-16, committed and tagged **before** any forecasting model was fitted or
scored and before any metric/harness code that touches real data was written. After the tag this file may only be
changed by appending rows to the deviations log (section 15), each in its own commit with date and reason.

Status of the repository when this protocol was written: raw data downloaded and hashed (`data/MANIFEST.json`),
series list fixed (`data/series_list.json`), raw sanity plots made (`results/figures/raw_*.png`). Nothing else.

## 1. Question and hypotheses

Question: on real monthly and daily business/economic series, when does a more complex model beat strong simple
baselines, and are its prediction intervals honest?

Confirmatory stratum: the 30 monthly FRED series (section 3). Secondary stratum: the 30 daily electricity series.
The secondary stratum is analysed with the same rules but is **not** part of the Holm family and its verdicts are
labelled "secondary replication".

* **H1 (point accuracy).** The gradient-boosted global model (`ML-G`, section 6) beats seasonal naive (`SNAIVE`) on
  MASE at horizon 12 by at least 0.03 MASE on at least 60% of the 30 monthly series.
* **H2 (statistical models).** The statistical model selected for each series on the validation window (the better
  of `ETS` and `ARIMA`, section 6.4) beats `SNAIVE` on MASE at horizon 12 (margin 0) on at least 60% of series.
* **H3 (intervals).** (a) Split-conformal intervals (section 8) of `ETS`, `ARIMA` and `ML-G` reach pooled empirical
  coverage within 3 percentage points of nominal 80% and of nominal 95%. (b) The model-based intervals of `ETS`
  and `ARIMA` under-cover, meaning pooled coverage more than 3 points below nominal at both levels. (b) is to be
  tested, not assumed.
* **H4 (simple wins somewhere).** There is at least one pre-defined series type (section 11.4) for which no complex
  model beats `SNAIVE` at horizon 12 (rule in section 11.4). The report names which.

Negative results are reported as plainly as positive ones. "Seasonal naive wins often" is a valid finding.

## 2. Allowed and forbidden actions (data-handling rules)

Allowed before the confirmatory run: counts, frequency, missing values, date ranges, raw sanity plots; unit tests on
synthetic data; running forecasters on the **validation window only** (section 5) for tuning, and on synthetic data
for smoke tests.

Forbidden: scoring any forecaster on a test origin's target (section 5) before the confirmatory run; choosing
anything (models, grids, series, thresholds, horizons, metrics) after seeing test-period results; dropping a series
for a model failure after the fact (failures use the fallback in section 6.6).

The confirmatory run is a single execution of `scripts/run_confirmatory.py` (Phase 2) from a clean checkout at a
recorded commit hash, after all model code is frozen. A rerun caused by a bug is allowed only if logged in section
15 with the reason and the commit hash; results of the aborted run are kept and mentioned in the report.

## 3. Data

All series are listed with reasons in `data/series_list.json` (written before any model was run). Hashes, dates,
row counts and licence notes are in `data/MANIFEST.json`.

### 3.1 Monthly series (confirmatory), window 1992-01 to 2025-09 (405 observations)

The common window ends at 2025-09 because the Oct-Nov 2025 US data-collection gap leaves a missing October 2025 value
in the CPI series; one common end date avoids interpolating actuals. All 30 series have 405 non-missing positive
observations in this window (checked by the download script).

| # | FRED id | Group | Adj. | Why chosen (fixed before any model was run) |
|---|---|---|---|---|
| 1 | RSAFSNA | retail | NSA | Headline consumer demand; strong December seasonality |
| 2 | RSAFS | retail | SA | SA twin of #1; contrasts seasonal vs de-seasonalised behaviour |
| 3 | RSMVPDN | retail | NSA | Largest durable-goods retail category; cyclical |
| 4 | RSGCSN | retail | NSA | Staple goods; stable level and seasonality |
| 5 | RSCCASN | retail | NSA | Highly seasonal discretionary category; pandemic shock |
| 6 | RSNSRN | retail | NSA | E-commerce proxy; strong trend, evolving seasonality |
| 7 | RSBMGESDN | retail | NSA | Weather-driven seasonality tied to housing |
| 8 | RSFSDPN | retail | NSA | Services-like category; 2020 collapse and recovery |
| 9 | RSGASSN | retail | NSA | Energy-price volatility on top of seasonality |
| 10 | RSFHFSN | retail | NSA | Housing-linked discretionary category |
| 11 | INDPRO | industrial | SA | Standard real-activity indicator |
| 12 | IPMAN | industrial | SA | Manufacturing; supply-chain relevance |
| 13 | IPB50001N | industrial | NSA | Raw total index with seasonal pattern |
| 14 | IPG2211N | industrial | NSA | Electric power generation index; strong seasonality, stable level |
| 15 | IPG3361T3N | industrial | NSA | Volatile durable-goods manufacturing |
| 16 | CPIAUCNS | cpi | NSA | Headline inflation; trending |
| 17 | CPILFESL | cpi | SA | Core inflation; smooth |
| 18 | CUUR0000SAF11 | cpi | NSA | Food-at-home prices |
| 19 | CUUR0000SEHA | cpi | NSA | Rent; very smooth, persistent |
| 20 | CUUR0000SETB01 | cpi | NSA | Gasoline; highly volatile |
| 21 | CUUR0000SEHF01 | cpi | NSA | Electricity prices (energy analytics) |
| 22 | HOUST | housing | SA | Housing starts; noisy cycle |
| 23 | HOUSTNSA | housing | NSA | NSA twin of #22; weather seasonality |
| 24 | PERMIT | housing | SA | Leading housing indicator |
| 25 | TOTALSA | auto | SA | Auto-sector demand |
| 26 | TOTALNSA | auto | NSA | NSA twin of #25 |
| 27 | PAYNSA | labor_activity | NSA | Large, smooth level series; mild seasonality |
| 28 | UNRATENSA | labor_activity | NSA | A bounded rate rather than a level |
| 29 | TRFVOLUSM227NFWA | labor_activity | NSA | Mobility/activity; strong seasonality |
| 30 | M2SL | money | SA | Smooth, strongly trending, non-seasonal |

Candidates probed for completeness and not used (completeness/duplication only, no model performance involved):
RSXFSN (near-duplicate of #1), IPUTIL and IPG2211A2N (utility indices overlapping #14), CPIAUCSL and CPIENGNS
(overlap with #16/#20), CPIMEDNS and CPIAPPNS (missing values inside the window), UMCSENT (gaps), AIRRPMTSID11
(starts 2000, ends 2026-06), DGORDER, AMTMNO and TTLCON (start after 1992-01), PERMITNSA, PAYEMS, UNRATE, PCEPI,
ALTSALES, IPG334N, IPG325N (surplus to the target of at most 30 series).

### 3.2 Daily electricity series (secondary)

Monash Time Series Forecasting Repository "Electricity Hourly Dataset" (Zenodo record 4656140, CC BY 4.0, 11.8 MB
zip, 321 hourly series 2012-01-01 to 2014-12-31, derived from UCI ElectricityLoadDiagrams20112014, CC BY 4.0).
Aggregation: sum of the 24 hourly values of each calendar day (1096 days). Selection rule (fixed before any model
was run): eligible = at most 3 zero hourly values over the whole span (tolerates the zeros at the March clock-change
hour) and no missing values; 233 of 321 series are eligible; the 30 used are the eligible series at positions
floor(i * 233 / 30), i = 0..29, in file order. The selected ids are listed in `data/MANIFEST.json`
(`series_ids`). All daily totals are positive (0 zero daily totals).

### 3.3 Data inspection log (what was looked at before this protocol)

Only counts, frequency, missing values, date ranges and the raw plots in `results/figures/raw_*.png` were inspected.
From the plots, before writing this protocol: the monthly series contain a sharp 2020 shock (e.g. retail, industrial
production, unemployment) and a 2021-2023 inflation surge (CPI series, rent, electricity); several daily series
show level shifts or flat stretches (e.g. T116, T283), and year-end holiday dips. The test window below was fixed by a
calendar rule (last 8 half-yearly origins ending at the final observation), not by looking at model performance, but
the author knew a pandemic shock exists in the sample. No forecaster has been fitted.

## 4. Notation

A series is `y_1..y_T`. A **forecast origin** `t` is the last observed time index: a forecast made at origin `t`
may use `y_1..y_t` and nothing else (no value with index `t+1` or later, no value from any other split). It
produces forecasts `yhat_{t+h|t}` for horizons `h = 1..H`. `m` is the seasonal period: 12 (monthly), 7 (daily).
All indices are 1-based.

## 5. Splits and rolling-origin schedule

Expanding window: the training window of origin `t` is always `y_1..y_t`.

| | Monthly (30 series, T = 405) | Daily (30 series, T = 1096) |
|---|---|---|
| Horizons | 1, 3, 6, 12 | 1, 7, 14, 28 |
| Max horizon H | 12 | 28 |
| Seasonal period m | 12 | 7 |
| Origin spacing | 6 months | 28 days |
| **Validation origins** (tuning only) | t = 300, 306, 312, 318, 324 (2016-12, 2017-06, 2017-12, 2018-06, 2018-12) | t = 732, 760, 788, 816, 844 (2014-01-01 ... 2014-04-23) |
| Last validation target index | 324 + 12 = 336 (2019-12) | 844 + 28 = 872 |
| **Test origins** (confirmatory, 8) | t = 351, 357, 363, 369, 375, 381, 387, 393 (2021-03, 2021-09, 2022-03, 2022-09, 2023-03, 2023-09, 2024-03, 2024-09) | t = 872, 900, 928, 956, 984, 1012, 1040, 1068 (2014-05-21 ... 2014-12-03) |
| Test target indices | 352..405 (2021-04..2025-09) | 873..1096 |
| Min. training length | 300 (validation), 351 (test) | 732 (validation), 872 (test) |

Rules: (i) the last test origin plus the maximum horizon equals T exactly (393 + 12 = 405; 1068 + 28 = 1096);
(ii) every validation target index is at most the first test origin (so validation never touches test targets);
(iii) a test target `y_{t+h}` is never inside any training window used for hyperparameter selection or fitting of a
forecast whose origin is at or before it. The validation window deliberately precedes the 2020 shock for monthly
series (a design choice committed in advance; it means tuned configurations were not chosen on shock data, and it
is listed as a threat in section 14).

Hyperparameters, order grids and model-choice rules are tuned **once, on the validation origins only**, before the
first test origin is touched; the selected configuration is then fixed. At each test origin the parameters of the
fixed configuration are re-estimated on the expanding training window (ETS/ARIMA/Theta smoothing and ARMA
parameters; gradient-boosted trees are refitted).

## 6. Models

Fixed library versions (`requirements.lock`): Python 3.11.2, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1,
statsmodels 0.15.0, scikit-learn 1.9.1.

### 6.1 Baselines (no tuning)

* `NAIVE`: `yhat_{t+h|t} = y_t`.
* `SNAIVE`: `yhat_{t+h|t} = y_{t+h-m*k}` with `k = ceil(h/m)` (the last observed value of the same season).
* `DRIFT`: `yhat_{t+h|t} = y_t + h * (y_t - y_1)/(t - 1)`.
* `SMEAN` (seasonal mean): mean of the `K = 5` most recent observations of the same season,
  `yhat_{t+h|t} = (1/5) * sum_{j=0..4} y_{t+h-m*(k+j)}`, `k = ceil(h/m)`.

### 6.2 ETS

`statsmodels.tsa.exponential_smoothing.ets.ETSModel`, seasonal period `m`. Grid (18 configurations): error in
{additive, multiplicative} x trend in {none, additive, additive damped} x seasonal in {none, additive,
multiplicative}. Series are strictly positive so multiplicative forms are admissible. Maximum-likelihood fit with the
library defaults (`disp=False`, `maxiter=1000`).

### 6.3 ARIMA

`statsmodels.tsa.statespace.sarimax.SARIMAX` (no exogenous regressors, no trend term, no log transform,
`enforce_stationarity=True`, `enforce_invertibility=True`). Grid (12 configurations): non-seasonal order `(p,d,q)` in
{(0,1,1), (1,1,0), (1,1,1), (2,1,1)} x seasonal order `(P,D,Q)_m` in {(0,0,0), (0,1,1), (1,1,1)}.

### 6.4 Selection rule for ETS and ARIMA, and `STAT*`

For each series and each of ETS and ARIMA, the configuration with the lowest **validation score** is selected, where
the validation score is the mean of the scaled absolute error (section 9) over the 5 validation origins and the 4
horizons of the frequency. `STAT*` for series `i` is whichever of {ETS, ARIMA} (each with its selected configuration)
has the lower validation score for that series (ties go to ETS). It is fixed before the test run.

### 6.5 Theta

`statsmodels.tsa.forecasting.theta.ThetaModel(y, period=m, deseasonalize=True, method="auto")`, theta = 2 (library
default). No tuning.

### 6.6 Failure handling

If a configuration fails to fit or returns non-finite forecasts at a validation origin, that configuration is
excluded for that series. If every configuration of a model fails for a series, `SNAIVE` forecasts are substituted
for that model/series (flagged `fallback=1` in the results and counted in the report). If a fit fails at a test
origin, the same fallback is used for that origin only. No series is dropped.

### 6.7 Gradient-boosted models

`sklearn.ensemble.HistGradientBoostingRegressor` (loss squared error, `min_samples_leaf=20`,
`l2_regularization=0`, `early_stopping=False`, `random_state=20261016`).

* `ML-G` (global): one model per horizon `h` in the frequency's horizon set (direct strategy), trained on the
  stacked rows of all series of the frequency (monthly: the 30 FRED series; daily: the 30 electricity series).
* `ML-L` (per-series): same features and the same selected configuration, trained on one series only.
* Grid (8 configurations): `learning_rate` in {0.05, 0.1} x `max_leaf_nodes` in {8, 31} x `max_iter` in {100, 300}.
  One configuration per frequency group is chosen for `ML-G` by the lowest validation score (mean scaled absolute
  error over all series, the 5 validation origins and the 4 horizons); `ML-L` uses the same configuration.
* No hyperparameter is chosen on any test origin.

### 6.8 Combination

`COMB`: the arithmetic mean of the point forecasts of `ETS`, `ARIMA` (each with its selected configuration) and
`ML-G`. Always run; included in all analyses (it is in the model list of H4).

## 7. Features for the ML models (no look-ahead)

Let `s` be a feature origin (a last observed index). Define the scale `c_s = mean(y_{s-w0+1..s})` with `w0 = 12`
(monthly) or `28` (daily), and the maximum lookback `L = 24` (monthly) or `56` (daily). Features at origin `s`:

| Feature | Monthly | Daily |
|---|---|---|
| lags | `y_{s-k}/c_s`, k = 0..23 | `y_{s-k}/c_s`, k = 0..27 |
| rolling means | `mean(y_{s-w+1..s})/c_s`, w in {3, 6, 24} | w in {7, 14, 56} |
| rolling std | `std(y_{s-11..s})/c_s` (ddof=1) | `std(y_{s-27..s})/c_s` |
| seasonal ratio | `y_s / y_{s-12}` | `y_s / y_{s-7}` |
| calendar of the target | month number of `s+h` | day-of-week and month number of `s+h` |

Target: `y_{s+h}/c_s`; forecasts are multiplied back by `c_s`. No series identifier, no future information. Every
feature at origin `s` is a function of `y_1..y_s` only; calendar features depend only on the known target date.

Training rows for a model fitted at origin `t` and horizon `h`: all `(series, s)` with `L <= s` and `s + h <= t`
(i.e. every target lies inside the training window; rows whose target would be later than `t` are never built).
Row construction and the "no feature at origin `s` uses `y_j`, `j > s`" property are unit-tested
(`tests/test_features.py`, synthetic data; the test perturbs all values after `s` and asserts identical features).

## 8. Prediction intervals

Nominal levels 80% (quantiles 0.1/0.9) and 95%.

* **Model-based** (`ETS`, `ARIMA`, `Theta`): the library's analytic or simulated prediction intervals
  (`ETSResults.get_prediction(...).summary_frame(alpha)` with `simulate_repetitions=1000` and a fixed seed;
  `SARIMAXResults.get_forecast(...).conf_int(alpha)`; `ThetaModel.prediction_intervals`), `alpha` = 0.2 and 0.05.
* **Split-conformal** (all models including baselines, `ML-G`, `ML-L`, `COMB`): for origin `t`, horizon `h`, level
  `1-alpha`:
  1. Calibration length `n_cal` = 36 (monthly) or 112 (daily). The proper-training window is `y_1..y_{t-n_cal}`;
     the model (with the already-selected configuration) is fitted on it once. Models with parameters (ETS, ARIMA,
     Theta, ML) keep those parameters fixed; ETS/ARIMA states and ML features at calibration origin `c` are updated
     with data up to `c` only (baselines simply use data up to `c`).
  2. For each calibration origin `c = t-n_cal, ..., t-h`, compute the h-step forecast from origin `c` and the
     absolute residual `r_c = |y_{c+h} - yhat_{c+h|c}|` (every `c+h <= t`). `n = n_cal - h + 1` residuals.
  3. `q = ` the `k`-th smallest residual with `k = min(n, ceil((n+1)(1-alpha)))`.
  4. Interval for the real forecast (model fitted on the full window `y_1..y_t`): `[yhat - q, yhat + q]`.
  Residuals are in original units, one `q` per (series, origin, horizon, model, level). Time-series residuals are not
  exchangeable, so finite-sample coverage guarantees do not hold; coverage is evaluated empirically.

## 9. Metrics (exact definitions)

For a series, origin `t`, horizon `h`, target `y = y_{t+h}`, forecast `f`, lower/upper bounds `l`, `u`:

* In-sample seasonal-naive scales over the training window:
  `S1(t) = (1/(t-m)) * sum_{i=m+1..t} |y_i - y_{i-m}|`, `S2(t) = (1/(t-m)) * sum_{i=m+1..t} (y_i - y_{i-m})^2`.
* **Scaled absolute error** `ASE = |y - f| / S1(t)`. **MASE(series, h)** = mean of ASE over the series' origins
  (primary metric; every comparison uses MASE at horizon 12 monthly / 28 daily unless stated).
* **sMAPE**(series, h) = mean over origins of `200 * |y - f| / (|y| + |f|)` (percent; 0 if both are 0).
* **RMSSE**(series, h) = `sqrt( mean over origins of (y - f)^2 / S2(t) )`.
* **Pinball loss** at level `tau`: `rho_tau(y, q) = max(tau (y - q), (tau - 1)(y - q))`, reported divided by `S1(t)`.
  Quantile forecasts at the 80% level: `q_0.1 = l`, `q_0.5 = f` (the point forecast), `q_0.9 = u`. The reported
  figure is the mean of the three scaled losses.
* **Coverage** = mean of `1[l <= y <= u]`; **width** = mean of `(u - l)/S1(t)`.
* **Pooled coverage** `C(model, level, stratum)` = unweighted mean over series of the series' coverage computed over
  all its origins and all 4 horizons of the frequency (each series has the same number of forecasts, so this equals
  the pooled mean over all forecasts).

## 10. Statistics

* **Bootstrap over series**: `B = 10000` resamples of the series (with replacement) using
  `numpy.random.default_rng(20261016)`; the same resampled index sets are used for every statistic. This treats series
  as exchangeable units and therefore ignores their cross-correlation (a threat, section 14).
* **Wilcoxon signed-rank** (two-sided, `scipy.stats.wilcoxon`) on the paired per-series MASE differences
  `MASE_SNAIVE - MASE_model` at the reference horizon: a secondary check, **not** used for any decision.
* **Holm correction** across the four hypotheses (section 11). Reported adjusted p-values: `p_adj_(i) =
  min(1, max_{j<=i} (m - j + 1) p_(j))` over the ascending p-values, `m = 4`.

## 11. Decision rules

For each hypothesis `H` there are elementary claims `c`. Each claim has a point criterion (true/false on the
observed data) and two one-sided bootstrap p-values: `p+_c` for rejecting "claim false" (evidence for the claim) and
`p-_c` for rejecting "claim true" (evidence against). Hypothesis-level quantities:

* `p+_H = max_c p+_c` (intersection-union: every claim must be supported).
* `p-_H = min( 1, (number of claims whose criterion is false) * min over those claims of p-_c )`.

The four values `p+_H` are Holm-adjusted together; the four values `p-_H` are Holm-adjusted together (separately).
Verdict for `H` (alpha = 0.05):

* **Supported**: every claim's point criterion is true and Holm-adjusted `p+_H < 0.05`.
* **Not supported**: at least one claim's point criterion is false and Holm-adjusted `p-_H < 0.05`.
* **Inconclusive**: everything else. If all point criteria hold but `p+_H >= 0.05`, the report must say
  "point estimates meet the criterion but the claim is not statistically established" (likely for H3a, see below).

### 11.1 H1

One claim. `win_i = 1[ MASE_SNAIVE,i(12) - MASE_ML-G,i(12) >= 0.03 ]`, `W1 = mean_i win_i` over the 30 monthly series.
Criterion: `W1 >= 0.60`. Bootstrap replicate `W1*`: `p+ = share of replicates with W1* <= 0.60`;
`p- = share of replicates with W1* >= 0.60`.

### 11.2 H2

One claim. `win_i = 1[ MASE_SNAIVE,i(12) - MASE_STAT*,i(12) > 0 ]`, `W2 = mean_i win_i`. Criterion `W2 >= 0.60`; p-values
as for H1 with `W2`. (Descriptive extras, not decisions: ETS-only and ARIMA-only win rates, and the share of series
where `STAT*` beats `ML-G`.)

### 11.3 H3

Monthly stratum; pooled coverage `C` (section 9) per cell, nominal level `nu` in {0.80, 0.95}; bootstrap replicate
`C*` computed on the resampled series.

* **H3a** (6 claims: models `ETS`, `ARIMA`, `ML-G` x 2 levels, split-conformal intervals). Criterion for a cell:
  `|C - nu| <= 0.03`. `p+_c = max(share of C* <= nu - 0.03, share of C* >= nu + 0.03)` (equivalence-style: both
  tails of the band must be rejected). If `C < nu - 0.03`: `p-_c = share of C* >= nu - 0.03`; if `C > nu + 0.03`:
  `p-_c = share of C* <= nu + 0.03`.
* **H3b** (4 claims: `ETS`, `ARIMA` x 2 levels, model-based intervals). Criterion for a cell: `C < nu - 0.03`.
  `p+_c = share of C* >= nu - 0.03`; `p-_c = share of C* < nu - 0.03`.
* Power note, fixed in advance: with 30 series the standard error of `C` is probably about 2 points, so H3a cannot be
  statistically established unless coverage is very close to nominal and stable; "inconclusive with all point
  criteria met" is an expected and acceptable outcome for H3a. H3's claims are the 10 cells; its verdict uses the
  hypothesis-level rules above with `m = 4` hypotheses in the Holm family.

### 11.4 H4

One claim (existence). Model list `M = {ETS, ARIMA, THETA, ML-G, ML-L, COMB}`. Per series, with `d_{i,M} =
MASE_SNAIVE,i(12) - MASE_M,i(12)`. Types (a type needs at least 3 series; each is a pre-specified subset of the 30
monthly series):

* by `group` in the series list (retail, industrial, cpi, housing, labor_activity; `auto` and `money` have fewer
  than 3 series and are not types);
* by adjustment: `SA`, `NSA`;
* by seasonal strength: `high` if `F_S >= 0.6`, else `low`, where `F_S = max(0, 1 - Var(R)/Var(S + R))` from
  `statsmodels.tsa.seasonal.STL(log y, period=12, robust=False)` applied to `y_1..y_351` (the first test origin's
  training window);
* by volatility: `low` if `sd( log(y_t/y_{t-12}) , t = 13..351 )` is at most the median of that quantity across the 30
  series, else `high`;
* the cross type `high seasonal strength AND low volatility`.

Type `S` satisfies the rule `R(S)` if `|S| >= 3` and, for every model `M` in the list: (i) the share of series in `S`
with `d_{i,M} >= 0.03` is at most 1/3, and (ii) the median of `d_{i,M}` over `S` is at most 0. Criterion: at least one
type satisfies `R`. Bootstrap (stability, not a significance test of a specific type): in each replicate the
resampled series keep their type labels and `R` is evaluated on all types with at least 3 resampled series; `s` =
share of replicates in which at least one type satisfies `R`. `p+ = 1 - s`; `p- = s`. The report lists every type
that satisfies `R` and states that the type search is data-dependent (12 types examined).

### 11.5 Secondary stratum (daily electricity, no Holm, labelled replication)

The same rules with the reference horizon 28 (H1, H2, with `ML-G`/`STAT*` fitted on the daily series), H3 with the
daily pooled coverage, `n_cal = 112` and the three-model/two-level cells, and descriptive per-series results for H4
(all daily series share one group, so no type analysis).

## 12. Seeds and compute budget

Master seed 20261016. numpy `default_rng(20261016)` for the bootstrap; scikit-learn `random_state=20261016`; any
simulation in ETS intervals seeded from the same value through `numpy.random.seed(20261016)` immediately before each
call. Single process, `OMP_NUM_THREADS=1`. Budget: at most 6 hours wall-clock CPU in total, peak resident memory at
most about 1.2 GB (series processed one at a time, results written to `results/cache/` per series). If the ARIMA grid
cannot finish within budget the order grid is **not** reduced silently: the shortfall and any change are logged in
section 15 before results are used.

## 13. Outputs

`scripts/make_tables.py` (one script) regenerates every table and figure in the report from the saved raw forecasts:
error by horizon, win-rate bars with bootstrap CIs, coverage plots, example forecasts with intervals, the type table
for H4, the full table of validation-selected configurations, and the fallback counts.

## 14. Threats to validity (stated in advance)

1. **Dependence between series**: SA/NSA twins and total/component pairs are not independent; the series bootstrap
   ignores this and understates uncertainty. Origins of one series (6 months apart, horizons up to 12) overlap in
   their target windows.
2. **Small samples**: 30 series x 8 origins per stratum; per-cell coverage standard errors of about 2 points; win-rate
   resolution 1/30.
3. **Test period regime**: monthly test targets (2021-04..2025-09) include post-pandemic inflation, supply-chain and
   rate shocks; results are conditional on that regime and need not hold elsewhere. Training windows for test origins
   contain the 2020 shock.
4. **Validation/test regime gap**: the monthly validation window (targets to 2019-12) deliberately avoids 2020 but
   is 1.25 or more years before the first test origin; tuned configurations may be stale. The daily validation window is
   January-May 2014 and the test window June-December 2014 (different season of the year).
5. **FRED revisions**: values can be revised; the manifest hash pins the data used. Data are US only.
6. **Electricity data**: only 3 years of data, so no annual seasonality can be learnt for the first year; the clock-change
   hour contributes zeros that slightly distort daily totals in 3 days; some series have level shifts or flat stretches
   (possibly imputed); the 30 series are chosen by a deterministic rule from 233 eligible ones (88 of 321 were excluded
   for many zero hours) so results describe "clean, long-lived load series" rather than all clients.
7. **Intervals**: time-series residuals break exchangeability; conformal coverage is empirical only. ETS multiplicative
   intervals are simulated.
8. **Multiplicity**: many metrics and models are reported; only H1-H4 are confirmatory. The H4 type search covers
   12 data-dependent types.
9. **Excluded methods**: no deep learning, no foundation models, no exogenous regressors; conclusions concern the
   models listed in section 6 only.
10. **Implementation risk**: library defaults (statsmodels ETS/SARIMAX optimisers) can fail silently on some series;
    the fallback count is reported.

## 15. Deviations log

Append-only. Each row is added in a separate commit after the tag, with date and reason.

| # | Date | Section | Change | Reason | Effect on analysis | Commit |
|---|------|---------|--------|--------|--------------------|--------|
| | | | | | | |
