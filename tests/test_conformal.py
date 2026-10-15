"""Split-conformal code: rank rule, no-lookahead, and an empirical coverage test on synthetic data."""
import math

import numpy as np
import pytest

from forecasting_study import metrics
from forecasting_study.baselines import naive, snaive
from forecasting_study.conformal import (calibration_residuals, conformal_interval, conformal_quantile,
                                         conformal_rank, split_conformal_forecast)


def test_rank_rule_hand_computed():
    assert conformal_rank(9, 0.1) == 9          # ceil(10 * 0.9) = 9
    assert conformal_rank(19, 0.1) == 18        # ceil(20 * 0.9) = 18
    assert conformal_rank(25, 0.05) == 25       # ceil(26 * 0.95) = 25
    assert conformal_rank(25, 0.2) == 21        # ceil(26 * 0.8) = 21
    assert conformal_rank(10, 0.05) == 10       # ceil(11 * 0.95) = 11 -> clipped to n
    with pytest.raises(ValueError):
        conformal_rank(0, 0.1)


def test_quantile_hand_computed():
    r = [3.0, -1.0, 2.0, -5.0, 4.0, 0.5, 1.5, -2.5, 3.5]            # |.| sorted: .5 1 1.5 2 2.5 3 3.5 4 5
    assert conformal_quantile(r, 0.1) == 5.0                          # rank 9
    assert conformal_quantile(r, 0.3) == 3.5                          # ceil(10*0.7)=7 -> 3.5
    lo, up = conformal_interval(10.0, 2.0)
    assert (lo, up) == (8.0, 12.0)


def test_iid_residuals_reach_nominal_coverage():
    """Textbook exchangeable case: coverage of a fresh residual should be >= 1-alpha (within MC error)."""
    rng = np.random.default_rng(0)
    n, trials = 25, 20000
    for alpha in (0.2, 0.05):
        hits = 0
        for _ in range(trials):
            res = rng.standard_t(df=5, size=n)
            q = conformal_quantile(res, alpha)
            hits += abs(rng.standard_t(df=5)) <= q
        cov = hits / trials
        assert 1 - alpha - 0.01 <= cov <= 1 - alpha + 0.035, (alpha, cov)


def _seasonal_series(rng, n=260, m=12, noise=1.0):
    t = np.arange(n)
    return 100 + 0.05 * t + 8 * np.sin(2 * np.pi * t / m) + rng.normal(scale=noise, size=n)


def test_calibration_residuals_hand_computed_and_no_lookahead():
    y = np.arange(1.0, 21.0)                                          # y_i = i
    t, h, n_cal = 20, 2, 6
    res = calibration_residuals(lambda hist, hh: naive(hist, hh)[-1], y, t, h, n_cal)
    # origins c = 14..18; naive forecast = y_c = c; target y_{c+2} = c+2 -> residual 2 each; n = n_cal - h + 1 = 5
    np.testing.assert_allclose(res, [2.0] * 5)
    # values after t are never read
    y2 = y.copy()
    y2[t:] = 1e9
    res2 = calibration_residuals(lambda hist, hh: naive(hist, hh)[-1], y2, t, h, n_cal)
    np.testing.assert_array_equal(res, res2)
    # forecaster is only ever shown a history that ends at or before c <= t - h
    seen = []
    calibration_residuals(lambda hist, hh: seen.append(len(hist)) or hist[-1], y, t, h, n_cal)
    assert max(seen) == t - h and min(seen) == t - n_cal


def test_split_conformal_interval_uses_only_window_data():
    rng = np.random.default_rng(3)
    y = _seasonal_series(rng)
    t, h = 200, 3
    fn = lambda hist, hh: snaive(hist, hh, 12)[-1]
    a = split_conformal_forecast(fn, y, t, h, 36, 0.2)
    y2 = y.copy()
    y2[t:] = 0.0
    b = split_conformal_forecast(fn, y2, t, h, 36, 0.2)
    assert a == b


@pytest.mark.parametrize("alpha", [0.2, 0.05])
def test_empirical_coverage_on_synthetic_seasonal_series(alpha):
    """Many synthetic seasonal series with iid noise, snaive point forecast, n_cal=36, h in {1, 6, 12}.

    Seasonal-naive errors are differences of iid noise (exchangeable enough), so split-conformal coverage should be
    close to nominal. Tolerance +-4 points (finite calibration samples, shared noise terms).
    """
    rng = np.random.default_rng(42)
    m, n_cal = 12, 36
    fn = lambda hist, hh: snaive(hist, hh, m)[-1]
    hits, tot = 0, 0
    for _ in range(150):
        y = _seasonal_series(rng, n=300)
        for t in range(200, 288, 11):
            for h in (1, 6, 12):
                _, lo, up = split_conformal_forecast(fn, y, t, h, n_cal, alpha)
                hits += int(lo <= y[t + h - 1] <= up)
                tot += 1
    cov = hits / tot
    assert abs(cov - (1 - alpha)) <= 0.04, cov


def test_coverage_metric_agrees_with_interval_construction():
    rng = np.random.default_rng(7)
    y = _seasonal_series(rng, n=300)
    fn = lambda hist, hh: snaive(hist, hh, 12)[-1]
    los, ups, acts = [], [], []
    for t in range(200, 288):
        _, lo, up = split_conformal_forecast(fn, y, t, 1, 36, 0.2)
        los.append(lo), ups.append(up), acts.append(y[t])
    c = metrics.coverage(acts, los, ups)
    assert 0.6 < c < 1.0 and math.isfinite(metrics.width(los, ups))
