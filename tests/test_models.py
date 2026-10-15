"""ETS / SARIMAX / Theta wrappers on synthetic series with known structure (no real data)."""
import numpy as np
import pytest

from forecasting_study.models import (ARIMAConfig, ETSConfig, FitFailure, arima_grid, ets_grid, fit_arima, fit_ets,
                                      fit_theta, snaive_analytic_interval)

M = 12


def clean_seasonal(n=240, noise=0.0, seed=0, trend=0.1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 100 + trend * t + 10 * np.sin(2 * np.pi * t / M) + rng.normal(scale=noise, size=n) if noise else \
        100 + trend * t + 10 * np.sin(2 * np.pi * t / M)


def truth(n=240, H=12, trend=0.1):
    t = np.arange(n, n + H)
    return 100 + trend * t + 10 * np.sin(2 * np.pi * t / M)


def test_grids_have_the_protocol_sizes_and_unique_names():
    assert len(ets_grid()) == 18 and len({c.name for c in ets_grid()}) == 18
    assert len(arima_grid()) == 12 and len({c.name for c in arima_grid()}) == 12
    assert ETSConfig.from_name("ETS(A,Ad,M)") == ETSConfig("add", "add", True, "mul")
    assert ARIMAConfig.from_name("ARIMA(1,1,1)x(0,1,1)").seasonal == (0, 1, 1)
    # trend options: none / additive / additive damped
    assert {(c.trend, c.damped) for c in ets_grid()} == {(None, False), ("add", False), ("add", True)}


def test_ets_recovers_clean_seasonal_series():
    y = clean_seasonal(noise=0.3)
    fit = fit_ets(y, ETSConfig("add", "add", False, "add"), M)
    f = fit.forecast(12)
    assert f.shape == (12,)
    assert np.max(np.abs(f - truth())) < 2.5      # amplitude is 10, noise sd 0.3


def test_ets_multiplicative_seasonal_works_on_positive_series():
    y = clean_seasonal(noise=0.3)
    f = fit_ets(y, ETSConfig("mul", "add", False, "mul"), M).forecast(12)
    assert np.max(np.abs(f - truth())) < 3.5


def test_ets_forecast_uses_only_the_training_window():
    y = clean_seasonal(n=260, noise=0.3)
    f1 = fit_ets(y[:240], ETSConfig("add", "add", False, "add"), M).forecast(12)
    y2 = y.copy()
    y2[240:] += 1000.0                       # perturb everything after the origin
    f2 = fit_ets(y2[:240], ETSConfig("add", "add", False, "add"), M).forecast(12)
    np.testing.assert_array_equal(f1, f2)


def test_ets_frozen_refilter_matches_full_fit_and_uses_prefix_only():
    y = clean_seasonal(noise=0.3)
    fit = fit_ets(y, ETSConfig("add", "add", False, "add"), M)
    np.testing.assert_allclose(fit.forecast_from(y, 12), fit.forecast(12), rtol=1e-6)
    f_pref = fit.forecast_from(y[:200], 12)
    assert f_pref.shape == (12,)
    assert np.max(np.abs(f_pref - truth(n=200))) < 4.0   # frozen parameters, state re-filtered to t=200


def test_ets_intervals_are_ordered_wider_at_95_and_seeded():
    y = clean_seasonal(noise=1.0)
    fit = fit_ets(y, ETSConfig("mul", "add", False, "mul"), M)       # simulated intervals
    a = fit.intervals(12)
    b = fit.intervals(12)
    for al in (0.2, 0.05):
        lo, up = a[al]
        assert lo.shape == up.shape == (12,) and np.all(lo < up)
        np.testing.assert_array_equal(lo, b[al][0])                   # fixed seed -> reproducible
    assert np.all(a[0.05][1] - a[0.05][0] > a[0.2][1] - a[0.2][0])
    exact = fit_ets(y, ETSConfig("add", "add", False, "add"), M).intervals(12)   # analytic intervals
    assert np.all(exact[0.05][1] - exact[0.05][0] > exact[0.2][1] - exact[0.2][0])


def test_arima_seasonal_model_recovers_clean_series():
    y = clean_seasonal(n=240, noise=0.3)
    fit = fit_arima(y, ARIMAConfig((0, 1, 1), (0, 1, 1)), M)
    f = fit.forecast(12)
    assert f.shape == (12,)
    assert np.max(np.abs(f - truth())) < 3.0
    iv = fit.intervals(12)
    lo, up = iv[0.2]
    assert np.all(lo < f) and np.all(f < up)
    assert np.all((up - lo)[1:] >= (up - lo)[:-1] - 1e-9)          # interval width does not shrink with h
    assert np.all(iv[0.05][1] - iv[0.05][0] > up - lo)


def test_arima_no_lookahead_and_frozen_refilter():
    y = clean_seasonal(n=260, noise=0.3)
    cfg = ARIMAConfig((1, 1, 0), (0, 0, 0))
    f1 = fit_arima(y[:240], cfg, M).forecast(6)
    y2 = y.copy()
    y2[240:] = -5.0
    f2 = fit_arima(y2[:240], cfg, M).forecast(6)
    np.testing.assert_array_equal(f1, f2)
    fit = fit_arima(y[:240], cfg, M)
    np.testing.assert_allclose(fit.forecast_from(y[:240], 6), fit.forecast(6), rtol=1e-6)
    assert fit.forecast_from(y[:200], 3).shape == (3,)


def test_theta_forecasts_a_clean_seasonal_trend_series():
    y = clean_seasonal(noise=0.3)
    fit = fit_theta(y, M)
    f = fit.forecast(12)
    assert f.shape == (12,)
    assert np.max(np.abs(f - truth())) < 6.0                       # Theta halves the trend: loose bound
    lo, up = fit.intervals(12)[0.2]
    assert np.all(lo < f) and np.all(f < up)
    assert fit.forecast_from(y[:200], 12).shape == (12,)


def test_failures_raise_fitfailure_for_the_fallback():
    bad = clean_seasonal().copy()
    bad[50] = np.nan
    with pytest.raises(FitFailure):
        fit_ets(bad, ETSConfig("add", None, False, "add"), M)
    bad[50] = np.inf                                                # non-finite data -> non-finite output
    with pytest.raises(FitFailure):
        fit_arima(bad, ARIMAConfig((1, 1, 1), (0, 0, 0)), M)
    with pytest.raises(FitFailure):
        fit_theta(bad, M)
    with pytest.raises(FitFailure):                                 # far too short for a seasonal model
        fit_ets(clean_seasonal()[:6], ETSConfig("add", "add", False, "add"), M)


def test_snaive_analytic_interval_formula():
    rng = np.random.default_rng(0)
    y = 50 + 5 * np.sin(2 * np.pi * np.arange(120) / M) + rng.normal(scale=1.0, size=120)
    lo, up = snaive_analytic_interval(y, 24, M, 0.2)
    s2 = np.mean((y[M:] - y[:-M]) ** 2)
    z = 1.2815515655446004
    np.testing.assert_allclose((up - lo)[0], 2 * z * np.sqrt(s2), rtol=1e-9)
    np.testing.assert_allclose((up - lo)[12], 2 * z * np.sqrt(2 * s2), rtol=1e-9)   # h = 13 -> two seasons
    np.testing.assert_allclose(lo + (up - lo) / 2, np.concatenate([y[-M:], y[-M:]]), rtol=1e-9)
