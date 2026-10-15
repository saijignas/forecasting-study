"""Leakage tests for the lag-feature builder (synthetic data only; docs/protocol.md section 7).

Property under test: the feature row for origin s depends on y_1..y_s only. Equivalently, changing any value with
index > s must leave that row unchanged, and a row built from the full series equals the row built from y[:s].
"""
import numpy as np
import pandas as pd
import pytest

from forecasting_study.features import (DAILY_SPEC, MONTHLY_SPEC, feature_matrix, features_at, scale_at,
                                        training_rows)

RNG = np.random.default_rng(123)


def _series(n, seed, base=100.0):
    r = np.random.default_rng(seed)
    t = np.arange(n)
    return base + 0.2 * t + 10 * np.sin(2 * np.pi * t / 12) + r.normal(scale=2.0, size=n)


N = 200
H = 12
DATES = pd.date_range("2000-01-01", periods=N + H, freq="MS")
Y = _series(N, 0)


@pytest.mark.parametrize("s", [24, 30, 77, 120, 199])
def test_row_from_full_series_equals_row_from_truncated_series(s):
    X, scales, origins = feature_matrix(Y, DATES, MONTHLY_SPEC, H)
    row_full = X[list(origins).index(s)]
    row_trunc = features_at(Y, DATES, s, H, MONTHLY_SPEC)
    np.testing.assert_array_equal(row_full, row_trunc)


@pytest.mark.parametrize("s", [24, 50, 100, 150])
def test_future_values_do_not_change_features_at_origin(s):
    """Corrupt every observation after s with extreme garbage: features, scale and row must be identical."""
    X, scales, origins = feature_matrix(Y, DATES, MONTHLY_SPEC, H)
    i = list(origins).index(s)
    y2 = Y.copy()
    y2[s:] = RNG.normal(loc=1e6, scale=1e5, size=N - s)      # 0-based y2[s:] are indices s+1.. (1-based)
    X2, scales2, origins2 = feature_matrix(y2, DATES, MONTHLY_SPEC, H)
    np.testing.assert_array_equal(X[i], X2[i])
    assert scales[i] == scales2[i]
    # also true for every earlier origin
    np.testing.assert_array_equal(X[: i + 1], X2[: i + 1])


def test_detector_is_sensitive_to_data_at_or_before_origin():
    """Sanity check that the leakage test could fail: changing y_s itself must change the row."""
    s = 100
    i = list(feature_matrix(Y, DATES, MONTHLY_SPEC, H)[2]).index(s)
    y2 = Y.copy()
    y2[s - 1] += 50.0                                         # y_s (1-based) is element s-1
    X, _, _ = feature_matrix(Y, DATES, MONTHLY_SPEC, H)
    X2, _, _ = feature_matrix(y2, DATES, MONTHLY_SPEC, H)
    assert not np.array_equal(X[i], X2[i])
    # and a leaky implementation (using y_{s+1}) WOULD be caught: emulate one and check it differs after corruption
    leaky = lambda yy: yy[s]                                  # y_{s+1}
    y3 = Y.copy()
    y3[s:] = 0.0
    assert leaky(Y) != leaky(y3)


def test_feature_semantics_hand_checked():
    """Spot-check feature values at one origin against direct computation."""
    s = 60
    X, scales, origins = feature_matrix(Y, DATES, MONTHLY_SPEC, H)
    row = X[list(origins).index(s)]
    window12 = Y[s - 12: s]                                    # y_{s-11..s}
    c = window12.mean()
    assert scales[list(origins).index(s)] == pytest.approx(c)
    assert scale_at(Y, s, MONTHLY_SPEC) == pytest.approx(c)
    names = MONTHLY_SPEC.feature_names
    get = lambda n: row[names.index(n)]
    assert get("lag0") == pytest.approx(Y[s - 1] / c)
    assert get("lag5") == pytest.approx(Y[s - 6] / c)
    assert get("rmean3") == pytest.approx(Y[s - 3: s].mean() / c)
    assert get("rmean24") == pytest.approx(Y[s - 24: s].mean() / c)
    assert get("rstd12") == pytest.approx(window12.std(ddof=1) / c)
    assert get("ratio12") == pytest.approx(Y[s - 1] / Y[s - 13])
    assert get("target_month") == DATES[s + H - 1].month
    assert len(row) == len(names) == 24 + 3 + 3


def test_training_rows_targets_stay_inside_training_window():
    """Rows for a model fitted at origin t, horizon h: every target index s+h <= t, and values after t are unused."""
    panel = {"a": _series(N, 1), "b": _series(N, 2, base=50)}
    t, h = 150, 6
    X, tgt, names, origins = training_rows(panel, DATES, t, h, MONTHLY_SPEC)
    assert origins.max() + h <= t and origins.min() >= MONTHLY_SPEC.max_lookback
    # corrupting everything after t changes nothing
    panel2 = {k: v.copy() for k, v in panel.items()}
    for v in panel2.values():
        v[t:] = -999.0
    X2, tgt2, names2, origins2 = training_rows(panel2, DATES, t, h, MONTHLY_SPEC)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(tgt, tgt2)
    # target is y_{s+h}/c_s computed from the in-window values
    j = 5
    s = origins[j]
    y = panel[names[j]]
    assert tgt[j] == pytest.approx(y[s + h - 1] / y[s - 12: s].mean())
    # and the number of rows equals sum over series of (t - h - L + 1)
    assert len(tgt) == 2 * (t - h - MONTHLY_SPEC.max_lookback + 1)


def test_calendar_features_use_target_date_only():
    """Different observed values, same calendar: month feature identical and equals month of the target date."""
    X1, _, o1 = feature_matrix(_series(N, 10), DATES, MONTHLY_SPEC, 3)
    X2, _, o2 = feature_matrix(_series(N, 11), DATES, MONTHLY_SPEC, 3)
    np.testing.assert_array_equal(X1[:, -1], X2[:, -1])
    np.testing.assert_array_equal(X1[:, -1], DATES[o1 - 1 + 3].month)


def test_daily_spec_leakage_and_dow():
    n, h = 400, 28
    dates = pd.date_range("2012-01-01", periods=n + h, freq="D")
    y = 1000 + 50 * np.sin(2 * np.pi * np.arange(n) / 7) + np.random.default_rng(5).normal(scale=5, size=n)
    s = 300
    X, scales, origins = feature_matrix(y, dates, DAILY_SPEC, h)
    i = list(origins).index(s)
    y2 = y.copy()
    y2[s:] = 5000.0
    X2, _, _ = feature_matrix(y2, dates, DAILY_SPEC, h)
    np.testing.assert_array_equal(X[i], X2[i])
    names = DAILY_SPEC.feature_names
    assert X[i, names.index("target_dow")] == dates[s + h - 1].dayofweek
    assert X[i, names.index("ratio7")] == pytest.approx(y[s - 1] / y[s - 8])


def test_too_short_series_or_dates_raise():
    with pytest.raises(ValueError):
        feature_matrix(Y[:10], DATES, MONTHLY_SPEC, H)
    with pytest.raises(ValueError):
        feature_matrix(Y, DATES[:N], MONTHLY_SPEC, H)             # dates do not cover the target
