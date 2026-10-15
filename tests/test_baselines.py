"""The four baselines against hand-computed forecasts (docs/protocol.md section 6.1)."""
import numpy as np
import pytest

from forecasting_study.baselines import BASELINES, drift, naive, seasonal_mean, snaive

Y = np.arange(1.0, 9.0)  # y_1..y_8 = 1..8, t = 8


def test_naive():
    np.testing.assert_allclose(naive(Y, 3), [8, 8, 8])


def test_drift_hand_computed():
    # slope = (8-1)/(8-1) = 1
    np.testing.assert_allclose(drift(Y, 3), [9, 10, 11])
    # y = 2, 4, 10: slope = (10-2)/2 = 4
    np.testing.assert_allclose(drift([2, 4, 10], 2), [14, 18])


def test_snaive_hand_computed_m4():
    # h=1..5 -> y_{8+h-4k}: h=1:y5=5, h=2:y6=6, h=3:y7=7, h=4:y8=8, h=5 (k=2): y_{8+5-8}=y5=5
    np.testing.assert_allclose(snaive(Y, 5, 4), [5, 6, 7, 8, 5])


def test_snaive_m1_equals_naive():
    np.testing.assert_allclose(snaive(Y, 4, 1), naive(Y, 4))


def test_seasonal_mean_hand_computed():
    # m=4, K=2: h=1: mean(y5, y1) = 3 ; h=4: mean(y8, y4) = 6 ; h=5 (k=2): mean(y5, y1) = 3
    np.testing.assert_allclose(seasonal_mean(Y, 5, 4, K=2), [3, 4, 5, 6, 3])


def test_seasonal_mean_default_k5_on_long_series():
    y = np.tile([10.0, 20.0, 30.0], 10)  # period 3, constant per season
    np.testing.assert_allclose(seasonal_mean(y, 3, 3), [10, 20, 30])


def test_short_history_raises():
    with pytest.raises(ValueError):
        snaive([1.0, 2.0], 3, 4)
    with pytest.raises(ValueError):
        seasonal_mean(Y, 2, 4, K=5)


def test_forecasts_ignore_values_not_in_window():
    """Baselines only see the array they are given: appending future data must not change forecasts from y[:t]."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=200).cumsum() + 100
    t = 150
    for name, fn in BASELINES.items():
        a = fn(y[:t], 6, 12)
        y2 = y.copy()
        y2[t:] = 1e9
        b = fn(y2[:t], 6, 12)
        np.testing.assert_array_equal(a, b, err_msg=name)


def test_all_baselines_have_right_shape():
    y = np.random.default_rng(1).normal(size=120) + 50
    for fn in BASELINES.values():
        assert fn(y, 12, 12).shape == (12,)
