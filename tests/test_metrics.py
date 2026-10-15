"""Metrics checked against hand-computed values (docs/protocol.md section 9)."""
import math

import numpy as np
import pytest

from forecasting_study import metrics as M

TRAIN = [1, 3, 2, 5, 4, 7]


def test_scale_mae_m1_hand_computed():
    # diffs 2,-1,3,-1,3 -> |.| = 2,1,3,1,3 -> mean 10/5 = 2
    assert M.scale_mae(TRAIN, 1) == pytest.approx(2.0)


def test_scale_mse_m1_hand_computed():
    # squares 4,1,9,1,9 -> 24/5 = 4.8
    assert M.scale_mse(TRAIN, 1) == pytest.approx(4.8)


def test_scale_m2_hand_computed():
    # y_i - y_{i-2}: 2-1=1, 5-3=2, 4-2=2, 7-5=2
    assert M.scale_mae(TRAIN, 2) == pytest.approx(7 / 4)
    assert M.scale_mse(TRAIN, 2) == pytest.approx(13 / 4)


def test_scale_requires_more_than_m_points():
    with pytest.raises(ValueError):
        M.scale_mae([1, 2, 3], 3)


def test_mase_hand_computed():
    # |10-9|/2 = 0.5, |12-15|/2 = 1.5 -> mean 1.0
    assert M.mase([10, 12], [9, 15], 2.0) == pytest.approx(1.0)
    np.testing.assert_allclose(M.ase([10, 12], [9, 15], 2.0), [0.5, 1.5])


def test_mase_per_element_scale():
    # errors 1 and 3, scales 1 and 3 -> ASE 1 and 1
    assert M.mase([10, 12], [9, 15], [1.0, 3.0]) == pytest.approx(1.0)


def test_mase_of_seasonal_naive_in_sample_is_one():
    # a forecaster that equals the in-sample seasonal naive error has MASE 1 by construction
    y = np.array(TRAIN, dtype=float)
    m = 2
    f = y[:-m]
    assert M.mase(y[m:], f, M.scale_mae(y, m)) == pytest.approx(1.0)


def test_smape_hand_computed():
    # 200*20/220 = 18.1818..., second term 0 -> mean 9.0909...
    assert M.smape([100, 50], [120, 50]) == pytest.approx((200 * 20 / 220) / 2)


def test_smape_zero_zero_is_zero_and_bounded():
    assert M.smape([0.0], [0.0]) == 0.0
    assert M.smape([1.0], [0.0]) == pytest.approx(200.0)


def test_rmsse_hand_computed():
    # errors 1 and 3 -> squares 1, 9 -> mean(1/4.8, 9/4.8) = 5/4.8
    assert M.rmsse([10, 12], [9, 15], 4.8) == pytest.approx(math.sqrt(5 / 4.8))


def test_pinball_hand_computed():
    # y=10: q=8 -> d=+2 ; q=12 -> d=-2
    assert M.pinball([10], [8], 0.9) == pytest.approx(1.8)
    assert M.pinball([10], [12], 0.9) == pytest.approx(0.2)
    assert M.pinball([10], [8], 0.1) == pytest.approx(0.2)
    assert M.pinball([10], [12], 0.1) == pytest.approx(1.8)
    assert M.pinball([10], [8], 0.5) == pytest.approx(1.0)
    assert M.pinball([10], [8], 0.5, scale=4.0) == pytest.approx(0.25)


def test_pinball_rejects_bad_tau():
    with pytest.raises(ValueError):
        M.pinball([1], [1], 1.0)


def test_mean_pinball_80_hand_computed():
    # y=10, lower 8, point 9, upper 12: rho.1(8)=0.2, rho.5(9)=0.5, rho.9(12)=0.2 -> mean 0.3
    assert M.mean_pinball_80([10], [8], [9], [12]) == pytest.approx(0.3)


def test_coverage_and_width_hand_computed():
    y = [1, 2, 3, 4]
    lo = [0, 2.5, 2, 5]
    up = [2, 3, 2.5, 6]
    assert M.coverage(y, lo, up) == pytest.approx(0.25)
    assert M.width(lo, up) == pytest.approx(1.0)
    assert M.width(lo, up, scale=2.0) == pytest.approx(0.5)


def test_coverage_endpoints_are_inclusive():
    assert M.coverage([1.0, 3.0], [1.0, 2.0], [2.0, 3.0]) == 1.0
