"""Gradient-boosted wrappers on synthetic panels: shapes, no look-ahead, learns clean seasonality."""
import numpy as np
import pandas as pd

from forecasting_study import ml
from forecasting_study.features import MONTHLY_SPEC, feature_matrix, training_rows

DATES = pd.date_range("1990-01-01", periods=400, freq="MS")
CFG = {"learning_rate": 0.1, "max_leaf_nodes": 8, "max_iter": 100}


def panel(n_series=4, n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    out = {}
    for k in range(n_series):
        level = 50 + 30 * k
        out[f"s{k}"] = level * (1 + 0.002 * t) * (1 + 0.2 * np.sin(2 * np.pi * t / 12 + k)) * (
            1 + rng.normal(scale=0.01, size=n))
    return out


def test_grid_and_config_naming():
    assert len(ml.HGB_GRID) == 8
    assert {(c["learning_rate"], c["max_leaf_nodes"], c["max_iter"]) for c in ml.HGB_GRID} == {
        (lr, ml_, it) for lr in (0.05, 0.1) for ml_ in (8, 31) for it in (100, 300)}
    m = ml.make_hgb(ml.HGB_GRID[0])
    assert (m.min_samples_leaf, m.l2_regularization, m.early_stopping, m.random_state) == (20, 0.0, False, 20261016)


def test_origin_forecasts_shapes_and_accuracy_on_clean_seasonality():
    p = panel()
    t, hs, n_cal = 150, (1, 3, 6, 12), 24
    out = ml.origin_forecasts({k: v[:t] for k, v in p.items()}, DATES, MONTHLY_SPEC, t, hs, CFG, n_cal)
    for name, y in p.items():
        assert out["point"][name].shape == (4,) and out["cal"][name].shape == (n_cal, 4)
        actual = y[[t + h - 1 for h in hs]]
        rel = np.abs(out["point"][name] - actual) / actual
        assert rel.max() < 0.08, (name, rel)                 # level ~ 50-170, seasonal swing 20%, noise 1%
        # calibration forecasts are for targets c+h; compare the h=1 column to the realised y_{c+1}
        c = np.arange(t - n_cal, t)
        assert np.median(np.abs(out["cal"][name][:, 0] - y[c]) / y[c]) < 0.08


def test_no_lookahead_perturbing_the_future_changes_nothing():
    p = panel()
    t, hs, n_cal = 150, (1, 6), 24
    a = ml.origin_forecasts({k: v[:t] for k, v in p.items()}, DATES, MONTHLY_SPEC, t, hs, CFG, n_cal)
    q = {k: v.copy() for k, v in p.items()}
    for v in q.values():
        v[t:] *= 7.0
    b = ml.origin_forecasts({k: v[:t] for k, v in q.items()}, DATES, MONTHLY_SPEC, t, hs, CFG, n_cal)
    for name in p:
        np.testing.assert_array_equal(a["point"][name], b["point"][name])
        np.testing.assert_array_equal(a["cal"][name], b["cal"][name])


def test_calibration_model_never_trains_on_calibration_targets():
    """Rows of the proper-training window have every target at or before t - n_cal."""
    p = panel()
    t, n_cal, h = 150, 24, 6
    _, _, _, origins = training_rows({k: v[:t] for k, v in p.items()}, DATES, t - n_cal, h, MONTHLY_SPEC)
    assert origins.max() + h <= t - n_cal


def test_cal_forecast_equals_recomputation_from_truncated_history():
    p = panel(n_series=2)
    t, hs, n_cal = 150, (3,), 12
    out = ml.origin_forecasts({k: v[:t] for k, v in p.items()}, DATES, MONTHLY_SPEC, t, hs, CFG, n_cal)
    name, y = "s0", p["s0"]
    X, tgt, _, _ = training_rows({k: v[:t] for k, v in p.items()}, DATES, t - n_cal, 3, MONTHLY_SPEC)
    model = ml.make_hgb(CFG).fit(X, tgt)
    c = t - n_cal + 5
    Xf, sc, origins = feature_matrix(y[:c], DATES, MONTHLY_SPEC, 3)   # history truncated at c: no later value seen
    expected = model.predict(Xf[-1:])[0] * sc[-1]
    assert origins[-1] == c
    np.testing.assert_allclose(out["cal"][name][5, 0], expected, rtol=1e-9)


def test_point_forecasts_at_matches_origin_forecasts_point():
    p = {k: v[:150] for k, v in panel(n_series=3).items()}
    res = ml.point_forecasts_at(p, DATES, MONTHLY_SPEC, 150, 6, [CFG])[0]
    out = ml.origin_forecasts(p, DATES, MONTHLY_SPEC, 150, (6,), CFG, 24)
    for name in p:
        assert res[name] == out["point"][name][0]
