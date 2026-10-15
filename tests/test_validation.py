"""Validation-window tuning logic on synthetic series (selection rules, failure exclusion, truncation guard)."""
import numpy as np
import pandas as pd
import pytest

from forecasting_study import ml, validation as V
from forecasting_study.models import ARIMAConfig, ETSConfig, fit_arima, fit_ets
from forecasting_study.splits import Schedule

SCHED = Schedule(name="monthly", T=200, m=12, horizons=(1, 3, 6, 12), spacing=6,
                 val_origins=(120, 126), test_origins=(150, 156), n_cal=24)
DATES = pd.date_range("1990-01-01", periods=200, freq="MS")


def seasonal(n=138, noise=0.5, seed=0):
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    return 100 + 0.1 * t + 12 * np.sin(2 * np.pi * t / 12) + rng.normal(scale=noise, size=n)


def test_last_validation_target_and_truncation_guard():
    assert V.last_validation_target(SCHED) == 138
    from forecasting_study.splits import DAILY, MONTHLY
    assert V.last_validation_target(MONTHLY) == 336 and V.last_validation_target(DAILY) == 872
    with pytest.raises(AssertionError):
        V.config_validation_score(seasonal(n=139), SCHED, lambda s: fit_ets(s, ETSConfig("add", None, False, None), 12))


def test_seasonal_model_beats_non_seasonal_on_a_seasonal_series():
    y = seasonal()
    s_seas = V.config_validation_score(y, SCHED, lambda s: fit_ets(s, ETSConfig("add", None, False, "add"), 12))
    s_flat = V.config_validation_score(y, SCHED, lambda s: fit_ets(s, ETSConfig("add", None, False, None), 12))
    assert s_seas < 0.6 * s_flat
    a_seas = V.config_validation_score(y, SCHED, lambda s: fit_arima(s, ARIMAConfig((1, 1, 0), (0, 1, 1)), 12))
    a_flat = V.config_validation_score(y, SCHED, lambda s: fit_arima(s, ARIMAConfig((1, 1, 0), (0, 0, 0)), 12))
    assert a_seas < 0.6 * a_flat


def test_failing_configuration_is_excluded_not_scored():
    y = seasonal()
    y[10] = np.inf
    assert V.config_validation_score(y, SCHED, lambda s: fit_ets(s, ETSConfig("add", None, False, "add"), 12)) is None


def test_selection_rules_and_ties():
    names = ["a", "b", "c"]
    assert V.select_config({"a": 0.5, "b": 0.5, "c": 0.6}, names) == ("a", 0.5)      # tie -> earlier in grid order
    assert V.select_config({"a": None, "b": 0.7, "c": 0.6}, names) == ("c", 0.6)
    assert V.select_config({"a": None, "b": None, "c": None}, names) == (None, None)
    from forecasting_study.models import arima_grid, ets_grid
    e, a = [c.name for c in ets_grid()], [c.name for c in arima_grid()]
    base = {n: None for n in e + a}
    assert V.select_for_series({"scores": {**base, e[3]: 0.4, a[2]: 0.4}})["STAT_star"] == "ETS"       # tie -> ETS
    assert V.select_for_series({"scores": {**base, e[3]: 0.5, a[2]: 0.4}})["STAT_star"] == "ARIMA"
    assert V.select_for_series({"scores": {**base, a[2]: 0.4}})["STAT_star"] == "ARIMA"
    assert V.select_for_series({"scores": base})["STAT_star"] == "SNAIVE"
    assert V.select_for_series({"scores": {**base, e[0]: 0.3}})["n_failed_configs"] == 29


def test_ml_global_validation_picks_the_lowest_scoring_config(monkeypatch):
    t = np.arange(138)
    rng = np.random.default_rng(1)
    panel = {f"s{k}": (60 + 15 * k) * (1 + 0.1 * np.sin(2 * np.pi * t / 12 + k)) * (1 + rng.normal(scale=0.01, size=138))
             for k in range(3)}
    small = [{"learning_rate": 0.1, "max_leaf_nodes": 8, "max_iter": 50},
             {"learning_rate": 0.05, "max_leaf_nodes": 8, "max_iter": 10}]     # the second is clearly under-fitted
    monkeypatch.setattr(ml, "HGB_GRID", small)
    monkeypatch.setattr(V.ml, "HGB_GRID", small)
    res = V.validate_ml_global(panel, DATES, "monthly", SCHED)
    assert res["chosen"] == small[0]
    assert res["scores"][ml.cfg_name(small[0])] < res["scores"][ml.cfg_name(small[1])]
