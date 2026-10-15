"""Descriptive tables on the synthetic forecast table of test_stats."""
import numpy as np
import pandas as pd
import pytest

from forecasting_study import stats as S
from forecasting_study import tables as T
from forecasting_study.evaluate import add_stat_star, per_series_mase
from test_stats import synthetic_forecast_table


def test_error_by_horizon_hand_values():
    df = synthetic_forecast_table()
    eh = T.error_by_horizon(df)
    r = eh[(eh.model == "SNAIVE") & (eh.h == 12)].iloc[0]
    assert r.MASE_mean == pytest.approx(1.0) and r.RMSSE_mean == pytest.approx(1.0)
    assert r.sMAPE_mean == pytest.approx(200 * 1.0 / (100 + 101))          # |100-101| / (100 + 101) * 200
    assert len(eh) == 10 * 4


def test_interval_table_values_and_ci():
    df = synthetic_forecast_table()
    it = T.interval_table(df, S.boot_indices(30, 300))
    r = it[(it.interval == "conformal") & (it.level == 80) & (it.model == "ETS")].iloc[0]
    assert r.coverage == pytest.approx(26 / 32) and r.ci95_lo == pytest.approx(r.ci95_hi)
    assert set(it[it.interval == "model-based"].model) == {"ETS", "ARIMA", "THETA"}
    w = it[(it.interval == "conformal") & (it.level == 80) & (it.model == "ML-G")].iloc[0]
    assert w.mean_scaled_width == pytest.approx(2 * (0.5 + 0.5) * 26 / 32 + 2 * (0.5 - 0.5) * 6 / 32)
    assert not np.isnan(w.mean_pinball80)


def test_best_model_and_fallback_tables_and_markdown():
    df = synthetic_forecast_table()
    m = add_stat_star(per_series_mase(df, 12), {f"S{i}": "ETS" for i in range(30)})
    b = T.best_model_table(m)
    assert (b.best_model == "ML-G").all() and not b.snaive_beats_all_complex.any()
    assert b.best_complex_minus_snaive.iloc[0] == pytest.approx(-0.5)
    df["fb_point"], df["fb_cal"], df["fb_mb"] = 0, 0, 0
    df.loc[(df.model == "ETS") & (df.series == "S0"), "fb_point"] = 1
    fb = T.fallback_table(df)
    assert fb.loc["ETS", "fb_point"] == 32 and fb.loc["SNAIVE", "fb_point"] == 0
    md = T.md_table(fb)
    assert md.splitlines()[0].startswith("| ") and "ETS" in md
