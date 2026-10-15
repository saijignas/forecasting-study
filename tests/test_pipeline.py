"""Per-series records, caching and assembly on a tiny synthetic panel (cheap configurations only)."""
import numpy as np
import pandas as pd
import pytest

from forecasting_study import pipeline
from forecasting_study.assemble import ALL_MODELS, assemble
from forecasting_study.baselines import snaive
from forecasting_study.conformal import conformal_quantile
from forecasting_study.data import Panel
from forecasting_study.pipeline import (compute_ml_global, compute_series, ml_global_cache_path, save_npz_atomic,
                                        series_cache_path, stat_record)
from forecasting_study.splits import Schedule

SCHED = Schedule(name="monthly", T=170, m=12, horizons=(1, 3, 6, 12), spacing=6,
                 val_origins=(120,), test_origins=(140, 146, 152), n_cal=24)
DATES = pd.date_range("1990-01-01", periods=170, freq="MS")
CFG = {"learning_rate": 0.1, "max_leaf_nodes": 8, "max_iter": 50}
SELECTED = {"ETS": "ETS(A,N,A)", "ARIMA": "ARIMA(1,1,0)x(0,0,0)"}


def make_panel(n_series=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(170)
    series = {f"S{k}": (80 + 20 * k) * (1 + 0.001 * t) * (1 + 0.15 * np.sin(2 * np.pi * t / 12 + k))
              * (1 + rng.normal(scale=0.02, size=170)) for k in range(n_series)}
    return Panel("monthly", DATES, series, {k: {} for k in series})


def test_records_do_not_look_ahead():
    p = make_panel(1)
    y = p.series["S0"]
    y2 = y.copy()
    y2[140:] *= 50.0
    for name, cfg in (("ETS", "ETS(A,N,A)"), ("ARIMA", "ARIMA(1,1,0)x(0,0,0)"), ("THETA", None)):
        a = stat_record(name, y, 140, SCHED, cfg)
        b = stat_record(name, y2, 140, SCHED, cfg)
        np.testing.assert_array_equal(a["point"], b["point"])
        np.testing.assert_array_equal(a["cal"], b["cal"])
        for lv in ("80", "95"):
            np.testing.assert_array_equal(a["mb"][lv][0], b["mb"][lv][0])
        assert a["point"].shape == (4,) and a["cal"].shape == (24, 4)
        assert np.all(a["mb"]["95"][1] - a["mb"]["95"][0] > a["mb"]["80"][1] - a["mb"]["80"][0])
        assert a["fb_point"] == a["fb_cal"] == a["fb_mb"] == 0


def test_missing_configuration_triggers_the_snaive_fallback():
    y = make_panel(1).series["S0"]
    rec = stat_record("ETS", y, 140, SCHED, None)
    assert (rec["fb_point"], rec["fb_cal"], rec["fb_mb"]) == (1, 1, 1)
    np.testing.assert_array_equal(rec["point"], snaive(y[:140], 12, 12)[[0, 2, 5, 11]])
    assert np.all(rec["mb"]["80"][0] < rec["point"]) and np.all(rec["point"] < rec["mb"]["80"][1])


def test_end_to_end_assembly(tmp_path):
    p = make_panel(3)
    origins = SCHED.test_origins
    for oi, t in enumerate(origins):
        arr, meta = compute_ml_global(p.series, DATES, "monthly", t, SCHED, CFG)
        save_npz_atomic(ml_global_cache_path(tmp_path, "monthly", t), arr, meta)
    for name, y in p.series.items():
        arr, meta = compute_series("monthly", name, y, DATES, SCHED, origins, SELECTED, CFG)
        save_npz_atomic(series_cache_path(tmp_path, "monthly", name), arr, meta)
    df = assemble("monthly", p, SCHED, origins, tmp_path)
    assert len(df) == 3 * len(ALL_MODELS) * 3 * 4
    assert set(df.model) == set(ALL_MODELS)
    # targets are real observations, never beyond the series end
    assert (df.target_idx <= 170).all()
    for name, y in p.series.items():
        sub = df[df.series == name]
        np.testing.assert_allclose(sub.actual, y[sub.target_idx.to_numpy() - 1])
    # COMB = mean of ETS, ARIMA, ML-G point forecasts
    key = ["series", "origin", "h"]
    pt = df.pivot_table(index=key, columns="model", values="point")
    np.testing.assert_allclose(pt["COMB"], pt[["ETS", "ARIMA", "ML-G"]].mean(axis=1))
    # conformal interval for SNAIVE matches a by-hand calculation (h = 1, level 80%)
    name, y, t = "S1", p.series["S1"], 146
    row = df[(df.series == name) & (df.model == "SNAIVE") & (df.origin == t) & (df.h == 1)].iloc[0]
    res = [abs(y[c] - y[c - 12]) for c in range(t - 24, t)]       # y_{c+1} vs y_{c+1-12}, c = t-24..t-1
    q = conformal_quantile(res, 0.2)
    assert row.cf_hi80 - row.point == pytest.approx(q) and row.point - row.cf_lo80 == pytest.approx(q)
    # h = 12: only c <= t - 12 enter, i.e. n = 13 residuals
    row = df[(df.series == name) & (df.model == "SNAIVE") & (df.origin == t) & (df.h == 12)].iloc[0]
    res = [abs(y[c + 11] - y[c + 11 - 12]) for c in range(t - 24, t - 12 + 1)]
    assert len(res) == 13
    assert row.cf_hi95 - row.point == pytest.approx(conformal_quantile(res, 0.05))
    # model-based intervals exist only for ETS, ARIMA, THETA
    has_mb = df.groupby("model").mb_lo80.apply(lambda s: s.notna().all())
    assert set(has_mb[has_mb].index) == {"ETS", "ARIMA", "THETA"}
    assert (df.groupby("model").cf_lo80.apply(lambda s: s.notna().all())).all()


def test_cache_roundtrip_is_atomic_and_exact(tmp_path):
    arr = {"a|point": np.arange(6.0).reshape(2, 3)}
    path = tmp_path / "x" / "f.npz"
    save_npz_atomic(path, arr, {"k": 1})
    a, m = pipeline.load_npz(path)
    np.testing.assert_array_equal(a["a|point"], arr["a|point"])
    assert m == {"k": 1}
    assert not list(path.parent.glob("*.tmp*"))
