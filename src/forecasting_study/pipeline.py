"""Per-series, per-origin forecast production and per-series caching (protocol sections 5-8, 12).

A "record" holds, for one (series, model, origin): the point forecasts at the schedule's horizons, the
calibration-origin forecasts needed by split-conformal (``cal[i, j]`` = forecast from origin ``c = t - n_cal + i``
for horizon ``horizons[j]``), optional model-based interval bounds and fallback flags. Conformal quantiles, the
combination and all scoring are computed later from these raw arrays (``assemble.py``), so that a crash only ever
loses the series in progress.

The same code path runs at validation origins (dry runs / interval checks on the validation window) and at test
origins (the single confirmatory run); only the origin tuple differs.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import ml
from .baselines import BASELINES, snaive
from .models import (ARIMAConfig, ETSConfig, FitFailure, fit_arima, fit_ets, fit_theta,
                     snaive_analytic_interval)
from .splits import Schedule

BASELINE_NAMES = ["NAIVE", "SNAIVE", "DRIFT", "SMEAN"]
MODEL_BASED = ["ETS", "ARIMA", "THETA"]
SERIES_MODELS = BASELINE_NAMES + MODEL_BASED + ["ML-L"]      # produced per series (ML-G is global)
ALPHAS = (0.2, 0.05)                                         # 80% and 95% intervals
LEVELS = ("80", "95")


def _empty_record(n_h: int, n_cal: int) -> dict:
    return {"point": np.full(n_h, np.nan), "cal": np.full((n_cal, n_h), np.nan),
            "mb": {lv: (np.full(n_h, np.nan), np.full(n_h, np.nan)) for lv in LEVELS},
            "fb_point": 0, "fb_cal": 0, "fb_mb": 0, "has_mb": False, "err": ""}


def _pick(a: np.ndarray, horizons) -> np.ndarray:
    return np.asarray([a[h - 1] for h in horizons])


def baseline_record(name: str, y, t: int, sched: Schedule) -> dict:
    fn = BASELINES[name]
    H, m, n_cal, hs = sched.H, sched.m, sched.n_cal, sched.horizons
    rec = _empty_record(len(hs), n_cal)
    rec["point"] = _pick(fn(y[:t], H, m), hs)
    for i in range(n_cal):
        c = t - n_cal + i
        rec["cal"][i] = _pick(fn(y[:c], H, m), hs)
    return rec


def stat_record(name: str, y, t: int, sched: Schedule, cfg_name: str | None) -> dict:
    """ETS / ARIMA (selected configuration ``cfg_name``) / THETA at origin ``t`` (uses ``y[:t]`` only)."""
    H, m, n_cal, hs = sched.H, sched.m, sched.n_cal, sched.horizons
    rec = _empty_record(len(hs), n_cal)
    rec["has_mb"] = True
    hist = np.asarray(y[:t], dtype="float64")
    sn_point = snaive(hist, H, m)

    def fit(series):
        if name == "ETS":
            return fit_ets(series, ETSConfig.from_name(cfg_name), m)
        if name == "ARIMA":
            return fit_arima(series, ARIMAConfig.from_name(cfg_name), m)
        return fit_theta(series, m)

    # real forecast + model-based intervals from the full window y_1..y_t
    try:
        if name != "THETA" and cfg_name is None:
            raise FitFailure("no configuration survived validation")
        full = fit(hist)
        point = full.forecast(H)
        rec["point"] = _pick(point, hs)
        try:
            iv = full.intervals(H, ALPHAS)
            for lv, a in zip(LEVELS, ALPHAS):
                rec["mb"][lv] = (_pick(iv[a][0], hs), _pick(iv[a][1], hs))
        except Exception as e:  # noqa: BLE001
            rec["fb_mb"] = 1
            rec["err"] += f"interval:{type(e).__name__}:{e}; "
    except FitFailure as e:
        rec["fb_point"] = 1
        rec["fb_mb"] = 1
        rec["err"] += f"fit:{e}; "
        rec["point"] = _pick(sn_point, hs)
    if rec["fb_mb"]:
        for lv, a in zip(LEVELS, ALPHAS):
            lo, up = snaive_analytic_interval(hist, H, m, a)
            rec["mb"][lv] = (_pick(lo, hs), _pick(up, hs))

    # calibration forecasts: model fitted ONCE on the proper-training window y_1..y_{t-n_cal}, parameters frozen
    try:
        if name != "THETA" and cfg_name is None:
            raise FitFailure("no configuration survived validation")
        proper = fit(hist[: t - n_cal])
        for i in range(n_cal):
            c = t - n_cal + i
            rec["cal"][i] = _pick(proper.forecast_from(hist[:c], H), hs)
    except FitFailure as e:
        rec["fb_cal"] = 1
        rec["err"] += f"cal:{e}; "
        for i in range(n_cal):
            c = t - n_cal + i
            rec["cal"][i] = _pick(snaive(hist[:c], H, m), hs)
    return rec


def ml_record(panel_one: dict[str, np.ndarray], dates, stratum: str, t: int, sched: Schedule, cfg: dict) -> dict:
    """ML-L: same features/config as ML-G trained on a single series."""
    ((name, y),) = panel_one.items()
    rec = _empty_record(len(sched.horizons), sched.n_cal)
    out = ml.origin_forecasts(panel_one, dates, ml.spec_for(stratum), t, sched.horizons, cfg, sched.n_cal)
    rec["point"], rec["cal"] = out["point"][name], out["cal"][name]
    return rec


# ------------------------------------------------------------------------------------------------ cache I/O
def save_npz_atomic(path: Path, arrays: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays, _meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def load_npz(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["_meta"]))
        arrays = {k: z[k] for k in z.files if k != "_meta"}
    return arrays, meta


def pack_records(records: dict[str, list[dict]]) -> dict:
    """records[model] = list over origins of record dicts -> flat arrays [n_origins, ...]."""
    arr = {}
    for model, recs in records.items():
        arr[f"{model}|point"] = np.stack([r["point"] for r in recs])
        arr[f"{model}|cal"] = np.stack([r["cal"] for r in recs])
        arr[f"{model}|fb"] = np.array([[r["fb_point"], r["fb_cal"], r["fb_mb"]] for r in recs], dtype="int8")
        if recs[0]["has_mb"]:
            for lv in LEVELS:
                arr[f"{model}|mb_lo{lv}"] = np.stack([r["mb"][lv][0] for r in recs])
                arr[f"{model}|mb_hi{lv}"] = np.stack([r["mb"][lv][1] for r in recs])
    return arr


def series_cache_path(cache_dir: Path, stratum: str, series: str) -> Path:
    return cache_dir / stratum / f"series_{series}.npz"


def ml_global_cache_path(cache_dir: Path, stratum: str, origin: int) -> Path:
    return cache_dir / stratum / f"mlg_origin{origin}.npz"


def compute_series(stratum: str, name: str, y_full, dates, sched: Schedule, origins, selected: dict,
                   ml_cfg: dict, meta_extra: dict | None = None) -> tuple[dict, dict]:
    """All per-series models at every origin. ``selected`` = {'ETS': cfg_name|None, 'ARIMA': cfg_name|None}.
    Returns (arrays, meta) ready for ``save_npz_atomic``."""
    records: dict[str, list[dict]] = {m: [] for m in SERIES_MODELS}
    timings = {m: 0.0 for m in SERIES_MODELS}
    for t in origins:
        for model in SERIES_MODELS:
            t0 = time.perf_counter()
            if model in BASELINE_NAMES:
                rec = baseline_record(model, y_full, t, sched)
            elif model == "ML-L":
                try:
                    rec = ml_record({name: y_full[:t]}, dates, stratum, t, sched, ml_cfg)
                except Exception as e:  # noqa: BLE001
                    rec = baseline_record("SNAIVE", y_full, t, sched)
                    rec["fb_point"], rec["fb_cal"] = 1, 1
                    rec["err"] = f"ML-L failed: {type(e).__name__}: {e}"
            else:
                rec = stat_record(model, y_full, t, sched, selected.get(model))
            timings[model] += time.perf_counter() - t0
            records[model].append(rec)
    meta = {"stratum": stratum, "series": name, "origins": list(origins), "selected": selected,
            "timings_s": timings, "errors": {m: [r["err"] for r in recs if r["err"]] for m, recs in records.items()
                                              if any(r["err"] for r in recs)}}
    meta.update(meta_extra or {})
    return pack_records(records), meta


def compute_ml_global(panel: dict[str, np.ndarray], dates, stratum: str, t: int, sched: Schedule, ml_cfg: dict):
    """ML-G arrays for one origin: point [series, n_h], cal [series, n_cal, n_h] (series in ``panel`` order)."""
    out = ml.origin_forecasts({k: v[:t] for k, v in panel.items()}, dates, ml.spec_for(stratum), t, sched.horizons,
                              ml_cfg, sched.n_cal)
    names = list(panel)
    arrays = {"point": np.stack([out["point"][n] for n in names]), "cal": np.stack([out["cal"][n] for n in names])}
    return arrays, {"series": names, "origin": t, "cfg": ml_cfg}
