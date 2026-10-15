"""Turn the raw per-series caches into one long table of forecasts, conformal intervals and scales.

The table has one row per (series, model, origin, horizon). It is the only input of the statistics and of
``scripts/make_tables.py``. Conformal quantiles use ``conformal.py`` (protocol section 8), scales use ``metrics.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .conformal import conformal_quantile
from .metrics import scale_mae, scale_mse
from .pipeline import (BASELINE_NAMES, LEVELS, ALPHAS, MODEL_BASED, SERIES_MODELS, load_npz, ml_global_cache_path,
                       series_cache_path)
from .splits import Schedule

ALL_MODELS = BASELINE_NAMES + MODEL_BASED + ["ML-G", "ML-L", "COMB"]
COLUMNS = ["stratum", "series", "model", "origin", "h", "target_idx", "actual", "point", "s1", "s2",
           "fb_point", "fb_cal", "fb_mb",
           "mb_lo80", "mb_hi80", "mb_lo95", "mb_hi95", "cf_lo80", "cf_hi80", "cf_lo95", "cf_hi95"]


def _model_arrays(series_arr: dict, mlg: dict, k: int) -> dict[str, dict]:
    """model -> {'point' [n_o, n_h], 'cal' [n_o, n_cal, n_h], 'fb' [n_o, 3], optional mb arrays}."""
    out = {}
    for m in SERIES_MODELS:
        d = {"point": series_arr[f"{m}|point"], "cal": series_arr[f"{m}|cal"], "fb": series_arr[f"{m}|fb"]}
        for lv in LEVELS:
            if f"{m}|mb_lo{lv}" in series_arr:
                d[f"mb_lo{lv}"] = series_arr[f"{m}|mb_lo{lv}"]
                d[f"mb_hi{lv}"] = series_arr[f"{m}|mb_hi{lv}"]
        out[m] = d
    n_o = out["ETS"]["point"].shape[0]
    out["ML-G"] = {"point": mlg["point"][:, k, :], "cal": mlg["cal"][:, k, :, :],
                   "fb": np.zeros((n_o, 3), dtype="int8")}
    comps = [out["ETS"], out["ARIMA"], out["ML-G"]]
    out["COMB"] = {"point": np.mean([c["point"] for c in comps], axis=0),
                   "cal": np.mean([c["cal"] for c in comps], axis=0),
                   "fb": np.max([c["fb"] for c in comps], axis=0).astype("int8")}
    return out


def assemble(stratum: str, panel, sched: Schedule, origins, cache_dir: Path) -> pd.DataFrame:
    """``panel`` is the FULL panel (actuals are read from it); ``origins`` the origins that were run."""
    origins = list(origins)
    hs = sched.horizons
    mlg_parts = []
    for t in origins:
        arr, _ = load_npz(ml_global_cache_path(cache_dir, stratum, t))
        mlg_parts.append(arr)
    mlg = {"point": np.stack([p["point"] for p in mlg_parts]),               # [n_o, series, n_h]
           "cal": np.stack([p["cal"] for p in mlg_parts])}                   # [n_o, series, n_cal, n_h]
    rows = []
    for k, name in enumerate(panel.names):
        y = panel.series[name]
        series_arr, _ = load_npz(series_cache_path(cache_dir, stratum, name))
        marr = _model_arrays(series_arr, mlg, k)
        for model in ALL_MODELS:
            d = marr[model]
            for oi, t in enumerate(origins):
                s1, s2 = scale_mae(y[:t], sched.m), scale_mse(y[:t], sched.m)
                for j, h in enumerate(hs):
                    cal = d["cal"][oi][:, j]
                    c_idx = np.arange(sched.n_cal)                   # c = t - n_cal + i
                    ok = c_idx <= sched.n_cal - h                    # c + h <= t
                    targets = y[(t - sched.n_cal + c_idx[ok]) + h - 1]  # y_{c+h} (1-based c+h -> 0-based c+h-1)
                    res = np.abs(targets - cal[ok])
                    row = {"stratum": stratum, "series": name, "model": model, "origin": t, "h": h,
                           "target_idx": t + h, "actual": float(y[t + h - 1]), "point": float(d["point"][oi][j]),
                           "s1": s1, "s2": s2, "fb_point": int(d["fb"][oi][0]), "fb_cal": int(d["fb"][oi][1]),
                           "fb_mb": int(d["fb"][oi][2])}
                    for lv, a in zip(LEVELS, ALPHAS):
                        row[f"mb_lo{lv}"] = float(d[f"mb_lo{lv}"][oi][j]) if f"mb_lo{lv}" in d else np.nan
                        row[f"mb_hi{lv}"] = float(d[f"mb_hi{lv}"][oi][j]) if f"mb_hi{lv}" in d else np.nan
                        q = conformal_quantile(res, a)
                        row[f"cf_lo{lv}"] = row["point"] - q
                        row[f"cf_hi{lv}"] = row["point"] + q
                    rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df
