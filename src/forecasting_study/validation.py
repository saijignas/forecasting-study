"""Tuning on the VALIDATION window only (protocol sections 5, 6.4, 6.7).

Everything here receives series truncated at the last validation target (``Schedule.val_origins[-1] + H``), so no
test-period value is reachable. The validation score is the mean scaled absolute error (ASE, section 9) over the 5
validation origins and the 4 horizons (and, for ML-G, over all series).
"""
from __future__ import annotations

import time

import numpy as np

from . import ml
from .metrics import scale_mae
from .models import (ARIMAConfig, ETSConfig, FitFailure, arima_grid, ets_grid, fit_arima, fit_ets)
from .splits import Schedule


def last_validation_target(sched: Schedule) -> int:
    return sched.val_origins[-1] + sched.H


def _pick(a: np.ndarray, horizons) -> np.ndarray:
    return np.asarray([a[h - 1] for h in horizons])


def config_validation_score(y, sched: Schedule, fit_fn) -> float | None:
    """Mean ASE of one configuration over the validation origins and horizons; ``None`` if it fails anywhere
    (protocol section 6.6: such a configuration is excluded for the series). ``y`` must be truncated at the last
    validation target."""
    y = np.asarray(y, dtype="float64")
    assert len(y) <= last_validation_target(sched), "series reaches beyond the validation window"
    errs = []
    for t in sched.val_origins:
        try:
            f = _pick(fit_fn(y[:t]).forecast(sched.H), sched.horizons)
        except FitFailure:
            return None
        s1 = scale_mae(y[:t], sched.m)
        actual = y[[t + h - 1 for h in sched.horizons]]
        errs.append(np.abs(actual - f) / s1)
    return float(np.mean(errs))


def validate_series_stat(y, sched: Schedule) -> dict:
    """Scores of all 18 ETS and 12 ARIMA configurations for one series, plus per-config seconds."""
    scores, seconds = {}, {}
    for cfg in ets_grid():
        t0 = time.perf_counter()
        scores[cfg.name] = config_validation_score(y, sched, lambda s, c=cfg: fit_ets(s, c, sched.m))
        seconds[cfg.name] = time.perf_counter() - t0
    for cfg in arima_grid():
        t0 = time.perf_counter()
        scores[cfg.name] = config_validation_score(y, sched, lambda s, c=cfg: fit_arima(s, c, sched.m))
        seconds[cfg.name] = time.perf_counter() - t0
    return {"scores": scores, "seconds": seconds}


def select_config(scores: dict[str, float | None], names: list[str]) -> tuple[str | None, float | None]:
    """Lowest score among ``names`` (ties go to the earlier name in grid order); (None, None) if all failed."""
    best, best_s = None, None
    for n in names:
        s = scores.get(n)
        if s is not None and (best_s is None or s < best_s):
            best, best_s = n, s
    return best, best_s


def select_for_series(res: dict) -> dict:
    ets, ets_s = select_config(res["scores"], [c.name for c in ets_grid()])
    ari, ari_s = select_config(res["scores"], [c.name for c in arima_grid()])
    if ets is not None and (ari is None or ets_s <= ari_s):      # ties go to ETS (protocol 6.4)
        star = "ETS"
    elif ari is not None:
        star = "ARIMA"
    else:
        star = "SNAIVE"
    n_failed = sum(v is None for v in res["scores"].values())
    return {"ETS": ets, "ETS_val_score": ets_s, "ARIMA": ari, "ARIMA_val_score": ari_s, "STAT_star": star,
            "n_failed_configs": n_failed}


def validate_ml_global(panel: dict[str, np.ndarray], dates, stratum: str, sched: Schedule) -> dict:
    """Validation score of each of the 8 HGB configurations for ML-G (mean ASE over all series, origins, horizons)."""
    spec = ml.spec_for(stratum)
    cfgs = ml.HGB_GRID
    errs = [[] for _ in cfgs]
    seconds = [0.0] * len(cfgs)
    for t in sched.val_origins:
        s1 = {n: scale_mae(a[:t], sched.m) for n, a in panel.items()}
        for h in sched.horizons:
            t0 = time.perf_counter()
            sub = {n: a[:t] for n, a in panel.items()}
            preds = ml.point_forecasts_at(sub, dates, spec, t, h, cfgs)
            dt = time.perf_counter() - t0
            for k, pr in enumerate(preds):
                seconds[k] += dt / len(cfgs)
                errs[k].extend(abs(panel[n][t + h - 1] - pr[n]) / s1[n] for n in panel)
    scores = {ml.cfg_name(c): float(np.mean(e)) for c, e in zip(cfgs, errs)}
    best = min(range(len(cfgs)), key=lambda k: (scores[ml.cfg_name(cfgs[k])], k))   # ties -> earlier config
    return {"scores": scores, "chosen": cfgs[best], "chosen_name": ml.cfg_name(cfgs[best]),
            "seconds": dict(zip([ml.cfg_name(c) for c in cfgs], seconds))}
