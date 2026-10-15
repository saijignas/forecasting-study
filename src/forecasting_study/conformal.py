"""Split-conformal prediction intervals for time-series forecasts (protocol section 8).

``calibration_residuals`` works for forecasters that can be called on a history ``y_1..y_c`` (baselines and any
model with frozen parameters wrapped in a closure). The caller supplies ``forecast_fn(history, h) -> float``.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


def conformal_rank(n: int, alpha: float) -> int:
    """k = min(n, ceil((n + 1)(1 - alpha))): the (1-based) order statistic used as the interval half-width."""
    if n < 1:
        raise ValueError("need at least one calibration residual")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return min(n, math.ceil((n + 1) * (1.0 - alpha)))


def conformal_quantile(residuals, alpha: float) -> float:
    """The k-th smallest absolute residual, k = conformal_rank(n, alpha)."""
    r = np.sort(np.abs(np.asarray(residuals, dtype="float64")))
    return float(r[conformal_rank(len(r), alpha) - 1])


def calibration_residuals(forecast_fn: Callable[[np.ndarray, int], float], y, t: int, h: int, n_cal: int) -> np.ndarray:
    """Absolute h-step residuals for calibration origins c = t - n_cal, ..., t - h.

    The forecast at origin c is ``forecast_fn(y[:c], h)`` and is compared with ``y_{c+h}`` (1-based), so every
    target index c + h <= t; nothing after index ``t`` is ever read (the slice ``y[:t]`` is taken first).
    """
    a = np.asarray(y, dtype="float64")[:t]
    if n_cal < h:
        raise ValueError("n_cal must be at least h")
    res = []
    for c in range(t - n_cal, t - h + 1):
        f = forecast_fn(a[:c], h)
        res.append(abs(a[c + h - 1] - f))
    return np.array(res)


def conformal_interval(point: float, quantile: float) -> tuple[float, float]:
    """Symmetric interval [point - q, point + q]."""
    return point - quantile, point + quantile


def split_conformal_forecast(forecast_fn: Callable[[np.ndarray, int], float], y, t: int, h: int, n_cal: int,
                             alpha: float) -> tuple[float, float, float]:
    """(point, lower, upper) for origin t, horizon h, level 1 - alpha using only y_1..y_t."""
    a = np.asarray(y, dtype="float64")[:t]
    q = conformal_quantile(calibration_residuals(forecast_fn, a, t, h, n_cal), alpha)
    point = float(forecast_fn(a, h))
    lo, up = conformal_interval(point, q)
    return point, lo, up
