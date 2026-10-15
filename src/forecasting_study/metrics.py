"""Forecast accuracy and interval metrics, exactly as defined in docs/protocol.md section 9.

All functions are pure numpy. "scale" arguments are the in-sample seasonal-naive scales S1 (MAE) or S2 (MSE)
computed on the *training window of the forecast origin* (see ``scale_mae`` / ``scale_mse``).
"""
from __future__ import annotations

import numpy as np


def _as_float(a) -> np.ndarray:
    return np.asarray(a, dtype="float64")


def scale_mae(train, m: int) -> float:
    """S1(t) = (1/(t-m)) * sum_{i=m+1..t} |y_i - y_{i-m}| for a training window y_1..y_t."""
    y = _as_float(train)
    if m < 1 or len(y) <= m:
        raise ValueError("need len(train) > m >= 1")
    return float(np.mean(np.abs(y[m:] - y[:-m])))


def scale_mse(train, m: int) -> float:
    """S2(t) = (1/(t-m)) * sum_{i=m+1..t} (y_i - y_{i-m})^2."""
    y = _as_float(train)
    if m < 1 or len(y) <= m:
        raise ValueError("need len(train) > m >= 1")
    return float(np.mean((y[m:] - y[:-m]) ** 2))


def ase(actual, forecast, scale) -> np.ndarray:
    """Scaled absolute error |y - f| / S1, elementwise (``scale`` may be a scalar or per-element array)."""
    a, f, s = _as_float(actual), _as_float(forecast), _as_float(scale)
    return np.abs(a - f) / s


def mase(actual, forecast, scale) -> float:
    """MASE: mean of the scaled absolute errors."""
    return float(np.mean(ase(actual, forecast, scale)))


def smape(actual, forecast) -> float:
    """sMAPE in percent: mean of 200*|y-f|/(|y|+|f|); a term is 0 when y and f are both 0."""
    a, f = _as_float(actual), _as_float(forecast)
    denom = np.abs(a) + np.abs(f)
    num = 200.0 * np.abs(a - f)
    terms = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
    return float(np.mean(terms))


def rmsse(actual, forecast, scale_sq) -> float:
    """RMSSE = sqrt(mean((y-f)^2 / S2)); ``scale_sq`` scalar or per-element array of S2 values."""
    a, f, s = _as_float(actual), _as_float(forecast), _as_float(scale_sq)
    return float(np.sqrt(np.mean((a - f) ** 2 / s)))


def pinball(actual, quantile_forecast, tau: float, scale=1.0) -> float:
    """Mean pinball loss rho_tau(y, q) = max(tau*(y-q), (tau-1)*(y-q)), divided by ``scale`` (S1)."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1)")
    a, q, s = _as_float(actual), _as_float(quantile_forecast), _as_float(scale)
    d = a - q
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d) / s))


def mean_pinball_80(actual, lower, point, upper, scale=1.0) -> float:
    """Mean of the scaled pinball losses at tau = 0.1 (lower), 0.5 (point forecast), 0.9 (upper)."""
    return float(np.mean([pinball(actual, lower, 0.1, scale), pinball(actual, point, 0.5, scale),
                          pinball(actual, upper, 0.9, scale)]))


def coverage(actual, lower, upper) -> float:
    """Share of actuals inside the closed interval [lower, upper]."""
    a, lo, up = _as_float(actual), _as_float(lower), _as_float(upper)
    return float(np.mean((lo <= a) & (a <= up)))


def width(lower, upper, scale=1.0) -> float:
    """Mean interval width (upper - lower) / scale."""
    lo, up, s = _as_float(lower), _as_float(upper), _as_float(scale)
    return float(np.mean((up - lo) / s))
