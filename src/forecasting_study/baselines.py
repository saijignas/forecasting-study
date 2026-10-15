"""The four baselines of protocol section 6.1. ``y`` is the training window y_1..y_t (a 1-d array)."""
from __future__ import annotations

import math

import numpy as np


def _prep(y, horizon: int) -> np.ndarray:
    a = np.asarray(y, dtype="float64")
    if a.ndim != 1 or len(a) < 2:
        raise ValueError("y must be 1-d with at least 2 observations")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return a


def naive(y, horizon: int, m: int = 1) -> np.ndarray:
    """yhat_{t+h|t} = y_t for h = 1..horizon."""
    a = _prep(y, horizon)
    return np.full(horizon, a[-1])


def drift(y, horizon: int, m: int = 1) -> np.ndarray:
    """yhat_{t+h|t} = y_t + h * (y_t - y_1) / (t - 1)."""
    a = _prep(y, horizon)
    slope = (a[-1] - a[0]) / (len(a) - 1)
    return a[-1] + slope * np.arange(1, horizon + 1)


def snaive(y, horizon: int, m: int) -> np.ndarray:
    """yhat_{t+h|t} = y_{t+h-m*k}, k = ceil(h/m): the last observed value of the same season."""
    a = _prep(y, horizon)
    t = len(a)
    out = np.empty(horizon)
    for h in range(1, horizon + 1):
        k = math.ceil(h / m)
        idx = t + h - m * k  # 1-based
        if idx < 1:
            raise ValueError("training window shorter than one season")
        out[h - 1] = a[idx - 1]
    return out


def seasonal_mean(y, horizon: int, m: int, K: int = 5) -> np.ndarray:
    """Mean of the K most recent same-season observations: (1/K) sum_{j<K} y_{t+h-m*(k+j)}, k = ceil(h/m)."""
    a = _prep(y, horizon)
    t = len(a)
    out = np.empty(horizon)
    for h in range(1, horizon + 1):
        k = math.ceil(h / m)
        idxs = [t + h - m * (k + j) for j in range(K)]
        if min(idxs) < 1:
            raise ValueError("training window too short for K seasonal cycles")
        out[h - 1] = np.mean([a[i - 1] for i in idxs])
    return out


BASELINES = {"NAIVE": naive, "SNAIVE": snaive, "DRIFT": drift, "SMEAN": seasonal_mean}
