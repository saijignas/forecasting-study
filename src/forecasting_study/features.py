"""Lag-feature builder for the ML models (protocol section 7).

Leakage rule: a feature row for origin ``s`` is a function of ``y_1..y_s`` only. ``feature_matrix`` computes all rows
of a series at once (for speed) with sliding windows; ``features_at`` recomputes a single row from the truncated
series ``y[:s]`` so tests can verify the two agree and that perturbing anything after ``s`` changes nothing.
Calendar features depend on the (known) date of the target, never on observed values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    n_lags: int                 # lags y_{s-k}/c_s for k = 0..n_lags-1
    roll_windows: tuple[int, ...]
    std_window: int
    season_lag: int             # ratio y_s / y_{s-season_lag}
    scale_window: int           # c_s = mean(y_{s-scale_window+1..s})
    max_lookback: int           # L: smallest usable feature origin
    use_dow: bool

    @property
    def feature_names(self) -> list[str]:
        names = [f"lag{k}" for k in range(self.n_lags)]
        names += [f"rmean{w}" for w in self.roll_windows]
        names += [f"rstd{self.std_window}", f"ratio{self.season_lag}", "target_month"]
        if self.use_dow:
            names.append("target_dow")
        return names


MONTHLY_SPEC = FeatureSpec("monthly", n_lags=24, roll_windows=(3, 6, 24), std_window=12, season_lag=12,
                           scale_window=12, max_lookback=24, use_dow=False)
DAILY_SPEC = FeatureSpec("daily", n_lags=28, roll_windows=(7, 14, 56), std_window=28, season_lag=7,
                         scale_window=28, max_lookback=56, use_dow=True)


def _check_spec(spec: FeatureSpec) -> None:
    longest = max(spec.n_lags, *spec.roll_windows, spec.std_window, spec.season_lag + 1, spec.scale_window)
    if longest > spec.max_lookback:
        raise ValueError("max_lookback smaller than the longest window")


def scale_at(y, s: int, spec: FeatureSpec) -> float:
    """c_s = mean of the last ``scale_window`` observations up to and including index s (1-based)."""
    a = np.asarray(y, dtype="float64")[:s]
    return float(a[-spec.scale_window:].mean())


def feature_matrix(y, dates, spec: FeatureSpec, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features for every feature origin s = L..len(y) (1-based), for a horizon-h direct model.

    Returns (X, scales, origins) where X has one row per origin and ``scales`` are the c_s values used to
    normalise (and later de-normalise) the target. ``dates`` is a DatetimeIndex (or array of datetimes) of the
    calendar dates for indices 1..len(y)+h at least; it is used ONLY for calendar features of the target date.
    """
    _check_spec(spec)
    a = np.asarray(y, dtype="float64")
    n = len(a)
    L = spec.max_lookback
    if n < L:
        raise ValueError("series shorter than max_lookback")
    dates = pd.DatetimeIndex(dates)
    if len(dates) < n + h:
        raise ValueError("dates must cover the target dates (len(y) + h entries)")
    origins = np.arange(L, n + 1)  # 1-based s
    cols = []
    # scale c_s: mean of window ending at s
    sc = sliding_window_view(a, spec.scale_window)  # window i covers a[i .. i+w-1], ends at 1-based s = i+w
    scales = sc.mean(axis=1)[origins - spec.scale_window]
    for k in range(spec.n_lags):
        cols.append(a[origins - 1 - k] / scales)
    for w in spec.roll_windows:
        rw = sliding_window_view(a, w).mean(axis=1)[origins - w]
        cols.append(rw / scales)
    sd = sliding_window_view(a, spec.std_window).std(axis=1, ddof=1)[origins - spec.std_window]
    cols.append(sd / scales)
    cols.append(a[origins - 1] / a[origins - 1 - spec.season_lag])
    tdates = dates[origins - 1 + h]  # date of index s+h (1-based) is dates[s+h-1]
    cols.append(tdates.month.to_numpy().astype("float64"))
    if spec.use_dow:
        cols.append(tdates.dayofweek.to_numpy().astype("float64"))
    return np.column_stack(cols), scales, origins


def features_at(y, dates, s: int, h: int, spec: FeatureSpec) -> np.ndarray:
    """The feature row for origin s, recomputed from ``y[:s]`` only (never sees y_{s+1}, y_{s+2}, ...)."""
    X, _, origins = feature_matrix(np.asarray(y, dtype="float64")[:s], dates, spec, h)
    assert origins[-1] == s
    return X[-1]


def training_rows(panel: dict[str, np.ndarray], dates, t: int, h: int, spec: FeatureSpec):
    """Stacked training rows for a model fitted at origin ``t`` for horizon ``h``.

    For each series, rows are the feature origins s with L <= s <= t - h, so every target y_{s+h} lies inside the
    training window y_1..y_t. Only ``y[:t]`` of each series is read. Returns (X, target, series_names, origins)
    where target = y_{s+h} / c_s.
    """
    Xs, ts, names, orgs = [], [], [], []
    for name, y in panel.items():
        a = np.asarray(y, dtype="float64")[:t]
        X, scales, origins = feature_matrix(a, dates, spec, h)
        keep = origins <= t - h
        if not keep.any():
            continue
        tgt = a[origins[keep] + h - 1] / scales[keep]
        Xs.append(X[keep])
        ts.append(tgt)
        names.extend([name] * int(keep.sum()))
        orgs.append(origins[keep])
    return np.vstack(Xs), np.concatenate(ts), np.array(names), np.concatenate(orgs)
