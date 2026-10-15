"""Gradient-boosted forecasters ML-G (global) and ML-L (per-series), protocol sections 6.7 and 7.

Direct strategy: one ``HistGradientBoostingRegressor`` per horizon. Features and training rows come from
``features.py`` (leakage-tested). The same function serves both: ``ML-G`` passes the whole panel of a frequency,
``ML-L`` passes a one-series panel. Only one fitted model is alive at a time.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import DAILY_SPEC, MONTHLY_SPEC, FeatureSpec, feature_matrix, training_rows

SEED = 20261016

HGB_GRID: list[dict] = [{"learning_rate": lr, "max_leaf_nodes": ml, "max_iter": it}
                        for lr in (0.05, 0.1) for ml in (8, 31) for it in (100, 300)]


def cfg_name(cfg: dict) -> str:
    return f"HGB(lr={cfg['learning_rate']},leaves={cfg['max_leaf_nodes']},iter={cfg['max_iter']})"


def make_hgb(cfg: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="squared_error", min_samples_leaf=20, l2_regularization=0.0,
                                         early_stopping=False, random_state=SEED, **cfg)


def spec_for(stratum: str) -> FeatureSpec:
    return {"monthly": MONTHLY_SPEC, "daily": DAILY_SPEC}[stratum]


def _fit(X, y, cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_hgb(cfg).fit(X, y)


def point_forecasts_at(panel: dict[str, np.ndarray], dates, spec: FeatureSpec, t: int, h: int,
                       cfgs: list[dict]) -> list[dict[str, float]]:
    """For each config: the h-step forecast of every series from origin ``t`` (model trained on rows with
    ``s + h <= t``). Used for tuning on the validation window and for the full-window model of a test origin."""
    X, y, _, _ = training_rows(panel, dates, t, h, spec)
    feats = {}
    for name, a in panel.items():
        Xf, sc, _ = feature_matrix(np.asarray(a, dtype="float64")[:t], dates, spec, h)
        feats[name] = (Xf[-1:], float(sc[-1]))
    out = []
    for cfg in cfgs:
        model = _fit(X, y, cfg)
        out.append({name: float(model.predict(xf)[0]) * sc for name, (xf, sc) in feats.items()})
    return out


def origin_forecasts(panel: dict[str, np.ndarray], dates, spec: FeatureSpec, t: int, horizons, cfg: dict,
                     n_cal: int) -> dict[str, dict[str, np.ndarray]]:
    """Everything the split-conformal procedure needs for one origin ``t`` (protocol sections 7-8).

    Returns ``{'point': {name: array(n_h)}, 'cal': {name: array(n_cal, n_h)}}`` where
    * ``point[name][j]`` is yhat_{t+h_j|t} from the model trained on rows with ``s + h <= t``;
    * ``cal[name][i, j]`` is yhat_{c+h_j|c}, ``c = t - n_cal + i``, from the model trained on the proper-training
      window (rows with ``s + h <= t - n_cal``), with features at ``c`` built from ``y_1..y_c`` only.
    """
    names = list(panel)
    point = {n: np.empty(len(horizons)) for n in names}
    cal = {n: np.empty((n_cal, len(horizons))) for n in names}
    for j, h in enumerate(horizons):
        feats = {}
        for name, a in panel.items():
            feats[name] = feature_matrix(np.asarray(a, dtype="float64")[:t], dates, spec, h)
        # full-window model -> real forecast at origin t
        X, y, _, _ = training_rows(panel, dates, t, h, spec)
        model = _fit(X, y, cfg)
        for name in names:
            Xf, sc, _ = feats[name]
            point[name][j] = float(model.predict(Xf[-1:])[0]) * float(sc[-1])
        # proper-training model (frozen) -> forecasts at calibration origins c = t - n_cal .. t - 1
        X, y, _, _ = training_rows(panel, dates, t - n_cal, h, spec)
        model = _fit(X, y, cfg)
        for name in names:
            Xf, sc, origins = feats[name]
            sel = (origins >= t - n_cal) & (origins <= t - 1)
            assert sel.sum() == n_cal
            cal[name][:, j] = model.predict(Xf[sel]) * sc[sel]
    return {"point": point, "cal": cal}
