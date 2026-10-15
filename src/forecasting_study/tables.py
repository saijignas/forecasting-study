"""Descriptive tables computed from the long forecast table (protocol section 13). No inference here."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as S
from .evaluate import per_series_coverage

MODEL_ORDER = ["NAIVE", "SNAIVE", "DRIFT", "SMEAN", "ETS", "ARIMA", "THETA", "ML-G", "ML-L", "COMB"]


def _pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1.0) * d)


def error_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, horizon): mean and median over series of MASE, mean sMAPE and mean RMSSE."""
    d = df.copy()
    d["ase"] = (d.actual - d.point).abs() / d.s1
    d["smape"] = 200.0 * (d.actual - d.point).abs() / (d.actual.abs() + d.point.abs())
    d["sq"] = (d.actual - d.point) ** 2 / d.s2
    g = d.groupby(["model", "h", "series"]).agg(mase=("ase", "mean"), smape=("smape", "mean"), msq=("sq", "mean"))
    g["rmsse"] = np.sqrt(g.msq)
    out = g.groupby(["model", "h"]).agg(MASE_mean=("mase", "mean"), MASE_median=("mase", "median"),
                                         sMAPE_mean=("smape", "mean"), RMSSE_mean=("rmsse", "mean")).reset_index()
    out["model"] = pd.Categorical(out.model, MODEL_ORDER, ordered=True)
    return out.sort_values(["model", "h"]).reset_index(drop=True)


def interval_table(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """Pooled coverage (unweighted mean over series) with bootstrap 95% CI, mean scaled width and, at 80%, the mean
    of the three scaled pinball losses, for every (interval type, model, level)."""
    rows = []
    for kind, label in (("cf", "conformal"), ("mb", "model-based")):
        for lv in ("80", "95"):
            lo, up = f"{kind}_lo{lv}", f"{kind}_hi{lv}"
            sub = df.dropna(subset=[lo, up]).copy()
            if sub.empty:
                continue
            cov = per_series_coverage(df, kind, lv)
            sub["w"] = (sub[up] - sub[lo]) / sub.s1
            if lv == "80":
                sub["pin"] = (_pinball(sub.actual, sub[lo], 0.1) + _pinball(sub.actual, sub.point, 0.5)
                              + _pinball(sub.actual, sub[up], 0.9)) / 3.0 / sub.s1
            for m in cov.columns:
                c = cov[m].to_numpy()
                cb = c[idx].mean(axis=1)
                ss = sub[sub.model == m]
                rows.append({"interval": label, "level": int(lv), "model": m, "coverage": float(c.mean()),
                             "ci95_lo": float(np.percentile(cb, 2.5)), "ci95_hi": float(np.percentile(cb, 97.5)),
                             "mean_scaled_width": float(ss.w.mean()),
                             "mean_pinball80": float(ss.pin.mean()) if lv == "80" else np.nan})
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out.model, MODEL_ORDER, ordered=True)
    return out.sort_values(["interval", "level", "model"]).reset_index(drop=True)


def coverage_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind in ("cf", "mb"):
        for lv in ("80", "95"):
            lo, up = f"{kind}_lo{lv}", f"{kind}_hi{lv}"
            sub = df.dropna(subset=[lo, up]).copy()
            sub["hit"] = ((sub[lo] <= sub.actual) & (sub.actual <= sub[up])).astype(float)
            g = sub.groupby(["model", "h", "series"]).hit.mean().groupby(["model", "h"]).mean()
            for (m, h), v in g.items():
                rows.append({"interval": "conformal" if kind == "cf" else "model-based", "level": int(lv),
                             "model": m, "h": h, "coverage": float(v)})
    return pd.DataFrame(rows)


def best_model_table(mase_ref: pd.DataFrame) -> pd.DataFrame:
    """Per series MASE at the reference horizon, the best model and whether SNAIVE is at least as good as every
    complex model (descriptive companion of H4)."""
    cols = [m for m in MODEL_ORDER if m in mase_ref.columns]
    t = mase_ref[cols].copy()
    t["best_model"] = t[cols].idxmin(axis=1)
    complex_ = [m for m in ("ETS", "ARIMA", "THETA", "ML-G", "ML-L", "COMB") if m in cols]
    t["snaive_beats_all_complex"] = t["SNAIVE"] <= t[complex_].min(axis=1)
    t["best_complex_minus_snaive"] = t[complex_].min(axis=1) - t["SNAIVE"]
    return t


def fallback_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("model")[["fb_point", "fb_cal", "fb_mb"]].sum().astype(int)
    g["n_forecast_rows"] = df.groupby("model").size()
    return g.reindex([m for m in MODEL_ORDER if m in g.index])


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal GitHub-markdown table (no external dependency)."""
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns] if df.index.name or not isinstance(df.index, pd.RangeIndex) \
        else [str(c) for c in df.columns]
    use_index = len(cols) == len(df.columns) + 1

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for i, row in df.iterrows():
        cells = ([str(i)] if use_index else []) + [fmt(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
