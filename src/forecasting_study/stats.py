"""Statistics and decision rules of protocol sections 10-11 (series bootstrap, Wilcoxon, Holm, verdicts, H4 types).

All claims are implemented literally as written in the protocol; the code takes per-series quantities (MASE
differences, coverages) as input so it can be tested on synthetic numbers. Floating-point tolerance ``EPS`` is
used only so that e.g. 18/30 counts as "at least 0.60".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sps
from statsmodels.tsa.seasonal import STL

SEED = 20261016
B_DEFAULT = 10000
EPS = 1e-9
ALPHA = 0.05
H4_MODELS = ["ETS", "ARIMA", "THETA", "ML-G", "ML-L", "COMB"]


def boot_indices(n: int, B: int = B_DEFAULT, seed: int = SEED) -> np.ndarray:
    """B resamples of the n series with replacement; one index matrix reused for every statistic."""
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


@dataclass
class Claim:
    name: str
    value: float                     # the point estimate (win rate / pooled coverage / existence indicator)
    criterion: bool                  # point criterion on the observed data
    p_plus: float                    # evidence FOR the claim (reject "claim false")
    p_minus: float                   # evidence AGAINST the claim (reject "claim true"); 1.0 if not applicable
    extra: dict = field(default_factory=dict)


def holm(p: list[float] | np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (protocol section 10), returned in the input order."""
    p = np.asarray(p, dtype="float64")
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj_sorted
    return out


# ------------------------------------------------------------------------------------------------ H1 / H2
def claim_win_rate(win: np.ndarray, idx: np.ndarray, name: str, thr: float = 0.60) -> Claim:
    """Share of series that 'win' must be at least ``thr`` (protocol 11.1 / 11.2)."""
    win = np.asarray(win, dtype="float64")
    W = float(win.mean())
    Wb = win[idx].mean(axis=1)
    return Claim(name, W, W >= thr - EPS, float(np.mean(Wb <= thr + EPS)), float(np.mean(Wb >= thr - EPS)),
                 {"n_wins": int(win.sum()), "n": len(win), "ci95": [float(np.percentile(Wb, 2.5)),
                                                                     float(np.percentile(Wb, 97.5))]})


# ---------------------------------------------------------------------------------------------------- H3
def claim_h3a(cov: np.ndarray, nu: float, idx: np.ndarray, name: str, tol: float = 0.03) -> Claim:
    """Conformal coverage within ``tol`` of nominal (equivalence-style, protocol 11.3)."""
    cov = np.asarray(cov, dtype="float64")
    C = float(cov.mean())
    Cb = cov[idx].mean(axis=1)
    lo, hi = nu - tol, nu + tol
    p_plus = max(float(np.mean(Cb <= lo + EPS)), float(np.mean(Cb >= hi - EPS)))
    if C < lo - EPS:
        p_minus = float(np.mean(Cb >= lo - EPS))
    elif C > hi + EPS:
        p_minus = float(np.mean(Cb <= hi + EPS))
    else:
        p_minus = 1.0
    ci = [float(np.percentile(Cb, 2.5)), float(np.percentile(Cb, 97.5))]
    return Claim(name, C, abs(C - nu) <= tol + EPS, p_plus, p_minus, {"nominal": nu, "ci95": ci})


def claim_h3b(cov: np.ndarray, nu: float, idx: np.ndarray, name: str, tol: float = 0.03) -> Claim:
    """Model-based coverage more than ``tol`` below nominal (protocol 11.3)."""
    cov = np.asarray(cov, dtype="float64")
    C = float(cov.mean())
    Cb = cov[idx].mean(axis=1)
    thr = nu - tol
    ci = [float(np.percentile(Cb, 2.5)), float(np.percentile(Cb, 97.5))]
    return Claim(name, C, C < thr - EPS, float(np.mean(Cb >= thr - EPS)), float(np.mean(Cb < thr - EPS)),
                 {"nominal": nu, "ci95": ci})


# ---------------------------------------------------------------------------------------------------- H4
def seasonal_strength(y, period: int = 12) -> float:
    """F_S = max(0, 1 - Var(R) / Var(S + R)) from STL of log y (robust=False), protocol 11.4."""
    r = STL(np.log(np.asarray(y, dtype="float64")), period=period, robust=False).fit()
    return float(max(0.0, 1.0 - np.var(r.resid) / np.var(r.seasonal + r.resid)))


def yoy_volatility(y) -> float:
    """sd of log(y_t / y_{t-12}), t = 13..len(y)."""
    a = np.log(np.asarray(y, dtype="float64"))
    return float(np.std(a[12:] - a[:-12], ddof=1))


def h4_type_masks(names: list[str], meta: dict[str, dict], windows: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """The 12 pre-defined series types as boolean masks over ``names`` (protocol 11.4) and the label table.

    ``windows[name]`` is y_1..y_351 (the first test origin's training window)."""
    fs = np.array([seasonal_strength(windows[n]) for n in names])
    vol = np.array([yoy_volatility(windows[n]) for n in names])
    med = float(np.median(vol))
    high_s = fs >= 0.6
    low_v = vol <= med
    groups = np.array([meta[n]["group"] for n in names])
    adj = np.array([meta[n]["adjustment"] for n in names])
    masks = {}
    for g in ("retail", "industrial", "cpi", "housing", "labor_activity"):
        masks[f"group:{g}"] = groups == g
    masks["adj:SA"] = adj == "SA"
    masks["adj:NSA"] = adj == "NSA"
    masks["seasonal:high"] = high_s
    masks["seasonal:low"] = ~high_s
    masks["volatility:low"] = low_v
    masks["volatility:high"] = ~low_v
    masks["cross:high-seasonal&low-volatility"] = high_s & low_v
    labels = {n: {"F_S": float(a), "yoy_sd": float(b), "high_seasonal": bool(c), "low_volatility": bool(d)}
              for n, a, b, c, d in zip(names, fs, vol, high_s, low_v)}
    labels["_median_yoy_sd"] = med
    return masks, labels


def rule_R(D: np.ndarray, members: np.ndarray, margin: float = 0.03, share: float = 1 / 3) -> bool:
    """R(S): |S| >= 3 and for EVERY model: share(d >= margin) <= 1/3 and median(d) <= 0.
    ``D`` is [series, models]; ``members`` an index array (may repeat series, as in a bootstrap replicate)."""
    if len(members) < 3:
        return False
    d = D[members]
    return bool((d >= margin - EPS).mean(axis=0).max() <= share + EPS and np.median(d, axis=0).max() <= EPS)


def claim_h4(D: np.ndarray, masks: dict[str, np.ndarray], idx: np.ndarray, name: str = "H4") -> Claim:
    """Existence claim: at least one type satisfies R; bootstrap share ``s`` of replicates in which some type does."""
    types_ok = [t for t, mk in masks.items() if rule_R(D, np.flatnonzero(mk))]
    crit = len(types_ok) > 0
    hit = 0
    for row in idx:
        for mk in masks.values():
            sel = row[mk[row]]
            if rule_R(D, sel):
                hit += 1
                break
    s = hit / len(idx)
    return Claim(name, float(crit), crit, 1.0 - s, s, {"types_satisfying_R": types_ok, "bootstrap_share_s": s,
                                                       "n_types_examined": len(masks)})


# ------------------------------------------------------------------------------------------- verdict logic
def combine_claims(claims: list[Claim]) -> dict:
    """Hypothesis-level quantities (protocol 11): intersection-union p+, Bonferroni-style p- over false claims."""
    false = [c for c in claims if not c.criterion]
    p_plus = max(c.p_plus for c in claims)
    p_minus = min(1.0, len(false) * min(c.p_minus for c in false)) if false else 1.0
    return {"all_criteria_true": not false, "any_criterion_false": bool(false), "p_plus_H": p_plus,
            "p_minus_H": p_minus, "n_claims": len(claims), "n_false": len(false)}


def verdict(all_true: bool, any_false: bool, p_plus_adj: float, p_minus_adj: float, alpha: float = ALPHA) -> str:
    if all_true and p_plus_adj < alpha:
        return "supported"
    if any_false and p_minus_adj < alpha:
        return "not supported"
    return "inconclusive"


def wilcoxon_secondary(diff: np.ndarray) -> dict:
    """Two-sided signed-rank test of the paired per-series differences (secondary; never used for a decision)."""
    d = np.asarray(diff, dtype="float64")
    try:
        r = sps.wilcoxon(d)
        return {"statistic": float(r.statistic), "p": float(r.pvalue), "n_nonzero": int(np.sum(d != 0))}
    except ValueError:
        return {"statistic": float("nan"), "p": float("nan"), "n_nonzero": int(np.sum(d != 0))}
