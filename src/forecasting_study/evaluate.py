"""From the long forecast table to per-series metrics and the hypothesis verdicts (protocol sections 9-11)."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from . import stats as S

COMPLEX = ["ETS", "ARIMA", "THETA", "ML-G", "ML-L", "COMB"]
REF_H = {"monthly": 12, "daily": 28}


def per_series_mase(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """MASE(series, h) = mean over origins of |y - f| / S1(t); rows = series, columns = models."""
    d = df[df.h == h].copy()
    d["ase"] = (d.actual - d.point).abs() / d.s1
    return d.groupby(["series", "model"]).ase.mean().unstack("model")


def add_stat_star(mase: pd.DataFrame, selection: dict[str, str]) -> pd.DataFrame:
    """Add column ``STAT*``: for each series the MASE of whichever of ETS/ARIMA the frozen validation chose."""
    out = mase.copy()
    out["STAT*"] = [mase.loc[s, selection[s] if selection[s] in ("ETS", "ARIMA") else "SNAIVE"] for s in mase.index]
    return out


def per_series_coverage(df: pd.DataFrame, kind: str, level: str) -> pd.DataFrame:
    """Coverage per series over all its origins and all horizons; rows = series, columns = models.
    ``kind`` is 'cf' (conformal) or 'mb' (model-based; only ETS, ARIMA, THETA have it)."""
    lo, up = f"{kind}_lo{level}", f"{kind}_hi{level}"
    d = df.dropna(subset=[lo, up]).copy()
    d["hit"] = ((d[lo] <= d.actual) & (d.actual <= d[up])).astype(float)
    return d.groupby(["series", "model"]).hit.mean().unstack("model")


def evaluate_stratum(df: pd.DataFrame, stratum: str, selection: dict[str, str], meta: dict[str, dict] | None,
                     windows: dict[str, np.ndarray] | None, B: int = S.B_DEFAULT) -> dict:
    """All claim-level quantities for one stratum. Holm is applied by ``apply_holm`` (monthly only)."""
    h = REF_H[stratum]
    mase = add_stat_star(per_series_mase(df, h), selection)
    names = list(mase.index)
    idx = S.boot_indices(len(names), B)
    snaive = mase["SNAIVE"].to_numpy()
    d = {m: snaive - mase[m].to_numpy() for m in COMPLEX + ["STAT*"]}
    res: dict = {"stratum": stratum, "ref_h": h, "n_series": len(names), "B": B,
                 "mean_mase_at_ref_h": {m: float(mase[m].mean()) for m in mase.columns},
                 "median_mase_at_ref_h": {m: float(mase[m].median()) for m in mase.columns}}

    # H1, H2
    c1 = S.claim_win_rate(d["ML-G"] >= 0.03 - S.EPS, idx, "H1: ML-G beats SNAIVE by >= 0.03 MASE")
    c2 = S.claim_win_rate(d["STAT*"] > 0, idx, "H2: STAT* beats SNAIVE")
    res["H1"] = {"claims": [asdict(c1)], **S.combine_claims([c1])}
    res["H2"] = {"claims": [asdict(c2)], **S.combine_claims([c2]),
                 "descriptive": {"ETS_only_win_rate": float(np.mean(d["ETS"] > 0)),
                                 "ARIMA_only_win_rate": float(np.mean(d["ARIMA"] > 0)),
                                 "STAT*_beats_ML-G_share": float(np.mean(mase["STAT*"].to_numpy()
                                                                         < mase["ML-G"].to_numpy()))}}
    # H3 (a: conformal ETS/ARIMA/ML-G; b: model-based ETS/ARIMA)
    claims3 = []
    for kind, fn in (("cf", S.claim_h3a), ("mb", S.claim_h3b)):
        for lv, nu in (("80", 0.80), ("95", 0.95)):
            cov = per_series_coverage(df, kind, lv)
            for m in (["ETS", "ARIMA", "ML-G"] if kind == "cf" else ["ETS", "ARIMA"]):
                label = f"H3{'a' if kind == 'cf' else 'b'}: {m} {'conformal' if kind == 'cf' else 'model-based'} {lv}%"
                claims3.append(fn(cov[m].to_numpy(), nu, idx, label))
    res["H3"] = {"claims": [asdict(c) for c in claims3], **S.combine_claims(claims3)}
    # H4
    D = np.column_stack([d[m] for m in S.H4_MODELS])
    if stratum == "monthly":
        masks, labels = S.h4_type_masks(names, {n: meta[n] for n in names}, windows)
        c4 = S.claim_h4(D, masks, idx)
        res["H4"] = {"claims": [asdict(c4)], **S.combine_claims([c4]), "labels": labels,
                     "type_sizes": {t: int(m.sum()) for t, m in masks.items()},
                     "type_table": {t: {"n": int(m.sum()),
                                        "median_d": {mm: float(np.median(D[m][:, j])) for j, mm in enumerate(S.H4_MODELS)}
                                        if m.sum() else None,
                                        "share_d_ge_0.03": {mm: float(np.mean(D[m][:, j] >= 0.03 - S.EPS))
                                                            for j, mm in enumerate(S.H4_MODELS)} if m.sum() else None,
                                        "satisfies_R": bool(S.rule_R(D, np.flatnonzero(m)))}
                                  for t, m in masks.items()}}
    else:
        res["H4"] = {"descriptive_only": True,
                     "n_series_where_no_complex_model_beats_snaive_by_0.03": int(np.sum(~(D >= 0.03 - S.EPS).any(axis=1))),
                     "per_series_d": {n: {m: float(D[i, j]) for j, m in enumerate(S.H4_MODELS)}
                                      for i, n in enumerate(names)}}
    res["wilcoxon_secondary"] = {m: S.wilcoxon_secondary(d[m]) for m in COMPLEX + ["STAT*"]}
    res["win_rates_vs_snaive_margin0"] = {m: float(np.mean(d[m] > 0)) for m in COMPLEX + ["STAT*"]}
    res["win_rates_vs_snaive_margin0.03"] = {m: float(np.mean(d[m] >= 0.03 - S.EPS)) for m in COMPLEX + ["STAT*"]}
    res["win_rate_ci95_margin0"] = {m: [float(np.percentile((d[m] > 0)[idx].mean(axis=1), 2.5)),
                                        float(np.percentile((d[m] > 0)[idx].mean(axis=1), 97.5))]
                                    for m in COMPLEX + ["STAT*"]}
    res["mase_snaive_minus_model"] = {m: d[m].tolist() for m in d}
    res["series"] = names
    return res


def apply_holm(res: dict, primary: bool, alpha: float = S.ALPHA) -> dict:
    """Holm across H1-H4 for the confirmatory stratum (separately for p+ and p-); unadjusted for the secondary one."""
    hyps = ["H1", "H2", "H3", "H4"]
    if primary:
        pp = S.holm([res[h]["p_plus_H"] for h in hyps])
        pm = S.holm([res[h]["p_minus_H"] for h in hyps])
    else:
        pp = np.array([res[h]["p_plus_H"] for h in hyps if "p_plus_H" in res[h]])
        pm = np.array([res[h]["p_minus_H"] for h in hyps if "p_minus_H" in res[h]])
        hyps = [h for h in hyps if "p_plus_H" in res[h]]
    out = {}
    for h, a, b in zip(hyps, pp, pm):
        r = res[h]
        v = S.verdict(r["all_criteria_true"], r["any_criterion_false"], float(a), float(b), alpha)
        note = ""
        if v == "inconclusive" and r["all_criteria_true"]:
            note = "point estimates meet the criterion but the claim is not statistically established"
        out[h] = {"p_plus_H": r["p_plus_H"], "p_minus_H": r["p_minus_H"], "p_plus_adj": float(a),
                  "p_minus_adj": float(b), "adjusted": primary, "verdict": v, "note": note}
    res["decisions"] = out
    return res
