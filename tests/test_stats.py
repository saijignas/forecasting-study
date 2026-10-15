"""Statistics, decision rules and the evaluation layer on synthetic data (protocol sections 10-11)."""
import numpy as np
import pandas as pd
import pytest

from forecasting_study import evaluate as E
from forecasting_study import stats as S


# ------------------------------------------------------------------------------------------------- primitives
def test_holm_known_values_and_order():
    adj = S.holm([0.01, 0.04, 0.03, 0.2])
    np.testing.assert_allclose(adj, [0.04, 0.09, 0.09, 0.2])         # sorted: .01*4, .03*3, .04*2 -> max(.09), .2*1
    np.testing.assert_allclose(S.holm([0.3, 0.001, 0.8, 0.9]), [0.9, 0.004, 1.0, 1.0])
    assert S.holm([0.4, 0.6, 0.0, 0.5])[2] == 0.0
    np.testing.assert_allclose(S.holm([0.2, 0.2, 0.2, 0.2]), 0.8)   # 4*0.2, then monotone max


def test_bootstrap_indices_are_deterministic_and_valid():
    a, b = S.boot_indices(30, 200), S.boot_indices(30, 200)
    assert a.shape == (200, 30) and a.min() >= 0 and a.max() <= 29
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(S.boot_indices(30, 200, seed=1), a)
    assert S.B_DEFAULT == 10000 and S.SEED == 20261016


def test_win_rate_claim_directions():
    idx = S.boot_indices(30, 4000)
    strong = S.claim_win_rate(np.r_[np.ones(26), np.zeros(4)], idx, "x")          # W = 26/30 = 0.867
    assert strong.criterion and strong.p_plus < 0.05 and strong.p_minus > 0.9
    weak = S.claim_win_rate(np.r_[np.ones(8), np.zeros(22)], idx, "x")            # W = 0.267
    assert not weak.criterion and weak.p_minus < 0.01 and weak.p_plus > 0.99
    edge = S.claim_win_rate(np.r_[np.ones(18), np.zeros(12)], idx, "x")           # exactly 0.60 meets the criterion
    assert edge.criterion and edge.value == pytest.approx(0.6)
    assert 0.3 < edge.p_plus < 0.8 and 0.3 < edge.p_minus < 0.8                  # the 0.6 boundary is a coin flip


def test_h3a_equivalence_logic():
    idx = S.boot_indices(30, 4000)
    exact = S.claim_h3a(np.full(30, 0.80), 0.80, idx, "x")
    assert exact.criterion and exact.p_plus == 0.0                                # every replicate is inside the band
    rng = np.random.default_rng(0)
    noisy = S.claim_h3a(np.clip(0.80 + rng.normal(0, 0.12, 30), 0, 1), 0.80, idx, "x")
    assert noisy.p_plus > 0.05                                                    # cannot be established (power note)
    under = S.claim_h3a(np.full(30, 0.60), 0.80, idx, "x")
    assert not under.criterion and under.p_minus == 0.0 and under.p_plus == 1.0
    over = S.claim_h3a(np.full(30, 0.99), 0.95, idx, "x")
    assert not over.criterion and over.p_minus == 0.0


def test_h3b_undercoverage_logic():
    idx = S.boot_indices(30, 4000)
    under = S.claim_h3b(np.full(30, 0.60), 0.80, idx, "x")
    assert under.criterion and under.p_plus == 0.0 and under.p_minus == 1.0
    fine = S.claim_h3b(np.full(30, 0.79), 0.80, idx, "x")                         # 1 point below: not "more than 3"
    assert not fine.criterion and fine.p_minus == 0.0 and fine.p_plus == 1.0
    edge = S.claim_h3b(np.full(30, 0.77), 0.80, idx, "x")                         # exactly 3 points below: strict <
    assert not edge.criterion


def test_combine_and_verdict_rules():
    c_true = S.Claim("a", 1, True, 0.01, 1.0)
    c_false = S.Claim("b", 0, False, 0.9, 0.02)
    c_false2 = S.Claim("c", 0, False, 0.9, 0.04)
    h = S.combine_claims([c_true, c_false, c_false2])
    assert h["p_plus_H"] == 0.9 and h["n_false"] == 2 and h["p_minus_H"] == pytest.approx(0.04)   # 2 * min(.02,.04)
    assert S.combine_claims([c_true])["p_minus_H"] == 1.0
    assert S.verdict(True, False, 0.01, 1.0) == "supported"
    assert S.verdict(True, False, 0.2, 1.0) == "inconclusive"
    assert S.verdict(False, True, 1.0, 0.01) == "not supported"
    assert S.verdict(False, True, 1.0, 0.2) == "inconclusive"


def test_wilcoxon_secondary():
    r = S.wilcoxon_secondary(np.linspace(0.1, 1.0, 30))
    assert r["p"] < 1e-5 and r["n_nonzero"] == 30
    pz = S.wilcoxon_secondary(np.zeros(5))
    assert pz["n_nonzero"] == 0 and (np.isnan(pz["p"]) or pz["p"] == 1.0)


# ------------------------------------------------------------------------------------------------------- H4
def test_seasonal_strength_separates_seasonal_from_non_seasonal():
    rng = np.random.default_rng(0)
    t = np.arange(240)
    seasonal = np.exp(5 + 0.2 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.01, 240))
    walk = np.exp(5 + np.cumsum(rng.normal(0, 0.02, 240)))
    assert S.seasonal_strength(seasonal) > 0.9
    assert S.seasonal_strength(walk) < 0.5
    assert 0.0 <= S.seasonal_strength(walk) <= 1.0


def test_rule_R_cases():
    D = np.array([[-0.5] * 6, [-0.2] * 6, [0.0] * 6, [0.5] * 6, [-1.0] * 6])
    assert S.rule_R(D, np.array([0, 1, 2]))                       # all losses/ties
    assert not S.rule_R(D, np.array([0, 1]))                      # fewer than 3 series
    assert S.rule_R(D, np.array([0, 1, 3]))                       # exactly 1/3 wins and median < 0: allowed
    assert not S.rule_R(D, np.array([0, 3, 3, 3]))                # 3 of 4 series are wins
    assert S.rule_R(D, np.array([0, 1, 2, 4]))
    D3 = np.full((6, 6), -0.4)
    D3[:3, 5] = 0.4                                               # one model wins on half the series -> rule fails
    assert not S.rule_R(D3, np.arange(6))
    D4 = np.full((6, 6), -0.4)
    D4[:4, :] = 0.01                                              # wins below the 0.03 margin do not count as wins ...
    assert not S.rule_R(D4, np.arange(6))                         # ... but the median is positive here
    D5 = np.full((6, 6), -0.4)
    D5[:2, :] = 0.01
    assert S.rule_R(D5, np.arange(6))                             # 2 of 6 slightly positive: share(>=.03)=0, median<0


def test_h4_claim_existence_and_bootstrap_share():
    n = 30
    D = np.full((n, 6), 0.5)                                       # complex models win everywhere ...
    D[:10, :] = -0.5                                               # ... except series 0-9 (SNAIVE wins everywhere)
    masks = {"A": np.arange(n) < 10, "B": np.arange(n) >= 10, "C": np.arange(n) < 5}
    idx = S.boot_indices(n, 1000)
    c = S.claim_h4(D, masks, idx)
    assert c.criterion and c.extra["types_satisfying_R"] == ["A", "C"]
    assert c.extra["bootstrap_share_s"] > 0.9 and c.p_plus == pytest.approx(1 - c.extra["bootstrap_share_s"])
    c0 = S.claim_h4(np.full((n, 6), 0.5), masks, idx)             # nobody loses to SNAIVE -> no type
    assert not c0.criterion and c0.extra["bootstrap_share_s"] == 0.0 and c0.p_minus == 0.0


def test_type_masks_have_twelve_types():
    rng = np.random.default_rng(1)
    t = np.arange(351)
    names = [f"S{i}" for i in range(30)]
    groups = ["retail"] * 10 + ["industrial"] * 5 + ["cpi"] * 6 + ["housing"] * 3 + ["labor_activity"] * 3 + ["auto"] * 2 + ["money"]
    meta = {n: {"group": g, "adjustment": "SA" if i % 3 == 0 else "NSA"} for i, (n, g) in enumerate(zip(names, groups))}
    win = {n: np.exp(5 + (0.3 if i % 2 else 0.0) * np.sin(2 * np.pi * t / 12) + np.cumsum(rng.normal(0, 0.01, 351)))
           for i, n in enumerate(names)}
    masks, labels = S.h4_type_masks(names, meta, win)
    assert len(masks) == 12
    assert masks["group:retail"].sum() == 10 and masks["adj:SA"].sum() + masks["adj:NSA"].sum() == 30
    assert (masks["seasonal:high"] ^ masks["seasonal:low"]).all()
    assert (masks["volatility:low"] ^ masks["volatility:high"]).all()
    assert masks["volatility:low"].sum() >= 15                    # at-or-below the median
    assert masks["seasonal:high"].sum() == 15                     # odd-indexed series are strongly seasonal
    np.testing.assert_array_equal(masks["cross:high-seasonal&low-volatility"],
                                  masks["seasonal:high"] & masks["volatility:low"])


# ------------------------------------------------------------------------------------------ evaluation layer
def synthetic_forecast_table(ml_err=0.5, stat_err=2.0, snaive_err=1.0, n_series=30):
    models = ["NAIVE", "SNAIVE", "DRIFT", "SMEAN", "ETS", "ARIMA", "THETA", "ML-G", "ML-L", "COMB"]
    err = {m: 3.0 for m in models}
    err.update({"SNAIVE": snaive_err, "ETS": stat_err, "ARIMA": stat_err, "ML-G": ml_err})
    rows = []
    for i in range(n_series):
        for m in models:
            k = 0
            for o in (351, 357, 363, 369, 375, 381, 387, 393):
                for h in (1, 3, 6, 12):
                    k += 1
                    hit_cf80, hit_cf95 = (k % 10) < 8, (k % 20) != 19              # 26/32 and 31/32 of 32 forecasts
                    hit_mb80, hit_mb95 = (k % 2) == 0, (k % 2) == 0                # 50%
                    f = 100.0 + err[m]
                    w = lambda hit: err[m] + 0.5 if hit else err[m] - 0.5          # noqa: E731
                    cf_w80, cf_w95, mb_w80, mb_w95 = w(hit_cf80), w(hit_cf95), w(hit_mb80), w(hit_mb95)
                    rows.append({"series": f"S{i}", "model": m, "origin": o, "h": h, "actual": 100.0, "point": f,
                                 "s1": 1.0, "s2": 1.0, "cf_lo80": f - cf_w80, "cf_hi80": f + cf_w80, "cf_lo95": f - cf_w95, "cf_hi95": f + cf_w95,
                                 "mb_lo80": f - mb_w80 if m in ("ETS", "ARIMA", "THETA") else np.nan,
                                 "mb_hi80": f + mb_w80 if m in ("ETS", "ARIMA", "THETA") else np.nan,
                                 "mb_lo95": f - mb_w95 if m in ("ETS", "ARIMA", "THETA") else np.nan,
                                 "mb_hi95": f + mb_w95 if m in ("ETS", "ARIMA", "THETA") else np.nan})
    return pd.DataFrame(rows)


def synthetic_meta_windows(n_series=30):
    rng = np.random.default_rng(2)
    t = np.arange(351)
    names = [f"S{i}" for i in range(n_series)]
    groups = ["retail"] * 10 + ["industrial"] * 5 + ["cpi"] * 6 + ["housing"] * 3 + ["labor_activity"] * 3 + ["auto"] * 2 + ["money"]
    meta = {n: {"group": g, "adjustment": "SA" if i % 3 == 0 else "NSA"} for i, (n, g) in enumerate(zip(names, groups))}
    win = {n: np.exp(5 + (0.3 if i % 2 else 0.0) * np.sin(2 * np.pi * t / 12) + np.cumsum(rng.normal(0, 0.01, 351)))
           for i, n in enumerate(names)}
    return meta, win


def test_per_series_helpers():
    df = synthetic_forecast_table()
    m = E.per_series_mase(df, 12)
    assert m.shape == (30, 10)
    assert m["SNAIVE"].iloc[0] == pytest.approx(1.0) and m["ML-G"].iloc[0] == pytest.approx(0.5)
    sel = {f"S{i}": ("ETS" if i % 2 else "ARIMA") for i in range(30)}
    assert (E.add_stat_star(m, sel)["STAT*"] == 2.0).all()
    cov = E.per_series_coverage(df, "cf", "80")
    assert cov.shape == (30, 10) and np.allclose(cov.to_numpy(), 26 / 32)
    mb = E.per_series_coverage(df, "mb", "80")
    assert set(mb.columns) == {"ETS", "ARIMA", "THETA"}


def test_evaluate_stratum_synthetic_with_known_answers():
    df = synthetic_forecast_table()
    meta, win = synthetic_meta_windows()
    sel = {f"S{i}": "ETS" for i in range(30)}
    res = E.apply_holm(E.evaluate_stratum(df, "monthly", sel, meta, win, B=1000), primary=True)
    assert res["H1"]["claims"][0]["value"] == 1.0                  # ML-G beats SNAIVE by 0.5 on every series
    assert res["H2"]["claims"][0]["value"] == 0.0                  # STAT* loses on every series
    assert res["H3"]["n_claims"] == 10 and res["H3"]["all_criteria_true"]
    assert res["H4"]["all_criteria_true"] is False
    d = res["decisions"]
    assert d["H1"]["verdict"] == "supported" and d["H2"]["verdict"] == "not supported"
    assert d["H3"]["verdict"] == "supported" and d["H4"]["verdict"] == "not supported"
    assert d["H1"]["adjusted"] is True
    assert res["win_rates_vs_snaive_margin0"]["ML-G"] == 1.0 and res["win_rates_vs_snaive_margin0"]["ETS"] == 0.0


def test_evaluate_stratum_inconclusive_when_wins_are_split():
    """Half the series favour ML-G: W1 = 0.5 < 0.6 but the bootstrap cannot reject '>= 0.6' at the Holm level."""
    df = synthetic_forecast_table(ml_err=0.5)
    bad = df[(df.model == "ML-G") & (df.series.isin([f"S{i}" for i in range(15)]))].index
    df.loc[bad, "point"] = 100.0 + 5.0
    meta, win = synthetic_meta_windows()
    sel = {f"S{i}": "ETS" for i in range(30)}
    res = E.apply_holm(E.evaluate_stratum(df, "monthly", sel, meta, win, B=1000), primary=True)
    assert res["H1"]["claims"][0]["value"] == pytest.approx(0.5)
    assert res["H1"]["claims"][0]["criterion"] is False
    assert res["decisions"]["H1"]["verdict"] == "inconclusive"                    # p- ~ 0.18 > 0.05
    assert res["decisions"]["H1"]["p_minus_adj"] >= res["H1"]["p_minus_H"]


def test_secondary_stratum_is_not_holm_adjusted_and_h4_is_descriptive():
    df = synthetic_forecast_table()
    df["h"] = df["h"].map({1: 1, 3: 7, 6: 14, 12: 28})
    sel = {f"S{i}": "ETS" for i in range(30)}
    res = E.apply_holm(E.evaluate_stratum(df, "daily", sel, None, None, B=500), primary=False)
    assert res["ref_h"] == 28 and res["H4"]["descriptive_only"]
    assert set(res["decisions"]) == {"H1", "H2", "H3"} and res["decisions"]["H1"]["adjusted"] is False
    assert res["decisions"]["H1"]["p_plus_adj"] == res["H1"]["p_plus_H"]
