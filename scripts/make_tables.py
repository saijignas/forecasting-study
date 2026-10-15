"""Regenerate every table and figure of the report from the saved raw forecasts (protocol section 13).

    python -I scripts/make_tables.py            # reads results/forecasts_*.csv.gz + config/frozen_configs.json

Writes results/hypotheses_{stratum}.json, results/tables/*.csv|md and results/figures/*.png. Example-forecast panels
use a rule fixed before looking at results: the first series of the retail, industrial, cpi and housing groups
(monthly; origin 369) and the first four series (daily; origin 956).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecasting_study import evaluate as E  # noqa: E402
from forecasting_study import stats as S  # noqa: E402
from forecasting_study import tables as T  # noqa: E402
from forecasting_study.data import load_panel  # noqa: E402
from forecasting_study.splits import SCHEDULES  # noqa: E402

RES = Path(os.environ.get("FS_RESULTS_DIR", ROOT / "results"))      # override only for smoke tests
FROZEN = Path(os.environ.get("FS_FROZEN", ROOT / "config" / "frozen_configs.json"))
TAB = RES / "tables"
FIG = RES / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
COLORS = {"SNAIVE": "#0b0b0b", "NAIVE": "#8a8984", "DRIFT": "#8a8984", "SMEAN": "#8a8984", "ETS": "#2a78d6",
          "ARIMA": "#eb6834", "THETA": "#1baf7a", "ML-G": "#4a3aa7", "ML-L": "#e87ba4", "COMB": "#008300",
          "STAT*": "#2a78d6"}
STYLES = {"NAIVE": "--", "DRIFT": ":", "SMEAN": "-.", "SNAIVE": "-"}
MARK = {"SNAIVE": "o", "NAIVE": "x", "DRIFT": "+", "SMEAN": "d", "ETS": "s", "ARIMA": "^", "THETA": "v", "ML-G": "D",
        "ML-L": "P", "COMB": "*"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def style_ax(ax):
    ax.set_axisbelow(True)


# ------------------------------------------------------------------------------------------------- figures
def fig_error_by_horizon(eh: pd.DataFrame, stratum: str):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    hs = sorted(eh.h.unique())
    for m in ["NAIVE", "SNAIVE", "DRIFT", "SMEAN", "ETS", "ARIMA", "THETA", "ML-G", "ML-L", "COMB"]:
        s = eh[eh.model == m].set_index("h").MASE_mean.reindex(hs)
        ax.plot(range(len(hs)), s.values, STYLES.get(m, "-"), color=COLORS[m], marker=MARK[m], ms=4,
                lw=2.0 if m == "SNAIVE" else 1.3, label=m)
    ax.set_xticks(range(len(hs)))
    ax.set_xticklabels([f"h={h}" for h in hs])
    ax.set_ylabel("MASE (mean over series)")
    ax.set_title(f"{stratum.capitalize()}: error by horizon (test origins)")
    ax.legend(ncol=2, fontsize=7)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / f"error_by_horizon_{stratum}.png", dpi=130)
    plt.close(fig)


def fig_win_rates(res: dict, stratum: str):
    models = E.COMPLEX + ["STAT*"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    ref = res["ref_h"]
    for ax, key, ttl in ((axes[0], "win_rates_vs_snaive_margin0", f"beats SNAIVE (margin 0), h={ref}"),
                         (axes[1], "win_rates_vs_snaive_margin0.03", f"beats SNAIVE by >= 0.03 MASE, h={ref}")):
        vals = [res[key][m] for m in models]
        idx = S.boot_indices(res["n_series"], res["B"])
        d = {m: np.array(res["mase_snaive_minus_model"][m]) for m in models}
        thr = 0.0 if "margin0 " in key or key.endswith("margin0") else 0.03
        cis = []
        for m in models:
            w = (d[m] > 0) if thr == 0.0 else (d[m] >= thr - S.EPS)
            wb = w[idx].mean(axis=1)
            cis.append((np.percentile(wb, 2.5), np.percentile(wb, 97.5)))
        y = np.arange(len(models))
        ax.barh(y, vals, color=[COLORS[m] for m in models], height=0.55)
        ax.errorbar(vals, y, xerr=[[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        ax.axvline(0.6, color=INK, lw=1, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of series (bootstrap 95% CI)")
        ax.set_title(ttl, fontsize=8)
        style_ax(ax)
    axes[0].invert_yaxis()                                  # shared y axis: models listed top to bottom
    fig.suptitle(f"{stratum.capitalize()}: win rates against seasonal naive (dashed = 60% criterion)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"win_rates_{stratum}.png", dpi=130)
    plt.close(fig)


def fig_coverage(it: pd.DataFrame, stratum: str):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharey=False)
    cells = [("conformal", "ETS"), ("conformal", "ARIMA"), ("conformal", "ML-G"), ("model-based", "ETS"),
             ("model-based", "ARIMA"), ("model-based", "THETA"), ("conformal", "SNAIVE"), ("conformal", "COMB")]
    for ax, lv in zip(axes, (80, 95)):
        nu = lv / 100
        ax.axhspan(nu - 0.03, nu + 0.03, color=GRID, zorder=0)
        ax.axhline(nu, color=INK, lw=1, ls="--")
        for i, (kind, m) in enumerate(cells):
            r = it[(it.interval == kind) & (it.level == lv) & (it.model == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(i, r.coverage, yerr=[[r.coverage - r.ci95_lo], [r.ci95_hi - r.coverage]], fmt=MARK[m],
                        color=COLORS[m], mfc=COLORS[m] if kind == "conformal" else "white", ms=6, capsize=2, lw=1.2)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"{m}\n{'conf.' if k == 'conformal' else 'model'}" for k, m in cells], fontsize=7)
        ax.set_ylabel("pooled coverage")
        ax.set_title(f"nominal {lv}% (band = +/- 3 points)", fontsize=8)
        style_ax(ax)
    fig.suptitle(f"{stratum.capitalize()}: prediction-interval coverage (filled = split-conformal, open = model-based)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"coverage_{stratum}.png", dpi=130)
    plt.close(fig)


def fig_coverage_by_horizon(cb: pd.DataFrame, stratum: str):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, lv in zip(axes, (80, 95)):
        ax.axhline(lv / 100, color=INK, lw=1, ls="--")
        for kind, ls in (("conformal", "-"), ("model-based", ":")):
            for m in ("ETS", "ARIMA", "ML-G"):
                s = cb[(cb.interval == kind) & (cb.level == lv) & (cb.model == m)].sort_values("h")
                if s.empty:
                    continue
                ax.plot(range(len(s)), s.coverage.values, ls, color=COLORS[m], marker=MARK[m], ms=4, lw=1.3,
                        label=f"{m} {kind}")
                ax.set_xticks(range(len(s)))
                ax.set_xticklabels([f"h={h}" for h in s.h])
        ax.set_ylim(0.3, 1.02)
        ax.set_title(f"nominal {lv}%", fontsize=8)
        ax.set_ylabel("coverage (mean over series)")
        style_ax(ax)
    axes[0].legend(fontsize=6, ncol=2, loc="lower left")
    fig.suptitle(f"{stratum.capitalize()}: coverage by horizon", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"coverage_by_horizon_{stratum}.png", dpi=130)
    plt.close(fig)


def fig_examples(df: pd.DataFrame, panel, sched, stratum: str, picks: list[str], origin: int):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6))
    back = 48 if stratum == "monthly" else 90
    for ax, name in zip(axes.ravel(), picks):
        y = panel.series[name]
        lo = max(0, origin - back)
        hi = origin + sched.H
        x = np.arange(lo + 1, hi + 1)
        ax.plot(x, y[lo:hi], color=INK, lw=1.2, label="actual")
        ax.axvline(origin, color=MUTED, lw=0.8, ls=":")
        sub = df[(df.series == name) & (df.origin == origin)]
        for k, (m, dx) in enumerate((("SNAIVE", -0.25), ("ETS", 0.0), ("ML-G", 0.25))):
            r = sub[sub.model == m].sort_values("h")
            xs = origin + r.h.to_numpy() + dx * (0.8 if stratum == "monthly" else 2.5)
            ax.plot(xs, r.point, MARK[m], color=COLORS[m], ms=5, ls="none", label=m)
            if m != "SNAIVE":
                for lvl, lw in (("95", 1.0), ("80", 2.4)):
                    ax.vlines(xs, r[f"cf_lo{lvl}"], r[f"cf_hi{lvl}"], color=COLORS[m], lw=lw, alpha=0.9 if lvl == "80" else 0.5)
        ax.set_title(f"{name} (origin {origin}; bars: conformal 80% thick, 95% thin)", fontsize=7)
        style_ax(ax)
    axes[0, 0].legend(fontsize=7, ncol=4, loc="upper left")
    fig.suptitle(f"{stratum.capitalize()}: example forecasts (series chosen by a rule fixed in advance)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"examples_{stratum}.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- main
def decisions_md(all_res: dict) -> str:
    lines = []
    for stratum, res in all_res.items():
        prim = stratum == "monthly"
        lines.append(f"### {stratum} ({'confirmatory, Holm-adjusted' if prim else 'secondary replication, no Holm'})\n")
        lines.append("| Hypothesis | Point criteria | p+ (raw) | p- (raw) | p+ adj | p- adj | Verdict |")
        lines.append("|---|---|---|---|---|---|---|")
        for h, d in res["decisions"].items():
            r = res[h]
            lines.append(f"| {h} | {r['n_claims'] - r['n_false']}/{r['n_claims']} true | {d['p_plus_H']:.4f} | "
                         f"{d['p_minus_H']:.4f} | {d['p_plus_adj']:.4f} | {d['p_minus_adj']:.4f} | "
                         f"{d['verdict']}{' (' + d['note'] + ')' if d['note'] else ''} |")
        lines.append("")
    return "\n".join(lines)


def claims_md(res: dict) -> str:
    lines = ["| Claim | Value | Criterion met | CI95 | p+ | p- |", "|---|---|---|---|---|---|"]
    for h in ("H1", "H2", "H3", "H4"):
        if h not in res or "claims" not in res[h]:
            continue
        for c in res[h]["claims"]:
            ci = c["extra"].get("ci95")
            lines.append(f"| {c['name']} | {c['value']:.3f} | {c['criterion']} | "
                         f"{'[%.3f, %.3f]' % tuple(ci) if ci else ''} | {c['p_plus']:.4f} | {c['p_minus']:.4f} |")
    return "\n".join(lines) + "\n"


def main():
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    meta_run = json.loads((RES / "run_meta.json").read_text(encoding="utf-8"))
    all_res, md = {}, [f"# Results tables\n\nGenerated from forecasts produced at commit "
                       f"`{meta_run['commit_at_final_invocation']}` (see results/run_meta.json).\n"]
    for stratum in ("monthly", "daily"):
        fpath = RES / f"forecasts_{stratum}.csv.gz"
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        sched = SCHEDULES[stratum]
        panel = load_panel(stratum)
        sel = {n: frozen[stratum]["series"][n]["STAT_star"] for n in panel.names}
        windows = {n: panel.series[n][:351] for n in panel.names} if stratum == "monthly" else None
        res = E.evaluate_stratum(df, stratum, sel, panel.meta if stratum == "monthly" else None, windows)
        res = E.apply_holm(res, primary=(stratum == "monthly"))
        all_res[stratum] = res
        (RES / f"hypotheses_{stratum}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

        idx = S.boot_indices(len(panel.names))
        eh = T.error_by_horizon(df)
        eh.to_csv(TAB / f"error_by_horizon_{stratum}.csv", index=False, float_format="%.4f")
        it = T.interval_table(df, idx)
        it.to_csv(TAB / f"intervals_{stratum}.csv", index=False, float_format="%.4f")
        cb = T.coverage_by_horizon(df)
        cb.to_csv(TAB / f"coverage_by_horizon_{stratum}.csv", index=False, float_format="%.4f")
        mase_ref = E.add_stat_star(E.per_series_mase(df, res["ref_h"]), sel)
        best = T.best_model_table(mase_ref)
        best.to_csv(TAB / f"per_series_mase_ref_{stratum}.csv", float_format="%.4f")
        fb = T.fallback_table(df)
        fb.to_csv(TAB / f"fallbacks_{stratum}.csv")
        selected = pd.DataFrame(frozen[stratum]["series"]).T
        selected.to_csv(TAB / f"selected_configs_{stratum}.csv")
        wr = pd.DataFrame({"win_rate_margin0": res["win_rates_vs_snaive_margin0"],
                           "win_rate_margin0.03": res["win_rates_vs_snaive_margin0.03"],
                           "mean_MASE_ref_h": res["mean_mase_at_ref_h"],
                           "wilcoxon_p_secondary": {m: v["p"] for m, v in res["wilcoxon_secondary"].items()}}
                          ).reindex(E.COMPLEX + ["STAT*"])
        wr.to_csv(TAB / f"win_rates_{stratum}.csv", float_format="%.4f")
        if stratum == "monthly":
            pd.DataFrame(res["H4"]["type_table"]).T.to_json(TAB / "h4_types.json", indent=1)
            rows = [{"type": t, "n": v["n"], "satisfies_R": v["satisfies_R"],
                     **{f"median_d_{m}": v["median_d"][m] for m in S.H4_MODELS},
                     **{f"share_d>=0.03_{m}": v["share_d_ge_0.03"][m] for m in S.H4_MODELS}}
                    for t, v in res["H4"]["type_table"].items()]
            pd.DataFrame(rows).to_csv(TAB / "h4_types.csv", index=False, float_format="%.3f")
            pd.DataFrame(res["H4"]["labels"]).T.drop(index="_median_yoy_sd").to_csv(TAB / "h4_series_labels.csv",
                                                                                      float_format="%.4f")

        fig_error_by_horizon(eh, stratum)
        fig_win_rates(res, stratum)
        fig_coverage(it, stratum)
        fig_coverage_by_horizon(cb, stratum)
        if stratum == "monthly":
            picks = [next(n for n in panel.names if panel.meta[n]["group"] == g)
                     for g in ("retail", "industrial", "cpi", "housing")]
            fig_examples(df, panel, sched, stratum, picks, 369)
        else:
            fig_examples(df, panel, sched, stratum, panel.names[:4], 956)

        md.append(f"## {stratum}\n")
        md.append("### Error by horizon\n\n" + T.md_table(eh.set_index(["model", "h"]).reset_index().assign(
            model=lambda d: d.model.astype(str))))
        md.append("### Win rates against SNAIVE\n\n" + T.md_table(wr))
        md.append("### Intervals\n\n" + T.md_table(it.assign(model=it.model.astype(str))))
        md.append("### Fallback counts\n\n" + T.md_table(fb))
        md.append("### Claims\n\n" + claims_md(res))
    md.insert(1, "## Hypothesis decisions\n\n" + decisions_md(all_res))
    (TAB / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("tables and figures written to", TAB, FIG)


if __name__ == "__main__":
    main()
