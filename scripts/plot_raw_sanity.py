"""Raw-data sanity plots (no forecasting). Reads data/raw + data/processed, writes results/figures/raw_*.png."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
spec = json.loads((ROOT / "data" / "series_list.json").read_text(encoding="utf-8"))
w = spec["window"]

fig, axes = plt.subplots(6, 5, figsize=(16, 13), sharex=True)
for ax, s in zip(axes.ravel(), spec["fred"]):
    df = pd.read_csv(ROOT / "data" / "raw" / "fred" / f"{s['id']}.csv", parse_dates=[0], index_col=0)
    v = pd.to_numeric(df.iloc[:, 0], errors="coerce").loc[w["start"]: w["end"]]
    ax.plot(v.index, v.values, lw=0.7, color="black")
    ax.set_title(f"{s['id']} ({s['adjustment']})", fontsize=8)
    ax.tick_params(labelsize=6)
fig.suptitle("FRED monthly series, 1992-01..2025-09 (raw, no modelling)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "raw_fred_monthly.png", dpi=80)
plt.close(fig)

d = pd.read_csv(ROOT / "data" / "processed" / "electricity_daily_30.csv", parse_dates=[0], index_col=0)
fig, axes = plt.subplots(6, 5, figsize=(16, 13), sharex=True)
for ax, c in zip(axes.ravel(), d.columns):
    ax.plot(d.index, d[c].values, lw=0.4, color="black")
    ax.set_title(c, fontsize=8)
    ax.tick_params(labelsize=6)
fig.suptitle("Electricity daily load (30 selected series, 2012-01..2014-12; raw, no modelling)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "raw_electricity_daily.png", dpi=80)
print("ok")
