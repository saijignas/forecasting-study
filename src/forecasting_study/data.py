"""Load the study panels (protocol section 3). Raw data are git-ignored; run scripts/download_data.py first.

``upto`` truncates every series to its first ``upto`` observations. The validation code always passes
``upto = last validation target index`` so that no test-period value can reach tuning code by construction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Panel:
    stratum: str                       # 'monthly' | 'daily'
    dates: pd.DatetimeIndex            # dates of indices 1..len (truncated like the series)
    series: dict[str, np.ndarray]      # name -> float64 array
    meta: dict[str, dict]              # name -> {group, adjustment} (monthly only)

    @property
    def names(self) -> list[str]:
        return list(self.series)


def load_panel(stratum: str, upto: int | None = None, root: Path = ROOT) -> Panel:
    data = root / "data"
    if stratum == "monthly":
        spec = json.loads((data / "series_list.json").read_text(encoding="utf-8"))
        w = spec["window"]
        series, meta, dates = {}, {}, None
        for s in spec["fred"]:
            df = pd.read_csv(data / "raw" / "fred" / f"{s['id']}.csv", parse_dates=[0], index_col=0)
            v = pd.to_numeric(df.iloc[:, 0], errors="coerce").loc[w["start"]: w["end"]]
            if len(v) != w["n_obs"] or v.isna().any() or (v <= 0).any():
                raise ValueError(f"series {s['id']} violates the protocol window checks")
            if dates is None:
                dates = pd.DatetimeIndex(v.index)
            elif not dates.equals(pd.DatetimeIndex(v.index)):
                raise ValueError("series do not share one monthly index")
            series[s["id"]] = v.to_numpy(dtype="float64")
            meta[s["id"]] = {"group": s["group"], "adjustment": s["adjustment"]}
    elif stratum == "daily":
        d = pd.read_csv(data / "processed" / "electricity_daily_30.csv", parse_dates=[0], index_col=0)
        if len(d) != 1096 or d.isna().any().any() or (d <= 0).any().any():
            raise ValueError("daily panel violates the protocol checks")
        dates = pd.DatetimeIndex(d.index)
        series = {c: d[c].to_numpy(dtype="float64") for c in d.columns}
        meta = {c: {} for c in d.columns}
    else:
        raise ValueError("stratum must be 'monthly' or 'daily'")
    if upto is not None:
        series = {k: v[:upto].copy() for k, v in series.items()}
        dates = dates[:upto]
    return Panel(stratum, dates, series, meta)
