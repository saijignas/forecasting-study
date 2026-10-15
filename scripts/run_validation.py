"""Tune ETS/ARIMA configurations and the ML-G configuration on the VALIDATION window only (protocol 5, 6.4, 6.7).

Usage (repo root):
    python -I scripts/run_validation.py --stratum monthly            # full grid, results cached per series
    python -I scripts/run_validation.py --stratum both --freeze      # also write config/frozen_configs.json
    python -I scripts/run_validation.py --stratum monthly --dry-run --series RSAFSNA INDPRO   # timing only

Series are truncated at the last validation target before any model sees them (monthly 336, daily 872), so test
targets are unreachable from this script. Per-series results are cached as JSON (resumable); only one series is in
memory at a time.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecasting_study import validation as V  # noqa: E402
from forecasting_study.data import load_panel  # noqa: E402
from forecasting_study.runinfo import (free_memory_mb, git_commit, library_versions,  # noqa: E402
                                       write_json_atomic)
from forecasting_study.splits import SCHEDULES  # noqa: E402

MIN_FREE_MB = 250


def run_stratum(stratum: str, cache_dir: Path, only: list[str] | None, dry: bool) -> dict:
    sched = SCHEDULES[stratum]
    panel = load_panel(stratum, upto=V.last_validation_target(sched))
    assert len(panel.dates) == V.last_validation_target(sched)
    names = only or panel.names
    out = {"series": {}, "ml_global": None}
    t_all = time.perf_counter()
    for name in names:
        path = cache_dir / stratum / f"{name}.json"
        if path.exists():
            out["series"][name] = json.loads(path.read_text(encoding="utf-8"))
            print(f"[{stratum}] {name}: cached", flush=True)
            continue
        if free_memory_mb() < MIN_FREE_MB:
            sys.exit(f"only {free_memory_mb():.0f} MB free; stop and retry later (cache keeps finished series)")
        t0 = time.perf_counter()
        res = V.validate_series_stat(panel.series[name], sched)
        res["selected"] = V.select_for_series(res)
        res["total_seconds"] = time.perf_counter() - t0
        write_json_atomic(path, res)
        out["series"][name] = res
        s = res["selected"]
        print(f"[{stratum}] {name}: {res['total_seconds']:.1f}s  ETS={s['ETS']} ARIMA={s['ARIMA']} "
              f"-> {s['STAT_star']}  failed={s['n_failed_configs']}", flush=True)
    mlpath = cache_dir / stratum / "_ml_global.json"
    if mlpath.exists():
        out["ml_global"] = json.loads(mlpath.read_text(encoding="utf-8"))
        print(f"[{stratum}] ML-G: cached", flush=True)
    else:
        t0 = time.perf_counter()
        res = V.validate_ml_global({n: panel.series[n] for n in names}, panel.dates, stratum, sched)
        res["total_seconds"] = time.perf_counter() - t0
        write_json_atomic(mlpath, res)
        out["ml_global"] = res
        print(f"[{stratum}] ML-G: {res['total_seconds']:.1f}s  chosen {res['chosen_name']}", flush=True)
    out["wall_seconds"] = time.perf_counter() - t_all
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratum", choices=["monthly", "daily", "both"], required=True)
    ap.add_argument("--series", nargs="*", help="restrict to these series (dry runs)")
    ap.add_argument("--dry-run", action="store_true", help="timing run: separate cache, no freezing, no ML-G")
    ap.add_argument("--freeze", action="store_true", help="write config/frozen_configs.json from the full caches")
    args = ap.parse_args()
    cache_dir = ROOT / "results" / "cache" / ("validation_dryrun" if args.dry_run else "validation")
    strata = ["monthly", "daily"] if args.stratum == "both" else [args.stratum]
    results = {s: run_stratum(s, cache_dir, args.series, args.dry_run) for s in strata}
    for s, r in results.items():
        print(f"[{s}] validation wall time {r['wall_seconds'] / 60:.1f} min for {len(r['series'])} series")
    if args.freeze:
        if args.dry_run or args.series:
            sys.exit("--freeze needs a full, non-dry run")
        frozen = {"protocol_tag": "protocol-v1", "created_with_commit": git_commit(),
                  "libraries": library_versions(),
                  "note": "Chosen on the validation window only (protocol sections 5, 6.4, 6.7). Frozen before the "
                          "confirmatory run; the confirmatory script refuses to run if this file is not committed."}
        for s, r in results.items():
            frozen[s] = {"ml_g": {"config": r["ml_global"]["chosen"], "name": r["ml_global"]["chosen_name"],
                                  "validation_scores": r["ml_global"]["scores"]},
                         "series": {n: r["series"][n]["selected"] for n in results[s]["series"]}}
        import pandas as pd
        vdir = ROOT / "results" / "validation"
        vdir.mkdir(parents=True, exist_ok=True)
        for s, r in results.items():
            pd.DataFrame({n: v["scores"] for n, v in r["series"].items()}).T.to_csv(
                vdir / f"config_scores_{s}.csv", float_format="%.5f")
            pd.Series(r["ml_global"]["scores"], name="validation_MASE").to_csv(
                vdir / f"ml_g_scores_{s}.csv", float_format="%.5f")
        write_json_atomic(ROOT / "config" / "frozen_configs.json", frozen)
        print("wrote config/frozen_configs.json")


if __name__ == "__main__":
    main()
