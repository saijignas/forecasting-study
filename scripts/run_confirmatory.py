"""The confirmatory backtest (protocol section 2: a single execution from a clean commit, after configs are frozen).

    python -I scripts/run_confirmatory.py --stratum both

Refuses to run unless (i) the working tree is clean (``results/`` excluded), (ii) ``config/frozen_configs.json`` is
tracked in git. Per-series and per-origin results are cached in ``results/cache/confirmatory`` (git-ignored), so a
crash resumes from the cache; every cache file records the commit it was produced at. The run writes
``results/forecasts_{stratum}.csv.gz`` and ``results/run_meta.json`` (commit hash, library versions, timings).

    python -I scripts/run_confirmatory.py --stratum monthly --dry-run --series RSAFSNA INDPRO

is a timing/smoke run of the same code path on the VALIDATION origins and validation-truncated series, using the
dry-run validation caches; it reads no test-period value and writes only to ``results/cache/confirmatory_dryrun``.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecasting_study import pipeline as P  # noqa: E402
from forecasting_study.assemble import assemble  # noqa: E402
from forecasting_study.data import load_panel  # noqa: E402
from forecasting_study.runinfo import (free_memory_mb, git, git_commit, git_dirty, library_versions,  # noqa: E402
                                       write_json_atomic)
from forecasting_study.splits import SCHEDULES  # noqa: E402
from forecasting_study.validation import last_validation_target  # noqa: E402

MIN_FREE_MB = 250
FROZEN = ROOT / "config" / "frozen_configs.json"


def frozen_for(stratum: str, dry: bool, names: list[str]) -> tuple[dict, dict, dict]:
    """(ml_cfg, selected-per-series, meta) from the frozen JSON, or from the dry-run validation caches."""
    if not dry:
        fz = json.loads(FROZEN.read_text(encoding="utf-8"))
        s = fz[stratum]
        return s["ml_g"]["config"], {n: {"ETS": s["series"][n]["ETS"], "ARIMA": s["series"][n]["ARIMA"]}
                                     for n in names}, fz
    vdir = ROOT / "results" / "cache" / "validation_dryrun" / stratum
    ml_cfg = json.loads((vdir / "_ml_global.json").read_text(encoding="utf-8"))["chosen"]
    sel = {}
    for n in names:
        r = json.loads((vdir / f"{n}.json").read_text(encoding="utf-8"))["selected"]
        sel[n] = {"ETS": r["ETS"], "ARIMA": r["ARIMA"]}
    return ml_cfg, sel, {}


def run_stratum(stratum: str, cache_dir: Path, dry: bool, only: list[str] | None, commit: str) -> dict:
    sched = SCHEDULES[stratum]
    panel = load_panel(stratum, upto=last_validation_target(sched) if dry else None)
    origins = sched.val_origins if dry else sched.test_origins
    names = only or panel.names
    if dry:
        panel.series = {n: panel.series[n] for n in names}
    ml_cfg, selected, _ = frozen_for(stratum, dry, names)
    timing = {"ml_global_s": 0.0, "series_s": {}}
    t_wall = time.perf_counter()

    # --- stage 1: ML-G (one global model per horizon; one origin at a time, nothing kept in memory)
    for t in origins:
        path = P.ml_global_cache_path(cache_dir, stratum, t)
        if path.exists():
            print(f"[{stratum}] ML-G origin {t}: cached", flush=True)
            continue
        if free_memory_mb() < MIN_FREE_MB:
            sys.exit(f"only {free_memory_mb():.0f} MB free; stop and retry (the cache keeps finished work)")
        t0 = time.perf_counter()
        arr, meta = P.compute_ml_global(panel.series, panel.dates, stratum, t, sched, ml_cfg)
        meta.update({"commit": commit, "seconds": time.perf_counter() - t0})
        P.save_npz_atomic(path, arr, meta)
        print(f"[{stratum}] ML-G origin {t}: {meta['seconds']:.1f}s", flush=True)

    # --- stage 2: per-series models (baselines, ETS, ARIMA, Theta, ML-L)
    for name in names:
        path = P.series_cache_path(cache_dir, stratum, name)
        if path.exists():
            print(f"[{stratum}] {name}: cached", flush=True)
            continue
        if free_memory_mb() < MIN_FREE_MB:
            sys.exit(f"only {free_memory_mb():.0f} MB free; stop and retry (the cache keeps finished series)")
        t0 = time.perf_counter()
        arr, meta = P.compute_series(stratum, name, panel.series[name], panel.dates, sched, origins, selected[name],
                                     ml_cfg, {"commit": commit})
        meta["seconds"] = time.perf_counter() - t0
        P.save_npz_atomic(path, arr, meta)
        fb = {m: int(arr[f"{m}|fb"].sum()) for m in ("ETS", "ARIMA", "THETA")}
        print(f"[{stratum}] {name}: {meta['seconds']:.1f}s  fallbacks(ETS/ARIMA/THETA)={fb}", flush=True)

    # --- collect timings and commits from the caches (valid across resumed runs)
    commits, ml_s, series_s = set(), 0.0, {}
    for t in origins:
        _, m = P.load_npz(P.ml_global_cache_path(cache_dir, stratum, t))
        commits.add(m["commit"])
        ml_s += m["seconds"]
    for name in names:
        _, m = P.load_npz(P.series_cache_path(cache_dir, stratum, name))
        commits.add(m["commit"])
        series_s[name] = m["seconds"]
    out = {"stratum": stratum, "n_series": len(names), "origins": list(origins), "commits_in_cache": sorted(commits),
           "compute_seconds_ml_global": ml_s, "compute_seconds_series_total": sum(series_s.values()),
           "compute_seconds_series": series_s, "this_invocation_wall_s": time.perf_counter() - t_wall}
    if dry:
        print(f"[{stratum}] DRY RUN: ML-G {ml_s:.1f}s; per-series mean {sum(series_s.values()) / len(series_s):.1f}s "
              f"(origins {list(origins)})")
        return out
    df = assemble(stratum, panel, sched, origins, cache_dir)
    outpath = ROOT / "results" / f"forecasts_{stratum}.csv.gz"
    df.to_csv(outpath, index=False, float_format="%.10g", compression="gzip")
    out["forecast_file"] = outpath.name
    out["forecast_rows"] = len(df)
    out["fallback_counts"] = {m: {"point": int(g.fb_point.sum()), "cal": int(g.fb_cal.sum()), "mb": int(g.fb_mb.sum())}
                              for m, g in df.groupby("model")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratum", choices=["monthly", "daily", "both"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--series", nargs="*")
    args = ap.parse_args()
    if args.dry_run:
        cache_dir = ROOT / "results" / "cache" / "confirmatory_dryrun"
    else:
        if args.series:
            sys.exit("--series is only for dry runs")
        dirty = git_dirty()
        if dirty:
            sys.exit("working tree is not clean (outside results/): commit or stash first:\n" + "\n".join(dirty))
        try:
            git("ls-files", "--error-unmatch", "config/frozen_configs.json")
        except Exception:  # noqa: BLE001
            sys.exit("config/frozen_configs.json is not committed")
        cache_dir = ROOT / "results" / "cache" / "confirmatory"
    commit = git_commit()
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    strata = ["monthly", "daily"] if args.stratum == "both" else [args.stratum]
    results = {s: run_stratum(s, cache_dir, args.dry_run, args.series, commit) for s in strata}
    if args.dry_run:
        return
    meta = {"protocol_tag": "protocol-v1", "commit_at_final_invocation": commit,
            "git_status_clean_outside_results": True, "invocation_started_utc": started,
            "invocation_finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "frozen_config_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
            "libraries": library_versions(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "strata": results}
    write_json_atomic(ROOT / "results" / "run_meta.json", meta)
    log = ROOT / "results" / "run_log.jsonl"
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"commit": commit, "started": started, "finished": meta["invocation_finished_utc"],
                            "strata": {s: r["this_invocation_wall_s"] for s, r in results.items()}}) + "\n")
    print("done; commit", commit)


if __name__ == "__main__":
    main()
