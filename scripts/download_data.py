"""Download the study data and write data/MANIFEST.json.

Usage (from the repository root):
    python -I scripts/download_data.py            # download (skips files already present) and write the manifest
    python -I scripts/download_data.py --verify   # re-hash local files and compare with data/MANIFEST.json

Only standard library + numpy + pandas are used. Nothing is fitted or scored here: the script downloads,
hashes, counts rows, checks completeness and (for the electricity data) aggregates hourly to daily.

Raw and processed data are NOT committed (data/raw and data/processed are git-ignored); the manifest records
source URL, download date, SHA-256, rows, frequency, date range and a licence/terms note for every file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_FRED = DATA / "raw" / "fred"
RAW_ELEC = DATA / "raw" / "electricity"
PROCESSED = DATA / "processed"
MANIFEST = DATA / "MANIFEST.json"
SERIES_LIST = DATA / "series_list.json"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
# NOTE: FRED drops connections that carry a custom or browser User-Agent; the default urllib UA works (observed 2026-10-16).
MAX_ZIP_BYTES = 50 * 1024 * 1024  # hard cap, plan says keep under ~50 MB

FRED_LICENCE_NOTE = (
    "Data obtained through FRED (Federal Reserve Bank of St. Louis) public CSV endpoint. FRED terms "
    "(https://fred.stlouisfed.org/legal/): free for personal, non-commercial and educational use with citation of "
    "the source agency 'via FRED'; copyrighted third-party series need the holder's permission beyond that; FRED "
    "also prohibits bulk scraping/mirroring/archiving. Underlying series are from US statistical agencies. "
    "Raw files are therefore NOT redistributed in this repository; only this script and SHA-256 hashes are."
)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url: str, tries: int = 4, timeout: int = 120, max_bytes: int | None = None) -> bytes:
    last: Exception | None = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read(max_bytes + 1 if max_bytes else -1)
            if max_bytes and len(data) > max_bytes:
                raise RuntimeError(f"response exceeds {max_bytes} bytes")
            return data
        except Exception as e:  # noqa: BLE001 - retry on any network error
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


def now_utc_date() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def fred_entry(spec: dict, window: dict, force: bool) -> dict:
    sid = spec["id"]
    url = FRED_URL.format(id=sid)
    path = RAW_FRED / f"{sid}.csv"
    RAW_FRED.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raw = path.read_bytes()
        date = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).date().isoformat()
    else:
        raw = http_get(url)
        path.write_bytes(raw)
        date = now_utc_date()
        time.sleep(1.0)  # polite rate limit
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED writes '.' for missing
    valid = df.dropna(subset=["value"])
    w = df[(df["date"] >= window["start"]) & (df["date"] <= window["end"])]
    expected = pd.date_range(window["start"], window["end"], freq="MS")
    complete = bool(len(w) == len(expected) and w["value"].notna().all() and (w["date"].values == expected.values).all())
    steps = df["date"].diff().dt.days.dropna()
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "series_id": sid,
        "title": spec["title"],
        "group": spec["group"],
        "adjustment": spec["adjustment"],
        "source_agency": spec["source"],
        "source_url": url,
        "download_date_utc": date,
        "sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "rows": int(len(df)),
        "rows_nonmissing": int(len(valid)),
        "frequency": "monthly",
        "date_range_file": [valid["date"].min().date().isoformat(), valid["date"].max().date().isoformat()],
        "window": [window["start"], window["end"]],
        "window_rows": int(len(w)),
        "window_missing": int(w["value"].isna().sum()),
        "window_nonpositive": int((w["value"] <= 0).sum()),
        "window_complete_monthly": complete,
        "median_step_days": float(steps.median()),
        "licence_note": FRED_LICENCE_NOTE,
    }


def parse_tsf(text: str) -> tuple[dict, list[tuple[str, str, np.ndarray]]]:
    meta: dict = {}
    series: list[tuple[str, str, np.ndarray]] = []
    in_data = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not in_data:
            if line.lower() == "@data":
                in_data = True
            elif line.startswith("@"):
                parts = line[1:].split(None, 1)
                meta[parts[0].lower()] = parts[1] if len(parts) > 1 else ""
            continue
        name, start, vals = line.split(":", 2)
        arr = np.array([np.nan if v == "?" else float(v) for v in vals.split(",")], dtype="float64")
        series.append((name, start, arr))
    return meta, series


def electricity_entries(spec: dict, force: bool) -> list[dict]:
    RAW_ELEC.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    zpath = RAW_ELEC / spec["file_name"]
    if zpath.exists() and not force:
        date = dt.datetime.fromtimestamp(zpath.stat().st_mtime, dt.timezone.utc).date().isoformat()
    else:
        zpath.write_bytes(http_get(spec["file_url"], max_bytes=MAX_ZIP_BYTES))
        date = now_utc_date()
    zbytes = zpath.read_bytes()
    md5 = hashlib.md5(zbytes).hexdigest()  # noqa: S324 - integrity check against Zenodo's published md5 only
    md5_ok = md5 == spec["zenodo_md5"]
    if not md5_ok:
        raise RuntimeError(f"md5 mismatch for {zpath}: got {md5}, expected {spec['zenodo_md5']}")
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        tsf_names = [n for n in zf.namelist() if n.lower().endswith(".tsf") and ".." not in n and not n.startswith("/")]
        if len(tsf_names) != 1:
            raise RuntimeError(f"expected exactly one .tsf in archive, found {zf.namelist()}")
        info = zf.getinfo(tsf_names[0])
        if info.file_size > 300 * 1024 * 1024:
            raise RuntimeError("tsf member unexpectedly large")
        tsf_bytes = zf.read(tsf_names[0])
    (RAW_ELEC / "electricity_hourly_dataset.tsf").write_bytes(tsf_bytes)
    meta, series = parse_tsf(tsf_bytes.decode("utf-8"))
    n_series = len(series)
    lengths = {len(a) for _, _, a in series}
    starts = {s for _, s, _ in series}
    assert len(lengths) == 1 and len(starts) == 1, "unequal lengths/starts not handled"
    L = lengths.pop()
    start_str = starts.pop()  # e.g. '2012-01-01 00-00-00'
    # The Monash file stamps the first hour as '2012-01-01 00-00-01' (a 1-second quirk); treat as midnight.
    assert start_str[11:16] == "00-00", f"series do not start at midnight: {start_str}"
    start = pd.Timestamp(start_str[:10])
    hourly = np.vstack([a for _, _, a in series])  # (n_series, L)
    n_missing = int(np.isnan(hourly).sum())
    zeros_per_series = (hourly == 0).sum(axis=1)
    eligible = [i for i in range(n_series) if zeros_per_series[i] <= 3 and not np.isnan(hourly[i]).any()]
    k = spec["n_series"]
    chosen = [eligible[int(np.floor(i * len(eligible) / k))] for i in range(k)]
    n_days = L // 24
    assert L % 24 == 0
    daily = hourly[:, : n_days * 24].reshape(n_series, n_days, 24).sum(axis=2)
    idx = pd.date_range(start, periods=n_days, freq="D")
    out = pd.DataFrame(daily[chosen].T, index=idx, columns=[series[i][0] for i in chosen])
    out.index.name = "date"
    ppath = PROCESSED / "electricity_daily_30.csv"
    out.to_csv(ppath, float_format="%.6g")
    common = {
        "source_url": spec["record_url"],
        "download_date_utc": date,
        "licence_note": spec["licence"] + ". Attribution required: " + spec["citation"],
    }
    return [
        {
            "file": zpath.relative_to(ROOT).as_posix(),
            "description": "Monash/Zenodo archive of 321 hourly electricity-load series (UCI ElectricityLoadDiagrams, 2012-2014)",
            "download_url": spec["file_url"],
            "sha256": sha256_bytes(zbytes),
            "md5_verified_against_zenodo": True,
            "bytes": len(zbytes),
            "rows": int(L),
            "n_series": n_series,
            "frequency": "hourly",
            "date_range": [str(start.date()), str((start + pd.Timedelta(hours=L - 1)).date())],
            "missing_values": n_missing,
            "n_series_zero_hours_gt3": int((zeros_per_series > 3).sum()),
            "n_eligible_series": len(eligible),
            **common,
        },
        {
            "file": ppath.relative_to(ROOT).as_posix(),
            "description": "Derived: daily sums of 30 selected series (selection rule in data/series_list.json); not committed",
            "derived_from": zpath.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(ppath),
            "bytes": ppath.stat().st_size,
            "rows": int(len(out)),
            "n_series": int(out.shape[1]),
            "frequency": "daily",
            "date_range": [str(out.index.min().date()), str(out.index.max().date())],
            "series_ids": list(out.columns),
            "selected_positions_in_file": chosen,
            "selection_rule": spec["selection_rule"],
            "aggregation": spec["aggregation"],
            "n_zero_daily_totals": int((out.values == 0).sum()),
            "source_url": spec["record_url"],
            "download_date_utc": date,
            "licence_note": common["licence_note"],
        },
    ]


def build(force: bool) -> None:
    spec = json.loads(SERIES_LIST.read_text(encoding="utf-8"))
    entries = []
    for s in spec["fred"]:
        e = fred_entry(s, spec["window"], force)
        flag = "" if (e["window_complete_monthly"] and e["window_nonpositive"] == 0) else "  <-- PROBLEM"
        print(f"{e['series_id']:18s} rows={e['rows']:5d} window_rows={e['window_rows']} missing={e['window_missing']}{flag}")
        entries.append(e)
    entries.extend(electricity_entries(spec["electricity"], force))
    manifest = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/download_data.py",
        "python": sys.version.split()[0],
        "window_monthly": spec["window"],
        "note": "FRED values can be revised; the SHA-256 pins the exact bytes used. Raw/processed data are git-ignored.",
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    bad = [e["series_id"] for e in entries if "series_id" in e and not (e["window_complete_monthly"] and e["window_nonpositive"] == 0)]
    print("manifest written:", MANIFEST, "| files:", len(entries), "| problems:", bad or "none")


def verify() -> int:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = 0
    for e in m["files"]:
        p = ROOT / e["file"]
        if not p.exists():
            print("MISSING", e["file"])
            bad += 1
            continue
        ok = sha256_file(p) == e["sha256"]
        print("ok  " if ok else "DIFF", e["file"])
        bad += 0 if ok else 1
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="re-hash local files against MANIFEST.json")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    a = ap.parse_args()
    sys.exit(verify() if a.verify else (build(a.force) or 0))
