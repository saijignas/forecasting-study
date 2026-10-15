"""Small helpers for run scripts: free memory, git state, library versions, atomic JSON writes."""
from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def free_memory_mb() -> float:
    """Free physical memory in MB (Windows; falls back to /proc/meminfo; ``inf`` if unknown)."""
    if os.name == "nt":
        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMSTAT()
        st.dwLength = ctypes.sizeof(MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / 2**20
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("inf")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_commit() -> str:
    return git("rev-parse", "HEAD")


def git_dirty(exclude: tuple[str, ...] = ("results",)) -> list[str]:
    """Porcelain lines of tracked-modified or untracked files, ignoring the given top-level paths."""
    pathspec = ["--", "."] + [f":(exclude){e}" for e in exclude]
    out = git("status", "--porcelain", *pathspec)
    return [ln for ln in out.splitlines() if ln.strip()]


def library_versions() -> dict[str, str]:
    import importlib.metadata as md
    pk = ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib"]
    return {"python": platform.python_version(), **{p: md.version(p) for p in pk}}


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


def require_env_single_thread() -> None:
    """The protocol fixes OMP_NUM_THREADS=1 (section 12); scripts set it before importing numpy-heavy modules."""
    if os.environ.get("OMP_NUM_THREADS") != "1":
        sys.exit("set OMP_NUM_THREADS=1 before importing numpy (protocol section 12)")
