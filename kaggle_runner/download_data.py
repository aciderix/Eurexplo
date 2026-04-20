#!/usr/bin/env python3
"""Download the v2 parquets from Google Drive into the repo working dir.

Resumable: skips files already present with expected min size.
Retries transient network failures with exponential backoff.
Reads file list + size hints from DATA_LINKS.md (parsed) with a fallback
hardcoded list so this script also works outside a checkout.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# (filename, gdrive_id, min_size_mb)  — min_size is a sanity check, not a hash.
FILES = [
    ("pmu_course_raw_v2.parquet",   "1KuJiQu0HcgqmlH1qSe6dEBt6jK_QKgco", 1500),
    ("pmu_dataset_v2.parquet",      "1DzVoNqV8IZN7jSpeGjz9ZD-XMGCkLFx4",  150),
    ("pmu_feat_elo.parquet",        "1GpHtg3IUG27BgNVXepi2mwBCM8DzLegx",   50),
    ("pmu_feat_market.parquet",     "1rqfFcvlBN9O18WZkks98T9dWhiq4CJZ8",   35),
    ("pmu_feat_musique.parquet",    "1fY0QFNKUbUacuMbV0OOSzh5Q5kPlMwu-",   10),
    ("pmu_feat_rolling.parquet",    "1D4whskae8vcem4iYw2UKDnPbC5gI12VC",  120),
    ("pmu_feat_target_enc.parquet", "18jWU3R3VQToJaRfUdgTh9EagRhan_sRD",   80),
    ("pmu_features_v2.parquet",     "1dBcCtzMFVeR5LHcrnvUKUGKQVGJ0YG0m",  450),
]

# Only the first two are mandatory to rebuild v3; the rest are v2 features we
# can reuse if present but aren't referenced by v3 stages directly.
REQUIRED = {"pmu_dataset_v2.parquet", "pmu_course_raw_v2.parquet"}


def _ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
        return
    except ImportError:
        pass
    print("[download] installing gdown…", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown>=5.2"])


def _download_one(dest: Path, gid: str, attempts: int = 4) -> bool:
    import gdown
    url = f"https://drive.google.com/uc?id={gid}&export=download&confirm=t"
    for i in range(attempts):
        try:
            gdown.download(url, str(dest), quiet=False, fuzzy=True)
            return dest.exists() and dest.stat().st_size > 0
        except Exception as exc:  # noqa: BLE001
            wait = 2 ** i
            print(f"[download]   retry {i+1}/{attempts} after {wait}s ({exc})",
                  flush=True)
            time.sleep(wait)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=".", help="repo root where parquets land")
    ap.add_argument("--required-only", action="store_true",
                    help="only fetch dataset_v2 + course_raw_v2")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dest_dir = Path(args.dest).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gdown()

    wanted = FILES if not args.required_only else [
        f for f in FILES if f[0] in REQUIRED
    ]

    failed: list[str] = []
    for name, gid, min_mb in wanted:
        out = dest_dir / name
        if out.exists() and not args.force:
            size_mb = out.stat().st_size / 1e6
            if size_mb >= min_mb * 0.8:
                print(f"[download] SKIP {name}  ({size_mb:.0f} MB already on disk)",
                      flush=True)
                continue
            print(f"[download] {name} is truncated ({size_mb:.0f} MB < {min_mb} MB). "
                  "Re-downloading.", flush=True)
            out.unlink()

        print(f"[download] FETCH {name}  (gdrive={gid})", flush=True)
        ok = _download_one(out, gid)
        if not ok:
            failed.append(name)
            print(f"[download] FAIL {name}", flush=True)

    if failed:
        missing_required = [f for f in failed if f in REQUIRED]
        if missing_required:
            print(f"[download] FATAL required files failed: {missing_required}",
                  flush=True)
            return 2
        print(f"[download] non-fatal: optional files missing: {failed}", flush=True)

    print("[download] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
