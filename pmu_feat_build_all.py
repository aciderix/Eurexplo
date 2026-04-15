#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator: build the final feature matrix by joining all pmu_feat_*.parquet.

Reads :
  pmu_dataset_v2.parquet      (base scalar columns)
  pmu_feat_musique.parquet
  pmu_feat_market.parquet
  pmu_feat_rolling.parquet
  pmu_feat_elo.parquet
  pmu_feat_target_enc.parquet

Writes: pmu_features_v2.parquet  (one row per (race, participant), ~300+ cols)

Usage:
  python pmu_feat_build_all.py
  python pmu_feat_build_all.py --run-missing    # run any feat module whose output is missing
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd


BASE = Path("pmu_dataset_v2.parquet")
OUT  = Path("pmu_features_v2.parquet")

FEAT_MODULES = [
    ("pmu_feat_musique.py",      Path("pmu_feat_musique.parquet")),
    ("pmu_feat_market.py",       Path("pmu_feat_market.parquet")),
    ("pmu_feat_rolling.py",      Path("pmu_feat_rolling.parquet")),
    ("pmu_feat_elo.py",          Path("pmu_feat_elo.parquet")),
    ("pmu_feat_target_enc.py",   Path("pmu_feat_target_enc.parquet")),
]


def ensure_features(run_missing: bool) -> None:
    for script, out_path in FEAT_MODULES:
        if out_path.exists():
            print(f"  OK  {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
            continue
        if not run_missing:
            raise SystemExit(f"Missing {out_path}. Run `python {script}` first (or use --run-missing).")
        print(f"  >>> running {script} (output missing)...")
        subprocess.check_call([sys.executable, script])


def join_all() -> None:
    t0 = time.time()
    print(f"Loading base {BASE}...")
    base = pq.read_table(BASE).to_pandas()
    print(f"  {len(base):,} rows, {base.shape[1]} cols in {time.time()-t0:.1f}s")

    for _, path in FEAT_MODULES:
        t1 = time.time()
        feat = pq.read_table(path).to_pandas()
        before = base.shape[1]
        base = base.merge(feat, on=["race_id", "num_pmu"], how="left")
        added = base.shape[1] - before
        print(f"  joined {path.name:28s} (+{added} cols)  in {time.time()-t1:.1f}s")

    print(f"\nFinal shape: {base.shape[0]:,} rows × {base.shape[1]} cols")
    t2 = time.time()
    base.to_parquet(OUT, compression="zstd", index=False)
    print(f"Wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB) in {time.time()-t2:.1f}s")
    print(f"Total orchestrator time: {time.time()-t0:.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-missing", action="store_true",
                    help="Run any pmu_feat_*.py whose output parquet is missing.")
    args = ap.parse_args()

    ensure_features(args.run_missing)
    join_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
