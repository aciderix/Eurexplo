#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator v3 — assemble la matrice finale pmu_features_v3.parquet.

Entrées :
  pmu_dataset_v2.parquet        base flat (1 ligne / (course, cheval))
  pmu_feat_musique.parquet      features v2 (scalars)
  pmu_feat_market.parquet       features v2
  pmu_feat_rolling.parquet      features v2
  pmu_feat_elo.parquet          features v2
  pmu_feat_target_enc.parquet   features v2
  pmu_feat_blobs_v3.parquet     NEW v3
  pmu_feat_elo_ctx_v3.parquet   NEW v3
  pmu_feat_h2h_v3.parquet       NEW v3
  pmu_feat_emb_v3.parquet       NEW v3 (optionnel)
  pmu_feat_musique_seq_v3.parquet NEW v3

Sortie : pmu_features_v3.parquet  (~400-500 colonnes)

Usage:
  python pmu_feat_build_all_v3.py
  python pmu_feat_build_all_v3.py --run-missing  # lance les feat_*.py manquants
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


BASE = Path("pmu_dataset_v2.parquet")
OUT  = Path("pmu_features_v3.parquet")

# (script, output_parquet, optional)
FEAT_MODULES = [
    # v2
    ("pmu_feat_musique.py",       Path("pmu_feat_musique.parquet"),       False),
    ("pmu_feat_market.py",        Path("pmu_feat_market.parquet"),        False),
    ("pmu_feat_rolling.py",       Path("pmu_feat_rolling.parquet"),       False),
    ("pmu_feat_elo.py",           Path("pmu_feat_elo.parquet"),           False),
    ("pmu_feat_target_enc.py",    Path("pmu_feat_target_enc.parquet"),    False),
    # v3
    ("pmu_feat_blobs.py",         Path("pmu_feat_blobs_v3.parquet"),      False),
    ("pmu_feat_elo_ctx.py",       Path("pmu_feat_elo_ctx_v3.parquet"),    False),
    ("pmu_feat_h2h.py",           Path("pmu_feat_h2h_v3.parquet"),        False),
    ("pmu_feat_embeddings.py",    Path("pmu_feat_emb_v3.parquet"),        True),   # lourd (optionnel)
    ("pmu_feat_musique_seq.py",   Path("pmu_feat_musique_seq_v3.parquet"), False),
]


def ensure_features(run_missing: bool) -> list[Path]:
    paths = []
    for script, out_path, optional in FEAT_MODULES:
        if out_path.exists():
            print(f"  OK   {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
            paths.append(out_path)
            continue
        if optional:
            print(f"  SKIP {out_path}  (optionnel, absent)")
            continue
        if not run_missing:
            raise SystemExit(f"Missing {out_path}. Run `python {script}` first (ou utilise --run-missing).")
        print(f"  >>> running {script} (sortie manquante)...")
        subprocess.check_call([sys.executable, script])
        paths.append(out_path)
    return paths


def join_all(paths: list[Path]) -> None:
    t0 = time.time()
    print(f"Loading base {BASE}...")
    base = pq.read_table(BASE).to_pandas()
    print(f"  {len(base):,} rows, {base.shape[1]} cols in {time.time()-t0:.1f}s")

    for path in paths:
        t1 = time.time()
        feat = pq.read_table(path).to_pandas()
        before = base.shape[1]
        # Drop overlapping feature cols from `feat` (keep keys)
        dup_cols = [c for c in feat.columns
                    if c in base.columns and c not in ("race_id", "num_pmu")]
        if dup_cols:
            print(f"  drop {len(dup_cols)} overlapping cols from {path.name}: {dup_cols[:3]}...")
            feat = feat.drop(columns=dup_cols)
        base = base.merge(feat, on=["race_id", "num_pmu"], how="left")
        added = base.shape[1] - before
        print(f"  joined {path.name:38s} (+{added} cols)  in {time.time()-t1:.1f}s")

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
    paths = ensure_features(args.run_missing)
    join_all(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
