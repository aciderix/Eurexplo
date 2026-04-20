#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature module v3 — ELO contextuel.

Au lieu d'un seul rating global par entité, on maintient un ELO par
(entité × contexte) où le contexte est défini par :
  - discipline (ATTELE / MONTE / PLAT / HAIE / STEEPLE / CROSS)
  - bucket de distance (< 1800 / 1800-2400 / 2400-3000 / > 3000 mètres)

Entités suivies : horse (nom), driver, entraineur.

Contexte = (discipline, distance_bucket). Un trotteur de 2100m ≠ 3600m.

Reads : pmu_dataset_v2.parquet
Writes: pmu_feat_elo_ctx_v3.parquet

Output colonnes (par participant) :
  elo_ctx_horse_pre, elo_ctx_driver_pre, elo_ctx_entraineur_pre
  elo_ctx_horse_ngames, elo_ctx_driver_ngames, elo_ctx_entraineur_ngames
  elo_ctx_horse_vs_field_avg   -- ELO horse - moyenne ELO field
  elo_ctx_driver_vs_field_avg
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


IN_PATH  = Path("pmu_dataset_v2.parquet")
OUT_PATH = Path("pmu_feat_elo_ctx_v3.parquet")

K_FACTOR = 24.0
BASE_RATING = 1500.0


def distance_bucket(d: float | None) -> str:
    if d is None or pd.isna(d):
        return "unk"
    if d < 1800: return "b_lt1800"
    if d < 2400: return "b_1800_2400"
    if d < 3000: return "b_2400_3000"
    return "b_ge3000"


def _elo_update(ratings: list[float], positions: list[int | None]) -> list[float]:
    n = len(ratings)
    if n < 2:
        return list(ratings)
    last_rank = max((p for p in positions if p is not None), default=0) + 1
    ranks = [p if p is not None else last_rank for p in positions]
    r_arr = np.array(ratings, dtype=np.float64)
    rk = np.array(ranks, dtype=np.float64)
    rk_i = rk[:, None]; rk_j = rk[None, :]
    S = np.where(rk_i < rk_j, 1.0, np.where(rk_i > rk_j, 0.0, 0.5))
    np.fill_diagonal(S, 0.0)
    r_i = r_arr[:, None]; r_j = r_arr[None, :]
    E = 1.0 / (1.0 + 10.0 ** ((r_j - r_i) / 400.0))
    np.fill_diagonal(E, 0.0)
    delta = (K_FACTOR / (n - 1)) * (S.sum(axis=1) - E.sum(axis=1))
    return list(r_arr + delta)


def run(limit: int | None = None) -> None:
    t0 = time.time()
    cols = ["race_id", "num_pmu", "file_date", "nom", "driver", "entraineur",
            "distance", "discipline", "finish_position"]
    df = pq.read_table(IN_PATH, columns=cols).to_pandas()
    if limit:
        df = df.head(limit).copy()
    df["date"] = pd.to_datetime(df["file_date"], format="%Y-%m-%d")
    df["dist_bucket"] = df["distance"].map(distance_bucket)
    df["ctx"] = df["discipline"].fillna("unk").astype(str) + "|" + df["dist_bucket"]
    df = df.sort_values(["date", "race_id", "num_pmu"], kind="stable").reset_index(drop=True)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    print(f"Contexts: {df['ctx'].nunique()}")

    # ratings[kind][ctx][name] = rating
    ratings: dict = {k: defaultdict(lambda: defaultdict(lambda: BASE_RATING))
                     for k in ("horse", "driver", "entraineur")}
    games:   dict = {k: defaultdict(lambda: defaultdict(int))
                     for k in ("horse", "driver", "entraineur")}
    col_map = {"horse": "nom", "driver": "driver", "entraineur": "entraineur"}

    n = len(df)
    out = {
        "elo_ctx_horse_pre":          np.full(n, np.nan),
        "elo_ctx_driver_pre":         np.full(n, np.nan),
        "elo_ctx_entraineur_pre":     np.full(n, np.nan),
        "elo_ctx_horse_ngames":       np.zeros(n, dtype=np.int32),
        "elo_ctx_driver_ngames":      np.zeros(n, dtype=np.int32),
        "elo_ctx_entraineur_ngames":  np.zeros(n, dtype=np.int32),
        "elo_ctx_horse_vs_field":     np.full(n, np.nan),
        "elo_ctx_driver_vs_field":    np.full(n, np.nan),
    }

    t1 = time.time()
    nr = 0
    for race_id, grp in df.groupby("race_id", sort=False):
        idx = grp.index.to_numpy()
        ctx = grp["ctx"].iloc[0]
        positions = [int(p) if pd.notna(p) else None for p in grp["finish_position"].tolist()]

        for kind in ("horse", "driver", "entraineur"):
            col = col_map[kind]
            names = grp[col].tolist()
            cur = [ratings[kind][ctx][n] if n else BASE_RATING for n in names]
            gam = [games[kind][ctx][n] if n else 0 for n in names]
            out[f"elo_ctx_{kind}_pre"][idx]    = cur
            out[f"elo_ctx_{kind}_ngames"][idx] = gam
            if kind in ("horse", "driver"):
                field_avg = float(np.mean(cur)) if cur else BASE_RATING
                out[f"elo_ctx_{kind}_vs_field"][idx] = np.array(cur) - field_avg

            # Mise à jour (uniquement si résultat connu)
            if any(p is not None for p in positions):
                new_r = _elo_update(cur, positions)
                for name, nr_ in zip(names, new_r):
                    if name:
                        ratings[kind][ctx][name] = nr_
                        games[kind][ctx][name] += 1

        nr += 1
        if nr % 20000 == 0:
            print(f"  {nr:>7,} races processed  ({nr/(time.time()-t1):.0f}/s)")

    print(f"\nDone {nr:,} races in {time.time()-t1:.1f}s")

    out_df = df[["race_id", "num_pmu"]].copy()
    for k, v in out.items():
        out_df[k] = v

    out_df.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
