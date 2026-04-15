#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature module: ELO rating for horses, drivers and entraineurs (no leakage).

Processes races in chronological order. For each race:
  1. Emit the CURRENT rating (pre-race) of each participant   → feature
  2. Update ratings using finish positions

Rating scheme: multi-competitor ELO where each finisher's score against every
other finisher is 1 if it finished higher, 0 if lower, 0.5 if equal; then
expected vs actual differential is used for the ELO update with K=24.

Entities tracked:
  - horse (by `nom`)
  - driver
  - entraineur

Reads : pmu_dataset_v2.parquet
Writes: pmu_feat_elo.parquet

Output columns (per participant):
  elo_horse_pre, elo_driver_pre, elo_entraineur_pre
  elo_horse_ngames, elo_driver_ngames, elo_entraineur_ngames
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
OUT_PATH = Path("pmu_feat_elo.parquet")

K_FACTOR = 24.0
BASE_RATING = 1500.0


def _elo_update(ratings: list[float], positions: list[int | None]) -> list[float]:
    """Return updated ratings for a single race.

    positions[i] = finish_position (int) or None for DNF. DNF participants are
    treated as finishing last (all tied).
    """
    n = len(ratings)
    if n < 2:
        return list(ratings)

    # Tie-break DNFs to last equal rank
    last_rank = max((p for p in positions if p is not None), default=0) + 1
    ranks = [p if p is not None else last_rank for p in positions]

    new_r = list(ratings)
    # Vectorize the pairwise expected score computation.
    r_arr = np.array(ratings, dtype=np.float64)
    rk = np.array(ranks, dtype=np.float64)

    # Actual score S_ij: 1 if rank_i < rank_j (better), 0 if worse, 0.5 if equal
    rk_i = rk[:, None]
    rk_j = rk[None, :]
    S = np.where(rk_i < rk_j, 1.0, np.where(rk_i > rk_j, 0.0, 0.5))
    np.fill_diagonal(S, 0.0)

    # Expected E_ij = 1 / (1 + 10^((R_j - R_i)/400))
    r_i = r_arr[:, None]
    r_j = r_arr[None, :]
    E = 1.0 / (1.0 + 10.0 ** ((r_j - r_i) / 400.0))
    np.fill_diagonal(E, 0.0)

    # Aggregate update: sum over opponents, scaled by K / (n-1)
    delta = (K_FACTOR / (n - 1)) * (S.sum(axis=1) - E.sum(axis=1))
    return list(r_arr + delta)


def run(limit: int | None = None) -> None:
    t0 = time.time()
    cols = ["race_id", "num_pmu", "file_date",
            "nom", "driver", "entraineur",
            "finish_position"]
    df = pq.read_table(IN_PATH, columns=cols).to_pandas()
    if limit:
        df = df.head(limit).copy()
    df["date"] = pd.to_datetime(df["file_date"], format="%Y-%m-%d")
    df = df.sort_values(["date", "race_id", "num_pmu"], kind="stable").reset_index(drop=True)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Group iteration
    ratings: dict[str, dict[str, float]] = {
        "horse":      defaultdict(lambda: BASE_RATING),
        "driver":     defaultdict(lambda: BASE_RATING),
        "entraineur": defaultdict(lambda: BASE_RATING),
    }
    games: dict[str, dict[str, int]] = {
        "horse":      defaultdict(int),
        "driver":     defaultdict(int),
        "entraineur": defaultdict(int),
    }
    entity_col = {"horse": "nom", "driver": "driver", "entraineur": "entraineur"}

    # Preallocate output columns
    n = len(df)
    out = {
        "elo_horse_pre":          np.full(n, np.nan, dtype=np.float64),
        "elo_driver_pre":         np.full(n, np.nan, dtype=np.float64),
        "elo_entraineur_pre":     np.full(n, np.nan, dtype=np.float64),
        "elo_horse_ngames":       np.zeros(n, dtype=np.int32),
        "elo_driver_ngames":      np.zeros(n, dtype=np.int32),
        "elo_entraineur_ngames":  np.zeros(n, dtype=np.int32),
    }

    t1 = time.time()
    n_races = 0
    for race_id, grp in df.groupby("race_id", sort=False):
        idx = grp.index.to_numpy()
        positions = grp["finish_position"].tolist()
        # Cast to Python int or None
        positions = [int(p) if pd.notna(p) else None for p in positions]

        for kind in ("horse", "driver", "entraineur"):
            col = entity_col[kind]
            names = grp[col].tolist()
            cur = [ratings[kind][name] if name else BASE_RATING for name in names]
            games_cur = [games[kind][name] if name else 0 for name in names]
            out[f"elo_{kind}_pre"][idx] = cur
            out[f"elo_{kind}_ngames"][idx] = games_cur

            # Only update if race has a known result (has at least one non-None position).
            if any(p is not None for p in positions):
                new_r = _elo_update(cur, positions)
                for name, nr in zip(names, new_r):
                    if name:
                        ratings[kind][name] = nr
                        games[kind][name] += 1

        n_races += 1
        if n_races % 20000 == 0:
            dt = time.time() - t1
            rate = n_races / max(dt, 1e-6)
            print(f"  {n_races:>7,} races processed  ({rate:.0f} races/s)")

    print(f"\nDone {n_races:,} races in {time.time()-t1:.1f}s")

    out_df = df[["race_id", "num_pmu"]].copy()
    for k, v in out.items():
        out_df[k] = v

    print("\nStats:")
    print(out_df.describe().T[["count", "mean", "std", "min", "max"]].to_string())

    out_df.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
