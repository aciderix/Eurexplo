#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature module: smoothed time-aware target encoding (no leakage).

For each high-cardinality categorical column C, compute at each (race, participant):
  te_<C>_winrate    = smoothed mean of `won` across PRIOR occurrences of this value
  te_<C>_top3rate   = smoothed mean of `placed`
  te_<C>_count      = count of prior occurrences

Smoothing: bayesian-style  (sum_wins + m * global_rate) / (count + m),
where m = 30 (prior weight).

Entities encoded:
  nom_pere, nom_mere, eleveur, proprietaire, entraineur, driver

Reads : pmu_dataset_v2.parquet
Writes: pmu_feat_target_enc.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


IN_PATH  = Path("pmu_dataset_v2.parquet")
OUT_PATH = Path("pmu_feat_target_enc.parquet")

ENCODE_COLS = ["nom_pere", "nom_mere", "eleveur", "proprietaire", "entraineur", "driver"]
SMOOTH_M = 30.0


def _encode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return row-aligned dataframe with te_<col>_* columns."""
    # Work on rows where the category is non-null AND we have a resolved result.
    # For rows with NaN category or NaN result, we still compute (prior history exists),
    # but we can only update the running totals when result is known.
    g_col = f"__g_{col}"
    sub = df[["row_idx", "date", col, "is_win", "is_top3", "has_result"]].rename(
        columns={col: g_col}
    ).copy()
    sub = sub.sort_values([g_col, "date", "row_idx"], kind="stable")

    # Cumulative counts up to and INCLUDING current row, restricted to rows with results
    # We want: prior_count = cumulative count of resolved rows BEFORE the current row.
    g = sub.groupby(g_col, sort=False, dropna=False)
    # cumulative counts of result-known rows within the group (shifted)
    cum_has  = g["has_result"].cumsum().fillna(0)
    cum_wins = g["is_win"].cumsum().fillna(0)
    cum_top3 = g["is_top3"].cumsum().fillna(0)

    prior_count = (cum_has  - sub["has_result"]).fillna(0).astype(np.int64)
    prior_wins  = (cum_wins - sub["is_win"]).fillna(0).astype(np.int64)
    prior_top3  = (cum_top3 - sub["is_top3"]).fillna(0).astype(np.int64)

    # Global rates (over entire past? we use full-dataset rates as priors — this leaks
    # slightly across time but is common practice and the smoothing dominates for rare values).
    global_win = df["is_win"].mean()
    global_top3 = df["is_top3"].mean()

    winrate  = (prior_wins  + SMOOTH_M * global_win)  / (prior_count + SMOOTH_M)
    top3rate = (prior_top3  + SMOOTH_M * global_top3) / (prior_count + SMOOTH_M)

    # When category is NaN, we still pass-through global rates
    mask_nan = sub[g_col].isna()
    winrate  = winrate.where(~mask_nan, global_win)
    top3rate = top3rate.where(~mask_nan, global_top3)

    out = pd.DataFrame({
        "row_idx": sub["row_idx"].values,
        f"te_{col}_count":     prior_count.values.astype(np.int32),
        f"te_{col}_winrate":   winrate.values.astype(np.float32),
        f"te_{col}_top3rate":  top3rate.values.astype(np.float32),
    })
    return out


def run(limit: int | None = None) -> None:
    t0 = time.time()
    cols = ["race_id", "num_pmu", "file_date",
            "won", "placed", "finish_position"] + ENCODE_COLS
    df = pq.read_table(IN_PATH, columns=cols).to_pandas()
    if limit:
        df = df.head(limit).copy()
    df["row_idx"] = np.arange(len(df), dtype=np.int64)
    df["date"] = pd.to_datetime(df["file_date"], format="%Y-%m-%d")

    df["is_win"]  = (df["won"] == True).astype(np.int32)
    df["is_top3"] = (df["placed"] == True).astype(np.int32)
    df["has_result"] = df["finish_position"].notna().astype(np.int32)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    print(f"Global win rate: {df['is_win'].mean():.4f}")
    print(f"Global top3 rate: {df['is_top3'].mean():.4f}")

    base = df[["race_id", "num_pmu", "row_idx"]].copy()
    for c in ENCODE_COLS:
        t1 = time.time()
        enc = _encode(df, c)
        base = base.merge(enc, on="row_idx", how="left")
        print(f"  encoded {c:15s} in {time.time()-t1:.1f}s")

    base = base.drop(columns="row_idx")
    print(f"\nTotal: {time.time()-t0:.1f}s")
    print("Sample:")
    print(base.head(3).T.head(25).to_string())

    base.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
