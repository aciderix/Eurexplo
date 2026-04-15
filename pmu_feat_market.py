#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature module: market / odds.

Derives per-(race, participant) features from the live odds columns
(drd_rapport, drr_rapport, drd_favoris, drd_grosse_prise, drd_nombre_indicateur_tendance).

Reads : pmu_dataset_v2.parquet
Writes: pmu_feat_market.parquet

Features:
  mkt_drd_rapport, mkt_drr_rapport  -- passthrough (cleaned)
  mkt_drd_rank    -- rank within race (1 = smallest rapport = favorite)
  mkt_drr_rank
  mkt_drd_prob    -- 1 / drd_rapport
  mkt_drr_prob
  mkt_drd_prob_norm  -- drd_prob / sum(drd_prob within race)  (overround-corrected)
  mkt_drr_prob_norm
  mkt_spread_drr_drd -- (drr - drd) / drr   ("money-in" indicator)
  mkt_favoris_flag
  mkt_grosse_prise_flag
  mkt_tendance_count
  mkt_race_nb_partants     -- number of declared runners with a drd quote
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
OUT_PATH = Path("pmu_feat_market.parquet")


def run(limit: int | None = None) -> None:
    t0 = time.time()
    cols = [
        "race_id", "num_pmu",
        "drd_rapport", "drr_rapport",
        "drd_favoris", "drd_grosse_prise", "drd_nombre_indicateur_tendance",
    ]
    df = pq.read_table(IN_PATH, columns=cols).to_pandas()
    if limit:
        df = df.head(limit).copy()
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    t1 = time.time()
    df["mkt_drd_rapport"] = df["drd_rapport"]
    df["mkt_drr_rapport"] = df["drr_rapport"]

    # Implied probabilities
    df["mkt_drd_prob"] = 1.0 / df["drd_rapport"].where(df["drd_rapport"] > 0)
    df["mkt_drr_prob"] = 1.0 / df["drr_rapport"].where(df["drr_rapport"] > 0)

    grp = df.groupby("race_id", sort=False)

    # Ranks (1 = smallest rapport = favorite). NaN stays NaN.
    df["mkt_drd_rank"] = grp["drd_rapport"].rank(method="min", ascending=True)
    df["mkt_drr_rank"] = grp["drr_rapport"].rank(method="min", ascending=True)

    # Normalized implied probability (remove overround)
    sum_drd = grp["mkt_drd_prob"].transform("sum")
    sum_drr = grp["mkt_drr_prob"].transform("sum")
    df["mkt_drd_prob_norm"] = df["mkt_drd_prob"] / sum_drd.where(sum_drd > 0)
    df["mkt_drr_prob_norm"] = df["mkt_drr_prob"] / sum_drr.where(sum_drr > 0)

    # Spread (money flow). Positive = horse took money vs reference.
    df["mkt_spread_drr_drd"] = (df["drr_rapport"] - df["drd_rapport"]) / df["drr_rapport"].where(df["drr_rapport"] > 0)

    # Scalars / flags
    df["mkt_favoris_flag"] = df["drd_favoris"].fillna(False).astype(bool).astype(np.int8)
    df["mkt_grosse_prise_flag"] = df["drd_grosse_prise"].fillna(False).astype(bool).astype(np.int8)
    df["mkt_tendance_count"] = df["drd_nombre_indicateur_tendance"].fillna(0).astype(np.int32)

    df["mkt_race_nb_partants"] = grp["drd_rapport"].transform("count").astype("Int32")

    print(f"Computed in {time.time()-t1:.1f}s")

    keep = [
        "race_id", "num_pmu",
        "mkt_drd_rapport", "mkt_drr_rapport",
        "mkt_drd_rank", "mkt_drr_rank",
        "mkt_drd_prob", "mkt_drr_prob",
        "mkt_drd_prob_norm", "mkt_drr_prob_norm",
        "mkt_spread_drr_drd",
        "mkt_favoris_flag", "mkt_grosse_prise_flag",
        "mkt_tendance_count", "mkt_race_nb_partants",
    ]
    out = df[keep]
    print("\nStats:")
    print(out.describe(include="all").T[["count", "mean", "std", "min", "max"]].to_string())

    out.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
