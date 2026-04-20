#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature module: rolling win/place rates per entity (no leakage).

For each (race, participant), compute aggregates over the entity's PRIOR races
(strict past only) on 30d / 90d / 365d windows.

Entities: driver, entraineur, nom (horse), proprietaire

For each entity E and window W, we emit:
  ent<E>_n_races_<W>       # number of prior races in window
  ent<E>_n_wins_<W>        # prior wins
  ent<E>_n_top3_<W>        # prior top3
  ent<E>_winrate_<W>       # wins / n_races (NaN if n_races == 0)
  ent<E>_top3rate_<W>      # top3 / n_races

Reads : pmu_dataset_v2.parquet
Writes: pmu_feat_rolling.parquet

Implementation: pure-pandas with pre-sort + expanding cumulative counts at the
entity level, then subtract the count at (now - W days) to get the window count.
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
OUT_PATH = Path("pmu_feat_rolling.parquet")

ENTITIES = ["driver", "entraineur", "nom", "proprietaire"]
WINDOWS_DAYS = [30, 90, 365]


def _compute_entity_rolling(df: pd.DataFrame, ent_col: str, prefix: str) -> pd.DataFrame:
    """Return a frame indexed like df with rolling features for this entity.

    We compute, per entity, the cumulative (n_races, n_wins, n_top3) up to each
    date (inclusive of prior rows only). For a window W, the feature at row i is
    `cum@(i-1)` minus `cum@(last index with date < date_i - W)`.

    This gives a strict-past rolling count without leakage.
    """
    # Work on a slimmed, sorted frame
    cols = ["row_idx", "date", ent_col, "is_win", "is_top3"]
    sub = df[cols].copy()
    sub = sub[sub[ent_col].notna()].sort_values([ent_col, "date", "row_idx"], kind="stable")

    # Cumulative counts up to and INCLUDING current row
    g = sub.groupby(ent_col, sort=False)
    sub["cum_n"]     = g.cumcount() + 1                    # count races (inc.)
    sub["cum_wins"]  = g["is_win"].cumsum().astype(np.int32)
    sub["cum_top3"]  = g["is_top3"].cumsum().astype(np.int32)

    # Shift so the counts represent strictly PRIOR races (exclude current)
    sub["prior_n"]    = (sub["cum_n"] - 1).astype(np.int32)
    sub["prior_wins"] = (sub["cum_wins"] - sub["is_win"]).astype(np.int32)
    sub["prior_top3"] = (sub["cum_top3"] - sub["is_top3"]).astype(np.int32)

    # For window-based counts, we need: prior_cum_at_date - prior_cum_at_(date - W)
    # We'll build it via merge_asof on the shifted cumulative sequence per entity.
    out_frames = [sub[["row_idx", "prior_n", "prior_wins", "prior_top3"]]]
    out_frames[0] = out_frames[0].rename(columns={
        "prior_n":     f"ent_{prefix}_n_races_all",
        "prior_wins":  f"ent_{prefix}_n_wins_all",
        "prior_top3":  f"ent_{prefix}_n_top3_all",
    })

    for W in WINDOWS_DAYS:
        # Window count  = (races for this entity with date < current_date)
        #              -  (races for this entity with date <= current_date - W)
        # The first term is `prior_n` (cumulative excluding current row).
        # The second term is `cum_n` of the latest past record with date <= cutoff.
        cutoff = sub["date"] - pd.Timedelta(days=W)
        left = sub[[ent_col, "date", "row_idx"]].assign(_cut=cutoff)
        right = sub[[ent_col, "date", "cum_n", "cum_wins", "cum_top3"]].rename(
            columns={"date": "_cut_date",
                     "cum_n":    "base_n",
                     "cum_wins": "base_wins",
                     "cum_top3": "base_top3"})
        left_sorted  = left.sort_values("_cut")
        right_sorted = right.sort_values("_cut_date")
        merged = pd.merge_asof(
            left_sorted, right_sorted,
            left_on="_cut", right_on="_cut_date",
            by=ent_col, direction="backward", allow_exact_matches=True,
        )
        merged = merged.set_index("row_idx")
        base_n    = merged["base_n"].fillna(0).astype(np.int32)
        base_wins = merged["base_wins"].fillna(0).astype(np.int32)
        base_top3 = merged["base_top3"].fillna(0).astype(np.int32)

        cur = sub.set_index("row_idx")[["prior_n", "prior_wins", "prior_top3"]]
        # Align base_* onto cur's index explicitly (avoid misalignment)
        base_n    = base_n.reindex(cur.index).fillna(0).astype(np.int32)
        base_wins = base_wins.reindex(cur.index).fillna(0).astype(np.int32)
        base_top3 = base_top3.reindex(cur.index).fillna(0).astype(np.int32)

        n     = (cur["prior_n"]    - base_n).clip(lower=0).astype(np.int32)
        wins  = (cur["prior_wins"] - base_wins).clip(lower=0).astype(np.int32)
        top3  = (cur["prior_top3"] - base_top3).clip(lower=0).astype(np.int32)

        wr    = np.where(n > 0, wins / n, np.nan)
        t3r   = np.where(n > 0, top3 / n, np.nan)

        w_df = pd.DataFrame({
            f"ent_{prefix}_n_races_{W}":  n.values,
            f"ent_{prefix}_n_wins_{W}":   wins.values,
            f"ent_{prefix}_n_top3_{W}":   top3.values,
            f"ent_{prefix}_winrate_{W}":  wr,
            f"ent_{prefix}_top3rate_{W}": t3r,
        }, index=cur.index).reset_index()
        out_frames.append(w_df)

    # Join all on row_idx
    base = out_frames[0].reset_index(drop=True)
    for f in out_frames[1:]:
        base = base.merge(f, on="row_idx", how="left")
    return base


def run(limit: int | None = None) -> None:
    t0 = time.time()
    cols = ["race_id", "num_pmu", "file_date",
            "nom", "driver", "entraineur", "proprietaire",
            "won", "placed", "finish_position"]
    df = pq.read_table(IN_PATH, columns=cols).to_pandas()
    if limit:
        df = df.head(limit).copy()
    df["row_idx"] = np.arange(len(df), dtype=np.int64)
    df["date"] = pd.to_datetime(df["file_date"], format="%Y-%m-%d")

    # Targets: we need a row to COUNT a prior race only if finish_position is known
    # (otherwise DNF — still count as a race but not as win).
    df["is_win"]  = (df["won"] == True).astype(np.int32)
    df["is_top3"] = (df["placed"] == True).astype(np.int32)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    features = [df[["race_id", "num_pmu", "row_idx"]]]
    for ent_col in ENTITIES:
        prefix = "horse" if ent_col == "nom" else ent_col
        t1 = time.time()
        feat = _compute_entity_rolling(df, ent_col, prefix)
        features.append(feat)
        print(f"  {prefix:12s}: {len(feat):,} rows, {time.time()-t1:.1f}s")

    out = features[0]
    for f in features[1:]:
        out = out.merge(f, on="row_idx", how="left")
    out = out.drop(columns="row_idx")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("Sample:")
    print(out.head(3).T.head(30).to_string())

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
