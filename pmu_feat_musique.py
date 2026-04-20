#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature module: musique parsing.

The `musique` string encodes past races as space-separated tokens like
"2m 1m 3a 0a Da (23) 4m 1m ..." where:
  - digit = finish position (0 = out of top 10)
  - D = disqualifié, T = tombé, A = arrêté, R = retiré
  - suffix letter = discipline (m=monte, a=attelé, h=haie, s=steeple, p=plat, c=cross)
  - "(23)" = year separator (2023 etc.)

Reads : pmu_dataset_v2.parquet
Writes: pmu_feat_musique.parquet (race_id, num_pmu, plus mus_* features)

Usage:
  python pmu_feat_musique.py
  python pmu_feat_musique.py --limit 100000
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


IN_PATH  = Path("pmu_dataset_v2.parquet")
OUT_PATH = Path("pmu_feat_musique.parquet")

# Token like "1m", "0a", "Dh", "(23)"
_TOKEN_RE = re.compile(r"\(\d+\)|[0-9DdTtAaRr][a-zA-Z]?")


def parse_musique(mus: str | None) -> dict:
    """Return a dict of scalar features for the musique string."""
    out = {
        "mus_n_races": 0,
        "mus_n_wins": 0,
        "mus_n_top2": 0,
        "mus_n_top3": 0,
        "mus_n_out": 0,          # position 0 = out of top N
        "mus_n_dnf": 0,
        "mus_mean_pos": np.nan,
        "mus_last_pos": np.nan,
        "mus_last_is_win": 0,
        "mus_recent3_wins": 0,
        "mus_recent5_wins": 0,
        "mus_recent3_top3": 0,
        "mus_recent5_top3": 0,
        "mus_n_years": 0,
        "mus_disc_attele": 0,
        "mus_disc_monte": 0,
        "mus_disc_plat": 0,
        "mus_disc_haie": 0,
        "mus_disc_steeple": 0,
        "mus_disc_cross": 0,
    }
    if mus is None or (isinstance(mus, float) and np.isnan(mus)) or not isinstance(mus, str) or not mus.strip():
        return out

    tokens = _TOKEN_RE.findall(mus)
    if not tokens:
        return out

    positions: list[int | None] = []   # None for DNF
    for tk in tokens:
        if tk.startswith("("):          # year marker
            out["mus_n_years"] += 1
            continue
        head = tk[0]
        disc = tk[1:].lower() if len(tk) > 1 else ""
        if head.isdigit():
            pos = int(head)
            positions.append(pos)
            out["mus_n_races"] += 1
            if pos == 1:
                out["mus_n_wins"] += 1
            if pos in (1, 2):
                out["mus_n_top2"] += 1
            if pos in (1, 2, 3):
                out["mus_n_top3"] += 1
            if pos == 0:
                out["mus_n_out"] += 1
        else:
            positions.append(None)
            out["mus_n_races"] += 1
            out["mus_n_dnf"] += 1
        # discipline counters
        if disc == "a":   out["mus_disc_attele"] += 1
        elif disc == "m": out["mus_disc_monte"] += 1
        elif disc == "p": out["mus_disc_plat"] += 1
        elif disc == "h": out["mus_disc_haie"] += 1
        elif disc == "s": out["mus_disc_steeple"] += 1
        elif disc == "c": out["mus_disc_cross"] += 1

    numeric = [p for p in positions if p is not None and p > 0]
    if numeric:
        out["mus_mean_pos"] = float(np.mean(numeric))
    if positions:
        last = positions[0]            # musique is recent-first
        out["mus_last_pos"] = float(last) if last is not None else np.nan
        out["mus_last_is_win"] = 1 if last == 1 else 0

    for k, n in ((3, "mus_recent3_wins"), (5, "mus_recent5_wins")):
        out[n] = sum(1 for p in positions[:k] if p == 1)
    for k, n in ((3, "mus_recent3_top3"), (5, "mus_recent5_top3")):
        out[n] = sum(1 for p in positions[:k] if p is not None and 1 <= p <= 3)

    return out


def run(limit: int | None = None) -> None:
    t0 = time.time()
    cols = ["race_id", "num_pmu", "musique"]
    print(f"Reading {IN_PATH} ({', '.join(cols)})...")
    df = pq.read_table(IN_PATH, columns=cols).to_pandas()
    if limit:
        df = df.head(limit).copy()
    print(f"  {len(df):,} rows in {time.time()-t0:.1f}s")

    t1 = time.time()
    parsed = df["musique"].map(parse_musique)
    feats = pd.DataFrame(parsed.tolist(), index=df.index)
    out = pd.concat([df[["race_id", "num_pmu"]].reset_index(drop=True),
                     feats.reset_index(drop=True)], axis=1)
    print(f"Parsed in {time.time()-t1:.1f}s")

    print("Sample:")
    print(out.head(3).to_string())
    print("\nNull rate mus_mean_pos:",
          out["mus_mean_pos"].isna().mean() * 100, "%")
    print("Mean n_races:", out["mus_n_races"].mean())
    print("Mean n_wins:", out["mus_n_wins"].mean())

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
