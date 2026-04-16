#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature module v3 — Head-to-Head (H2H).

Pour chaque (course, cheval), on construit des features qui résument
l'historique direct du cheval face à ses adversaires présents dans la
course courante.

Idée : quand chevaux A et B se sont déjà affrontés, qui a gagné ?
Feature : part de victoires de A vs les N autres chevaux de la course.

On maintient un dict sparse {(horseA, horseB): [wins_A, wins_B, total]},
mis à jour strictement avant la course courante (pas de fuite).

Reads : pmu_dataset_v2.parquet
Writes: pmu_feat_h2h_v3.parquet

Output colonnes (par participant) :
  h2h_n_meets_sum        -- nb total d'affrontements passés vs les adversaires du jour
  h2h_wins_sum           -- nb victoires directes
  h2h_winrate_mean       -- moyenne winrate pairwise vs adversaires (Laplace smoothed)
  h2h_winrate_max        -- meilleur winrate vs un adversaire (si >= 2 rencontres)
  h2h_winrate_min
  h2h_n_opponents_known  -- nb d'adversaires déjà rencontrés

Temps d'exec : 30-60 min sur ~200k courses. La clé est de pas matérialiser
la matrice dense ; on garde un dict clé=tuple chevaux triés.

Usage:
  python pmu_feat_h2h.py
  python pmu_feat_h2h.py --limit 10000   # debug
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
OUT_PATH = Path("pmu_feat_h2h_v3.parquet")

LAPLACE_A = 1.0   # lissage du winrate (Laplace)
LAPLACE_B = 2.0


def _key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def run(limit: int | None = None) -> None:
    t0 = time.time()
    cols = ["race_id", "num_pmu", "file_date", "nom", "finish_position"]
    df = pq.read_table(IN_PATH, columns=cols).to_pandas()
    if limit:
        df = df.head(limit).copy()
    df["date"] = pd.to_datetime(df["file_date"], format="%Y-%m-%d")
    df = df.sort_values(["date", "race_id", "num_pmu"], kind="stable").reset_index(drop=True)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # h2h[(horseA, horseB)] = (wins_A, wins_B, total)  avec A < B (ordre lexicographique)
    h2h: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])

    n = len(df)
    out = {
        "h2h_n_meets_sum":       np.zeros(n, dtype=np.int32),
        "h2h_wins_sum":          np.zeros(n, dtype=np.int32),
        "h2h_winrate_mean":      np.full(n, np.nan),
        "h2h_winrate_max":       np.full(n, np.nan),
        "h2h_winrate_min":       np.full(n, np.nan),
        "h2h_n_opponents_known": np.zeros(n, dtype=np.int32),
    }

    t1 = time.time()
    nr = 0
    for race_id, grp in df.groupby("race_id", sort=False):
        idx    = grp.index.to_numpy()
        names  = grp["nom"].tolist()
        positions = [int(p) if pd.notna(p) else None for p in grp["finish_position"].tolist()]

        # Pour chaque cheval : agréger son passé vs les autres du groupe
        for i, name in enumerate(names):
            if not name:
                continue
            meets = 0
            wins  = 0
            n_opp = 0
            winrates = []
            for j, opp in enumerate(names):
                if i == j or not opp or opp == name:
                    continue
                stat = h2h.get(_key(name, opp))
                if stat is None or stat[2] == 0:
                    continue
                wa, wb, tot = stat
                if name < opp:
                    w_self = wa
                else:
                    w_self = wb
                meets += tot
                wins  += w_self
                winrates.append((w_self + LAPLACE_A) / (tot + LAPLACE_A + LAPLACE_B))
                n_opp += 1
            out["h2h_n_meets_sum"][idx[i]]       = meets
            out["h2h_wins_sum"][idx[i]]          = wins
            out["h2h_n_opponents_known"][idx[i]] = n_opp
            if winrates:
                out["h2h_winrate_mean"][idx[i]] = float(np.mean(winrates))
                out["h2h_winrate_max"][idx[i]]  = float(np.max(winrates))
                out["h2h_winrate_min"][idx[i]]  = float(np.min(winrates))

        # Mise à jour post-course (si résultat connu) : pour chaque paire (i, j)
        if any(p is not None for p in positions):
            last_rank = max((p for p in positions if p is not None), default=0) + 1
            ranks = [p if p is not None else last_rank for p in positions]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    if not a or not b or a == b:
                        continue
                    ra, rb = ranks[i], ranks[j]
                    if ra == rb:
                        # ex-æquo strict : on ne touche PAS le dict (pas d'entrée fantôme)
                        continue
                    stat = h2h[_key(a, b)]
                    stat[2] += 1
                    # winner in key order (lexico)
                    if (a < b and ra < rb) or (a >= b and rb < ra):
                        stat[0] += 1
                    else:
                        stat[1] += 1

        nr += 1
        if nr % 20000 == 0:
            rate = nr / (time.time() - t1)
            print(f"  {nr:>7,} races | pairs stored = {len(h2h):,} | {rate:.0f} races/s")

    print(f"\nDone {nr:,} races in {time.time()-t1:.1f}s  (pairs = {len(h2h):,})")

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
