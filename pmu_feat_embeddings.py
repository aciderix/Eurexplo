#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature module v3 — entity2vec (Word2Vec sur séquences de courses).

Pour chaque course, on forme la "phrase" = liste des noms (cheval, driver,
entraîneur) qui y ont participé.  Word2Vec apprend ainsi que certains
chevaux "vont ensemble" (même niveau, mêmes hippodromes…).

Une fois les embeddings appris, on ajoute pour chaque participant :
  emb_horse_0..N     -- vecteur embedding cheval (dim D)
  emb_driver_0..N    -- vecteur driver
  emb_field_centroid_0..N  -- centroïde des embeddings des autres
  emb_horse_cos_field      -- cos(horse, centroïde) = à quel point
                              le cheval ressemble à la concurrence

D est faible (16 par défaut) pour limiter la dimensionnalité.

Reads : pmu_dataset_v2.parquet
Writes: pmu_feat_emb_v3.parquet

Dépendances : gensim>=4.3

Usage:
  python pmu_feat_embeddings.py
  python pmu_feat_embeddings.py --dim 16 --epochs 5
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
OUT_PATH = Path("pmu_feat_emb_v3.parquet")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def train_emb(sentences: list[list[str]], dim: int, epochs: int, min_count: int) -> dict:
    from gensim.models import Word2Vec
    print(f"  corpus: {len(sentences):,} races, dim={dim}, epochs={epochs}")
    t0 = time.time()
    model = Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=32,        # une course peut avoir 20 chevaux, on voit tout
        min_count=min_count,
        sg=1,             # skip-gram
        workers=4,
        epochs=epochs,
    )
    print(f"  trained in {time.time()-t0:.1f}s; vocab={len(model.wv):,}")
    return {w: model.wv[w] for w in model.wv.index_to_key}


def run(dim: int, epochs: int, min_count: int, limit: int | None) -> None:
    t0 = time.time()
    cols = ["race_id", "num_pmu", "file_date", "nom", "driver", "entraineur"]
    df = pq.read_table(IN_PATH, columns=cols).to_pandas()
    if limit:
        df = df.head(limit).copy()
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Phrases pour chaque type d'entité
    sent_horse  : list[list[str]] = []
    sent_driver : list[list[str]] = []
    sent_train  : list[list[str]] = []

    for _, grp in df.groupby("race_id", sort=False):
        sent_horse.append([s for s in grp["nom"].tolist() if s])
        sent_driver.append([s for s in grp["driver"].tolist() if s])
        sent_train.append([s for s in grp["entraineur"].tolist() if s])

    print("Training embeddings: horse")
    emb_h = train_emb(sent_horse, dim, epochs, min_count)
    print("Training embeddings: driver")
    emb_d = train_emb(sent_driver, dim, epochs, min_count)
    print("Training embeddings: entraineur")
    emb_t = train_emb(sent_train, dim, epochs, min_count)

    # ── Assign per-row vectors ──
    n = len(df)
    Z = np.zeros(dim, dtype=np.float32)
    def vec(d, k): return d.get(k, Z)

    emb_h_arr = np.stack([vec(emb_h, s) for s in df["nom"].tolist()])
    emb_d_arr = np.stack([vec(emb_d, s) for s in df["driver"].tolist()])
    emb_t_arr = np.stack([vec(emb_t, s) for s in df["entraineur"].tolist()])

    # Centroïde du "field" par course (moyenne horse embeddings)
    print("Computing field centroids + cos similarities...")
    df_idx = df.reset_index().rename(columns={"index": "_idx"})
    centroid_h = np.zeros_like(emb_h_arr)
    cos_h_field = np.zeros(n, dtype=np.float32)
    for _, grp in df_idx.groupby("race_id", sort=False):
        idx = grp["_idx"].to_numpy()
        if len(idx) < 2:
            continue
        sub = emb_h_arr[idx]
        mean = sub.mean(axis=0)
        # centroïde sans le cheval lui-même (leave-one-out)
        sum_ = sub.sum(axis=0)
        for k, i in enumerate(idx):
            lo = (sum_ - sub[k]) / (len(idx) - 1)
            centroid_h[i] = lo
            cos_h_field[i] = _cos(sub[k], lo)

    # ── Build output df ──
    out = df[["race_id", "num_pmu"]].copy()
    for i in range(dim):
        out[f"emb_horse_{i}"]       = emb_h_arr[:, i]
        out[f"emb_driver_{i}"]      = emb_d_arr[:, i]
        out[f"emb_entraineur_{i}"]  = emb_t_arr[:, i]
        out[f"emb_field_horse_{i}"] = centroid_h[:, i]
    out["emb_horse_cos_field"]     = cos_h_field

    out.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB) -- cols {out.shape[1]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim",       type=int, default=16)
    ap.add_argument("--epochs",    type=int, default=5)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--limit",     type=int, default=None)
    args = ap.parse_args()
    run(args.dim, args.epochs, args.min_count, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
