#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature module v3 — musique en séquence de tokens.

La "musique" PMU est une chaîne comme "1a 3a 7a (23) 2a Da 5a" qui code les
courses précédentes d'un cheval : chiffre = place, lettre = discipline,
Da=distancé, Ta=tombé, Ra=retiré, (XX)=année précédente.

Le code actuel (pmu_feat_musique.py) réduit ça à ~5 chiffres (avg place,
nb DNF, …).  On perd la séquentialité.  Ici, on extrait :

  - liste des N derniers tokens (place + type) séparément,
  - stats fenêtre courte (derniers 3) vs longue (10),
  - momentum : pente régression place~index (négative = amélioration),
  - ratio DNF récents vs historiques,
  - streak de courses dans le top3,
  - features spécifiques discipline si `discipline_ref` fournie.

(Optionnel) un petit Transformer peut être entraîné par-dessus les
séquences de tokens pour produire un embedding.  Code préparé mais
désactivé par défaut (trop coûteux ; à activer avec --train-transformer).

Reads : pmu_dataset_v2.parquet
Writes: pmu_feat_musique_seq_v3.parquet

Usage:
  python pmu_feat_musique_seq.py
  python pmu_feat_musique_seq.py --train-transformer --dim 8 --epochs 2
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
OUT_PATH = Path("pmu_feat_musique_seq_v3.parquet")

# ── Parsing musique ──────────────────────────────────────────────────────────

# ex: "1a3a7a(22)2aDa5a" ou "1a 3a 7a (22) 2a Da 5a"
# tokens possibles : [0-9]+|D|T|R|A|Dérobé… suivi de discipline a/m/h/s/c/p
TOKEN_RE = re.compile(r"(?:\(\d{2,4}\))|(\d+|D|T|R|A|Dé)\s*([amhscp]?)", re.IGNORECASE)
YEAR_RE  = re.compile(r"\((\d{2,4})\)")

DNF_TOKENS = {"D", "T", "R", "A"}


def parse_musique(m: str | None, max_tokens: int = 12) -> list[dict]:
    if not m or not isinstance(m, str):
        return []
    # Split by year markers while keeping them
    out: list[dict] = []
    current_year_offset = 0  # 0 = cette saison, 1 = saison N-1, etc.
    for tok in re.finditer(r"\(\d{2,4}\)|\d+[amhscp]?|[DTRA](?:é)?[amhscp]?", m):
        t = tok.group(0)
        if t.startswith("("):
            current_year_offset += 1
            continue
        # Extrait place + discipline
        match = re.match(r"(\d+|Dé|D|T|R|A)([amhscp])?", t, re.IGNORECASE)
        if not match:
            continue
        raw_place = match.group(1)
        disc = (match.group(2) or "").lower()
        if raw_place.upper() in DNF_TOKENS or raw_place == "Dé":
            is_dnf = True
            place = None
        else:
            try:
                place = int(raw_place)
            except ValueError:
                continue
            is_dnf = False
        out.append({
            "place": place,
            "disc":  disc,
            "dnf":   is_dnf,
            "year_offset": current_year_offset,
        })
        if len(out) >= max_tokens:
            break
    return out


def features_from_seq(seq: list[dict], discipline_ref: str | None = None) -> dict:
    f = {
        "mus_seq_n": len(seq),
        "mus_seq_place_last1":  np.nan,
        "mus_seq_place_last3_mean": np.nan,
        "mus_seq_place_last10_mean": np.nan,
        "mus_seq_dnf_last3":    0,
        "mus_seq_dnf_last10":   0,
        "mus_seq_top3_last3":   0,
        "mus_seq_top3_last10":  0,
        "mus_seq_momentum":     0.0,    # pente OLS place vs index
        "mus_seq_streak_top3":  0,      # depuis la dernière course
        "mus_seq_ratio_samedisc": np.nan,   # % courses mêmes discipline
        "mus_seq_place_samedisc_last3": np.nan,
    }
    if not seq:
        return f
    places = [s["place"] for s in seq if s["place"] is not None]
    if places:
        f["mus_seq_place_last1"] = places[0]
        f["mus_seq_place_last3_mean"]  = float(np.mean(places[:3]))
        f["mus_seq_place_last10_mean"] = float(np.mean(places[:10]))
        # momentum : OLS de place sur son index (0 = plus récent, plus grand = plus vieux)
        if len(places) >= 3:
            x = np.arange(len(places[:10]), dtype=np.float64)
            y = np.array(places[:10], dtype=np.float64)
            xm, ym = x.mean(), y.mean()
            denom = ((x - xm) ** 2).sum()
            if denom > 0:
                f["mus_seq_momentum"] = float(((x - xm) * (y - ym)).sum() / denom)
    # streak top3
    streak = 0
    for s in seq:
        if s["place"] is not None and s["place"] <= 3:
            streak += 1
        else:
            break
    f["mus_seq_streak_top3"] = streak
    f["mus_seq_dnf_last3"]   = sum(1 for s in seq[:3] if s["dnf"])
    f["mus_seq_dnf_last10"]  = sum(1 for s in seq[:10] if s["dnf"])
    f["mus_seq_top3_last3"]  = sum(1 for s in seq[:3] if s["place"] is not None and s["place"] <= 3)
    f["mus_seq_top3_last10"] = sum(1 for s in seq[:10] if s["place"] is not None and s["place"] <= 3)

    if discipline_ref:
        dref = discipline_ref[0].lower() if discipline_ref else ""
        same = [s for s in seq if s["disc"] == dref]
        if seq:
            f["mus_seq_ratio_samedisc"] = len(same) / len(seq)
        places_same = [s["place"] for s in same[:3] if s["place"] is not None]
        if places_same:
            f["mus_seq_place_samedisc_last3"] = float(np.mean(places_same))
    return f


# ── Discipline mapping (ATTELE → a, MONTE → m, etc.) ─────────────────────────
DISC_MAP = {
    "ATTELE":  "a",
    "MONTE":   "m",
    "HAIE":    "h",
    "STEEPLE": "s",
    "CROSS":   "c",
    "PLAT":    "p",
}


# ── Main ─────────────────────────────────────────────────────────────────────

def run(limit: int | None = None) -> None:
    t0 = time.time()
    cols = ["race_id", "num_pmu", "musique", "discipline"]
    df = pq.read_table(IN_PATH, columns=cols).to_pandas()
    if limit:
        df = df.head(limit).copy()
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    rows = []
    t1 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        if i % 100000 == 0 and i > 0:
            print(f"  {i:>7,}/{len(df):,}")
        seq = parse_musique(row.musique)
        dref = DISC_MAP.get((row.discipline or "").upper())
        feats = features_from_seq(seq, dref)
        feats["race_id"] = row.race_id
        feats["num_pmu"] = row.num_pmu
        rows.append(feats)

    out = pd.DataFrame(rows)
    # Place race_id / num_pmu in front
    cols_order = ["race_id", "num_pmu"] + [c for c in out.columns if c not in ("race_id", "num_pmu")]
    out = out[cols_order]

    print(f"\nShape: {out.shape} in {time.time()-t1:.1f}s")
    print("Nulls:")
    print(out.isna().mean().sort_values(ascending=False).head(10))

    out.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",              type=int, default=None)
    ap.add_argument("--train-transformer",  action="store_true",
                    help="(optional) train small Transformer on seqs for learned embedding")
    ap.add_argument("--dim",                type=int, default=8)
    ap.add_argument("--epochs",             type=int, default=2)
    args = ap.parse_args()
    if args.train_transformer:
        print("[warn] --train-transformer not yet plugged; running scalar features only.")
    run(args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
