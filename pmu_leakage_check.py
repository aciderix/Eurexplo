#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leakage audit: vérifie si `nombre_courses`, `nombre_victoires`, `gains_carriere`
et `drd_rapport` sont des snapshots au moment de la course (OK, pas de fuite)
ou au moment du scrape (fuite catastrophique -- la valeur reflète toute la
carrière future du cheval).

Méthode
-------
Pour chaque cheval (nom) apparaissant dans ≥ 3 courses, on trie ses courses
par date et on regarde si les compteurs cumulatifs sont :
  - strictement croissants ou stables race-par-race  → snapshot race-time (OK)
  - constants sur toutes les courses                 → snapshot scrape-time (FUITE)
  - décroissants                                     → absurde, data bug

On compare aussi `nombre_victoires[course_t]` avec la somme des `won` observés
jusqu'à la course t. Écart = indicateur de fuite.

Pour `drd_rapport`, on vérifie la couverture + la distribution par année.

Reads : pmu_dataset_v2.parquet
Prints: rapport synthétique.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


IN_PATH = Path("pmu_dataset_v2.parquet")


def check_career_counters(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("TEST 1  --  `nombre_courses`, `nombre_victoires`, `gains_carriere`")
    print("=" * 80)

    # Focus sur chevaux avec ≥3 courses pour avoir un signal
    counts = df.groupby("nom").size()
    eligible = counts[counts >= 3].index
    print(f"Chevaux avec ≥3 courses : {len(eligible):,}  "
          f"(sur {len(counts):,} total)")

    # Sample 5000 chevaux pour rester rapide
    rng = np.random.default_rng(42)
    sample = rng.choice(eligible.values, size=min(5000, len(eligible)), replace=False)
    sub = df[df["nom"].isin(sample)].copy()
    sub = sub.sort_values(["nom", "file_date", "race_id"])

    grp = sub.groupby("nom", sort=False)

    # --- Pour chaque cheval, compter les transitions ---
    def analyze_group(g: pd.DataFrame) -> dict:
        nc = g["nombre_courses"].values
        nv = g["nombre_victoires"].values
        gc = g["gains_carriere"].values
        if len(nc) < 2:
            return None
        # % de transitions strictement croissantes / stables / décroissantes
        diffs_nc = np.diff(nc)
        diffs_nv = np.diff(nv)
        diffs_gc = np.diff(gc)
        return {
            "n_races":    len(nc),
            "nc_inc":     int((diffs_nc > 0).sum()),
            "nc_stable":  int((diffs_nc == 0).sum()),
            "nc_dec":     int((diffs_nc < 0).sum()),
            "nv_inc":     int((diffs_nv > 0).sum()),
            "nv_stable":  int((diffs_nv == 0).sum()),
            "gc_inc":     int((diffs_gc > 0).sum()),
            "gc_stable":  int((diffs_gc == 0).sum()),
            "gc_dec":     int((diffs_gc < 0).sum()),
            "nc_span":    int(nc[-1] - nc[0]),
            "nc_all_same": bool(np.all(nc == nc[0])),
            "nv_all_same": bool(np.all(nv == nv[0])),
            "gc_all_same": bool(np.all(gc == gc[0])),
        }

    results = []
    for _, g in grp:
        r = analyze_group(g)
        if r is not None:
            results.append(r)
    res = pd.DataFrame(results)

    total_trans = res["nc_inc"] + res["nc_stable"] + res["nc_dec"]
    print(f"\nTransitions nombre_courses (total {total_trans.sum():,}) :")
    print(f"  croissante  : {res['nc_inc'].sum():>9,}   "
          f"({res['nc_inc'].sum()/total_trans.sum()*100:5.1f}%)")
    print(f"  stable      : {res['nc_stable'].sum():>9,}   "
          f"({res['nc_stable'].sum()/total_trans.sum()*100:5.1f}%)")
    print(f"  décroissante: {res['nc_dec'].sum():>9,}   "
          f"({res['nc_dec'].sum()/total_trans.sum()*100:5.1f}%)")

    print(f"\nChevaux avec nombre_courses constant sur toutes leurs courses : "
          f"{res['nc_all_same'].sum():,} / {len(res):,}  "
          f"({res['nc_all_same'].mean()*100:.1f}%)")
    print(f"  -- >> si >50% = FUITE (scrape-time snapshot)")
    print(f"  -- >> si  <5% = OK (race-time snapshot)")

    print(f"\nChevaux avec nombre_victoires constant : "
          f"{res['nv_all_same'].sum():,} / {len(res):,}  "
          f"({res['nv_all_same'].mean()*100:.1f}%)")
    print(f"Chevaux avec gains_carriere constant  : "
          f"{res['gc_all_same'].sum():,} / {len(res):,}  "
          f"({res['gc_all_same'].mean()*100:.1f}%)")

    # --- Cross-check : somme(won) observée vs. nombre_victoires ---
    print("\n--- Cross-check : Σ won observé vs nombre_victoires ---")
    sub2 = sub[sub["won"].notna()].copy()
    sub2["won_int"] = sub2["won"].astype(int)
    sub2["cum_wins_obs"] = sub2.groupby("nom")["won_int"].cumsum()
    # Prior cum_wins = cum - current
    sub2["prior_wins_obs"] = sub2["cum_wins_obs"] - sub2["won_int"]
    # Si `nombre_victoires` = race-time snapshot, il devrait être ~= prior_wins_obs
    # Si = scrape-time snapshot, il sera TOUJOURS >= cum_wins_obs (final)
    sub2["diff_race_time"] = sub2["nombre_victoires"] - sub2["prior_wins_obs"]
    sub2["diff_scrape_time"] = sub2["nombre_victoires"] - sub2["cum_wins_obs"]

    print(f"  nombre_victoires - prior_wins_obs  (race-time snap = 0) :")
    print(f"      mean  = {sub2['diff_race_time'].mean():+.3f}")
    print(f"      median= {sub2['diff_race_time'].median():+.1f}")
    print(f"      |diff|<=0 : {(sub2['diff_race_time']<=0).mean()*100:.1f}%")
    print(f"      |diff|==0 : {(sub2['diff_race_time']==0).mean()*100:.1f}%")
    print(f"  nombre_victoires - cum_wins_obs   (scrape-time snap >=0, biais>0) :")
    print(f"      mean  = {sub2['diff_scrape_time'].mean():+.3f}")
    print(f"      median= {sub2['diff_scrape_time'].median():+.1f}")

    # Verdict
    print("\n>>> Interprétation :")
    if res["nc_all_same"].mean() > 0.5:
        print("    *** FUITE PROBABLE *** : >50% des chevaux ont nombre_courses constant")
        print("    → valeur = snapshot au moment du scrape (inclut courses futures)")
    elif res["nc_inc"].sum() > 0.5 * total_trans.sum():
        print("    OK : majorité de transitions croissantes = snapshot race-time")
    else:
        print("    AMBIGU : vérifier manuellement")


def check_odds_coverage(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("TEST 2  --  `drd_rapport`  (cote finale vs. matinée ?)")
    print("=" * 80)

    df_has = df[df["won"].notna()].copy()
    df_has["year"] = df_has["file_date"].str[:4].astype(int)

    cov = df_has.groupby("year").agg(
        n_rows=("drd_rapport", "size"),
        n_quoted=("drd_rapport", "count"),
        median=("drd_rapport", "median"),
        mean=("drd_rapport", "mean"),
    )
    cov["cov_pct"] = 100 * cov["n_quoted"] / cov["n_rows"]
    print("\nCouverture drd_rapport par année :")
    print(cov.to_string())

    # Écart drd vs drr (matinée vs moins d'une minute)
    pair = df_has[(df_has["drd_rapport"] > 0) & (df_has["drr_rapport"] > 0)].copy()
    pair["spread"] = (pair["drr_rapport"] - pair["drd_rapport"]) / pair["drr_rapport"]
    print(f"\nSpread (drr-drd)/drr :  n={len(pair):,}")
    print(f"  mean   = {pair['spread'].mean():+.3f}")
    print(f"  median = {pair['spread'].median():+.3f}")
    print(f"  std    = {pair['spread'].std():.3f}")
    print("  -- >> spread nul partout = drd==drr (rapports = même moment)")
    print("  -- >> spread non-nul     = drd (final) ≠ drr (référence) OK")

    # Win rate par rang de drd_rapport (si drd = cote finale, le favori gagne ~28%)
    pair["rank_drd"] = pair.groupby("race_id")["drd_rapport"].rank(method="min", ascending=True)
    wr_rank = pair.groupby("rank_drd")["won"].agg(["mean", "size"]).head(10)
    print("\nWin rate par rang de drd_rapport (rang 1 = favori) :")
    print(wr_rank.to_string())
    print("  -- attendu favori ~25-30%, rang 2 ~17%, rang 3 ~12%...")


def check_other_snapshots(df: pd.DataFrame) -> None:
    """Colonnes potentiellement 'snapshots scrape-time' : gains_annee_precedente,
    etc. gains_annee_precedente devrait n'évoluer qu'au tournant de l'année.
    """
    print("\n" + "=" * 80)
    print("TEST 3  --  gains_annee_precedente / gains_annee_en_cours")
    print("=" * 80)

    counts = df.groupby("nom").size()
    eligible = counts[counts >= 3].index
    rng = np.random.default_rng(42)
    sample = rng.choice(eligible.values, size=min(3000, len(eligible)), replace=False)
    sub = df[df["nom"].isin(sample)].copy()
    sub = sub.sort_values(["nom", "file_date", "race_id"])
    sub["year"] = sub["file_date"].str[:4].astype(int)

    # Pour chaque cheval+année, y a-t-il variation de gains_annee_precedente ?
    agg = sub.groupby(["nom", "year"])["gains_annee_precedente"].agg(["nunique", "size"])
    agg = agg[agg["size"] >= 2]
    print(f"Pour {len(agg):,} (cheval×année) avec ≥2 courses :")
    print(f"  gains_annee_precedente varie dans l'année : "
          f"{(agg['nunique']>1).sum():,} ({(agg['nunique']>1).mean()*100:.1f}%)")
    print("  -- >> devrait être ~0% (valeur figée une fois l'année passée)")

    # gains_annee_en_cours : devrait augmenter au cours de l'année
    def check_monotone(g):
        v = g["gains_annee_en_cours"].values
        if len(v) < 2: return None
        d = np.diff(v)
        return {"n": len(v), "inc": int((d > 0).sum()),
                "stable": int((d == 0).sum()),
                "dec": int((d < 0).sum())}
    rows = []
    for _, g in sub.groupby(["nom", "year"], sort=False):
        r = check_monotone(g)
        if r is not None:
            rows.append(r)
    if rows:
        res = pd.DataFrame(rows)
        total = (res["inc"] + res["stable"] + res["dec"]).sum()
        print(f"\nTransitions gains_annee_en_cours (intra-année, n={total:,}) :")
        print(f"  croissante : {res['inc'].sum()/total*100:5.1f}%")
        print(f"  stable     : {res['stable'].sum()/total*100:5.1f}%")
        print(f"  décroissante : {res['dec'].sum()/total*100:5.1f}%")


def main() -> int:
    t0 = time.time()
    cols = ["race_id", "file_date", "num_pmu", "nom",
            "nombre_courses", "nombre_victoires", "gains_carriere",
            "gains_annee_en_cours", "gains_annee_precedente",
            "drd_rapport", "drr_rapport", "won", "finish_position"]
    print(f"Loading {IN_PATH}...")
    df = pq.read_table(IN_PATH, columns=cols).to_pandas()
    print(f"  {len(df):,} rows in {time.time()-t0:.1f}s")

    check_career_counters(df)
    check_odds_coverage(df)
    check_other_snapshots(df)

    print("\n" + "=" * 80)
    print(f"Total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
