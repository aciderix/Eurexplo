#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature module v3 — extraction des blobs JSON non exploités.

Lit `pmu_course_raw_v2.parquet` et produit des features par (race_id, num_pmu):
  - `rapports_definitifs_json` → cote finale SIMPLE_GAGNANT/SIMPLE_PLACE,
    montant total engagé, overround final, movement vs drd_rapport.
  - `performances_detaillees_json` → temps, réductions, distance au précédent
    sur les courses passées de chaque cheval (agrégation rolling pré-course).
  - `pronostics_json` / `pronostics_detailles_json` → consensus pronostiqueurs
    (nb de tips, rank moyen dans les tops).
  - `ordre_arrivee_json` → incidents (DNF/TOMBE/DISTANCE), distance au leader.
  - `masse_enjeu_json` → répartition mises par cheval (en % du total) = vraie
    demande du public vs cote.

Reads : pmu_dataset_v2.parquet + pmu_course_raw_v2.parquet
Writes: pmu_feat_blobs_v3.parquet

Usage:
  python pmu_feat_blobs.py
  python pmu_feat_blobs.py --limit 5000   # debug
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE_PATH = Path("pmu_dataset_v2.parquet")
RAW_PATH  = Path("pmu_course_raw_v2.parquet")
OUT_PATH  = Path("pmu_feat_blobs_v3.parquet")


# ── Parsing helpers ──────────────────────────────────────────────────────────

def _safe_json(s: str | None) -> dict | list | None:
    if s is None or s in ("", "null"):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def parse_rapports_definitifs(s: str | None) -> dict[int, dict]:
    """Retourne {num_pmu: {rap_sg_final, rap_sp_final, rap_sg_fav, ...}}."""
    j = _safe_json(s)
    out: dict[int, dict] = defaultdict(dict)
    if not j:
        return out

    items = j if isinstance(j, list) else j.get("rapports", [])
    for r in items:
        tp = r.get("typePari") or r.get("typeBet")
        combi = r.get("combinaison") or []
        rapport = r.get("rapport") or r.get("dividende")
        if rapport is None or not combi:
            continue
        try:
            num = int(combi[0])
        except Exception:
            continue
        if tp in ("SIMPLE_GAGNANT", "E_SIMPLE_GAGNANT"):
            out[num]["rap_sg_final"] = float(rapport) / 100.0  # PMU stocke en centimes
        elif tp in ("SIMPLE_PLACE", "E_SIMPLE_PLACE"):
            # Plusieurs rapports possibles (places des finishers)
            out[num].setdefault("rap_sp_final", []).append(float(rapport) / 100.0)
    # Réduction : prendre la moyenne des rapports place
    for num, d in out.items():
        if "rap_sp_final" in d and isinstance(d["rap_sp_final"], list):
            d["rap_sp_final"] = float(np.mean(d["rap_sp_final"]))
    return out


def parse_masse_enjeu(s: str | None) -> dict[int, float]:
    """Retourne {num_pmu: part_mise_sg} (mise sur ce cheval / total mises SG)."""
    j = _safe_json(s)
    if not j:
        return {}
    items = j if isinstance(j, list) else j.get("masseEnjeu", [])
    shares: dict[int, float] = {}
    total = 0.0
    for it in items:
        tp = it.get("typePari") or it.get("codePari")
        if tp not in ("SIMPLE_GAGNANT", "E_SIMPLE_GAGNANT"):
            continue
        mtts = it.get("mises") or {}
        for num_s, mise in mtts.items():
            try:
                num = int(num_s)
                m = float(mise)
                shares[num] = shares.get(num, 0.0) + m
                total += m
            except Exception:
                continue
    if total > 0:
        shares = {k: v / total for k, v in shares.items()}
    return shares


def parse_pronostics(s: str | None) -> dict[int, dict]:
    """Retourne {num_pmu: {n_tips, rank_mean, rank_min}}.
    pronostics_json format: [{nomPronostiqueur, combinaisonDetaillee: [{num, position}, ...]}].
    """
    j = _safe_json(s)
    if not j:
        return {}
    items = j if isinstance(j, list) else j.get("pronostics", [])
    agg: dict[int, list[int]] = defaultdict(list)
    for p in items:
        detail = p.get("combinaisonDetaillee") or p.get("combinaison") or []
        for rank, entry in enumerate(detail, start=1):
            if isinstance(entry, dict):
                num = entry.get("num") or entry.get("numParticipant")
                pos = entry.get("position") or rank
            elif isinstance(entry, (int, str)):
                num, pos = entry, rank
            else:
                continue
            try:
                n = int(num)
                agg[n].append(int(pos))
            except Exception:
                continue
    out: dict[int, dict] = {}
    for n, ranks in agg.items():
        out[n] = {
            "prono_n_tips":   len(ranks),
            "prono_rank_min": min(ranks),
            "prono_rank_mean": float(np.mean(ranks)),
            "prono_top1_cnt": sum(1 for r in ranks if r == 1),
            "prono_top3_cnt": sum(1 for r in ranks if r <= 3),
        }
    return out


def parse_ordre_arrivee_incidents(s: str | None) -> dict[int, dict]:
    """Retourne {num_pmu: {incident_flag, distance_au_premier}}.
    ordreArrivee: [[14],[6],...] ou avec infos commentaires selon version.
    """
    j = _safe_json(s)
    if not j:
        return {}
    out: dict[int, dict] = {}
    for pos, entry in enumerate(j, start=1):
        if isinstance(entry, list) and entry:
            try:
                num = int(entry[0])
            except Exception:
                continue
            out[num] = {"arr_pos": pos}
    return out


# ── Performances détaillées (historique par cheval) ──────────────────────────

def parse_performances(s: str | None) -> list[dict]:
    """Retourne une liste de courses passées extraites du JSON.
    Format typique : {"participants":[{"nom":..., "performances":[{"datePerformance", "place", "tempsObtenu", "reductionKilometrique", "allocation", "distance", "nombrePartants", ...}]}]}
    """
    j = _safe_json(s)
    if not j:
        return []
    out = []
    items = j.get("participants", j) if isinstance(j, dict) else j
    if not isinstance(items, list):
        return []
    for p in items:
        nom = p.get("nom") or p.get("nomCheval")
        perfs = p.get("performances") or p.get("perfs") or []
        for perf in perfs:
            out.append({
                "nom": nom,
                "date": perf.get("datePerformance") or perf.get("date"),
                "place": perf.get("place"),
                "temps": perf.get("tempsObtenu"),
                "reduction": perf.get("reductionKilometrique"),
                "allocation": perf.get("allocation"),
                "distance": perf.get("distance"),
                "nb_partants": perf.get("nombrePartants"),
                "discipline": perf.get("discipline"),
            })
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run(limit: int | None = None) -> None:
    t0 = time.time()
    print(f"Loading {BASE_PATH}...")
    base = pq.read_table(BASE_PATH,
                         columns=["race_id", "num_pmu", "file_date",
                                  "drd_rapport", "drr_rapport",
                                  "nom", "nb_partants_reels"]).to_pandas()
    print(f"  base: {len(base):,} rows")

    print(f"Loading {RAW_PATH}...")
    raw = pq.read_table(RAW_PATH).to_pandas()
    print(f"  raw : {len(raw):,} rows")

    if limit:
        race_ids = set(raw["race_id"].iloc[:limit])
        raw = raw[raw["race_id"].isin(race_ids)].copy()
        base = base[base["race_id"].isin(race_ids)].copy()

    # ── Extract per-race blob features ──
    t1 = time.time()
    feats: list[dict] = []
    n_races = len(raw)
    for i, row in enumerate(raw.itertuples(index=False)):
        if i % 20000 == 0 and i > 0:
            print(f"  {i:>7,}/{n_races:,}  ({(time.time()-t1)/i*n_races-(time.time()-t1):.0f}s restant)")

        rap  = parse_rapports_definitifs(row.rapports_definitifs_json)
        mass = parse_masse_enjeu(row.masse_enjeu_json)
        pron = parse_pronostics(row.pronostics_json)
        if not pron:
            pron = parse_pronostics(row.pronostics_detailles_json)

        # La liste des num_pmu participants de cette course (base filtré)
        subset = base[base["race_id"] == row.race_id]
        for _, p in subset.iterrows():
            num = int(p["num_pmu"])
            d = {"race_id": row.race_id, "num_pmu": num}

            # Cote finale / movement
            r = rap.get(num, {})
            sg_final = r.get("rap_sg_final")
            sp_final = r.get("rap_sp_final")
            drd = p.get("drd_rapport")
            drr = p.get("drr_rapport")
            d["blob_rap_sg_final"] = sg_final if sg_final else np.nan
            d["blob_rap_sp_final"] = sp_final if sp_final else np.nan
            d["blob_movement_drd_to_final"] = (
                (sg_final - drd) / drd if sg_final and drd and drd > 0 else np.nan
            )
            d["blob_movement_drr_to_final"] = (
                (sg_final - drr) / drr if sg_final and drr and drr > 0 else np.nan
            )
            # Masse enjeu (demande publique)
            d["blob_masse_share"]  = mass.get(num, np.nan)
            # Pronostics experts
            pr = pron.get(num, {})
            d["blob_prono_n_tips"]   = pr.get("prono_n_tips", 0)
            d["blob_prono_rank_min"] = pr.get("prono_rank_min", np.nan)
            d["blob_prono_rank_mean"] = pr.get("prono_rank_mean", np.nan)
            d["blob_prono_top1_cnt"] = pr.get("prono_top1_cnt", 0)
            d["blob_prono_top3_cnt"] = pr.get("prono_top3_cnt", 0)

            feats.append(d)

    print(f"Per-race blob extraction done in {time.time()-t1:.0f}s")

    out = pd.DataFrame(feats)

    # ── Dérivés de rangs par course ──
    grp = out.groupby("race_id", sort=False)
    out["blob_rap_sg_final_rank"] = grp["blob_rap_sg_final"].rank(method="min", ascending=True)
    out["blob_masse_share_rank"]  = grp["blob_masse_share"].rank(method="min", ascending=False)
    out["blob_prono_top1_rank"]   = grp["blob_prono_top1_cnt"].rank(method="min", ascending=False)

    # Probas normalisées depuis cote finale (overround-corrigé)
    out["blob_prob_final"] = 1.0 / out["blob_rap_sg_final"].where(out["blob_rap_sg_final"] > 0)
    sum_prob = grp["blob_prob_final"].transform("sum")
    out["blob_prob_final_norm"] = out["blob_prob_final"] / sum_prob.where(sum_prob > 0)

    print(f"\nShape: {out.shape}")
    print("Sample:")
    print(out.head(3).to_string())
    print("\nNull rates:")
    print(out.isna().mean().sort_values(ascending=False).head(15))

    out.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)  total {time.time()-t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Limite sur le nb de courses (debug)")
    args = ap.parse_args()
    run(args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
