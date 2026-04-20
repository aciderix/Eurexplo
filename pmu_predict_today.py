#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMU Live Predictor -- prédit les courses du jour via l'API PMU.

1. Fetch programme + participants depuis l'API PMU
2. Construit les features (inline + lookup ELO/rolling/te depuis les parquets)
3. Charge pmu_model_v2.lgb et produit les probabilités de victoire
4. Affiche les courses classées par score

Usage:
  python pmu_predict_today.py                    # aujourd'hui
  python pmu_predict_today.py --date 15042026    # date DDMMYYYY
  python pmu_predict_today.py --min-edge 0.02    # filtre edge minimum
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

# ── Config ───────────────────────────────────────────────────────────────────

BASE_URL     = "https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"
MODEL_PATH   = Path("pmu_model_v2.lgb")
FEATS_PATH   = Path("pmu_model_v2.features.json")
ELO_PATH     = Path("pmu_feat_elo.parquet")
ROLLING_PATH = Path("pmu_feat_rolling.parquet")
TE_PATH      = Path("pmu_feat_target_enc.parquet")
BASE_PARQUET = Path("pmu_dataset_v2.parquet")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

CATEG_COLS = [
    "sexe", "allure", "race", "oeilleres", "robe_code",
    "discipline", "specialite", "corde", "condition_sexe",
    "categorie_particuliere", "hippodrome_code", "pays_code",
    "nature_reunion",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _s(v): return str(v) if v is not None else None
def _i(v):
    try: return int(v)
    except: return None
def _f(v):
    try: return float(v)
    except: return None


# ── API fetch ─────────────────────────────────────────────────────────────────

def fetch(url: str, session: requests.Session) -> dict | list | None:
    try:
        r = session.get(url, timeout=15, headers=HEADERS)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        print(f"  HTTP {r.status_code} : {url}")
        return None
    except Exception as e:
        print(f"  Erreur fetch {url}: {e}")
        return None


def fetch_day(date_str: str) -> list[dict]:
    """Récupère toutes les courses d'une journée. date_str = DDMMYYYY."""
    session = requests.Session()
    prog = fetch(f"{BASE_URL}/{date_str}", session)
    if not prog:
        print(f"Aucun programme pour {date_str}")
        return []

    programme = prog.get("programme", prog)
    reunions = programme.get("reunions", [])
    print(f"  {len(reunions)} reunions trouvées")

    rows = []
    for reunion in reunions:
        r_num = reunion.get("numOfficiel", 0)
        hippo = reunion.get("hippodrome") or {}
        pays  = reunion.get("pays") or {}

        reunion_info = fetch(f"{BASE_URL}/{date_str}/R{r_num}", session) or {}
        ri = (reunion_info.get("programme", {})
                          .get("reunions", [{}])[0] if "programme" in reunion_info
              else reunion_info)

        for course in reunion.get("courses", []):
            c_num = course.get("numOrdre", course.get("numExterne", 0))
            c_url = f"{BASE_URL}/{date_str}/R{r_num}/C{c_num}"

            ic_raw = fetch(c_url, session) or {}
            ic = ic_raw if not ic_raw.get("course") else ic_raw.get("course", ic_raw)

            parts_raw = fetch(f"{c_url}/participants", session) or {}
            participants = (parts_raw.get("participants") or
                            parts_raw.get("participants", {}).get("participants") or [])
            if isinstance(parts_raw, dict) and "participants" in parts_raw:
                inner = parts_raw["participants"]
                if isinstance(inner, list):
                    participants = inner
                elif isinstance(inner, dict):
                    participants = inner.get("participants", [])

            if not participants:
                continue

            for p in participants:
                gp   = p.get("gainsParticipant") or {}
                robe = p.get("robe") or {}
                drd  = p.get("dernierRapportDirect") or {}
                drr  = p.get("dernierRapportReference") or {}

                rows.append({
                    # ids
                    "race_id":    f"{date_str}_R{r_num}_C{c_num}",
                    "file_date":  f"20{date_str[4:]}-{date_str[2:4]}-{date_str[:2]}"
                                  if len(date_str) == 8 and date_str[4:6] != "20"
                                  else f"{date_str[4:]}-{date_str[2:4]}-{date_str[:2]}",
                    "reunion_num": r_num,
                    "course_num":  c_num,
                    "num_pmu":    _i(p.get("numPmu")),
                    # course
                    "libelle":    _s(ic.get("libelle") or course.get("libelle")),
                    "discipline": _s(ic.get("discipline") or course.get("discipline")),
                    "specialite": _s(ic.get("specialite") or course.get("specialite")),
                    "categorie_particuliere": _s(ic.get("categorieParticularite")),
                    "condition_sexe": _s(ic.get("conditionSexe")),
                    "distance":   _i(ic.get("distance") or course.get("distance")),
                    "distance_unit": _s(ic.get("distanceUnit")),
                    "corde":      _s(ic.get("corde")),
                    "parcours":   _s(ic.get("parcours")),
                    "nb_declares_partants": _i(ic.get("nombreDeclaresPartants")),
                    "nb_partants_reels": None,  # sera calculé
                    "heure_depart_ts": _i(ic.get("heureDepart") or course.get("heureDepart")),
                    "grand_prix_national_trot": bool(ic.get("grandPrixNationalTrot")),
                    "pari_multi_courses": bool(ic.get("pariMultiCourses")),
                    "pari_special": bool(ic.get("pariSpecial")),
                    "montant_prix": _i(ic.get("montantPrix")),
                    "montant_total_offert": _i(ic.get("montantTotalOffert")),
                    "montant_offert_1er": _i(ic.get("montantOffert1er")),
                    "montant_offert_2eme": _i(ic.get("montantOffert2eme")),
                    "montant_offert_3eme": _i(ic.get("montantOffert3eme")),
                    "montant_offert_4eme": _i(ic.get("montantOffert4eme")),
                    "montant_offert_5eme": _i(ic.get("montantOffert5eme")),
                    "conditions_text": _s(ic.get("conditions")),
                    "course_trackee": bool(ic.get("courseTrackee")),
                    "statut_course": _s(ic.get("statut")),
                    "hippodrome_code": _s((hippo or {}).get("code") or (hippo or {}).get("codeHippodrome")
                                         or (ic.get("hippodrome") or {}).get("codeHippodrome")),
                    "hippodrome_libelle": _s((hippo or {}).get("libelleCourt")),
                    "pays_code":  _s((pays or {}).get("code")),
                    "nature_reunion": _s(reunion.get("nature")),
                    "date_reunion_ts": _i(reunion.get("dateReunion")),
                    # participant
                    "nom":        _s(p.get("nom")),
                    "age":        _i(p.get("age")),
                    "sexe":       _s(p.get("sexe")),
                    "race":       _s(p.get("race")),
                    "statut_participant": _s(p.get("statut")),
                    "oeilleres":  _s(p.get("oeilleres")),
                    "proprietaire": _s(p.get("proprietaire")),
                    "entraineur": _s(p.get("entraineur")),
                    "driver":     _s(p.get("driver")),
                    "driver_change": bool(p.get("driverChange")),
                    "indicateur_inedit": bool(p.get("indicateurInedit")),
                    "nombre_courses":  _i(p.get("nombreCourses")),
                    "nombre_victoires": _i(p.get("nombreVictoires")),
                    "nombre_places":   _i(p.get("nombrePlaces")),
                    "nombre_places_second": _i(p.get("nombrePlacesSecond")),
                    "nombre_places_troisieme": _i(p.get("nombrePlacesTroisieme")),
                    "nom_pere":   _s(p.get("nomPere")),
                    "nom_mere":   _s(p.get("nomMere")),
                    "nom_pere_mere": _s(p.get("nomPereMere")),
                    "jument_pleine": bool(p.get("jumentPleine")),
                    "engagement": bool(p.get("engagement")),
                    "supplement": _i(p.get("supplement")),
                    "poids_condition_monte_change": bool(p.get("poidsConditionMonteChange")),
                    "allure":     _s(p.get("allure")),
                    "musique":    _s(p.get("musique")),
                    "handicap_distance": _i(p.get("handicapDistance")),
                    "handicap_poids":    _i(p.get("handicapPoids")),
                    "handicap_valeur":   _i(p.get("handicapValeur")),
                    "place_corde": _i(p.get("placeCorde")),
                    "eleveur":    _s(p.get("eleveur")),
                    "avis_entraineur": _s(p.get("avisEntraineur")),
                    # gains
                    "gains_annee_en_cours": _i(gp.get("gainsAnneeEnCours")),
                    "gains_annee_precedente": _i(gp.get("gainsAnneePrecedente")),
                    "gains_carriere": _i(gp.get("gainsCarriere")),
                    "gains_place":    _i(gp.get("gainsPlace")),
                    "gains_victoires": _i(gp.get("gainsVictoires")),
                    # robe
                    "robe_code":  _s(robe.get("code")),
                    # odds
                    "drd_rapport": _f(drd.get("rapport")),
                    "drr_rapport": _f(drr.get("rapport")),
                    "drd_favoris": bool(drd.get("favoris")),
                    "drd_grosse_prise": bool(drd.get("grossePrise")),
                    "drd_nombre_indicateur_tendance": _i(drd.get("nombreIndicateurTendance")),
                    "drd_permutation": _i(drd.get("permutation")),
                    "drd_date_rapport_ts": _i(drd.get("dateRapport")),
                    "drr_favoris": bool(drr.get("favoris")),
                    "drr_grosse_prise": bool(drr.get("grossePrise")),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # nb_partants_reels = nombre de participants par course
    nb = df.groupby("race_id")["num_pmu"].transform("count")
    df["nb_partants_reels"] = nb
    print(f"  {len(df)} participants dans {df['race_id'].nunique()} courses")
    return df


# ── Lookup tables depuis les parquets ─────────────────────────────────────────

def build_elo_lookup() -> dict[str, dict[str, float]]:
    """Retourne les derniers ELO connus par entité."""
    print("Chargement ELO lookup...")
    cols = ["race_id", "num_pmu", "elo_horse_pre", "elo_driver_pre",
            "elo_entraineur_pre", "elo_horse_ngames", "elo_driver_ngames",
            "elo_entraineur_ngames"]
    elo = pq.read_table(ELO_PATH, columns=cols).to_pandas()
    base = pq.read_table(BASE_PARQUET, columns=["race_id", "num_pmu", "file_date",
                                                  "nom", "driver", "entraineur"]).to_pandas()
    merged = elo.merge(base, on=["race_id", "num_pmu"])
    merged = merged.sort_values("file_date")

    horse_elo    = merged.groupby("nom")[["elo_horse_pre", "elo_horse_ngames"]].last()
    driver_elo   = merged.groupby("driver")[["elo_driver_pre", "elo_driver_ngames"]].last()
    entr_elo     = merged.groupby("entraineur")[["elo_entraineur_pre", "elo_entraineur_ngames"]].last()

    return {
        "horse":      horse_elo.to_dict("index"),
        "driver":     driver_elo.to_dict("index"),
        "entraineur": entr_elo.to_dict("index"),
    }


def build_rolling_lookup() -> dict[str, pd.DataFrame]:
    """Retourne les dernières stats rolling par entité."""
    print("Chargement rolling lookup...")
    roll = pq.read_table(ROLLING_PATH).to_pandas()
    base = pq.read_table(BASE_PARQUET,
                         columns=["race_id", "num_pmu", "file_date",
                                  "nom", "driver", "entraineur", "proprietaire"]).to_pandas()
    merged = roll.merge(base, on=["race_id", "num_pmu"])
    merged = merged.sort_values("file_date")

    ent_map = {"horse": "nom", "driver": "driver",
               "entraineur": "entraineur", "proprietaire": "proprietaire"}
    lookups = {}
    roll_cols = [c for c in roll.columns if c.startswith("ent_")]
    for ent, col in ent_map.items():
        sub = merged[[col] + roll_cols].copy()
        sub = sub.groupby(col).last()
        lookups[ent] = sub
    return lookups


def build_te_lookup() -> dict[str, pd.Series]:
    """Retourne les target encodings par valeur de colonne."""
    print("Chargement target-encoding lookup...")
    te = pq.read_table(TE_PATH).to_pandas()
    base = pq.read_table(BASE_PARQUET,
                         columns=["race_id", "num_pmu", "file_date",
                                  "nom_pere", "nom_mere", "entraineur",
                                  "driver", "nom", "proprietaire"]).to_pandas()
    merged = te.merge(base, on=["race_id", "num_pmu"])
    merged = merged.sort_values("file_date")

    te_cols = [c for c in te.columns if c.startswith("te_") and c not in ("race_id", "num_pmu")]
    entity_for_te = {
        "nom_pere": [c for c in te_cols if "pere" in c],
        "nom_mere": [c for c in te_cols if "mere" in c and "pere_mere" not in c],
        "nom_pere_mere": [c for c in te_cols if "pere_mere" in c],
        "nom": [c for c in te_cols if "_horse_" in c or ("_nom_" in c and "pere" not in c)],
        "entraineur": [c for c in te_cols if "entr" in c],
        "driver": [c for c in te_cols if "driver" in c],
        "proprietaire": [c for c in te_cols if "propri" in c],
    }
    lookups = {}
    for ent, cols in entity_for_te.items():
        if not cols or ent not in merged.columns:
            continue
        sub = merged[[ent] + cols].groupby(ent).last()
        lookups[ent] = sub
    return lookups


# ── Musique parser (même logique que pmu_feat_musique.py) ─────────────────────

_TOKEN_RE = re.compile(r"\(\d+\)|[0-9DdTtAaRr][a-zA-Z]?")

def _parse_musique_full(mus: str | None) -> dict:
    BASE = {"mus_n_races": 0, "mus_n_wins": 0, "mus_n_top2": 0, "mus_n_top3": 0,
            "mus_n_out": 0, "mus_n_dnf": 0, "mus_mean_pos": np.nan,
            "mus_last_pos": np.nan, "mus_last_is_win": 0,
            "mus_recent3_wins": 0, "mus_recent5_wins": 0,
            "mus_recent3_top3": 0, "mus_recent5_top3": 0,
            "mus_n_years": 0, "mus_disc_attele": 0, "mus_disc_monte": 0,
            "mus_disc_plat": 0, "mus_disc_haie": 0, "mus_disc_steeple": 0,
            "mus_disc_cross": 0}
    if not mus or not isinstance(mus, str):
        return BASE

    tokens = _TOKEN_RE.findall(mus)
    positions, discs, years = [], [], 0
    disc_map = {"a": "attele", "m": "monte", "p": "plat", "h": "haie",
                "s": "steeple", "c": "cross"}

    for t in tokens:
        if t.startswith("("):
            years += 1
            continue
        r = t[0].upper()
        d = t[1].lower() if len(t) > 1 else ""
        if d in disc_map:
            BASE[f"mus_disc_{disc_map[d]}"] += 1
        if r.isdigit():
            pos = int(r)
            positions.append(pos if pos > 0 else None)
        elif r in ("D", "T", "A", "R"):
            positions.append(None)

    valid = [p for p in positions if p is not None]
    r5 = positions[:5]
    r3 = positions[:3]

    BASE["mus_n_races"] = len(positions)
    BASE["mus_n_wins"]  = sum(1 for p in valid if p == 1)
    BASE["mus_n_top2"]  = sum(1 for p in valid if p <= 2)
    BASE["mus_n_top3"]  = sum(1 for p in valid if p <= 3)
    BASE["mus_n_out"]   = sum(1 for p in valid if p == 0)
    BASE["mus_n_dnf"]   = sum(1 for p in positions if p is None)
    BASE["mus_mean_pos"] = np.mean(valid) if valid else np.nan
    BASE["mus_last_pos"] = valid[0] if valid else np.nan
    BASE["mus_last_is_win"] = 1 if valid and valid[0] == 1 else 0
    BASE["mus_recent3_wins"] = sum(1 for p in r3 if p == 1)
    BASE["mus_recent5_wins"] = sum(1 for p in r5 if p == 1)
    BASE["mus_recent3_top3"] = sum(1 for p in r3 if p is not None and p <= 3)
    BASE["mus_recent5_top3"] = sum(1 for p in r5 if p is not None and p <= 3)
    BASE["mus_n_years"] = years
    return BASE


# ── Feature engineering ───────────────────────────────────────────────────────

def build_live_features(df: pd.DataFrame, elo_lkp: dict,
                        roll_lkp: dict, te_lkp: dict,
                        feat_names: list[str]) -> pd.DataFrame:
    df = df.copy()

    # ── Date ──
    df["year"]  = df["file_date"].str[:4].astype(int)
    df["month"] = df["file_date"].str[5:7].astype(int)
    df["heure_h"] = ((df["heure_depart_ts"].fillna(0) / 3_600_000) % 24).round(1)

    # ── Win / place rates ──
    nc = df["nombre_courses"].fillna(1).clip(lower=1)
    df["win_rate"]    = df["nombre_victoires"].fillna(0) / nc
    df["place_rate"]  = df["nombre_places"].fillna(0) / nc
    df["place2_rate"] = df["nombre_places_second"].fillna(0) / nc
    df["place3_rate"] = df["nombre_places_troisieme"].fillna(0) / nc

    # ── Gains ──
    for col in ["gains_carriere", "gains_annee_en_cours", "gains_annee_precedente",
                "gains_victoires", "gains_place"]:
        df[f"log_{col}"] = np.log1p(df[col].fillna(0))
    df["gains_per_start"] = df["gains_carriere"].fillna(0) / nc

    # ── Odds ──
    df["log_drd"]    = np.log1p(df["drd_rapport"].clip(lower=0))
    df["log_drr"]    = np.log1p(df["drr_rapport"].clip(lower=0))
    df["prob_drd"]   = 1.0 / df["drd_rapport"].clip(lower=1)
    df["prob_drr"]   = 1.0 / df["drr_rapport"].clip(lower=1)
    df["odds_diff"]  = df["drd_rapport"].fillna(0) - df["drr_rapport"].fillna(0)
    df["drd_favoris"]      = df["drd_favoris"].fillna(False).astype(int)
    df["drd_grosse_prise"] = df["drd_grosse_prise"].fillna(False).astype(int)
    df["drd_nb_tendance"]  = df["drd_nombre_indicateur_tendance"].fillna(0)
    df["drd_permutation"]  = df["drd_permutation"].fillna(0).astype(int)

    df["rank_odds"]     = df.groupby("race_id")["drd_rapport"].rank(method="min", ascending=True)
    df["rank_odds_pct"] = df.groupby("race_id")["drd_rapport"].rank(pct=True, ascending=True)
    min_odds = df.groupby("race_id")["drd_rapport"].transform("min")
    df["ratio_to_fav"] = df["drd_rapport"] / min_odds.clip(lower=0.1)

    # ── Booleans ──
    for col in ["driver_change", "indicateur_inedit", "jument_pleine",
                "grand_prix_national_trot", "pari_special"]:
        df[col] = df[col].fillna(False).astype(int)
    df["grand_prix"]   = df["grand_prix_national_trot"]
    df["pari_special"] = df["pari_special"]

    # ── Catégorielles ──
    for col in CATEG_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes.astype("int16")

    # ── Course ──
    df["log_distance"] = np.log1p(df["distance"].fillna(0))
    df["log_partants"] = np.log1p(df["nb_partants_reels"].fillna(0))
    df["log_prize"]    = np.log1p(df["montant_prix"].fillna(0))
    df["log_prize_1er"] = np.log1p(df["montant_offert_1er"].fillna(0))
    df["handicap_distance"] = df["handicap_distance"].fillna(0)
    df["handicap_poids"]    = df["handicap_poids"].fillna(0)
    df["handicap_valeur"]   = df["handicap_valeur"].fillna(0)
    df["place_corde"]       = df["place_corde"].fillna(0).astype(int)

    # ── Musique features ──
    mus_feats = df["musique"].apply(_parse_musique_full)
    for k in list(mus_feats.iloc[0].keys()):
        df[k] = [x[k] for x in mus_feats]

    # ── ELO lookup ──
    def get_elo(entity, key, feat, default=1500.0):
        lkp = elo_lkp.get(entity, {})
        return df[key].map(lambda v: lkp.get(v, {}).get(feat, default) if v else default)

    df["elo_horse_pre"]         = get_elo("horse",      "nom",        "elo_horse_pre")
    df["elo_driver_pre"]        = get_elo("driver",     "driver",     "elo_driver_pre")
    df["elo_entraineur_pre"]    = get_elo("entraineur", "entraineur", "elo_entraineur_pre")
    df["elo_horse_ngames"]      = get_elo("horse",      "nom",        "elo_horse_ngames", 0)
    df["elo_driver_ngames"]     = get_elo("driver",     "driver",     "elo_driver_ngames", 0)
    df["elo_entraineur_ngames"] = get_elo("entraineur", "entraineur", "elo_entraineur_ngames", 0)

    # ── Rolling lookup ──
    ent_col_map = {"horse": "nom", "driver": "driver",
                   "entraineur": "entraineur", "proprietaire": "proprietaire"}
    for ent, col in ent_col_map.items():
        lkp_df = roll_lkp.get(ent)
        if lkp_df is None or col not in df.columns:
            continue
        for feat_col in lkp_df.columns:
            df[feat_col] = df[col].map(lkp_df[feat_col])

    # ── Target encoding lookup ──
    te_ent_map = {
        "nom_pere":     "nom_pere",
        "nom_mere":     "nom_mere",
        "nom_pere_mere": "nom_pere_mere",
        "nom":          "nom",
        "entraineur":   "entraineur",
        "driver":       "driver",
        "proprietaire": "proprietaire",
    }
    for ent, col in te_ent_map.items():
        lkp_df = te_lkp.get(ent)
        if lkp_df is None or col not in df.columns:
            continue
        for te_col in lkp_df.columns:
            df[te_col] = df[col].map(lkp_df[te_col])

    # ── Market features (depuis pmu_feat_market) ──
    df["mkt_drd_rapport"]   = df["drd_rapport"]
    df["mkt_drr_rapport"]   = df["drr_rapport"]
    df["mkt_drd_rank"]      = df["rank_odds"]
    df["mkt_drd_prob"]      = df["prob_drd"]
    df["mkt_drr_prob"]      = df["prob_drr"]
    df["mkt_spread_drr_drd"] = df["odds_diff"]
    df["mkt_tendance_count"] = df["drd_nb_tendance"]
    field_sum = df.groupby("race_id")["prob_drd"].transform("sum").clip(lower=0.01)
    df["mkt_drd_prob_norm"] = df["prob_drd"] / field_sum
    field_sum2 = df.groupby("race_id")["prob_drr"].transform("sum").clip(lower=0.01)
    df["mkt_drr_prob_norm"] = df["prob_drr"] / field_sum2
    df["mkt_drd_favoris"]    = df["drd_favoris"]
    df["mkt_drd_grosse_prise"] = df["drd_grosse_prise"]
    df["mkt_drd_permutation"] = df["drd_permutation"]
    df["mkt_overround"]      = field_sum - 1.0

    # ── Inline target encoding (fallback si te_lkp incomplet) ──
    global_mean = 1.0 / df.groupby("race_id")["num_pmu"].transform("count")
    for col in ["nom_pere", "nom_mere", "entraineur", "driver"]:
        te_col = f"te_{col}"
        if te_col not in df.columns:
            df[te_col] = global_mean

    # ── Aligner sur feat_names (remplir NaN les colonnes manquantes) ──
    for f in feat_names:
        if f not in df.columns:
            df[f] = np.nan

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",     default=None, help="DDMMYYYY (défaut: aujourd'hui)")
    ap.add_argument("--min-edge", type=float, default=0.0, help="Edge min pour filtrer paris")
    ap.add_argument("--reunion",  type=int,   default=None, help="Filtrer une réunion")
    args = ap.parse_args()

    date_str = args.date or datetime.now().strftime("%d%m%Y")
    print(f"\nPMU Live Predictor — {date_str}\n{'='*60}")

    # Charger modèle + features
    model = lgb.Booster(model_file=str(MODEL_PATH))
    feat_names = json.loads(FEATS_PATH.read_text(encoding="utf-8"))
    print(f"Modèle chargé : {len(feat_names)} features")

    # Lookup tables
    elo_lkp  = build_elo_lookup()  if ELO_PATH.exists()     else {}
    roll_lkp = build_rolling_lookup() if ROLLING_PATH.exists() else {}
    te_lkp   = build_te_lookup()   if TE_PATH.exists()      else {}

    # Fetch programme du jour
    print(f"\nFetch programme {date_str}...")
    t0 = time.time()
    df_raw = fetch_day(date_str)
    if df_raw is None or (hasattr(df_raw, '__len__') and len(df_raw) == 0):
        print("Aucune donnée.")
        return 1
    print(f"  Fetché en {time.time()-t0:.1f}s")

    if args.reunion:
        df_raw = df_raw[df_raw["reunion_num"] == args.reunion]

    # Feature engineering
    print("\nFeature engineering...")
    df_fe = build_live_features(df_raw, elo_lkp, roll_lkp, te_lkp, feat_names)

    # Prédiction
    X = df_fe[feat_names].astype(np.float32).fillna(0).values
    proba = model.predict(X)
    df_raw["proba_win"] = proba
    df_raw["drd_rapport_f"] = df_raw["drd_rapport"].fillna(0)
    df_raw["edge"] = df_raw["proba_win"] - 1.0 / df_raw["drd_rapport_f"].clip(lower=0.1)

    # Affichage par course
    print(f"\n{'='*80}")
    print(f"{'RACE':<22} {'#':>3} {'NOM':<22} {'COTE':>6} {'PROBA':>6} {'EDGE':>7}  VERDICT")
    print(f"{'='*80}")

    for race_id, grp in df_raw.groupby("race_id", sort=True):
        grp = grp.sort_values("proba_win", ascending=False)
        parts = []
        info  = grp.iloc[0]
        race_label = f"R{info['reunion_num']}C{info['course_num']} {info.get('discipline','')[:4]}"

        for _, row in grp.iterrows():
            if args.min_edge > 0 and row["edge"] < args.min_edge:
                continue
            edge_flag = "*** BET ***" if row["edge"] > 0.05 else ("bet?" if row["edge"] > 0 else "")
            cote_str  = f"{row['drd_rapport_f']:.1f}" if row['drd_rapport_f'] > 0 else "N/A"
            parts.append(
                f"{race_label:<22} {int(row['num_pmu']):>3} "
                f"{str(row['nom'])[:22]:<22} {cote_str:>6} "
                f"{row['proba_win']:>6.3f} {row['edge']:>+7.3f}  {edge_flag}"
            )
        if parts:
            print(parts[0])  # top pick par course
            for line in parts[1:3]:  # 2ème et 3ème choix en grisé
                print("  " + line[2:])
        print()

    # Résumé des paris à edge positif
    bets = df_raw[df_raw["edge"] > 0].sort_values("edge", ascending=False)
    if not bets.empty:
        print(f"\n{'='*80}")
        print(f"PARIS EDGE POSITIF ({len(bets)}) :")
        for _, row in bets.head(10).iterrows():
            print(f"  R{int(row['reunion_num'])}C{int(row['course_num'])} "
                  f"#{int(row['num_pmu'])} {str(row['nom'])[:20]:<20} "
                  f"cote={row['drd_rapport_f']:.1f}  proba={row['proba_win']:.3f}  "
                  f"edge={row['edge']:+.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
