#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMU LightGBM Training V2 -- walk-forward CV sur pmu_dataset_v2.parquet

Features extraites (~60 colonnes) :
  - Cotes (drd/drr) : rapport log, probabilité implicite, rang dans la course
  - Forme : win rate, place rate, parse musique (5 dernières courses)
  - Caractéristiques cheval : age, sexe, allure, robe, oeilleres, gains
  - Course : distance, discipline, corde, nb_partants, montant_prix, heure
  - Lignée / entraîneur / driver : target-encoding (expanding window)
  - Signaux marché : drd_favoris, drd_grosse_prise, drd_nombre_indicateur_tendance

Walk-forward : train 2014→Y, val Y+1, pour Y = 2018..2024
Modèle final  : train 2014→2025, évalué sur 2026

Usage :
  python pmu_train_v2.py                       # full run
  python pmu_train_v2.py --fast                # 1 fold rapide (test)
  python pmu_train_v2.py --from-year 2022      # folds à partir de 2022
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

PARQUET   = Path("pmu_dataset_v2.parquet")
OUT_MODEL = Path("pmu_model_v2.lgb")
OUT_REPORT = Path("pmu_report_v2.json")

# ── Feature config ─────────────────────────────────────────────────────────────

# Colonnes chargées depuis le parquet (pas tout pour économiser RAM)
LOAD_COLS = [
    "race_id", "file_date", "num_pmu",
    # target
    "won", "finish_position",
    # odds
    "drd_rapport", "drr_rapport", "drd_favoris", "drd_grosse_prise",
    "drd_nombre_indicateur_tendance", "drd_permutation",
    # forme
    "nombre_courses", "nombre_victoires", "nombre_places",
    "nombre_places_second", "nombre_places_troisieme",
    "musique",
    # gains
    "gains_carriere", "gains_annee_en_cours", "gains_annee_precedente",
    "gains_victoires", "gains_place",
    # cheval
    "age", "sexe", "allure", "race", "oeilleres", "robe_code",
    "jument_pleine", "driver_change", "indicateur_inedit",
    "handicap_distance", "handicap_poids", "handicap_valeur",
    # course
    "discipline", "specialite", "distance", "corde",
    "nb_partants_reels", "nb_declares_partants",
    "montant_prix", "montant_total_offert", "montant_offert_1er",
    "condition_sexe", "categorie_particuliere",
    "heure_depart_ts", "grand_prix_national_trot", "pari_special",
    "place_corde",
    # lieu
    "hippodrome_code", "pays_code", "nature_reunion",
    # acteurs (pour target encoding)
    "nom_pere", "nom_mere", "entraineur", "driver",
]

CATEG_COLS = [
    "sexe", "allure", "race", "oeilleres", "robe_code",
    "discipline", "specialite", "corde", "condition_sexe",
    "categorie_particuliere", "hippodrome_code", "pays_code",
    "nature_reunion",
]

TARGET_ENCODE_COLS = ["nom_pere", "nom_mere", "entraineur", "driver"]

LGBM_PARAMS = {
    "objective":       "binary",
    "metric":          "auc",
    "n_estimators":    2000,
    "learning_rate":   0.05,
    "num_leaves":      127,
    "max_depth":       -1,
    "min_child_samples": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq":    5,
    "lambda_l1":       0.1,
    "lambda_l2":       0.1,
    "verbose":         -1,
    "n_jobs":          -1,
    "random_state":    42,
}


# ── Musique parser ─────────────────────────────────────────────────────────────

def _parse_musique(s: str | None, n: int = 5) -> dict:
    """Extrait les n derniers résultats de la musique (leftmost = most recent)."""
    if not s or not isinstance(s, str):
        return {"mus_wins": 0, "mus_places": 0, "mus_dnf": 0, "mus_avg_pos": np.nan}
    tokens = re.findall(r"[0-9D]", s)[:n]
    wins = sum(1 for t in tokens if t == "1")
    places = sum(1 for t in tokens if t in ("2", "3"))
    dnf = sum(1 for t in tokens if t in ("0", "D"))
    positions = [int(t) for t in tokens if t.isdigit() and t != "0"]
    avg_pos = np.mean(positions) if positions else np.nan
    return {"mus_wins": wins, "mus_places": places, "mus_dnf": dnf, "mus_avg_pos": avg_pos}


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Date → year/month/weekday ──
    df["year"]    = df["file_date"].str[:4].astype(int)
    df["month"]   = df["file_date"].str[5:7].astype(int)

    # ── Heure de départ ──
    df["heure_h"] = ((df["heure_depart_ts"] / 3_600_000) % 24).round(1)

    # ── Win rate / place rate ──
    nc = df["nombre_courses"].clip(lower=1)
    df["win_rate"]   = df["nombre_victoires"] / nc
    df["place_rate"] = df["nombre_places"]    / nc
    df["place2_rate"] = df["nombre_places_second"].fillna(0) / nc
    df["place3_rate"] = df["nombre_places_troisieme"].fillna(0) / nc

    # ── Gains (log scale) ──
    for col in ["gains_carriere", "gains_annee_en_cours", "gains_annee_precedente",
                "gains_victoires", "gains_place"]:
        df[f"log_{col}"] = np.log1p(df[col].fillna(0))
    df["gains_per_start"] = df["gains_carriere"].fillna(0) / nc

    # ── Odds ──
    df["log_drd"]   = np.log1p(df["drd_rapport"].clip(lower=0))
    df["log_drr"]   = np.log1p(df["drr_rapport"].clip(lower=0))
    df["prob_drd"]  = 1.0 / df["drd_rapport"].clip(lower=1)
    df["prob_drr"]  = 1.0 / df["drr_rapport"].clip(lower=1)
    df["odds_diff"] = df["drd_rapport"].fillna(0) - df["drr_rapport"].fillna(0)
    df["drd_favoris"]     = df["drd_favoris"].fillna(False).astype(int)
    df["drd_grosse_prise"] = df["drd_grosse_prise"].fillna(False).astype(int)
    df["drd_nb_tendance"]  = df["drd_nombre_indicateur_tendance"].fillna(0)
    df["drd_permutation"]  = df["drd_permutation"].fillna(0).astype(int)

    # ── Rang cote dans la course ──
    df["rank_odds"] = df.groupby("race_id")["drd_rapport"].rank(method="min", ascending=True)
    df["rank_odds_pct"] = df.groupby("race_id")["drd_rapport"].rank(pct=True, ascending=True)

    # ── Rapport au favori ──
    min_odds = df.groupby("race_id")["drd_rapport"].transform("min")
    df["ratio_to_fav"] = df["drd_rapport"] / min_odds.clip(lower=0.1)

    # ── Musique ──
    mus = df["musique"].apply(_parse_musique)
    df["mus_wins"]   = [x["mus_wins"]   for x in mus]
    df["mus_places"] = [x["mus_places"] for x in mus]
    df["mus_dnf"]    = [x["mus_dnf"]    for x in mus]
    df["mus_avg_pos"] = [x["mus_avg_pos"] for x in mus]

    # ── Booleans ──
    df["driver_change"]    = df["driver_change"].fillna(False).astype(int)
    df["indicateur_inedit"] = df["indicateur_inedit"].fillna(False).astype(int)
    df["jument_pleine"]     = df["jument_pleine"].fillna(False).astype(int)
    df["grand_prix"]        = df["grand_prix_national_trot"].fillna(False).astype(int)
    df["pari_special"]      = df["pari_special"].fillna(False).astype(int)

    # ── Catégorielles → codes ──
    for col in CATEG_COLS:
        df[col] = df[col].astype("category").cat.codes.astype("int16")

    # ── Champ (log distance, log partants) ──
    df["log_distance"]  = np.log1p(df["distance"].fillna(0))
    df["log_partants"]  = np.log1p(df["nb_partants_reels"].fillna(0))
    df["log_prize"]     = np.log1p(df["montant_prix"].fillna(0))
    df["log_prize_1er"] = np.log1p(df["montant_offert_1er"].fillna(0))

    # ── Handicap ──
    df["handicap_distance"] = df["handicap_distance"].fillna(0)
    df["handicap_poids"]    = df["handicap_poids"].fillna(0)
    df["handicap_valeur"]   = df["handicap_valeur"].fillna(0)
    df["place_corde"]       = df["place_corde"].fillna(0).astype(int)

    return df


def target_encode(df: pd.DataFrame, col: str, global_mean: float) -> np.ndarray:
    """Target encoding par expanding window triée par date (anti-fuite)."""
    df_sorted = df.sort_values("file_date")
    cum_sum   = df_sorted.groupby(col)["won"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    encoded = cum_sum.fillna(global_mean)
    return encoded.loc[df.index].values


def add_target_encodings(df: pd.DataFrame, global_mean: float | None = None) -> pd.DataFrame:
    if global_mean is None:
        global_mean = df["won"].mean()
    for col in TARGET_ENCODE_COLS:
        df[f"te_{col}"] = target_encode(df, col, global_mean)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {
        # ids
        "race_id", "file_date", "num_pmu",
        # targets / post-course (JAMAIS en feature)
        "won", "finish_position", "placed",
        "temps_obtenu", "reduction_kilometrique", "distance_cheval_precedent",
        "arrivee_definitive", "statut_course",
        # raw stats cumulées (remplacées par win_rate etc. calculés)
        "heure_depart_ts", "musique", "nombre_courses", "nombre_victoires",
        "nombre_places", "nombre_places_second", "nombre_places_troisieme",
        "gains_carriere", "gains_annee_en_cours", "gains_annee_precedente",
        "gains_victoires", "gains_place",
        # raw odds (remplacés par log_drd, prob_drd etc.)
        "drd_rapport", "drr_rapport",
        "drd_nombre_indicateur_tendance", "grand_prix_national_trot", "pari_special",
        # haute cardinalité texte (remplacés par target-encoding)
        "nom_pere", "nom_mere", "entraineur", "driver",
        "montant_total_offert",
    } | {c for c in df.columns if c.startswith("year") and c != "year"}
    return [c for c in df.columns if c not in drop]


# ── ROI simulation ─────────────────────────────────────────────────────────────

def simulate_roi(df_val: pd.DataFrame, preds: np.ndarray, strategy: str = "top1") -> dict:
    """
    Simule des paris à mise plate (1€) sur la ou les sélection(s) de la course.
    strategy='top1' : parie sur le cheval avec la proba la plus haute par course.
    strategy='top1_value' : parie uniquement si proba > 1/cote (edge positif).
    """
    df = df_val.copy()
    df["pred"] = preds
    df["cote"] = df["drd_rapport"].fillna(0)

    results = []
    for _, grp in df.groupby("race_id"):
        if grp["cote"].max() == 0:
            continue
        grp_valid = grp[grp["cote"] > 0]
        if grp_valid.empty:
            continue
        best = grp_valid.loc[grp_valid["pred"].idxmax()]
        won  = bool(best["won"])

        if strategy == "top1_value":
            edge = best["pred"] - 1.0 / best["cote"]
            if edge <= 0:
                continue

        results.append({
            "race_id": best["race_id"],
            "stake":   1.0,
            "cote":    best["cote"],
            "won":     won,
            "pnl":     (best["cote"] - 1.0) if won else -1.0,
        })

    if not results:
        return {"bets": 0, "roi": 0.0, "win_rate": 0.0}
    df_r = pd.DataFrame(results)
    total_stake = df_r["stake"].sum()
    total_pnl   = df_r["pnl"].sum()
    return {
        "bets":     len(df_r),
        "roi":      round(total_pnl / total_stake * 100, 2),
        "win_rate": round(df_r["won"].mean() * 100, 2),
        "total_pnl": round(total_pnl, 2),
    }


# ── Walk-forward training ─────────────────────────────────────────────────────

def run_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, fold_name: str) -> dict:
    global_mean = train_df["won"].mean()

    train_fe = add_target_encodings(build_features(train_df), global_mean)
    val_fe   = add_target_encodings(val_df,                   global_mean)
    val_fe   = build_features(val_fe)

    # Recalcul target encodings sur val avec les stats du train
    for col in TARGET_ENCODE_COLS:
        col_mean = (
            train_df.groupby(col)["won"].mean()
            .rename(f"te_{col}")
        )
        val_fe[f"te_{col}"] = val_df[col].map(col_mean).fillna(global_mean)

    feat_cols = feature_columns(train_fe)
    feat_cols = [c for c in feat_cols if c in val_fe.columns]

    # Keep only columns that can be safely cast to float32
    safe_cols = []
    for c in feat_cols:
        col = train_fe[c]
        if col.dtype == object:
            col = pd.to_numeric(col.map({True: 1, False: 0}).where(
                col.isin([True, False, 1, 0]), other=np.nan), errors="coerce")
            if col.notna().mean() < 0.1:
                continue  # true string column, skip
            train_fe[c] = col
            val_fe[c]   = pd.to_numeric(val_fe[c].map({True: 1, False: 0}).where(
                val_fe[c].isin([True, False, 1, 0]), other=np.nan), errors="coerce")
        try:
            train_fe[c].astype(np.float32)
            val_fe[c].astype(np.float32)
            safe_cols.append(c)
        except (ValueError, TypeError):
            pass  # string column in train or val, drop it
    feat_cols = safe_cols

    X_tr = train_fe[feat_cols].values.astype(np.float32)
    y_tr = train_fe["won"].fillna(0).astype(int).values
    X_va = val_fe[feat_cols].values.astype(np.float32)
    y_va = val_fe["won"].fillna(0).astype(int).values

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    preds = model.predict_proba(X_va)[:, 1]
    auc   = roc_auc_score(y_va, preds)
    roi_top1 = simulate_roi(val_df.assign(pred_=preds).merge(
        pd.DataFrame({"idx": val_df.index, "pred": preds}).set_index("idx"),
        left_index=True, right_index=True
    ) if False else val_df.copy().assign(**{"drd_rapport": val_df["drd_rapport"]}),
        preds, "top1"
    )
    # Simpler ROI: rebuild with odds
    val_roi = val_df[["race_id", "drd_rapport", "won"]].copy()
    val_roi["pred"] = preds
    roi_info  = simulate_roi(val_roi, preds, "top1")
    roi_value = simulate_roi(val_roi, preds, "top1_value")

    n_trees = model.best_iteration_ or LGBM_PARAMS["n_estimators"]
    print(f"  {fold_name}  AUC={auc:.4f}  trees={n_trees:>4}  "
          f"ROI_top1={roi_info['roi']:+.1f}%  ({roi_info['bets']} paris)  "
          f"ROI_value={roi_value['roi']:+.1f}% ({roi_value['bets']} paris)")

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)

    return {
        "fold":        fold_name,
        "auc":         round(auc, 4),
        "n_trees":     n_trees,
        "roi_top1":    roi_info,
        "roi_value":   roi_value,
        "feat_importance": fi.head(30).to_dict(),
        "model":       model,
        "feat_cols":   feat_cols,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast",      action="store_true", help="1 fold seulement (2023→2024)")
    ap.add_argument("--from-year", type=int, default=2018, help="Premier fold val (défaut 2018)")
    ap.add_argument("--parquet",   default=str(PARQUET))
    args = ap.parse_args()

    print("Chargement du parquet…")
    t0 = time.time()
    # Si le parquet est enrichi (pmu_features_v2), charger toutes les colonnes
    available = pq.read_schema(args.parquet).names
    is_enriched = any(c.startswith(("elo_", "ent_", "mus_n_", "te_")) for c in available)
    if is_enriched:
        print(f"  Parquet enrichi détecté ({len(available)} colonnes) — chargement complet")
        df = pq.read_table(args.parquet).to_pandas()
    else:
        cols_to_load = [c for c in LOAD_COLS if c in available]
        df = pq.read_table(args.parquet, columns=cols_to_load).to_pandas()
    print(f"  {len(df):,} lignes  {df.shape[1]} cols  ({time.time()-t0:.1f}s)")

    # Filtrer les lignes sans résultat connu (courses futures)
    df = df[df["won"].notna()].copy()
    df["won"] = df["won"].astype(int)
    df["year"] = df["file_date"].str[:4].astype(int)
    print(f"  {len(df):,} lignes avec résultat")

    # Définir les folds
    max_val_year = df["year"].max()
    fold_years = range(args.from_year, max_val_year)
    if args.fast:
        fold_years = [max_val_year - 1]

    print(f"\nWalk-forward CV  |  folds val: {list(fold_years)}\n")
    print("-" * 80)

    fold_results = []
    for val_year in fold_years:
        train_mask = df["year"] < val_year
        val_mask   = df["year"] == val_year
        if val_mask.sum() == 0:
            continue
        fold_name = f"train→{val_year-1}  val={val_year}"
        result = run_fold(df[train_mask], df[val_mask], fold_name)
        fold_results.append(result)

    print("-" * 80)
    aucs = [r["auc"] for r in fold_results]
    rois = [r["roi_top1"]["roi"] for r in fold_results]
    print(f"\nMoyenne AUC  : {np.mean(aucs):.4f}  (±{np.std(aucs):.4f})")
    print(f"Moyenne ROI top1 : {np.mean(rois):+.1f}%")

    # Modèle final : train tout sauf dernière année
    final_val_year = max_val_year
    print(f"\nModèle final : train 2014→{final_val_year-1}  val={final_val_year}")
    train_mask = df["year"] < final_val_year
    val_mask   = df["year"] == final_val_year
    final = run_fold(df[train_mask], df[val_mask], f"FINAL val={final_val_year}")
    fold_results.append(final)

    # Sauvegarder modèle
    final["model"].booster_.save_model(str(OUT_MODEL))
    print(f"\nModèle sauvegardé : {OUT_MODEL}")

    # Rapport JSON
    report = {
        "folds": [
            {k: v for k, v in r.items() if k not in ("model",)}
            for r in fold_results
        ],
        "mean_auc":      round(np.mean(aucs), 4),
        "std_auc":       round(np.std(aucs), 4),
        "mean_roi_top1": round(float(np.mean(rois)), 2),
        "final_auc":     final["auc"],
        "final_roi_top1": final["roi_top1"],
        "final_roi_value": final["roi_value"],
        "top30_features": final["feat_importance"],
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Rapport sauvegardé : {OUT_REPORT}\n")

    # Top features
    print("Top 20 features (modèle final) :")
    for i, (feat, imp) in enumerate(list(final["feat_importance"].items())[:20], 1):
        print(f"  {i:>2}. {feat:<35} {imp}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
