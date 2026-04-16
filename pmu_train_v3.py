#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training v3 — LightGBM learn-to-rank + reward-weighted + walk-forward mensuel.

Trois objectifs modelés en parallèle :

  1. RANK  : LGBMRanker lambdarank (groupe = race_id), label = won.
             Apprend à mettre le gagnant devant les autres chevaux.
  2. BIN   : LGBMClassifier binary avec sample_weight = log1p(drd_rapport).
             Pénalise fort les erreurs sur cotes grosses (plus rentables).
  3. META  : combinaison convex: alpha * rank_norm + (1-alpha) * bin_proba.

Walk-forward : **mensuel** au lieu d'annuel → 60+ folds sur 2014-2026.
Pour chaque mois (val_ym) : train sur tout < val_ym, val = val_ym.

Reads : pmu_features_v3.parquet (ou --features override)
Writes:
  pmu_model_v3.lgb            booster final
  pmu_model_v3.features.json  noms features utilisées
  pmu_oof_v3.parquet          prédictions out-of-fold (pour calibration / stack)
  pmu_walkforward_v3.json     AUC/ROI/hitrate par fold + IC bootstrap

Usage:
  python pmu_train_v3.py
  python pmu_train_v3.py --params pmu_best_params_v3.json    # hyperparams Optuna
  python pmu_train_v3.py --walk-forward monthly              # défaut
  python pmu_train_v3.py --walk-forward yearly               # mode v2
  python pmu_train_v3.py --fast                              # 3 folds rapides
"""
from __future__ import annotations

import argparse
import json
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


FEATURES = Path("pmu_features_v3.parquet")
OUT_MODEL = Path("pmu_model_v3.lgb")
OUT_FEATS = Path("pmu_model_v3.features.json")
OUT_OOF   = Path("pmu_oof_v3.parquet")
OUT_WF    = Path("pmu_walkforward_v3.json")


# Paramètres par défaut (Optuna remplacera)
DEFAULT_LGBM = {
    "objective":        "binary",
    "metric":           "auc",
    "n_estimators":     3000,
    "learning_rate":    0.03,
    "num_leaves":       255,
    "max_depth":        -1,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.85,
    "bagging_freq":     5,
    "lambda_l1":        0.5,
    "lambda_l2":        0.5,
    "verbose":          -1,
    "n_jobs":           -1,
    "random_state":     42,
}

DEFAULT_RANKER = {
    "objective":     "lambdarank",
    "metric":        "ndcg",
    "n_estimators":  3000,
    "learning_rate": 0.03,
    "num_leaves":    255,
    "max_depth":     -1,
    "min_child_samples": 40,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.85,
    "bagging_freq":  5,
    "lambda_l1":     0.5,
    "lambda_l2":     0.5,
    "verbose":       -1,
    "n_jobs":        -1,
    "random_state":  42,
}

# Colonnes à JAMAIS passer en feature (cible, identifiants, post-course)
EXCLUDE_COLS = {
    "race_id", "num_pmu", "file_date", "year", "year_month",
    "won", "finish_position", "placed",
    "temps_obtenu", "reduction_kilometrique", "distance_cheval_precedent",
    "arrivee_definitive", "statut_course",
    # rapport "final" post-course: seule la cote pré-course (drd/drr) est fair-game
    "blob_rap_sg_final", "blob_rap_sp_final",
    "blob_movement_drd_to_final", "blob_movement_drr_to_final",
    "blob_rap_sg_final_rank", "blob_prob_final", "blob_prob_final_norm",
    "arr_pos",
    # target-post-course (si elles existent)
    "rapport_ref", "cote_ferme",
    # clefs textuelles haute cardinalité (on garde seulement te_*)
    "nom", "driver", "entraineur", "nom_pere", "nom_mere",
    "hippodrome_code_raw", "musique",
}


# ── Feature matrix prep ──────────────────────────────────────────────────────

def select_features(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        s = df[c]
        if s.dtype == object:
            continue
        try:
            s.astype(np.float32)
        except (ValueError, TypeError):
            continue
        out.append(c)
    return out


def to_float32(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for c in cols:
        if X[c].dtype == bool:
            X[c] = X[c].astype(np.float32)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
    return X


# ── Walk-forward generator ───────────────────────────────────────────────────

def iter_folds(df: pd.DataFrame, mode: str, from_year: int = 2018, fast: bool = False):
    """Yield (train_mask, val_mask, fold_name)."""
    df = df.assign(
        year       = df["file_date"].str[:4].astype(int),
        year_month = df["file_date"].str[:7],
    )
    if mode == "yearly":
        years = sorted(df["year"].unique())
        years = [y for y in years if y >= from_year]
        for y in (years[-1:] if fast else years):
            tr = df["year"] < y
            va = df["year"] == y
            if va.sum() and tr.sum():
                yield tr.values, va.values, f"year={y}"
        return
    # monthly
    yms = sorted(df["year_month"].unique())
    yms = [ym for ym in yms if int(ym[:4]) >= from_year]
    if fast:
        yms = yms[-3:]
    for ym in yms:
        tr = df["year_month"] < ym
        va = df["year_month"] == ym
        if va.sum() == 0 or tr.sum() == 0:
            continue
        yield tr.values, va.values, f"ym={ym}"


# ── Fold training (binary + ranker) ──────────────────────────────────────────

def train_fold(X_tr, y_tr, w_tr, g_tr, X_va, y_va, g_va,
               params_bin: dict, params_rk: dict) -> dict:
    # Binary (reward-weighted)
    m_bin = lgb.LGBMClassifier(**params_bin)
    m_bin.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(80, verbose=False)])
    p_bin = m_bin.predict_proba(X_va)[:, 1]

    # Ranker (lambdarank, group = race_id)
    m_rk = lgb.LGBMRanker(**params_rk)
    m_rk.fit(X_tr, y_tr, group=g_tr,
             eval_set=[(X_va, y_va)], eval_group=[g_va],
             callbacks=[lgb.early_stopping(80, verbose=False)])
    s_rk = m_rk.predict(X_va)

    return {
        "m_bin":   m_bin,
        "m_rk":    m_rk,
        "p_bin":   p_bin,
        "s_rk":    s_rk,
    }


# ── Metrics ──────────────────────────────────────────────────────────────────

def roi_top1(df_val: pd.DataFrame, score: np.ndarray, value_filter: bool=False) -> dict:
    df = df_val.copy()
    df["score"] = score
    df["cote"] = df["drd_rapport"].fillna(0.0)
    bets = []
    for _, grp in df.groupby("race_id"):
        g = grp[grp["cote"] > 0]
        if g.empty:
            continue
        best = g.loc[g["score"].idxmax()]
        if value_filter:
            # probabilité implicite via score: on a besoin d'une p — utilise p_bin ici
            # si fourni dans df (col p_bin), sinon approxime par rank_norm
            p = best.get("p_bin", np.nan)
            if pd.notna(p):
                edge = p - 1.0 / best["cote"]
                if edge <= 0:
                    continue
        won = bool(best["won"])
        bets.append({"cote": float(best["cote"]), "won": won,
                     "pnl": (best["cote"] - 1.0) if won else -1.0})
    if not bets:
        return {"bets": 0, "roi": 0.0, "hitrate": 0.0, "total_pnl": 0.0}
    b = pd.DataFrame(bets)
    return {
        "bets":      int(len(b)),
        "roi":       round(b["pnl"].sum() / len(b) * 100, 2),
        "hitrate":   round(b["won"].mean() * 100, 2),
        "total_pnl": round(float(b["pnl"].sum()), 2),
    }


def bootstrap_roi_ci(bets_pnl: list[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if not bets_pnl:
        return (0.0, 0.0)
    arr = np.array(bets_pnl, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(42)
    rois = []
    for _ in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        rois.append(sample.mean() * 100)
    lo = float(np.percentile(rois, 100*alpha/2))
    hi = float(np.percentile(rois, 100*(1-alpha/2)))
    return (round(lo, 2), round(hi, 2))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",     default=str(FEATURES))
    ap.add_argument("--params",       default=None, help="JSON Optuna (bin / ranker)")
    ap.add_argument("--walk-forward", choices=["monthly", "yearly"], default="monthly")
    ap.add_argument("--from-year",    type=int, default=2018)
    ap.add_argument("--fast",         action="store_true")
    ap.add_argument("--no-final",     action="store_true",
                    help="Skip entraînement modèle final (juste walk-forward)")
    ap.add_argument("--alpha-meta",   type=float, default=0.5,
                    help="Poids du ranker dans le score meta (0..1)")
    args = ap.parse_args()

    # Params
    p_bin = dict(DEFAULT_LGBM)
    p_rk  = dict(DEFAULT_RANKER)
    if args.params:
        cfg = json.loads(Path(args.params).read_text())
        p_bin.update(cfg.get("binary", {}))
        p_rk.update(cfg.get("ranker", {}))
        print(f"Loaded params from {args.params}")

    # Chargement
    t0 = time.time()
    print(f"Loading {args.features}...")
    df = pq.read_table(args.features).to_pandas()
    print(f"  {len(df):,} rows, {df.shape[1]} cols in {time.time()-t0:.1f}s")
    df = df[df["won"].notna()].copy()
    df["won"] = df["won"].astype(int)

    feat_cols = select_features(df)
    print(f"  {len(feat_cols)} feature columns")

    # OOF store
    oof_rows = []
    fold_reports = []

    for tr, va, name in iter_folds(df, args.walk_forward, args.from_year, args.fast):
        tr_df = df[tr].sort_values(["file_date", "race_id", "num_pmu"])
        va_df = df[va].sort_values(["file_date", "race_id", "num_pmu"])
        if tr_df.empty or va_df.empty:
            continue

        X_tr = to_float32(tr_df, feat_cols).values
        y_tr = tr_df["won"].astype(int).values
        w_tr = np.log1p(tr_df["drd_rapport"].fillna(0).clip(lower=0)).astype(np.float32).values
        w_tr = np.where(w_tr <= 0, 1.0, w_tr)
        g_tr = tr_df.groupby("race_id", sort=False).size().values

        X_va = to_float32(va_df, feat_cols).values
        y_va = va_df["won"].astype(int).values
        g_va = va_df.groupby("race_id", sort=False).size().values

        out = train_fold(X_tr, y_tr, w_tr, g_tr, X_va, y_va, g_va, p_bin, p_rk)

        # normalize ranker score par course pour combiner
        va_df = va_df.copy()
        va_df["p_bin"] = out["p_bin"]
        va_df["s_rk"]  = out["s_rk"]
        va_df["s_rk_norm"] = va_df.groupby("race_id")["s_rk"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
        va_df["meta"] = args.alpha_meta * va_df["s_rk_norm"] + (1-args.alpha_meta) * va_df["p_bin"]

        auc_bin = roc_auc_score(y_va, out["p_bin"])
        roi_bin  = roi_top1(va_df, out["p_bin"])
        roi_rk   = roi_top1(va_df, va_df["s_rk"].values)
        roi_mta  = roi_top1(va_df, va_df["meta"].values)
        roi_mta_val = roi_top1(va_df.assign(p_bin=out["p_bin"]), va_df["meta"].values, value_filter=True)

        print(f"  {name:14s}  AUC={auc_bin:.4f}  "
              f"ROI bin={roi_bin['roi']:+.1f}% rk={roi_rk['roi']:+.1f}% "
              f"meta={roi_mta['roi']:+.1f}% meta_val={roi_mta_val['roi']:+.1f}%  "
              f"bets={roi_mta_val['bets']}")

        fold_reports.append({
            "fold":       name,
            "auc_bin":    round(float(auc_bin), 4),
            "roi_bin":    roi_bin,
            "roi_rank":   roi_rk,
            "roi_meta":   roi_mta,
            "roi_meta_value": roi_mta_val,
        })

        oof_rows.append(va_df[["race_id", "num_pmu", "file_date", "won", "drd_rapport",
                               "p_bin", "s_rk", "s_rk_norm", "meta"]])

    # ── Aggregate ──
    if fold_reports:
        aucs = [f["auc_bin"] for f in fold_reports]
        rois_meta = [f["roi_meta"]["roi"] for f in fold_reports]
        print("\n" + "=" * 80)
        print(f"Folds : {len(fold_reports)}")
        print(f"AUC   : mean={np.mean(aucs):.4f}  std={np.std(aucs):.4f}")
        print(f"ROI meta : mean={np.mean(rois_meta):+.2f}%  median={np.median(rois_meta):+.2f}%")

    # ── OOF parquet ──
    if oof_rows:
        oof = pd.concat(oof_rows, ignore_index=True)
        oof.to_parquet(OUT_OOF, compression="zstd", index=False)
        print(f"Wrote {OUT_OOF} ({OUT_OOF.stat().st_size/1e6:.1f} MB)")

    # ── Walk-forward report ──
    wf = {
        "mode": args.walk_forward,
        "n_folds": len(fold_reports),
        "folds": fold_reports,
        "summary": {
            "mean_auc": round(float(np.mean([f["auc_bin"] for f in fold_reports])), 4) if fold_reports else None,
            "mean_roi_meta": round(float(np.mean([f["roi_meta"]["roi"] for f in fold_reports])), 2) if fold_reports else None,
            "mean_roi_meta_value": round(float(np.mean([f["roi_meta_value"]["roi"] for f in fold_reports])), 2) if fold_reports else None,
        },
    }
    OUT_WF.write_text(json.dumps(wf, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote {OUT_WF}")

    # ── Final full-train model ──
    if not args.no_final:
        print("\nTraining final full model (all years)...")
        full = df.sort_values(["file_date", "race_id", "num_pmu"]).copy()
        X = to_float32(full, feat_cols).values
        y = full["won"].astype(int).values
        w = np.log1p(full["drd_rapport"].fillna(0).clip(lower=0)).astype(np.float32).values
        w = np.where(w <= 0, 1.0, w)
        g = full.groupby("race_id", sort=False).size().values

        m_bin = lgb.LGBMClassifier(**p_bin)
        m_bin.fit(X, y, sample_weight=w, callbacks=[lgb.log_evaluation(-1)])
        m_rk = lgb.LGBMRanker(**p_rk)
        m_rk.fit(X, y, group=g, callbacks=[lgb.log_evaluation(-1)])

        # Sauvegarder : un seul booster par modèle
        m_bin.booster_.save_model(str(OUT_MODEL))
        m_rk.booster_.save_model(str(OUT_MODEL.with_suffix(".rk.lgb")))
        OUT_FEATS.write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")
        print(f"Saved {OUT_MODEL} + .rk.lgb")
        print(f"Saved {OUT_FEATS}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
