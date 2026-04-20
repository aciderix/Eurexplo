#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacking v3 — 5 base models + logistic meta-learner.

Base learners (tous entraînés en walk-forward monthly) :
  1. LGBM ranker  (lambdarank)        → score rang
  2. LGBM binary  (reward-weighted)   → p_bin
  3. XGBoost binary                    → p_xgb
  4. CatBoost binary                   → p_cat
  5. MLP (sklearn)                     → p_mlp
  6. Baseline "fav only" : 1/drd_rapport normalisé → p_fav

Meta-learner : logistic regression sur OOF predictions de chaque model.
On apprend **par course** : features = (p_1..p_6, rank_p_1..rank_p_6)
puis softmax par race pour sortir une distribution.

Reads :
  pmu_features_v3.parquet
  pmu_best_params_v3.json  (si présent, sinon defaults)
  pmu_oof_v3.parquet       (optionnel — si absent, on ré-entraîne)

Writes:
  pmu_stack_v3.pkl         dict {models, meta, feat_cols}
  pmu_oof_stack_v3.parquet OOF finales du stack

Usage:
  python pmu_stack_v3.py
  python pmu_stack_v3.py --params pmu_best_params_v3.json
  python pmu_stack_v3.py --fast   # 3 folds
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


FEATURES = Path("pmu_features_v3.parquet")
PARAMS   = Path("pmu_best_params_v3.json")
OUT_PKL  = Path("pmu_stack_v3.pkl")
OUT_OOF  = Path("pmu_oof_stack_v3.parquet")


EXCLUDE_COLS = {
    "race_id", "num_pmu", "file_date", "year", "year_month",
    "won", "finish_position", "placed",
    "temps_obtenu", "reduction_kilometrique", "distance_cheval_precedent",
    "arrivee_definitive", "statut_course",
    "blob_rap_sg_final", "blob_rap_sp_final",
    "blob_movement_drd_to_final", "blob_movement_drr_to_final",
    "blob_rap_sg_final_rank", "blob_prob_final", "blob_prob_final_norm",
    "arr_pos", "nom", "driver", "entraineur", "nom_pere", "nom_mere",
    "musique",
}


def _select_features(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        try:
            df[c].astype(np.float32)
            out.append(c)
        except (ValueError, TypeError):
            continue
    return out


def _iter_months(df: pd.DataFrame, from_ym: str = "2020-01", fast: bool = False):
    yms = sorted(df["year_month"].unique())
    yms = [ym for ym in yms if ym >= from_ym]
    if fast:
        yms = yms[-3:]
    for ym in yms:
        tr = df["year_month"] < ym
        va = df["year_month"] == ym
        if tr.sum() == 0 or va.sum() == 0:
            continue
        yield tr.values, va.values, ym


def _prep_X(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).fillna(0.0).values
    return X


def train_stack_fold(tr_df, va_df, feat_cols: list[str], params: dict, use_mlp: bool = True):
    X_tr = _prep_X(tr_df, feat_cols)
    y_tr = tr_df["won"].astype(int).values
    w_tr = np.log1p(tr_df["drd_rapport"].fillna(0).clip(lower=0)).astype(np.float32).values
    w_tr = np.where(w_tr <= 0, 1.0, w_tr)
    g_tr = tr_df.groupby("race_id", sort=False).size().values

    X_va = _prep_X(va_df, feat_cols)
    y_va = va_df["won"].astype(int).values

    base_preds: dict[str, np.ndarray] = {}

    # 1 — LGBM ranker
    m_rk = lgb.LGBMRanker(**params["ranker"])
    m_rk.fit(X_tr, y_tr, group=g_tr, callbacks=[lgb.log_evaluation(-1)])
    s_rk = m_rk.predict(X_va)
    # normaliser par course
    va_df2 = va_df.copy()
    va_df2["s_rk"] = s_rk
    s_rk_norm = va_df2.groupby("race_id")["s_rk"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    ).values
    base_preds["rank"] = s_rk_norm

    # 2 — LGBM binary
    m_bin = lgb.LGBMClassifier(**params["binary"])
    m_bin.fit(X_tr, y_tr, sample_weight=w_tr, callbacks=[lgb.log_evaluation(-1)])
    base_preds["lgb"] = m_bin.predict_proba(X_va)[:, 1]

    # 3 — XGBoost
    if HAS_XGB:
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 8,
            "learning_rate": params["binary"].get("learning_rate", 0.03),
            "n_estimators": 1500,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 0,
        }
        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
        base_preds["xgb"] = m_xgb.predict_proba(X_va)[:, 1]
    else:
        m_xgb = None
        base_preds["xgb"] = base_preds["lgb"]

    # 4 — CatBoost
    if HAS_CAT:
        m_cat = cb.CatBoostClassifier(
            iterations=1500, depth=8, learning_rate=0.03,
            l2_leaf_reg=3, random_seed=42, verbose=False, thread_count=-1
        )
        m_cat.fit(X_tr, y_tr, sample_weight=w_tr)
        base_preds["cat"] = m_cat.predict_proba(X_va)[:, 1]
    else:
        m_cat = None
        base_preds["cat"] = base_preds["lgb"]

    # 5 — MLP
    if use_mlp:
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)
        m_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                              alpha=1e-3, batch_size=1024, max_iter=40,
                              early_stopping=True, validation_fraction=0.05,
                              random_state=42, verbose=False)
        m_mlp.fit(X_tr_s, y_tr)
        base_preds["mlp"] = m_mlp.predict_proba(X_va_s)[:, 1]
    else:
        sc = m_mlp = None
        base_preds["mlp"] = base_preds["lgb"]

    # 6 — Favori-only (simple baseline)
    base_preds["fav"] = (1.0 / va_df["drd_rapport"].replace(0, np.nan).fillna(9e9)).values

    return {
        "m_rk": m_rk, "m_bin": m_bin, "m_xgb": m_xgb, "m_cat": m_cat,
        "m_mlp": m_mlp, "scaler": sc,
        "base_preds": base_preds,
        "y_va": y_va,
    }


def build_meta_features(va_df: pd.DataFrame, base_preds: dict) -> pd.DataFrame:
    keep = ["race_id", "num_pmu", "won", "drd_rapport"]
    if "file_date" in va_df.columns:
        keep.insert(2, "file_date")
    df = va_df[keep].copy()
    for name, p in base_preds.items():
        df[f"p_{name}"] = p
        df[f"rank_{name}"] = df.groupby("race_id")[f"p_{name}"].rank(ascending=False)
        df[f"pct_{name}"] = df.groupby("race_id")[f"p_{name}"].transform(
            lambda x: x / (x.sum() + 1e-9)
        )
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(FEATURES))
    ap.add_argument("--params",   default=str(PARAMS))
    ap.add_argument("--fast",     action="store_true")
    ap.add_argument("--from-ym",  default="2020-01")
    ap.add_argument("--no-mlp",   action="store_true")
    args = ap.parse_args()

    # Load params
    if Path(args.params).exists():
        p_json = json.loads(Path(args.params).read_text())
    else:
        p_json = {
            "binary": {"objective": "binary", "metric": "auc", "n_estimators": 2000,
                       "learning_rate": 0.03, "num_leaves": 255, "min_child_samples": 40,
                       "feature_fraction": 0.8, "bagging_fraction": 0.85, "bagging_freq": 5,
                       "lambda_l1": 0.5, "lambda_l2": 0.5, "verbose": -1, "n_jobs": -1,
                       "random_state": 42},
            "ranker": {"objective": "lambdarank", "metric": "ndcg", "n_estimators": 2000,
                       "learning_rate": 0.03, "num_leaves": 255, "min_child_samples": 40,
                       "feature_fraction": 0.8, "bagging_fraction": 0.85, "bagging_freq": 5,
                       "lambda_l1": 0.5, "lambda_l2": 0.5, "verbose": -1, "n_jobs": -1,
                       "random_state": 42},
        }

    t0 = time.time()
    print(f"Loading {args.features}...")
    df = pq.read_table(args.features).to_pandas()
    df = df[df["won"].notna()].copy()
    df["won"] = df["won"].astype(int)
    df["year_month"] = df["file_date"].str[:7]
    feat_cols = _select_features(df)
    print(f"  {len(df):,} rows, {len(feat_cols)} features  ({time.time()-t0:.1f}s)")

    # ── Per-month OOF ──
    oof_parts: list[pd.DataFrame] = []
    for tr, va, ym in _iter_months(df, args.from_ym, args.fast):
        tr_df = df[tr].sort_values(["file_date", "race_id", "num_pmu"])
        va_df = df[va].sort_values(["file_date", "race_id", "num_pmu"])
        t1 = time.time()
        fold = train_stack_fold(tr_df, va_df, feat_cols, p_json, use_mlp=not args.no_mlp)
        meta_df = build_meta_features(va_df, fold["base_preds"])
        oof_parts.append(meta_df)

        # Quick metrics
        for name, p in fold["base_preds"].items():
            try:
                auc = roc_auc_score(fold["y_va"], p)
            except Exception:
                auc = float("nan")
        print(f"  {ym}  n_tr={tr.sum():>7,}  n_va={va.sum():>6,}  "
              f"AUC(lgb)={roc_auc_score(fold['y_va'], fold['base_preds']['lgb']):.4f}  "
              f"AUC(mlp)={roc_auc_score(fold['y_va'], fold['base_preds']['mlp']):.4f}  "
              f"{time.time()-t1:.0f}s")

    if not oof_parts:
        print("No folds ran. Exit.")
        return 1

    oof = pd.concat(oof_parts, ignore_index=True)

    # ── Meta learner ──
    # Utiliser toutes les cols meta (p_*, rank_*, pct_*) + log(drd)
    oof["log_drd"] = np.log1p(oof["drd_rapport"].clip(lower=0))
    meta_cols = [c for c in oof.columns if c.startswith(("p_", "rank_", "pct_"))] + ["log_drd"]
    meta_X = oof[meta_cols].fillna(0).values
    meta_y = oof["won"].astype(int).values

    # ── Meta = LogReg. Cross-fit pour des predictions OOF HONNÊTES
    # (sinon p_stack est in-sample → calibration/conformal biaisés).
    meta_base = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=500, class_weight="balanced", n_jobs=-1,
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof["p_stack"] = cross_val_predict(
        meta_base, meta_X, meta_y, cv=kf, method="predict_proba", n_jobs=-1,
    )[:, 1]
    # Modèle final (utilisé en inférence live) entraîné sur TOUT le stack
    meta = meta_base.fit(meta_X, meta_y)

    auc_stack = roc_auc_score(meta_y, oof["p_stack"])
    print(f"\nStack OOF (cross-fit) AUC: {auc_stack:.4f}")
    for c in [c for c in oof.columns if c.startswith("p_") and c != "p_stack"]:
        try:
            print(f"  vs {c}: AUC={roc_auc_score(meta_y, oof[c]):.4f}")
        except Exception:
            pass

    # ── Save ──
    oof.to_parquet(OUT_OOF, compression="zstd", index=False)
    print(f"Wrote {OUT_OOF} ({OUT_OOF.stat().st_size/1e6:.1f} MB)")

    with open(OUT_PKL, "wb") as fh:
        pickle.dump({
            "meta":      meta,
            "meta_cols": meta_cols,
            "feat_cols": feat_cols,
            "params":    p_json,
        }, fh)
    print(f"Wrote {OUT_PKL}  (meta coefs: {dict(zip(meta_cols, meta.coef_[0].round(3)))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
