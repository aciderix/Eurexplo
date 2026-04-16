#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna hyperparam search v3 — optimise AUC **et** ROI meta simultanément.

Utilise un critère composite : score = AUC + 0.002 * max(ROI_meta, -10)
  (donc AUC pondère beaucoup, ROI tire dans la bonne direction sans
   exploser la métrique sur les folds chanceux).

Strategie : TPE (Bayesien) avec pruning médiane, N trials, 3 folds
rolling pour chaque trial (train ≤ M-12, val = M-12 .. M-1 pour M = 2024-01..2026-03).

Reprenable : SQLite storage.

Reads : pmu_features_v3.parquet
Writes:
  pmu_optuna_study.db           SQLite storage Optuna
  pmu_best_params_v3.json       meilleurs hyperparams (binary + ranker)

Usage:
  python pmu_optuna_v3.py --n-trials 200 --study pmu_v3_main
  python pmu_optuna_v3.py --resume --study pmu_v3_main --n-trials 50
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

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("ERR: optuna non installé. pip install optuna optuna-integration")
    sys.exit(1)


FEATURES = Path("pmu_features_v3.parquet")
DB_PATH  = Path("pmu_optuna_study.db")
OUT_BEST = Path("pmu_best_params_v3.json")

# ── 3 folds rolling pour chaque trial (pas 12, sinon trop long) ──
# Pour chaque val_ym, train = TOUTES les lignes avec year_month < val_ym
# (donc depuis 2014). On ne fait pas de cutoff sliding — on exploite max data.
EVAL_MONTHS = ["2025-01", "2025-07", "2026-01"]


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
        s = df[c]
        if s.dtype == object:
            continue
        try:
            s.astype(np.float32)
        except (ValueError, TypeError):
            continue
        out.append(c)
    return out


def _roi_top1(df_val: pd.DataFrame, score: np.ndarray) -> float:
    df = df_val.copy()
    df["score"] = score
    df["cote"]  = df["drd_rapport"].fillna(0.0)
    pnl = []
    for _, grp in df.groupby("race_id"):
        g = grp[grp["cote"] > 0]
        if g.empty:
            continue
        best = g.loc[g["score"].idxmax()]
        won = bool(best["won"])
        pnl.append((best["cote"] - 1.0) if won else -1.0)
    if not pnl:
        return 0.0
    return float(np.mean(pnl) * 100)


def _train_eval(trial, df: pd.DataFrame, feat_cols: list[str]) -> float:
    # Sample hyperparams
    bin_params = {
        "objective":        "binary",
        "metric":           "auc",
        "n_estimators":     trial.suggest_int("n_estimators", 500, 4000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 31, 511),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 0, 10),
        "lambda_l1":        trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2":        trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "verbose":          -1, "n_jobs": -1, "random_state": 42,
    }
    rk_params = dict(bin_params)
    rk_params["objective"] = "lambdarank"
    rk_params["metric"]    = "ndcg"

    alpha_meta = trial.suggest_float("alpha_meta", 0.0, 1.0)
    reward_pow = trial.suggest_float("reward_pow", 0.0, 1.5)

    scores = []
    for i, val_ym in enumerate(EVAL_MONTHS):
        tr_df = df[df["year_month"] < val_ym].sort_values(["file_date", "race_id", "num_pmu"])
        va_df = df[df["year_month"] == val_ym].sort_values(["file_date", "race_id", "num_pmu"])
        if tr_df.empty or va_df.empty:
            continue

        X_tr = tr_df[feat_cols].astype(np.float32).values
        y_tr = tr_df["won"].astype(int).values
        w_tr = np.log1p(tr_df["drd_rapport"].fillna(0).clip(lower=0)).astype(np.float32).values ** reward_pow
        w_tr = np.where(w_tr <= 0, 1.0, w_tr)
        g_tr = tr_df.groupby("race_id", sort=False).size().values

        X_va = va_df[feat_cols].astype(np.float32).values
        y_va = va_df["won"].astype(int).values
        g_va = va_df.groupby("race_id", sort=False).size().values

        m_bin = lgb.LGBMClassifier(**bin_params)
        m_bin.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        p_bin = m_bin.predict_proba(X_va)[:, 1]

        m_rk = lgb.LGBMRanker(**rk_params)
        m_rk.fit(X_tr, y_tr, group=g_tr,
                 eval_set=[(X_va, y_va)], eval_group=[g_va],
                 callbacks=[lgb.early_stopping(50, verbose=False)])
        s_rk = m_rk.predict(X_va)

        auc = roc_auc_score(y_va, p_bin)

        va_df2 = va_df.copy()
        va_df2["s_rk"] = s_rk
        va_df2["s_rk_norm"] = va_df2.groupby("race_id")["s_rk"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
        meta = alpha_meta * va_df2["s_rk_norm"].values + (1-alpha_meta) * p_bin
        roi_m = _roi_top1(va_df2.assign(won=y_va), meta)

        composite = float(auc) + 0.002 * max(roi_m, -10)
        scores.append(composite)

        trial.report(np.mean(scores), i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    if not scores:
        return 0.0
    return float(np.mean(scores))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--study",    default="pmu_v3_main")
    ap.add_argument("--resume",   action="store_true")
    ap.add_argument("--features", default=str(FEATURES))
    args = ap.parse_args()

    print(f"Loading {args.features}...")
    t0 = time.time()
    df = pq.read_table(args.features).to_pandas()
    df = df[df["won"].notna()].copy()
    df["won"] = df["won"].astype(int)
    df["year_month"] = df["file_date"].str[:7]
    feat_cols = _select_features(df)
    print(f"  {len(df):,} rows, {len(feat_cols)} features  ({time.time()-t0:.1f}s)")

    storage = f"sqlite:///{DB_PATH}"
    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=20),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        load_if_exists=args.resume,
    )

    print(f"\nStarting {args.n_trials} trials  (existing: {len(study.trials)})")
    study.optimize(
        lambda t: _train_eval(t, df, feat_cols),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print("\nBest:")
    print(f"  score = {study.best_value:.4f}")
    print(f"  params = {study.best_params}")

    # Split params into bin / ranker form
    best = study.best_params
    shared_keys = ["n_estimators", "learning_rate", "num_leaves",
                   "min_child_samples", "feature_fraction", "bagging_fraction",
                   "bagging_freq", "lambda_l1", "lambda_l2"]
    bin_out = {k: best[k] for k in shared_keys if k in best}
    bin_out.update({"objective": "binary", "metric": "auc"})
    rk_out  = dict(bin_out); rk_out.update({"objective": "lambdarank", "metric": "ndcg"})

    OUT_BEST.write_text(json.dumps({
        "binary":     bin_out,
        "ranker":     rk_out,
        "alpha_meta": best.get("alpha_meta", 0.5),
        "reward_pow": best.get("reward_pow", 1.0),
        "study_best_value": round(float(study.best_value), 4),
    }, indent=2))
    print(f"Wrote {OUT_BEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
