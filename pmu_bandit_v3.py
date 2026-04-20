#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contextual bandit v3 — Thompson Sampling sur Bayesian Linear Regression.

Là où le classifieur apprend "quel cheval gagne ?", le bandit apprend
directement "quel cheval rapporte le plus ?" via le PnL observé.

Pour chaque course :
  - Chaque cheval i est un bras
  - Le contexte x_i = vecteur de features (features v3 + preds stack)
  - On modélise E[PnL_i | x_i] = x_i · β + bruit
  - β ~ N(μ, Σ) (prior puis posterior après chaque course)
  - Thompson : tirer β ~ posterior, choisir argmax_i x_i · β̃

Le PnL observé :
  - Si on parie sur i et i gagne : PnL = cote - 1
  - Si on parie sur i et i perd  : PnL = -1
  - Sinon : aucune observation (censuré) → on n'update que le bras choisi

Reads :
  pmu_features_v3.parquet
  pmu_oof_stack_v3.parquet     (pour inclure p_stack, p_lgb, etc. dans le contexte)

Writes:
  pmu_bandit_v3.pkl
  pmu_bandit_report_v3.json

Usage:
  python pmu_bandit_v3.py --train-years 2014-2024 --eval-year 2025
  python pmu_bandit_v3.py --train-end 2025-06 --eval-start 2025-07 --eval-end 2026-03
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FEATURES = Path("pmu_features_v3.parquet")
OOF      = Path("pmu_oof_stack_v3.parquet")
OUT_PKL  = Path("pmu_bandit_v3.pkl")
OUT_REPORT = Path("pmu_bandit_report_v3.json")


# Set restreint de features pour le bandit (sinon Σ trop grande)
BANDIT_FEATS = [
    "log_drd", "log_drr", "prob_drd", "rank_odds_pct", "ratio_to_fav",
    "log_distance", "log_partants", "nb_partants_reels",
    "win_rate", "place_rate", "log_gains_carriere", "gains_per_start",
    "age", "handicap_valeur", "place_corde",
    # v3 ajouts
    "elo_ctx_horse_vs_field", "elo_ctx_driver_vs_field",
    "h2h_winrate_mean", "h2h_n_opponents_known",
    "blob_masse_share", "blob_prono_top1_cnt", "blob_prono_rank_mean",
    "mus_seq_place_last3_mean", "mus_seq_momentum", "mus_seq_streak_top3",
]


class BayesianLinBandit:
    """Thompson Sampling / BLR avec prior N(0, λ⁻¹ I), bruit σ²."""

    def __init__(self, dim: int, lam: float = 1.0, sigma2: float = 4.0):
        self.dim = dim
        self.lam = lam
        self.sigma2 = sigma2
        # Sufficient stats
        self.XtX = lam * np.eye(dim)
        self.Xty = np.zeros(dim)
        self._mu  = np.zeros(dim)
        self._Sig = np.eye(dim) / lam
        self._dirty = False

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim == 1:
            X = X[None, :]
        self.XtX += X.T @ X / self.sigma2
        self.Xty += X.T @ y / self.sigma2
        self._dirty = True

    def _recompute(self):
        self._Sig = np.linalg.inv(self.XtX)
        self._mu  = self._Sig @ self.Xty
        self._dirty = False

    def sample_beta(self, rng: np.random.Generator) -> np.ndarray:
        if self._dirty:
            self._recompute()
        L = np.linalg.cholesky(self._Sig + 1e-6 * np.eye(self.dim))
        eps = rng.standard_normal(self.dim)
        return self._mu + L @ eps

    def mean_beta(self) -> np.ndarray:
        if self._dirty:
            self._recompute()
        return self._mu


def _prep_context(df: pd.DataFrame, oof: pd.DataFrame | None, feat_cols: list[str]) -> pd.DataFrame:
    x = df.copy()
    if oof is not None:
        # Merge p_stack / rank_stack
        oof_small = oof[["race_id", "num_pmu"] + [c for c in oof.columns if c.startswith(("p_", "rank_"))]]
        x = x.merge(oof_small, on=["race_id", "num_pmu"], how="left")
    # Sécurité : garder les colonnes disponibles
    cols = [c for c in feat_cols if c in x.columns]
    # Ajouter p_stack si présent
    for extra in ("p_stack", "p_lgb", "p_fav", "rank_stack"):
        if extra in x.columns:
            cols.append(extra)
    X = x[cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).fillna(0.0)
    # Standardiser grossièrement
    mu, sd = X.mean(), X.std().replace(0, 1)
    Xz = (X - mu) / sd
    Xz["_bias"] = 1.0
    return Xz, cols


def simulate(df: pd.DataFrame, oof: pd.DataFrame | None, feat_cols: list[str],
             train_end: str, eval_start: str, eval_end: str | None,
             seed: int = 42) -> dict:
    # Tri chronologique AVANT tout — sans quoi le bandit apprend dans le désordre.
    df = df.sort_values(["file_date", "race_id", "num_pmu"], kind="stable").reset_index(drop=True)
    if oof is not None:
        oof = oof.copy()  # pas trié car merge par clés

    Xz, cols_used = _prep_context(df, oof, feat_cols)
    dim = Xz.shape[1]
    print(f"Bandit dim = {dim} (cols: {cols_used[:5]}... + _bias)")

    df2 = df.copy()
    df2["year_month"] = df2["file_date"].str[:7]
    X_mat = Xz.values.astype(np.float64)

    rng = np.random.default_rng(seed)
    bandit = BayesianLinBandit(dim=dim)

    # Phase 1 — TRAIN : on traite les courses ≤ train_end en off-policy
    # On simule : pour chaque course, on tire un β, on choisit argmax. Si le
    # cheval choisi est dans l'arrivée connue, on met à jour avec le PnL observé.
    print(f"\nTrain phase  (<= {train_end})")
    nr_tr = 0
    t0 = time.time()
    for race_id, grp in df2[df2["year_month"] <= train_end].groupby("race_id", sort=False):
        if grp["won"].isna().all() or grp["drd_rapport"].max() <= 0:
            continue
        idx = grp.index.to_numpy()
        beta = bandit.sample_beta(rng)
        preds = X_mat[idx] @ beta
        best_i = int(np.argmax(preds))
        row = grp.iloc[best_i]
        cote = float(row["drd_rapport"])
        if cote <= 0:
            continue
        pnl = (cote - 1.0) if bool(row["won"]) else -1.0
        bandit.update(X_mat[idx[best_i]:idx[best_i]+1], np.array([pnl]))
        nr_tr += 1
        if nr_tr % 20000 == 0:
            print(f"  trained on {nr_tr:,} races  ({nr_tr/(time.time()-t0):.0f}/s)")
    print(f"Trained on {nr_tr:,} races.")

    # Phase 2 — EVAL : on utilise la MOYENNE (exploitation pure) sur la période test
    print(f"\nEval phase  ({eval_start} .. {eval_end or 'end'})")
    beta_hat = bandit.mean_beta()
    pnl_list = []
    bets = 0
    hits = 0
    ev_mask = df2["year_month"] >= eval_start
    if eval_end:
        ev_mask &= (df2["year_month"] <= eval_end)
    for race_id, grp in df2[ev_mask].groupby("race_id", sort=False):
        if grp["won"].isna().all() or grp["drd_rapport"].max() <= 0:
            continue
        idx = grp.index.to_numpy()
        preds = X_mat[idx] @ beta_hat
        best_i = int(np.argmax(preds))
        row = grp.iloc[best_i]
        cote = float(row["drd_rapport"])
        if cote <= 0:
            continue
        won = bool(row["won"])
        pnl = (cote - 1.0) if won else -1.0
        pnl_list.append(pnl)
        bets += 1
        hits += int(won)

    roi = float(np.mean(pnl_list) * 100) if pnl_list else 0.0
    hitrate = hits / bets * 100 if bets else 0.0
    print(f"Eval : bets={bets:,}  hits={hits:,}  hit={hitrate:.2f}%  ROI={roi:+.2f}%")

    return {
        "bandit":  bandit,
        "cols":    cols_used,
        "n_train": nr_tr,
        "n_bets":  bets,
        "roi":     roi,
        "hitrate": hitrate,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",    default=str(FEATURES))
    ap.add_argument("--oof",         default=str(OOF))
    ap.add_argument("--train-end",   default="2024-12")
    ap.add_argument("--eval-start",  default="2025-01")
    ap.add_argument("--eval-end",    default=None)
    # Short form : --train-years 2014-2024 --eval-year 2025
    ap.add_argument("--train-years", default=None,
                    help="Ex: 2014-2024 → train-end=2024-12, eval-year=2025")
    ap.add_argument("--eval-year",   default=None)
    args = ap.parse_args()

    if args.train_years:
        y_end = args.train_years.split("-")[-1]
        args.train_end = f"{y_end}-12"
    if args.eval_year:
        args.eval_start = f"{args.eval_year}-01"
        args.eval_end   = f"{args.eval_year}-12"

    t0 = time.time()
    print(f"Loading {args.features}...")
    df = pq.read_table(args.features).to_pandas()
    df = df[df["won"].notna()].copy()
    df["won"] = df["won"].astype(int)
    print(f"  {len(df):,} rows  ({time.time()-t0:.1f}s)")

    oof = None
    if Path(args.oof).exists():
        oof = pq.read_table(args.oof).to_pandas()
        print(f"  loaded OOF stack: {len(oof):,} rows")

    res = simulate(df, oof, BANDIT_FEATS,
                   train_end=args.train_end,
                   eval_start=args.eval_start,
                   eval_end=args.eval_end)

    with open(OUT_PKL, "wb") as fh:
        pickle.dump({
            "bandit":   res["bandit"],
            "cols":     res["cols"],
            "train_end": args.train_end,
            "eval_start": args.eval_start,
            "eval_end":   args.eval_end,
        }, fh)
    print(f"\nWrote {OUT_PKL}")

    OUT_REPORT.write_text(json.dumps({
        "n_train": res["n_train"],
        "n_bets":  res["n_bets"],
        "roi":     round(res["roi"], 2),
        "hitrate": round(res["hitrate"], 2),
        "train_end": args.train_end,
        "eval_start": args.eval_start,
        "eval_end":   args.eval_end,
    }, indent=2))
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
