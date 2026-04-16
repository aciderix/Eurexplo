#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prédiction live v3 — utilise le pipeline complet (stack + calibration + conformal).

Reprend l'architecture de `pmu_predict_today.py` (v2) mais :
  - charge pmu_features_v3.parquet pour dernières lookups (ELO ctx, H2H…)
  - applique le stack (6 modèles) + meta-learner
  - applique la calibration (isotonic/beta) si dispo
  - applique le conformal pour une borne basse p_lo
  - stratégie par défaut : `conformal_lo × cote > 1.05`

Reads live :
  - pmu_model_v3.lgb + .rk.lgb + .features.json
  - pmu_stack_v3.pkl (opt)
  - pmu_calibrator_v3.pkl (opt)
  - pmu_conformal_v3.json (opt)
  - pmu_features_v3.parquet (pour lookups)

Usage:
  python pmu_predict_today_v3.py --date 16042026
  python pmu_predict_today_v3.py --date 16042026 --strategy conformal
  python pmu_predict_today_v3.py --date 16042026 --strategy kelly_quarter --bankroll 100

Note : ce script fait l'inférence "snapshot" (pour une vraie prédiction
live, il FAUT fetcher l'API live pour les participants du jour — voir
pmu_predict_today.py pour le scaffolding fetch).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _load_models():
    feats_path = Path("pmu_model_v3.features.json")
    if not feats_path.exists():
        sys.exit("Missing pmu_model_v3.features.json. Run pmu_train_v3.py first.")
    feat_cols = json.loads(feats_path.read_text())

    bin_booster = lgb.Booster(model_file="pmu_model_v3.lgb")
    rk_path = Path("pmu_model_v3.rk.lgb")
    rk_booster = lgb.Booster(model_file=str(rk_path)) if rk_path.exists() else None

    stack = None
    if Path("pmu_stack_v3.pkl").exists():
        with open("pmu_stack_v3.pkl", "rb") as fh:
            stack = pickle.load(fh)

    calib = None
    if Path("pmu_calibrator_v3.pkl").exists():
        with open("pmu_calibrator_v3.pkl", "rb") as fh:
            calib = pickle.load(fh)

    conformal = None
    if Path("pmu_conformal_v3.json").exists():
        conformal = json.loads(Path("pmu_conformal_v3.json").read_text())

    return {
        "feat_cols":  feat_cols,
        "bin":        bin_booster,
        "rk":         rk_booster,
        "stack":      stack,
        "calib":      calib,
        "conformal":  conformal,
    }


def predict(df_features: pd.DataFrame, models: dict, alpha_meta: float = 0.5) -> pd.DataFrame:
    """Applique toute la chaîne stack + calib + conformal sur df_features."""
    feat_cols = [c for c in models["feat_cols"] if c in df_features.columns]
    X = df_features[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).fillna(0.0).values

    p_bin = models["bin"].predict(X)
    if models["rk"] is not None:
        s_rk = models["rk"].predict(X)
        df_features = df_features.copy()
        df_features["_s_rk"] = s_rk
        df_features["_s_rk_norm"] = df_features.groupby("race_id")["_s_rk"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
        meta = alpha_meta * df_features["_s_rk_norm"].values + (1 - alpha_meta) * p_bin
    else:
        meta = p_bin

    p = meta.copy()

    # Calibration
    if models["calib"]:
        cal = models["calib"]
        if cal["method"] == "isotonic":
            p = cal["model"].predict(p)
        else:
            p = cal["model"].predict(p.reshape(-1, 1))

    # Conformal bounds
    p_lo = p.copy()
    p_hi = p.copy()
    if models["conformal"]:
        q = float(models["conformal"]["q"])
        p_lo = np.maximum(p - q, 0)
        p_hi = np.minimum(p + q, 1)

    out = df_features[["race_id", "num_pmu", "drd_rapport"]].copy()
    out["p_bin"]  = p_bin
    out["p_meta"] = meta
    out["p_cal"]  = p
    out["p_lo"]   = p_lo
    out["p_hi"]   = p_hi
    return out


def recommend_bets(preds: pd.DataFrame, strategy: str = "conformal",
                   edge_margin: float = 0.05, bankroll: float = 100.0) -> pd.DataFrame:
    """Génère les paris recommandés selon la stratégie."""
    bets = []
    for race_id, grp in preds.groupby("race_id"):
        g = grp[grp["drd_rapport"] > 0].copy()
        if g.empty:
            continue

        if strategy == "conformal":
            g["edge"] = g["p_lo"] * g["drd_rapport"] - 1
            best = g.loc[g["edge"].idxmax()]
            if best["edge"] > edge_margin:
                stake = 1.0
                bets.append({**best.to_dict(), "stake": stake, "strategy": strategy})

        elif strategy == "kelly_quarter":
            g["b"] = g["drd_rapport"] - 1
            g["f"] = (g["b"] * g["p_cal"] - (1 - g["p_cal"])) / g["b"].clip(lower=0.01)
            g = g[g["f"] > 0]
            if g.empty:
                continue
            best = g.loc[g["f"].idxmax()]
            stake = max(best["f"] * 0.25 * bankroll, 0)
            if stake > 0:
                bets.append({**best.to_dict(), "stake": stake, "strategy": strategy})

        else:  # flat top1
            best = g.loc[g["p_cal"].idxmax()]
            bets.append({**best.to_dict(), "stake": 1.0, "strategy": strategy})

    return pd.DataFrame(bets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",     required=True, help="DDMMYYYY")
    ap.add_argument("--features", default="pmu_features_v3.parquet",
                    help="Parquet features v3 (ou live snapshot)")
    ap.add_argument("--strategy", default="conformal",
                    choices=["conformal", "kelly_quarter", "flat"])
    ap.add_argument("--edge-margin", type=float, default=0.05)
    ap.add_argument("--bankroll",    type=float, default=100.0)
    args = ap.parse_args()

    d = args.date
    file_date = f"{d[4:8]}-{d[2:4]}-{d[0:2]}"

    models = _load_models()

    # Filtrer les courses du jour
    print(f"Loading {args.features}...")
    df = pq.read_table(args.features).to_pandas()
    df_day = df[df["file_date"] == file_date].copy()
    if df_day.empty:
        sys.exit(f"No rows for date {file_date} in {args.features}. "
                 f"Need to fetch live with pmu_predict_today.py first.")

    print(f"  {len(df_day):,} rows for {file_date}")

    preds = predict(df_day, models)
    bets  = recommend_bets(preds, args.strategy, args.edge_margin, args.bankroll)

    print(f"\n=== Paris recommandés ({args.strategy}) ===")
    if bets.empty:
        print("Aucun pari rentable détecté.")
    else:
        print(bets[["race_id", "num_pmu", "drd_rapport", "p_cal", "p_lo", "stake"]]
              .to_string(index=False))
        print(f"\nTotal mise : {bets['stake'].sum():.2f} €")
        print(f"Gain espéré si tous gagnent : "
              f"{((bets['drd_rapport']-1) * bets['stake']).sum():.2f} €")
    return 0


if __name__ == "__main__":
    sys.exit(main())
