#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conformal prediction v3 — bornes d'incertitude non-paramétriques.

Pour chaque prédiction p (proba de gagner, calibrée), on retourne un
intervalle [p_lo, p_hi] tel que Pr(true ∈ [p_lo, p_hi]) ≥ 1 - α.

Méthode simple (inductive conformal, CP) :
  1. Sur un holdout (calibration set) on calcule les scores de non-conformité
     s_i = |p_i - y_i|
  2. q = quantile(1 - α) des s_i
  3. Pour un nouveau point p, l'intervalle est [max(0, p-q), min(1, p+q)]

Application aux paris :
  - On ne parie QUE si p_lo * (cote) > 1 + seuil_marge  (edge certain)
  - Réduit la variance, filtre les paris spéculatifs.

Reads : pmu_oof_stack_v3.parquet  + pmu_calibrator_v3.pkl (optionnel)
Writes: pmu_conformal_v3.json

Usage:
  python pmu_conformal_v3.py --alpha 0.1        # 90% coverage
  python pmu_conformal_v3.py --alpha 0.2        # 80% coverage (plus étroit)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


OOF_DEFAULT = Path("pmu_oof_stack_v3.parquet")
CAL_DEFAULT = Path("pmu_calibrator_v3.pkl")
OUT_JSON    = Path("pmu_conformal_v3.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof",     default=str(OOF_DEFAULT))
    ap.add_argument("--cal",     default=str(CAL_DEFAULT))
    ap.add_argument("--score-col", default="p_stack")
    ap.add_argument("--alpha",   type=float, default=0.1,
                    help="1-alpha = couverture voulue (0.1 ⇒ 90%)")
    ap.add_argument("--cal-frac", type=float, default=0.5)
    args = ap.parse_args()

    oof = pq.read_table(args.oof).to_pandas()

    # Appliquer calibrator si présent
    if Path(args.cal).exists():
        with open(args.cal, "rb") as fh:
            cal = pickle.load(fh)
        raw = oof[args.score_col].values
        if cal["method"] == "isotonic":
            p_cal = cal["model"].predict(raw)
        else:
            p_cal = cal["model"].predict(raw.reshape(-1, 1))
        oof["p_cal"] = p_cal
        score_col = "p_cal"
        print(f"Applied calibrator ({cal['method']})")
    else:
        score_col = args.score_col
        print("[warn] no calibrator — using raw scores")

    # Split ordre chronologique : 50% calibration, 50% test
    oof = oof.sort_values("race_id").reset_index(drop=True)
    ids = sorted(oof["race_id"].unique())
    cut = int(len(ids) * args.cal_frac)
    cal_ids = set(ids[:cut])
    cal_df  = oof[oof["race_id"].isin(cal_ids)]
    test_df = oof[~oof["race_id"].isin(cal_ids)]

    p_cal_set = cal_df[score_col].values
    y_cal     = cal_df["won"].astype(int).values
    nonconf   = np.abs(p_cal_set - y_cal)

    q = float(np.quantile(nonconf, 1 - args.alpha))
    print(f"Conformal q (alpha={args.alpha}) = {q:.4f}")

    # Coverage check on test
    p_te   = test_df[score_col].values
    y_te   = test_df["won"].astype(int).values
    lo     = np.maximum(p_te - q, 0)
    hi     = np.minimum(p_te + q, 1)
    covered = ((y_te >= lo) & (y_te <= hi)).mean()
    print(f"Empirical coverage on test : {covered*100:.1f}%  (target: {(1-args.alpha)*100:.1f}%)")

    # Betting simulation : p_lo × cote > 1 + margin
    for margin in (0.0, 0.05, 0.1, 0.15):
        cote = test_df["drd_rapport"].fillna(0).values
        ok = (cote > 0)
        edge = np.where(ok, lo * cote - 1, -99)
        bet_mask = edge > margin
        n_bets  = int(bet_mask.sum())
        if n_bets == 0:
            print(f"  margin {margin:.2f} : 0 bets")
            continue
        pnl = np.where(y_te[bet_mask] == 1,
                       cote[bet_mask] - 1,
                       -1.0)
        roi = float(pnl.mean() * 100)
        hit = float(y_te[bet_mask].mean() * 100)
        print(f"  margin {margin:.2f} : {n_bets:,} bets, hit={hit:.1f}%, ROI={roi:+.2f}%")

    report = {
        "alpha":    args.alpha,
        "q":        q,
        "coverage_empirical": round(float(covered), 4),
        "score_col": score_col,
        "n_cal":    int(len(cal_df)),
        "n_test":   int(len(test_df)),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
