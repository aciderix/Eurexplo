#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration v3 — probabilités calibrées via Isotonic + Beta.

La raw output du stack est mal calibrée (surconfiance sur les favoris).
On apprend f : p_raw → p_calibré tel que la fréquence empirique
correspond à la probabilité annoncée.

Deux méthodes :
  - Isotonic regression (monotone, non-paramétrique)
  - Beta calibration (Kull et al. 2017) — mieux adaptée aux proba
    bimodales (fav vs non-fav)

On garde celle avec le meilleur Brier score sur un holdout.

Reads : pmu_oof_stack_v3.parquet  (col p_stack + won)
Writes: pmu_calibrator_v3.pkl     ({method, model, brier_before, brier_after})

Usage:
  python pmu_calibrate_v3.py --oof pmu_oof_stack_v3.parquet
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

try:
    from betacal import BetaCalibration
    HAS_BETA = True
except ImportError:
    HAS_BETA = False


OOF_DEFAULT = Path("pmu_oof_stack_v3.parquet")
OUT_PKL    = Path("pmu_calibrator_v3.pkl")


def _reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins+1)
    idx  = np.digitize(p, bins) - 1
    idx  = np.clip(idx, 0, n_bins-1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin":    b,
            "p_low":  bins[b],
            "p_high": bins[b+1],
            "n":      int(mask.sum()),
            "p_mean": float(p[mask].mean()),
            "y_mean": float(y[mask].mean()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=str(OOF_DEFAULT))
    ap.add_argument("--score-col", default="p_stack")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    args = ap.parse_args()

    oof = pq.read_table(args.oof).to_pandas()
    oof = oof.sort_values("race_id").reset_index(drop=True)
    # Holdout temporel : on prend les DERNIERS race_ids comme test
    ids = sorted(oof["race_id"].unique())
    cut = int(len(ids) * (1 - args.holdout_frac))
    train_ids = set(ids[:cut])
    tr = oof[oof["race_id"].isin(train_ids)]
    ho = oof[~oof["race_id"].isin(train_ids)]
    print(f"Train {len(tr):,}  holdout {len(ho):,}")

    p_tr, y_tr = tr[args.score_col].values, tr["won"].astype(int).values
    p_ho, y_ho = ho[args.score_col].values, ho["won"].astype(int).values

    # ── Isotonic ──
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr, y_tr)
    p_ho_iso = iso.predict(p_ho)

    brier_raw = brier_score_loss(y_ho, p_ho)
    brier_iso = brier_score_loss(y_ho, p_ho_iso)
    ll_raw    = log_loss(y_ho, np.clip(p_ho, 1e-6, 1-1e-6))
    ll_iso    = log_loss(y_ho, np.clip(p_ho_iso, 1e-6, 1-1e-6))
    print(f"Isotonic: brier {brier_raw:.5f} → {brier_iso:.5f}  logloss {ll_raw:.4f} → {ll_iso:.4f}")

    # ── Beta ──
    method = "isotonic"
    best_model = iso
    best_brier = brier_iso
    if HAS_BETA:
        beta = BetaCalibration(parameters="abm")
        beta.fit(p_tr.reshape(-1, 1), y_tr)
        p_ho_beta = beta.predict(p_ho.reshape(-1, 1))
        brier_beta = brier_score_loss(y_ho, p_ho_beta)
        ll_beta    = log_loss(y_ho, np.clip(p_ho_beta, 1e-6, 1-1e-6))
        print(f"Beta:     brier {brier_raw:.5f} → {brier_beta:.5f}  logloss {ll_raw:.4f} → {ll_beta:.4f}")
        if brier_beta < best_brier:
            method = "beta"
            best_model = beta
            best_brier = brier_beta

    print(f"\nBest method : {method}  (brier={best_brier:.5f})")

    # Reliability diagram (text)
    rel = _reliability_table(y_ho, p_ho_iso if method == "isotonic" else p_ho)
    print("\nReliability (holdout, calibrated):")
    print(rel.to_string(index=False))

    with open(OUT_PKL, "wb") as fh:
        pickle.dump({
            "method":       method,
            "model":        best_model,
            "score_col":    args.score_col,
            "brier_raw":    float(brier_raw),
            "brier_cal":    float(best_brier),
            "holdout_frac": args.holdout_frac,
        }, fh)
    print(f"\nWrote {OUT_PKL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
