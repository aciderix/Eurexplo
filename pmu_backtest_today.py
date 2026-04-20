#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest "predict_today" sur les courses du jour :
  1. Fetch le programme + participants du jour (via API PMU)
  2. Applique le modèle pmu_model_v2.lgb + lookups ELO/rolling/TE
     (lookups construits à partir de pmu_dataset_v2.parquet qui s'arrête
     au 14/04/2026 donc **pas de leakage des résultats du jour**)
  3. Pour chaque course ayant un `ordreArrivee`, on compare le top-pick
     prédit au gagnant réel et on calcule le PnL (mise 1€ flat).
  4. Résumé ROI + couverture.

Usage:
  python pmu_backtest_today.py                  # date = aujourd'hui
  python pmu_backtest_today.py --date 15042026  # date explicite DDMMYYYY
  python pmu_backtest_today.py --min-edge 0.03  # seulement paris à edge>3%
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

# Réutilise tout le code live de pmu_predict_today.py
sys.path.insert(0, str(Path(__file__).parent))
from pmu_predict_today import (  # type: ignore
    BASE_URL, HEADERS, MODEL_PATH, FEATS_PATH,
    fetch, fetch_day,
    build_elo_lookup, build_rolling_lookup, build_te_lookup,
    build_live_features,
)


def fetch_results(date_str: str, race_ids: list[str]) -> dict[str, list[int]]:
    """Pour chaque race_id (=DDMMYYYY_RX_CY), retourne la liste ordonnée
    des num_pmu à l'arrivée (1er, 2e, 3e, ...).  Vide si course non courue.
    Le champ `ordreArrivee` est dans le JSON course lui-même.
    """
    session = requests.Session()
    out: dict[str, list[int]] = {}
    for rid in race_ids:
        try:
            _, R, C = rid.split("_")
            url = f"{BASE_URL}/{date_str}/{R}/{C}"
            data = fetch(url, session)
        except Exception:
            data = None
        if not data or not data.get("arriveeDefinitive"):
            out[rid] = []
            continue
        ordre = data.get("ordreArrivee") or []
        # ordre = [[14],[6],...] ou parfois [[5,2]] pour dead-heat
        arrivee = [int(x[0]) if isinstance(x, list) and x else int(x) for x in ordre]
        out[rid] = arrivee
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="DDMMYYYY (défaut: aujourd'hui)")
    ap.add_argument("--min-edge", type=float, default=0.0)
    ap.add_argument("--strategy", choices=["top1", "top1_value"],
                    default="top1_value")
    args = ap.parse_args()

    date_str = args.date or datetime.now().strftime("%d%m%Y")
    print(f"\nBacktest predict_today — {date_str}\n" + "=" * 70)

    # Chargement modèle + features + lookups (depuis dataset ≤ 14/04 donc no leak)
    model = lgb.Booster(model_file=str(MODEL_PATH))
    feat_names = json.loads(FEATS_PATH.read_text(encoding="utf-8"))
    print(f"Modèle chargé : {len(feat_names)} features  (coupure train=2025)")

    t0 = time.time()
    elo_lkp  = build_elo_lookup()
    roll_lkp = build_rolling_lookup()
    te_lkp   = build_te_lookup()
    print(f"Lookups construits en {time.time()-t0:.1f}s\n")

    # Fetch du programme du jour (live API)
    print(f"Fetch programme {date_str} ...")
    t1 = time.time()
    df_raw = fetch_day(date_str)
    if df_raw is None or len(df_raw) == 0:
        print("Aucune donnée.")
        return 1
    print(f"  {len(df_raw)} participants / {df_raw['race_id'].nunique()} courses "
          f"({time.time()-t1:.1f}s)")

    # Feature engineering + prédiction
    print("\nFeature engineering + prédiction...")
    df_fe = build_live_features(df_raw, elo_lkp, roll_lkp, te_lkp, feat_names)
    X = df_fe[feat_names].astype(np.float32).fillna(0).values
    df_raw["proba"] = model.predict(X)
    df_raw["cote"]  = df_raw["drd_rapport"].fillna(0.0)
    df_raw["edge"]  = df_raw["proba"] - 1.0 / df_raw["cote"].clip(lower=0.1)

    # Fetch résultats réels (ordre d'arrivée)
    print("\nFetch résultats réels des courses déjà courues...")
    race_ids = sorted(df_raw["race_id"].unique())
    results  = fetch_results(date_str, race_ids)
    n_run    = sum(1 for v in results.values() if v)
    print(f"  {n_run}/{len(race_ids)} courses avec arrivée connue")

    # Comparaison
    print("\n" + "=" * 90)
    print(f"{'RACE':<10} {'PICK#':>5} {'NOM':<22} {'COTE':>5} {'PROB':>5} "
          f"{'EDGE':>6}  {'WIN#':>4} {'HIT':>4}   PnL")
    print("=" * 90)

    rows = []
    for rid, grp in df_raw.groupby("race_id", sort=True):
        arrivee = results.get(rid, [])
        if not arrivee:
            continue
        winner_num = arrivee[0]

        grp = grp.sort_values("proba", ascending=False)
        valid = grp[grp["cote"] > 0].copy()
        if valid.empty:
            continue

        # Stratégie
        if args.strategy == "top1_value":
            pick = valid[valid["edge"] > args.min_edge]
            if pick.empty:
                # Pas de pari: skip
                continue
            pick = pick.sort_values("proba", ascending=False).iloc[0]
        else:  # top1
            pick = valid.iloc[0]

        hit = int(pick["num_pmu"]) == int(winner_num)
        pnl = (pick["cote"] - 1.0) if hit else -1.0
        rr, cc = rid.split("_")[1], rid.split("_")[2]
        label = f"{rr}{cc}"

        rows.append({
            "race": label,
            "pick_num": int(pick["num_pmu"]),
            "pick_nom": str(pick["nom"]),
            "cote":    pick["cote"],
            "proba":   pick["proba"],
            "edge":    pick["edge"],
            "winner":  winner_num,
            "hit":     hit,
            "pnl":     pnl,
        })

        mark = "✓" if hit else "✗"
        print(f"{label:<10} {int(pick['num_pmu']):>5} {str(pick['nom'])[:22]:<22} "
              f"{pick['cote']:>5.1f} {pick['proba']:>5.2f} {pick['edge']:>+6.3f}  "
              f"{winner_num:>4} {mark:>4}   {pnl:+6.2f}")

    if not rows:
        print("\nAucun pari pour cette stratégie/seuil.")
        return 0

    r = pd.DataFrame(rows)
    total_stake = len(r)
    total_pnl   = r["pnl"].sum()
    wins        = int(r["hit"].sum())
    print("\n" + "=" * 90)
    print(f"STRATÉGIE : {args.strategy}   min_edge={args.min_edge}")
    print(f"  Paris          : {total_stake}")
    print(f"  Victoires      : {wins}  ({wins/total_stake*100:.1f}%)")
    print(f"  Mise totale    : {total_stake:.2f} €")
    print(f"  PnL            : {total_pnl:+.2f} €")
    print(f"  ROI            : {total_pnl/total_stake*100:+.1f} %")
    print(f"  Cote moy. gagn.: {r[r['hit']]['cote'].mean() if wins else 0:.2f}")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    sys.exit(main())
