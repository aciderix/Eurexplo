#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest v3 multi-stratégies avec intervalles de confiance bootstrap.

Stratégies testées :
  - flat_top1            : 1€ sur le top du modèle
  - flat_top1_edge_X     : filtre edge ≥ X% (2, 5, 10)
  - kelly_full / kelly_1/4 : mise = fraction de Kelly × bankroll
  - top3_flat             : 1€ sur les 3 premiers
  - dutching_top3         : mise répartie pour gain constant
  - conformal_lo_margin_X : utilise borne basse conformal + marge

Intervalles de confiance : bootstrap (N=1000) sur les PnL mensuels.

Reads :
  pmu_oof_stack_v3.parquet       (+p_stack)
  pmu_calibrator_v3.pkl          (optionnel)
  pmu_conformal_v3.json          (optionnel)

Writes:
  pmu_backtest_v3.csv
  pmu_backtest_v3.png            (courbes PnL)
  pmu_backtest_v3.json           (stats résumées)

Usage:
  python pmu_backtest_v3.py --folds 12 --bootstrap 1000
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


OOF        = Path("pmu_oof_stack_v3.parquet")
CAL        = Path("pmu_calibrator_v3.pkl")
CONF       = Path("pmu_conformal_v3.json")
OUT_CSV    = Path("pmu_backtest_v3.csv")
OUT_PNG    = Path("pmu_backtest_v3.png")
OUT_JSON   = Path("pmu_backtest_v3.json")


# ── Stratégies ────────────────────────────────────────────────────────────────

def flat_top1(race: pd.DataFrame, edge: float = 0.0) -> list[dict]:
    g = race[race["cote"] > 0]
    if g.empty:
        return []
    best = g.loc[g["p"].idxmax()]
    if edge > 0:
        e = best["p"] - 1 / best["cote"]
        if e <= edge:
            return []
    return [{"num": int(best["num_pmu"]), "stake": 1.0, "cote": float(best["cote"]),
             "won": bool(best["won"]), "p": float(best["p"])}]


def kelly(race: pd.DataFrame, frac: float = 1.0, bankroll: float = 1.0) -> list[dict]:
    """Kelly simple : f = (b*p - q) / b où b = cote-1, q = 1-p."""
    g = race[race["cote"] > 0].copy()
    g["b"] = g["cote"] - 1
    g["f"] = (g["b"] * g["p"] - (1 - g["p"])) / g["b"].clip(lower=0.01)
    g = g[g["f"] > 0]
    if g.empty:
        return []
    best = g.loc[g["f"].idxmax()]
    stake = float(max(best["f"] * frac * bankroll, 0.0))
    if stake <= 0:
        return []
    return [{"num": int(best["num_pmu"]), "stake": stake, "cote": float(best["cote"]),
             "won": bool(best["won"]), "p": float(best["p"])}]


def top3_flat(race: pd.DataFrame) -> list[dict]:
    g = race[race["cote"] > 0].sort_values("p", ascending=False).head(3)
    out = []
    for _, r in g.iterrows():
        out.append({"num": int(r["num_pmu"]), "stake": 1.0, "cote": float(r["cote"]),
                    "won": bool(r["won"]), "p": float(r["p"])})
    return out


def dutching_top3(race: pd.DataFrame, budget: float = 3.0) -> list[dict]:
    g = race[race["cote"] > 0].sort_values("p", ascending=False).head(3).copy()
    if g.empty:
        return []
    # Mise proportionnelle à 1/cote pour gain constant si l'un gagne
    w = 1.0 / g["cote"].values
    w = w / w.sum()
    g["stake"] = budget * w
    out = []
    for _, r in g.iterrows():
        out.append({"num": int(r["num_pmu"]), "stake": float(r["stake"]),
                    "cote": float(r["cote"]), "won": bool(r["won"]), "p": float(r["p"])})
    return out


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_strategy(oof: pd.DataFrame, name: str, fn, **kwargs) -> pd.DataFrame:
    bets = []
    for race_id, grp in oof.groupby("race_id", sort=False):
        race = grp.rename(columns={"drd_rapport": "cote"})
        for b in fn(race, **kwargs):
            pnl = (b["cote"] - 1) * b["stake"] if b["won"] else -b["stake"]
            bets.append({
                "race_id":   race_id,
                "ym":        grp["ym"].iloc[0],
                "strategy":  name,
                "stake":     b["stake"],
                "cote":      b["cote"],
                "won":       b["won"],
                "pnl":       pnl,
            })
    return pd.DataFrame(bets)


def aggregate(bets: pd.DataFrame, name: str, n_boot: int, rng) -> dict:
    if bets.empty:
        return {"strategy": name, "n_bets": 0, "roi": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "hitrate": 0.0, "total_stake": 0.0, "total_pnl": 0.0}
    n = len(bets)
    stake = bets["stake"].sum()
    pnl   = bets["pnl"].sum()
    roi   = pnl / stake * 100

    # Bootstrap sur les paris
    roll = np.arange(n)
    rois = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = bets["stake"].values[idx].sum()
        p = bets["pnl"].values[idx].sum()
        if s > 0:
            rois.append(p / s * 100)
    if rois:
        ci_lo = float(np.percentile(rois, 2.5))
        ci_hi = float(np.percentile(rois, 97.5))
    else:
        ci_lo = ci_hi = float(roi)

    return {
        "strategy": name,
        "n_bets":   int(n),
        "total_stake": round(float(stake), 2),
        "total_pnl":   round(float(pnl), 2),
        "roi":         round(float(roi), 2),
        "ci_lo":       round(ci_lo, 2),
        "ci_hi":       round(ci_hi, 2),
        "hitrate":     round(float(bets["won"].mean() * 100), 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof",        default=str(OOF))
    ap.add_argument("--cal",        default=str(CAL))
    ap.add_argument("--score-col",  default="p_stack")
    ap.add_argument("--bootstrap",  type=int, default=1000)
    ap.add_argument("--folds",      type=int, default=12,
                    help="nb mois récents à montrer dans la courbe (pour affichage)")
    args = ap.parse_args()

    oof = pq.read_table(args.oof).to_pandas()
    # Déterminer year_month : priorité file_date > year_month existant > race_id
    if "file_date" in oof.columns:
        oof["ym"] = oof["file_date"].astype(str).str[:7]
    elif "year_month" in oof.columns:
        oof["ym"] = oof["year_month"]
    else:
        # race_id format : YYYY-MM-DD_R_C → [:7] = YYYY-MM
        oof["ym"] = oof["race_id"].astype(str).str[:7]

    # Calibration si dispo
    if Path(args.cal).exists():
        with open(args.cal, "rb") as fh:
            cal = pickle.load(fh)
        raw = oof[args.score_col].values
        if cal["method"] == "isotonic":
            oof["p"] = cal["model"].predict(raw)
        else:
            oof["p"] = cal["model"].predict(raw.reshape(-1, 1))
        print(f"Applied calibrator ({cal['method']})")
    else:
        oof["p"] = oof[args.score_col]

    # Normaliser p par course (somme = 1)
    oof["p_raw"] = oof["p"]
    oof["p"] = oof.groupby("race_id")["p_raw"].transform(
        lambda x: x / (x.sum() + 1e-9)
    )

    rng = np.random.default_rng(42)
    all_bets: dict[str, pd.DataFrame] = {}
    reports = []

    strategies = [
        ("flat_top1",           flat_top1, {}),
        ("flat_top1_edge2",     flat_top1, {"edge": 0.02}),
        ("flat_top1_edge5",     flat_top1, {"edge": 0.05}),
        ("flat_top1_edge10",    flat_top1, {"edge": 0.10}),
        ("kelly_full",          kelly,     {"frac": 1.0}),
        ("kelly_quarter",       kelly,     {"frac": 0.25}),
        ("top3_flat",           top3_flat, {}),
        ("dutching_top3",       dutching_top3, {"budget": 3.0}),
    ]

    for name, fn, kw in strategies:
        bets = simulate_strategy(oof, name, fn, **kw)
        all_bets[name] = bets
        r = aggregate(bets, name, args.bootstrap, rng)
        reports.append(r)
        print(f"  {name:24s}  n={r['n_bets']:>6}  ROI={r['roi']:+.2f}%  "
              f"CI95=[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  hit={r['hitrate']:.1f}%")

    # ── Save ──
    rep_df = pd.DataFrame(reports).sort_values("roi", ascending=False)
    rep_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    OUT_JSON.write_text(json.dumps(reports, indent=2))
    print(f"Wrote {OUT_JSON}")

    # ── Plot (optionnel si matplotlib dispo) ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        for name in ["flat_top1", "flat_top1_edge5", "kelly_quarter", "dutching_top3"]:
            bets = all_bets[name].sort_values("race_id")
            if bets.empty:
                continue
            cum = bets["pnl"].cumsum()
            ax.plot(range(len(cum)), cum.values, label=name)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Pari #")
        ax.set_ylabel("PnL cumulatif (€)")
        ax.set_title(f"PMU v3 backtest — {len(oof):,} observations")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=120)
        print(f"Wrote {OUT_PNG}")
    except ImportError:
        print("[info] matplotlib non dispo — skip plot")

    return 0


if __name__ == "__main__":
    sys.exit(main())
