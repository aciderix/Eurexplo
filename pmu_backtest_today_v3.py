#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest V3 en mode "predict_today" — fetch live depuis l'API PMU et
construit toutes les features v3 strictement pré-course, applique la chaîne
(LGB binary + LGB ranker blend → calibration → conformal), puis compare au
résultat réel pour mesurer le ROI.

Sans leak :
- Lookups (ELO, ELO ctx, rolling, TE, H2H) construits depuis les parquets
  historiques `pmu_dataset_v2.parquet` qui s'arrêtent au 14/04/2026.
- Les blobs (pronostics, masse_enjeu, performances) sont fetchés depuis
  l'API en live — les endpoints `performances-detaillees/pretty` et
  `pronostics*` ne renvoient que l'historique antérieur à la course.
- L'ordre d'arrivée n'est utilisé QUE pour la comparaison post-hoc.

Usage:
  python pmu_backtest_today_v3.py                   # date = aujourd'hui
  python pmu_backtest_today_v3.py --date 19042026   # DDMMYYYY
  python pmu_backtest_today_v3.py --date 19042026 --min-edge 0.05
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

sys.path.insert(0, str(Path(__file__).parent))
from pmu_predict_today import (  # type: ignore
    BASE_URL, HEADERS,
    fetch, fetch_day,
    build_elo_lookup, build_rolling_lookup, build_te_lookup,
    build_live_features,
)
from pmu_feat_blobs import (  # type: ignore
    parse_masse_enjeu, parse_pronostics, parse_performances_by_horse,
)
from pmu_feat_musique_seq import (  # type: ignore
    parse_musique as parse_musique_seq,
    features_from_seq, DISC_MAP,
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_PARQUET   = Path("pmu_dataset_v2.parquet")
ELO_CTX_PATH   = Path("pmu_feat_elo_ctx_v3.parquet")
H2H_PATH       = Path("pmu_feat_h2h_v3.parquet")
BIN_MODEL_PATH = Path("pmu_model_v3.lgb")
RK_MODEL_PATH  = Path("pmu_model_v3.rk.lgb")
FEATS_PATH     = Path("pmu_model_v3.features.json")
CAL_PATH       = Path("pmu_calibrator_v3.pkl")
CONF_PATH      = Path("pmu_conformal_v3.json")
BEST_PATH      = Path("pmu_best_params_v3.json")


# ── V3-specific lookup builders ──────────────────────────────────────────────

def build_elo_ctx_lookup() -> dict[str, dict]:
    """Dernier état ELO-contextuel par entité, toutes disciplines confondues.
    Simplification: on prend le dernier état sans filtrer par (discipline,
    distance_bucket) — le modèle apprend ça implicitement via la feature
    `discipline`/`distance` encodée ailleurs."""
    print("Chargement ELO ctx lookup...")
    cols = ["race_id", "num_pmu",
            "elo_ctx_horse_pre", "elo_ctx_driver_pre", "elo_ctx_entraineur_pre",
            "elo_ctx_horse_ngames", "elo_ctx_driver_ngames", "elo_ctx_entraineur_ngames"]
    ctx = pq.read_table(ELO_CTX_PATH, columns=cols).to_pandas()
    base = pq.read_table(BASE_PARQUET,
                         columns=["race_id", "num_pmu", "file_date",
                                  "nom", "driver", "entraineur"]).to_pandas()
    m = ctx.merge(base, on=["race_id", "num_pmu"]).sort_values("file_date")

    return {
        "horse":      m.groupby("nom")[["elo_ctx_horse_pre", "elo_ctx_horse_ngames"]].last().to_dict("index"),
        "driver":     m.groupby("driver")[["elo_ctx_driver_pre", "elo_ctx_driver_ngames"]].last().to_dict("index"),
        "entraineur": m.groupby("entraineur")[["elo_ctx_entraineur_pre", "elo_ctx_entraineur_ngames"]].last().to_dict("index"),
    }


def build_h2h_lookup() -> dict[str, dict]:
    """Dernier état H2H agrégé par cheval (moyenne de ses stats face à
    tous les adversaires rencontrés). Approximation live — dans le train,
    les stats sont calculées face aux adversaires PRÉSENTS dans chaque
    course. Ici on prend le dernier aggregate connu du cheval; c'est un
    proxy raisonnable quand il y a peu de surface disponible."""
    print("Chargement H2H lookup...")
    cols = ["race_id", "num_pmu",
            "h2h_n_meets_sum", "h2h_wins_sum",
            "h2h_winrate_mean", "h2h_winrate_max", "h2h_winrate_min",
            "h2h_n_opponents_known"]
    h2h = pq.read_table(H2H_PATH, columns=cols).to_pandas()
    base = pq.read_table(BASE_PARQUET,
                         columns=["race_id", "num_pmu", "file_date", "nom"]).to_pandas()
    m = h2h.merge(base, on=["race_id", "num_pmu"]).sort_values("file_date")
    h2h_cols = [c for c in cols if c.startswith("h2h_")]
    return {"horse": m.groupby("nom")[h2h_cols].last().to_dict("index")}


# ── Live fetch des blobs (endpoints API) ─────────────────────────────────────

def fetch_course_blobs(date_str: str, race_ids: list[str]) -> dict[str, dict]:
    """Pour chaque race_id = DDMMYYYY_RX_CY, fetch les 3 endpoints
    qui alimentent les blob_* features (les pronostics, la masse d'enjeu,
    les performances détaillées)."""
    session = requests.Session()
    out: dict[str, dict] = {}
    for rid in race_ids:
        try:
            _, R, C = rid.split("_")
            base = f"{BASE_URL}/{date_str}/{R}/{C}"
        except Exception:
            continue
        pron  = fetch(f"{base}/pronostics-detailles", session) or fetch(f"{base}/pronostics", session)
        # Live: per-horse mises via /combinaisons/E_SIMPLE_GAGNANT (et non /masse-enjeu qui ne donne que les totaux)
        mass  = fetch(f"{base}/combinaisons/E_SIMPLE_GAGNANT", session) or fetch(f"{base}/masse-enjeu", session)
        perfs = fetch(f"{base}/performances-detaillees/pretty", session)
        out[rid] = {
            "pronostics_json":            json.dumps(pron)  if pron  else None,
            "masse_enjeu_json":           json.dumps(mass)  if mass  else None,
            "performances_detaillees_json": json.dumps(perfs) if perfs else None,
        }
    return out


# ── Add v3-only features on top of v2 live features ─────────────────────────

def add_v3_features(df: pd.DataFrame, blobs: dict[str, dict],
                    elo_ctx_lkp: dict, h2h_lkp: dict) -> pd.DataFrame:
    df = df.copy()

    # ── mus_seq (from musique string + discipline) ──
    mus_seq_rows = []
    for mus, disc in zip(df["musique"], df.get("discipline", [""] * len(df))):
        seq = parse_musique_seq(mus)
        dref = DISC_MAP.get((disc or "").upper()) if isinstance(disc, str) else None
        mus_seq_rows.append(features_from_seq(seq, dref))
    mus_df = pd.DataFrame(mus_seq_rows, index=df.index)
    for c in mus_df.columns:
        df[c] = mus_df[c]

    # ── blobs (live, pre-race parse) ──
    blob_rows: list[dict] = []
    for rid, grp in df.groupby("race_id"):
        b = blobs.get(rid, {}) or {}
        file_date = grp["file_date"].iloc[0]
        pron  = parse_pronostics(b.get("pronostics_json"))
        mass  = parse_masse_enjeu(b.get("masse_enjeu_json"))
        perfs = parse_performances_by_horse(b.get("performances_detaillees_json"),
                                            current_date=file_date)
        for _, p in grp.iterrows():
            num = int(p["num_pmu"])
            nom = p["nom"]
            pr  = pron.get(num, {})
            hp  = perfs.get(nom, {})
            blob_rows.append({
                "race_id":                   rid,
                "num_pmu":                   num,
                "blob_masse_share":          mass.get(num, np.nan),
                "blob_prono_n_tips":         pr.get("prono_n_tips", 0),
                "blob_prono_rank_min":       pr.get("prono_rank_min", np.nan),
                "blob_prono_rank_mean":      pr.get("prono_rank_mean", np.nan),
                "blob_prono_top1_cnt":       pr.get("prono_top1_cnt", 0),
                "blob_prono_top3_cnt":       pr.get("prono_top3_cnt", 0),
                "blob_perf_n_past":          hp.get("blob_perf_n_past", 0),
                "blob_perf_place_mean":      hp.get("blob_perf_place_mean", np.nan),
                "blob_perf_place_last3":     hp.get("blob_perf_place_last3", np.nan),
                "blob_perf_reduction_best":  hp.get("blob_perf_reduction_best", np.nan),
                "blob_perf_reduction_last3": hp.get("blob_perf_reduction_last3", np.nan),
                "blob_perf_allocation_mean": hp.get("blob_perf_allocation_mean", np.nan),
                "blob_perf_temps_mean":      hp.get("blob_perf_temps_mean", np.nan),
            })
    if blob_rows:
        blob_df = pd.DataFrame(blob_rows)
        # Align dtypes with df to avoid silent merge failure
        blob_df["num_pmu"] = blob_df["num_pmu"].astype("int64")
        df["num_pmu"] = df["num_pmu"].astype("int64")
        blob_df["race_id"] = blob_df["race_id"].astype(str)
        df["race_id"] = df["race_id"].astype(str)
        # Ranks per race
        g = blob_df.groupby("race_id")
        blob_df["blob_masse_share_rank"] = g["blob_masse_share"].rank(method="min", ascending=False)
        blob_df["blob_prono_top1_rank"]  = g["blob_prono_top1_cnt"].rank(method="min", ascending=False)
        blob_df["blob_perf_place_rank"]  = g["blob_perf_place_mean"].rank(method="min", ascending=True)
        # Drop blob_* columns pre-filled as NaN by build_live_features (avoids _x/_y suffix)
        to_drop = [c for c in df.columns if c.startswith("blob_") and c in blob_df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
        df = df.merge(blob_df, on=["race_id", "num_pmu"], how="left")
        # Sanity: count how many rows still NaN on a core blob col
        _n_total = len(df)
        _n_nan = int(df["blob_masse_share"].isna().sum())
        if _n_nan == _n_total:
            print(f"[warn] blob merge: 100% NaN ({_n_nan}/{_n_total}) — check dtype/keys")
        else:
            print(f"[info] blob merge: {_n_total - _n_nan}/{_n_total} rows hydratées")

    # ── ELO ctx lookup ──
    for ent, key, pre_col, n_col in [
        ("horse",      "nom",        "elo_ctx_horse_pre",      "elo_ctx_horse_ngames"),
        ("driver",     "driver",     "elo_ctx_driver_pre",     "elo_ctx_driver_ngames"),
        ("entraineur", "entraineur", "elo_ctx_entraineur_pre", "elo_ctx_entraineur_ngames"),
    ]:
        lkp = elo_ctx_lkp.get(ent, {})
        df[pre_col] = df[key].map(lambda v: lkp.get(v, {}).get(pre_col, 1500.0) if v else 1500.0)
        df[n_col]   = df[key].map(lambda v: lkp.get(v, {}).get(n_col, 0) if v else 0)

    # vs_field: elo - moyenne elo du field dans cette course
    for col, out in [("elo_ctx_horse_pre",  "elo_ctx_horse_vs_field"),
                     ("elo_ctx_driver_pre", "elo_ctx_driver_vs_field")]:
        field_mean = df.groupby("race_id")[col].transform("mean")
        df[out] = df[col] - field_mean

    # ── H2H lookup (par cheval, agrégat latest) ──
    lkp_h = h2h_lkp.get("horse", {})
    for c in ["h2h_n_meets_sum", "h2h_wins_sum",
              "h2h_winrate_mean", "h2h_winrate_max", "h2h_winrate_min",
              "h2h_n_opponents_known"]:
        default = 0 if c.startswith("h2h_n_") or c == "h2h_wins_sum" else np.nan
        df[c] = df["nom"].map(lambda v: lkp_h.get(v, {}).get(c, default) if v else default)

    return df


# ── V3 prediction chain (LGB bin + rk blend → calib → conformal) ─────────────

def predict_v3(df_fe: pd.DataFrame, feat_cols: list[str], bin_m, rk_m,
               calib: dict | None, conformal: dict | None,
               alpha_meta: float) -> pd.DataFrame:
    # LGB gère NaN nativement; fillna(0) serait un biais (0 ≠ missing pour LGB)
    X = df_fe[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).values
    p_bin = bin_m.predict(X)
    if rk_m is not None:
        s_rk = rk_m.predict(X)
        df_fe = df_fe.copy()
        df_fe["_s_rk"] = s_rk
        df_fe["_s_rk_norm"] = df_fe.groupby("race_id")["_s_rk"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
        meta = alpha_meta * df_fe["_s_rk_norm"].values + (1 - alpha_meta) * p_bin
    else:
        meta = p_bin

    p = meta.copy()
    if calib:
        if calib["method"] == "isotonic":
            p = calib["model"].predict(p)
        else:
            p = calib["model"].predict(p.reshape(-1, 1))

    p_lo = p.copy()
    if conformal:
        q = float(conformal["q"])
        p_lo = np.maximum(p - q, 0)

    out = df_fe[["race_id", "num_pmu", "nom", "drd_rapport"]].copy()
    out["p_bin"]  = p_bin
    out["p_meta"] = meta
    out["p_cal"]  = p
    out["p_lo"]   = p_lo
    out["edge"]   = out["p_cal"] * out["drd_rapport"].fillna(0) - 1.0
    return out


# ── Récupération de l'ordre d'arrivée (post-hoc, pour compare seulement) ─────

def fetch_results(date_str: str, race_ids: list[str],
                  max_retries: int = 3, backoff: float = 2.0) -> dict[str, list[int]]:
    """Fetch l'ordre d'arrivée. Retry + backoff sur 503 (rate limit PMU)."""
    session = requests.Session()
    out: dict[str, list[int]] = {}
    for rid in race_ids:
        try:
            _, R, C = rid.split("_")
            url = f"{BASE_URL}/{date_str}/{R}/{C}"
        except Exception:
            out[rid] = []
            continue
        data = None
        for attempt in range(max_retries):
            try:
                r = session.get(url, timeout=15, headers=HEADERS)
                if r.status_code == 200:
                    data = r.json()
                    break
                if r.status_code == 503:
                    time.sleep(backoff * (2 ** attempt))
                    continue
                break
            except Exception:
                time.sleep(backoff)
        if not data or not data.get("arriveeDefinitive"):
            out[rid] = []
            continue
        ordre = data.get("ordreArrivee") or []
        arrivee = [int(x[0]) if isinstance(x, list) and x else int(x) for x in ordre]
        out[rid] = arrivee
        time.sleep(0.15)  # throttle doux pour éviter re-blocking
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",     default=None, help="DDMMYYYY (défaut: aujourd'hui)")
    ap.add_argument("--min-edge", type=float, default=0.0,
                    help="Edge min (p_cal*cote - 1) pour filtrer paris")
    ap.add_argument("--use-plo",  action="store_true",
                    help="Filter edge via p_lo (conformal lower bound) au lieu de p_cal")
    ap.add_argument("--predict-only", action="store_true",
                    help="Pas de fetch arrivée, juste les picks pré-course")
    args = ap.parse_args()

    date_str = args.date or datetime.now().strftime("%d%m%Y")
    print(f"\nBacktest V3 predict_today — {date_str}\n" + "=" * 70)

    # ── Load models ──
    feat_cols = json.loads(FEATS_PATH.read_text())
    bin_m = lgb.Booster(model_file=str(BIN_MODEL_PATH))
    rk_m  = lgb.Booster(model_file=str(RK_MODEL_PATH)) if RK_MODEL_PATH.exists() else None
    calib = pickle.loads(CAL_PATH.read_bytes()) if CAL_PATH.exists() else None
    conformal = json.loads(CONF_PATH.read_text()) if CONF_PATH.exists() else None
    best = json.loads(BEST_PATH.read_text()) if BEST_PATH.exists() else {}
    alpha_meta = float(best.get("alpha_meta", 0.5))
    print(f"Modèle v3: {len(feat_cols)} features | "
          f"bin+rank alpha_meta={alpha_meta:.3f} | "
          f"calib={'ok' if calib else 'off'} | "
          f"conformal={'ok' if conformal else 'off'}")

    # ── V2 lookups (base features) ──
    t0 = time.time()
    elo_lkp     = build_elo_lookup()
    roll_lkp    = build_rolling_lookup()
    te_lkp      = build_te_lookup()
    elo_ctx_lkp = build_elo_ctx_lookup()
    h2h_lkp     = build_h2h_lookup()
    print(f"Lookups prêts ({time.time()-t0:.1f}s, pas de leak: parquets coupés au 14/04/2026)\n")

    # ── Fetch programme du jour ──
    print(f"Fetch programme {date_str}...")
    t1 = time.time()
    df_raw = fetch_day(date_str)
    if df_raw is None or len(df_raw) == 0:
        print("Aucune donnée.")
        return 1
    print(f"  {len(df_raw)} participants / {df_raw['race_id'].nunique()} courses "
          f"({time.time()-t1:.1f}s)")

    # ── Fetch blobs (pronostics, masse, performances) ──
    race_ids = sorted(df_raw["race_id"].unique())
    print(f"\nFetch {len(race_ids)} blobs live (pronostics / masse / performances)...")
    t2 = time.time()
    blobs = fetch_course_blobs(date_str, race_ids)
    print(f"  blobs ok ({time.time()-t2:.0f}s)")

    # ── V2 live features ──
    print("\nFeature engineering v2...")
    df_fe = build_live_features(df_raw, elo_lkp, roll_lkp, te_lkp, feat_cols)

    # ── V3 features (mus_seq, blobs, elo_ctx, h2h) ──
    print("Feature engineering v3...")
    df_fe = add_v3_features(df_fe, blobs, elo_ctx_lkp, h2h_lkp)

    # Coverage diagnostic
    missing = [f for f in feat_cols if f not in df_fe.columns]
    if missing:
        print(f"  [warn] {len(missing)} features manquantes (mis à NaN): {missing[:5]}...")
        for f in missing:
            df_fe[f] = np.nan

    # ── Prédiction ──
    preds = predict_v3(df_fe, feat_cols, bin_m, rk_m, calib, conformal, alpha_meta)

    # Align drd_rapport to raw (live data)
    preds = preds.merge(df_raw[["race_id", "num_pmu", "reunion_num", "course_num"]],
                        on=["race_id", "num_pmu"], how="left")

    # ── Fetch ordre d'arrivée (sauf predict-only) ──
    if args.predict_only:
        print("\n[predict-only] Pas de fetch d'arrivée, picks pré-course uniquement.")
        results: dict[str, list[int]] = {}
    else:
        print("\nFetch ordres d'arrivée...")
        results = fetch_results(date_str, race_ids)
        n_run = sum(1 for v in results.values() if v)
        print(f"  {n_run}/{len(race_ids)} courses terminées")

    # ── Picks / Compare ──
    print("\n" + "=" * 95)
    if args.predict_only:
        print(f"{'RACE':<8} {'#':>3} {'NOM':<22} {'COTE':>5} {'p_cal':>6} {'p_lo':>6} {'EDGE':>6}")
    else:
        print(f"{'RACE':<8} {'#':>3} {'NOM':<22} {'COTE':>5} {'p_cal':>6} {'p_lo':>6} "
              f"{'EDGE':>6}  {'WIN#':>4} {'HIT':>4}   PnL")
    print("=" * 95)

    rows = []
    for rid, grp in preds.groupby("race_id", sort=True):
        arrivee = results.get(rid, [])
        if not arrivee and not args.predict_only:
            continue
        winner = arrivee[0] if arrivee else None

        valid = grp[grp["drd_rapport"].fillna(0) > 0].copy()
        if valid.empty:
            continue

        if args.use_plo:
            valid["edge_score"] = valid["p_lo"] * valid["drd_rapport"] - 1.0
        else:
            valid["edge_score"] = valid["edge"]

        if args.min_edge > 0:
            candidates = valid[valid["edge_score"] > args.min_edge]
            if candidates.empty:
                continue
        else:
            candidates = valid
        pick = candidates.sort_values("p_cal", ascending=False).iloc[0]

        r, c = rid.split("_")[1], rid.split("_")[2]
        label = f"{r}{c}"

        if args.predict_only:
            rows.append({
                "race": label,
                "pick_num": int(pick["num_pmu"]),
                "pick_nom": str(pick["nom"]),
                "cote":    pick["drd_rapport"],
                "p_cal":   pick["p_cal"],
                "p_lo":    pick["p_lo"],
                "edge":    pick["edge"],
            })
            print(f"{label:<8} {int(pick['num_pmu']):>3} {str(pick['nom'])[:22]:<22} "
                  f"{float(pick['drd_rapport']):>5.1f} {float(pick['p_cal']):>6.3f} "
                  f"{float(pick['p_lo']):>6.3f} {float(pick['edge']):>+6.3f}")
            continue

        hit = int(pick["num_pmu"]) == int(winner)
        pnl = (pick["drd_rapport"] - 1.0) if hit else -1.0

        rows.append({
            "race": label,
            "pick_num": int(pick["num_pmu"]),
            "pick_nom": str(pick["nom"]),
            "cote":    pick["drd_rapport"],
            "p_cal":   pick["p_cal"],
            "p_lo":    pick["p_lo"],
            "edge":    pick["edge"],
            "winner":  int(winner),
            "hit":     hit,
            "pnl":     pnl,
        })

        mark = "✓" if hit else "✗"
        print(f"{label:<8} {int(pick['num_pmu']):>3} {str(pick['nom'])[:22]:<22} "
              f"{float(pick['drd_rapport']):>5.1f} {float(pick['p_cal']):>6.3f} "
              f"{float(pick['p_lo']):>6.3f} {float(pick['edge']):>+6.3f}  "
              f"{int(winner):>4} {mark:>4}   {pnl:+6.2f}")

    if not rows:
        print("\nAucun pari (pas de course finie, ou filtre edge trop strict).")
        return 0

    if args.predict_only:
        r = pd.DataFrame(rows)
        print("\n" + "=" * 95)
        print(f"PREDICT v3 | min_edge={args.min_edge}  (use_plo={args.use_plo})")
        print(f"  Paris      : {len(r)}")
        print(f"  Cote moy.  : {r['cote'].mean():.2f}")
        print(f"  p_cal moy. : {r['p_cal'].mean():.3f}")
        print(f"  edge moy.  : {r['edge'].mean():+.3f}")
        print("=" * 95)
        return 0

    r = pd.DataFrame(rows)
    print("\n" + "=" * 95)
    print(f"STRATÉGIE: v3 stack-lite (bin+rank blend + calib + conformal) | "
          f"min_edge={args.min_edge}  (use_plo={args.use_plo})")
    print(f"  Paris     : {len(r)}")
    print(f"  Victoires : {int(r['hit'].sum())}  ({r['hit'].mean()*100:.1f}%)")
    print(f"  PnL       : {r['pnl'].sum():+.2f} €")
    print(f"  ROI       : {r['pnl'].sum()/len(r)*100:+.1f} %")
    if int(r["hit"].sum()) > 0:
        print(f"  Cote moy. gagnante : {r[r['hit']]['cote'].mean():.2f}")
    print("=" * 95)

    return 0


if __name__ == "__main__":
    sys.exit(main())
