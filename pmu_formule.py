"""
PMU — Formule Prédictive Contextuelle
======================================
Formule validée en walk-forward sur 5620 courses (2004-2026).
Edge mesuré : +2.99pp en Top-5 au-dessus du hasard pur.

Usage:
    python pmu_formule.py                    # Backtest complet
    python pmu_formule.py predict Attele 2700 Vincennes 16  # Prédiction
"""

import json
import sys
import numpy as np
from collections import Counter
from datetime import datetime


def load_races(path="pmu_races.json"):
    with open(path, "r") as f:
        data = json.load(f)

    type_map = {
        "Attelé": "Attele", "attelé": "Attele", "Monté": "Monte",
        "Haies": "Haies", "Steeple Chase": "Steeple",
        "Steeple-Chase": "Steeple", "Plat": "Plat", "PLat": "Plat",
    }

    races = []
    for r in data:
        arr = r.get("arrivee")
        if not isinstance(arr, list) or len(arr) < 5:
            continue
        if not all(isinstance(v, int) for v in arr):
            continue
        try:
            dt = datetime.strptime(r["date"], "%d/%m/%Y")
        except (ValueError, KeyError):
            continue
        p = r.get("partants")
        if isinstance(p, str):
            p = int(p) if p.isdigit() else None
        elif not isinstance(p, int):
            p = None

        type_course = (r.get("type") or "").strip()
        type_norm = type_map.get(type_course, "Inconnu")
        lieu = (r.get("lieu") or "").strip()

        races.append({
            "date": dt,
            "type": type_norm,
            "lieu": lieu,
            "distance": r.get("distance") or 0,
            "partants": p,
            "arrivee": arr[:5],
            "winner": arr[0],
        })

    races.sort(key=lambda x: x["date"])
    return races


def get_context(race):
    """Contexte = (type, distance arrondie à 500m, lieu simplifié)"""
    typ = race["type"]
    dist_bin = race["distance"] // 500 * 500
    top_lieux = ["Vincennes", "Auteuil", "Paris-Longchamp", "Deauville", "Chantilly"]
    lieu = race["lieu"] if race["lieu"] in top_lieux else "Autre"
    return (typ, dist_bin, lieu)


def score_formula(races, index, lookback=3000, w_ctx=0.3, w_uniform=0.7):
    """
    Score(k) = w_ctx × ContextFreq(k) + w_uniform × Uniform(k)

    ContextFreq(k) = fréquence de k parmi les gagnants de courses passées
                     ayant le même contexte (type, distance/500, lieu).
    Uniform(k) = 1 / nb_partants
    """
    race = races[index]
    n_partants = race["partants"] or 16
    ctx = get_context(race)

    # Historique contextuel
    ctx_winners = []
    for j in range(max(0, index - lookback), index):
        if get_context(races[j]) == ctx:
            ctx_winners.append(races[j]["winner"])

    scores = {}
    for k in range(1, n_partants + 1):
        # Contexte conditionnel
        if len(ctx_winners) >= 5:
            ctx_freq = sum(1 for w in ctx_winners if w == k) / len(ctx_winners)
        else:
            ctx_freq = 1.0 / n_partants

        # Uniforme
        uniform = 1.0 / n_partants

        scores[k] = w_ctx * ctx_freq + w_uniform * uniform

    # Normaliser en probabilités
    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()}


def predict(type_course, distance, lieu, n_partants, races, lookback=3000):
    """Prédit le classement probabiliste pour une nouvelle course."""
    fake_race = {
        "type": type_course,
        "distance": distance,
        "lieu": lieu,
        "partants": n_partants,
    }
    ctx = get_context(fake_race)

    # Chercher l'historique
    ctx_winners = [r["winner"] for r in races if get_context(r) == ctx]
    ctx_winners = ctx_winners[-lookback:]

    scores = {}
    for k in range(1, n_partants + 1):
        if len(ctx_winners) >= 5:
            ctx_freq = sum(1 for w in ctx_winners if w == k) / len(ctx_winners)
        else:
            ctx_freq = 1.0 / n_partants
        uniform = 1.0 / n_partants
        scores[k] = 0.3 * ctx_freq + 0.7 * uniform

    total = sum(scores.values())
    scores = {k: v / total for k, v in scores.items()}
    return sorted(scores.items(), key=lambda x: -x[1])


def backtest(races):
    """Walk-forward backtest complet."""
    step = 200
    all_results = []

    for start in range(0, len(races) - 1000 - step, step):
        hits1 = hits3 = hits5 = total = 0
        for i in range(start + 1000, min(start + 1000 + step, len(races))):
            scores = score_formula(races, i)
            sorted_k = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            winner = races[i]["winner"]
            total += 1
            if sorted_k[0] == winner:
                hits1 += 1
            if winner in sorted_k[:3]:
                hits3 += 1
            if winner in sorted_k[:5]:
                hits5 += 1
        if total > 0:
            period = f"{races[start+1000]['date'].strftime('%Y-%m')} → {races[min(start+1000+step-1, len(races)-1)]['date'].strftime('%Y-%m')}"
            all_results.append({
                "period": period,
                "top1": hits1 / total,
                "top3": hits3 / total,
                "top5": hits5 / total,
                "n": total,
            })

    return all_results


if __name__ == "__main__":
    races = load_races()
    print(f"Chargé: {len(races)} courses")

    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        if len(sys.argv) < 6:
            print("Usage: python pmu_formule.py predict <type> <distance> <lieu> <partants>")
            print("  Ex: python pmu_formule.py predict Attele 2700 Vincennes 16")
            sys.exit(1)

        type_c = sys.argv[2]
        dist = int(sys.argv[3])
        lieu = sys.argv[4]
        n_part = int(sys.argv[5])

        print(f"\nPrédiction: {type_c} | {dist}m | {lieu} | {n_part} partants")
        print(f"Contexte: {(type_c, dist // 500 * 500, lieu)}")

        ranking = predict(type_c, dist, lieu, n_part, races)
        print(f"\nClassement probabiliste:")
        for rank, (num, prob) in enumerate(ranking):
            uniform = 1.0 / n_part
            edge = (prob - uniform) / uniform * 100
            bar = "+" * max(0, int(edge / 2)) if edge > 0 else "-" * max(0, int(-edge / 2))
            print(f"  {rank+1:2d}. N°{num:2d}: {prob*100:5.2f}% (edge: {edge:+5.1f}%) {bar}")

        print(f"\n  TOP-5 recommandé: {[num for num, _ in ranking[:5]]}")

    else:
        print("\n=== BACKTEST WALK-FORWARD ===\n")
        results = backtest(races)

        for r in results:
            print(f"  {r['period']}: Top1={r['top1']*100:.1f}% Top3={r['top3']*100:.1f}% Top5={r['top5']*100:.1f}% ({r['n']}r)")

        avg1 = np.mean([r["top1"] for r in results])
        avg3 = np.mean([r["top3"] for r in results])
        avg5 = np.mean([r["top5"] for r in results])
        beat5 = sum(1 for r in results if r["top5"] > 0.3125)

        print(f"\n  MOYENNE: Top1={avg1*100:.2f}% Top3={avg3*100:.2f}% Top5={avg5*100:.2f}%")
        print(f"  BASELINE: Top1=6.25% Top3=18.75% Top5=31.25%")
        print(f"  EDGE:     Top1={avg1*100-6.25:+.2f}pp Top3={avg3*100-18.75:+.2f}pp Top5={avg5*100-31.25:+.2f}pp")
        print(f"  Périodes gagnantes (Top5 > hasard): {beat5}/{len(results)}")
