#!/usr/bin/env python3
"""
Tests de stratégies INATTENDUES (sans ML) sur les données pures.

Principe :
  Pour chaque "règle" empirique, on teste si elle produit un ROI positif sur
  le dataset complet, ET on valide par permutation test pour détecter les
  faux positifs (max-picking, data mining).

Stratégies testées :
  1. Drift négatif fort (steam) : cotes qui baissent fortement entre matin et final
  2. Drift positif extrême (drift-out) : cotes qui explosent vs matin
  3. Favori écrasant (odds_rank 1 avec cote très basse)
  4. Challenger du favori (odds_rank 2)
  5. Outsider avec bonne musique (cote > 15 + avg_last_5 < 4)
  6. Changement de jockey avec favori (jockey_changed=1 & odds_rank 1-2)
  7. Cheval au repos long (90-180j) qui revient (break return)
  8. Cheval jeune (age ≤ 4) avec nb_courses < 10 (émergent)
  9. Premier sur hippodrome avec cote basse
 10. Duo jockey+entraîneur avec WR > 0.15 (n_races ≥ 50)
 11. Contrarian favori (outsider extrême quand favori écrasant)
 12. Cote en chute libre (drift < -30%)
 13. Par discipline (plat/attele/monté/obstacle) — edge sur chaque
 14. Par hippodrome (top 20) — identifier biais spécifiques

Pour chaque stratégie :
  - ROI réel
  - Permutation test : mélange les gagnants au sein de chaque course
    20 fois, vérifie que ROI réel > ROI permuté (p < 0.05)
  - Nombre de paris (stat. significatif si ≥ 100)
"""
import json
import os
import time
import numpy as np
import pandas as pd

INPUT_PARQUET = "pmu_features.parquet"
OUTPUT_CSV = "unexpected_strategies.csv"

N_PERMUTATIONS = 20
RNG = np.random.default_rng(42)


def pnl_flat(df):
    """P&L à mise 1€ par pari : gain = div/100 - 1 si gagnant, -1 sinon."""
    won = df['won'].values == 1
    div = df['dividende_gagnant'].fillna(0).values / 100.0
    return np.where(won, div - 1, -1)


def eval_strategy(df_bets, stake=1.0):
    if len(df_bets) == 0:
        return {'n_bets': 0, 'win_rate': 0, 'avg_odds': 0, 'pnl': 0, 'roi': 0}
    pnl = pnl_flat(df_bets)
    total_stake = len(df_bets) * stake
    return {
        'n_bets': int(len(df_bets)),
        'win_rate': float(df_bets['won'].mean()),
        'avg_odds': float(df_bets['cote_direct'].mean()),
        'pnl': float(pnl.sum()),
        'roi': float(pnl.sum() / total_stake),
    }


def permutation_test(df_full, strategy_fn, n_perm=N_PERMUTATIONS):
    """Permute won/dividende_gagnant au sein de chaque race_id (vectorisé).

    Implémentation rapide :
      - On génère un random rank par ligne dans chaque race
      - On réordonne won/div selon ce rank
      - Le reste du df (features utilisées par la stratégie) reste tel quel
    Résultat : les paris sélectionnés par la stratégie sont les mêmes, mais
    les won/div sont aléatoirement réassignés parmi les chevaux de la même course.
    Si la stratégie n'a aucun edge prédictif, ROI perm ≈ ROI réel.
    """
    real = eval_strategy(strategy_fn(df_full))
    if real['n_bets'] == 0:
        return real, 0, [], None
    real_roi = real['roi']

    # Index d'origine par race (ordre deterministe)
    # On cache : pour chaque race_id, la liste des positions originales
    # race_ids sont contigus si on trie, donc on crée un mapping via cumcount
    base = df_full.reset_index(drop=True)
    race_idx = base.groupby('race_id', sort=False).cumcount().values  # pos dans course
    race_codes = pd.factorize(base['race_id'], sort=False)[0]
    # On trie les positions originales par (race_code, orig_idx)
    orig_order = np.argsort(race_codes * 10_000 + race_idx, kind='stable')
    # Taille de chaque course
    race_sizes = np.bincount(race_codes)
    # Pour chaque ligne : offset du début de sa course dans orig_order
    race_start = np.concatenate([[0], np.cumsum(race_sizes)[:-1]])
    row_race_start = race_start[race_codes]

    won_orig = base['won'].values
    div_orig = base['dividende_gagnant'].fillna(0).values

    perm_rois = []
    for i in range(n_perm):
        # Random key par ligne
        r = RNG.random(len(base))
        # On calcule le rank dans chaque course : tri (race_code, r)
        sort_key = race_codes.astype(np.int64) * np.int64(1e9) + (r * 1e9).astype(np.int64)
        perm_order = np.argsort(sort_key, kind='stable')
        # Position dans la course après perm
        # À la ligne i (dans df original), on veut connaître la "nouvelle" position
        # perm_order[k] donne l'index original de la k-ième ligne après tri
        # Donc pour chaque ligne originale i, sa nouvelle position dans sa course
        # est le rank de i dans perm_order au sein de sa course
        # Plus simple : on construit new_won[i] = won_orig[perm_order[ row_race_start[i] + (rank de i dans sa course) ]]
        # Utilisons un autre angle : pour chaque race, on shuffle won/div en place
        # Via bincount + tri

        # Plus simple et rapide : on fait un argsort par (race_code, random)
        # puis on reconstruit won_new[i] = won_orig[perm_order_inv[i]]
        # Mais perm_order réordonne les lignes selon (race, r), donc
        # won_orig[perm_order] donne la séquence perm. On veut réassigner
        # au sein de chaque race : le won du k-ième (tri original) va au k-ième (tri permuté)
        # Donc : won_new aligné sur l'ordre original devient won_orig[ordre_original → mappé]
        #
        # Soit orig_order_by_race[race_code, pos_in_race] = index_original
        # Et perm_order_by_race[race_code, pos_in_race] = index_perm
        # On veut won_new[orig_order_by_race[r, p]] = won_orig[perm_order_by_race[r, p]]

        # orig_order donne orig_order_by_race flatten; perm_order donne perm_order_by_race flatten
        won_new = np.empty_like(won_orig)
        div_new = np.empty_like(div_orig)
        won_new[orig_order] = won_orig[perm_order]
        div_new[orig_order] = div_orig[perm_order]

        df_perm = base.copy()
        df_perm['won'] = won_new
        df_perm['dividende_gagnant'] = div_new
        perm_bets = strategy_fn(df_perm)
        perm_rois.append(eval_strategy(perm_bets)['roi'])

    perm_rois = np.array(perm_rois)
    p_value = float((perm_rois >= real_roi).mean())
    return real, p_value, perm_rois.tolist(), float(perm_rois.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Strategy definitions (closures on df)
# ─────────────────────────────────────────────────────────────────────────────

def s_drift_negative(df):
    """Cotes qui chutent fortement (steam)."""
    return df[(df['drift_pct'] <= -0.25) & (df['cote_direct'] >= 2) & (df['cote_direct'] <= 20)]

def s_drift_positive_extreme(df):
    """Cotes qui explosent (marché qui abandonne)."""
    return df[(df['drift_pct'] >= 0.50) & (df['cote_direct'] >= 3) & (df['cote_direct'] <= 30)]

def s_crushing_fav(df):
    """Favori écrasant."""
    return df[(df['odds_rank_direct'] == 1) & (df['cote_direct'] <= 2.0)]

def s_challenger(df):
    """Challenger du favori."""
    return df[(df['odds_rank_direct'] == 2) & (df['cote_direct'] >= 3) & (df['cote_direct'] <= 8)]

def s_outsider_good_form(df):
    """Outsider avec bonne musique."""
    return df[(df['cote_direct'] >= 15) & (df['cote_direct'] <= 40) &
              (df['music_avg_last_5'] <= 3.5) & (df['music_len_musique'] >= 3)]

def s_jockey_change_fav(df):
    """Favori avec changement de jockey."""
    return df[(df['jockey_changed'] == 1) & (df['odds_rank_direct'] <= 2) &
              (df['cote_direct'] <= 8)]

def s_break_return(df):
    """Cheval qui revient après 90-180j."""
    return df[(df['horse_days_since_last'] >= 90) & (df['horse_days_since_last'] <= 180) &
              (df['cote_direct'] <= 12) & (df['odds_rank_direct'] <= 5)]

def s_young_emerging(df):
    """Cheval jeune (4 ans) émergent."""
    return df[(df['age'] == 4) & (df['horse_n_races'] <= 10) &
              (df['odds_rank_direct'] <= 3) & (df['cote_direct'] <= 10)]

def s_first_at_hippodrome(df):
    """Premier passage sur l'hippodrome avec cote basse."""
    return df[(df['first_at_hippodrome'] == 1) & (df['cote_direct'] <= 5) &
              (df['odds_rank_direct'] <= 2)]

def s_elite_duo(df):
    """Duo jockey+entraîneur avec fort WR historique (≥100 courses)."""
    return df[(df['duo_wr'] >= 0.15) & (df['duo_n_races'] >= 100) &
              (df['cote_direct'] <= 15) & (df['cote_direct'] >= 2)]

def s_drift_crash(df):
    """Drift violent négatif (cotes qui tombent)."""
    return df[(df['drift_pct'] <= -0.40) & (df['cote_direct'] >= 1.5) &
              (df['cote_direct'] <= 15)]

def s_elite_horse_comeback(df):
    """Cheval élite (WR ≥ 0.20, n ≥ 10) revenant après break."""
    return df[(df['horse_wr'] >= 0.20) & (df['horse_n_races'] >= 10) &
              (df['horse_days_since_last'] >= 60) & (df['cote_direct'] <= 10)]

def s_music_hot(df):
    """Cheval sur une série (2+ victoires dans les 5 dernières)."""
    return df[(df['music_wins_last_5'] >= 2) & (df['cote_direct'] <= 12) &
              (df['cote_direct'] >= 2)]

def s_distance_specialist(df):
    """Spécialiste de la distance (WR sur cette distance_bin ≥ 0.25 avec n ≥ 3)."""
    # horse_wr_distance_bin est calculé point-in-time, sur courses précédentes même distance
    return df[(df['horse_wr_distance_bin'] >= 0.25) & (df['cote_direct'] <= 10) &
              (df['cote_direct'] >= 2)]

def s_hippodrome_specialist(df):
    """Spécialiste de l'hippodrome."""
    return df[(df['horse_wr_hippodrome'] >= 0.25) & (df['horse_n_hippodrome'] >= 3) &
              (df['cote_direct'] <= 10)]

def s_jockey_hot(df):
    """Jockey en forme (WR 50 dernières > 0.18)."""
    return df[(df['jockey_wr_last50'] >= 0.18) & (df['cote_direct'] <= 10) &
              (df['odds_rank_direct'] <= 5)]

def s_trainer_elite(df):
    """Entraîneur élite (WR global ≥ 0.15, n ≥ 500)."""
    return df[(df['trainer_wr'] >= 0.15) & (df['trainer_n_races'] >= 500) &
              (df['cote_direct'] <= 15) & (df['odds_rank_direct'] <= 5)]

def s_tendance_up_strong(df):
    """Cotes stables/montantes mais favori."""
    return df[(df['tendance_enc'] >= 0) & (df['odds_rank_direct'] == 1) &
              (df['cote_direct'] <= 3)]

STRATEGIES = {
    'drift_negative_25': s_drift_negative,
    'drift_positive_50': s_drift_positive_extreme,
    'crushing_favorite': s_crushing_fav,
    'challenger_2nd': s_challenger,
    'outsider_good_form': s_outsider_good_form,
    'jockey_change_fav': s_jockey_change_fav,
    'break_90_180d': s_break_return,
    'young_4yo_emerging': s_young_emerging,
    'first_at_hippodrome': s_first_at_hippodrome,
    'elite_duo': s_elite_duo,
    'drift_crash_40': s_drift_crash,
    'elite_horse_comeback': s_elite_horse_comeback,
    'music_hot_2wins5': s_music_hot,
    'distance_specialist': s_distance_specialist,
    'hippodrome_specialist': s_hippodrome_specialist,
    'jockey_hot_last50': s_jockey_hot,
    'trainer_elite': s_trainer_elite,
    'tendance_up_fav': s_tendance_up_strong,
}


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:>6.1f}s] Loading features...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"          {len(df):,} rows")

    # Filter: only rows with cote_direct valid (sinon pas de pari possible)
    df = df[(df['cote_direct'] > 0) & df['finish_position'].notna()].copy()
    print(f"          {len(df):,} rows with cote and result")

    results = []
    for name, fn in STRATEGIES.items():
        t_s = time.time()
        real = eval_strategy(fn(df))
        if real['n_bets'] < 20:
            print(f"  [{name:25}] SKIP: n_bets={real['n_bets']}")
            continue

        # Permutation test (coûteux : on le fait seulement si n_bets >= 100)
        if real['n_bets'] >= 100:
            real, p_val, perm_rois, perm_roi_mean = permutation_test(df, fn, n_perm=N_PERMUTATIONS)
        else:
            p_val, perm_roi_mean = np.nan, np.nan

        results.append({
            'strategy': name, **real,
            'p_value': p_val,
            'perm_roi_mean': perm_roi_mean,
            'elapsed_s': round(time.time() - t_s, 1),
        })
        sig = "***" if (p_val is not None and not np.isnan(p_val) and p_val < 0.05) else ""
        print(f"  [{name:25}] n={real['n_bets']:>6} wr={real['win_rate']:.3f} "
              f"cote={real['avg_odds']:.2f} roi={real['roi']:+.3f} p={p_val if not np.isnan(p_val) else '-':.3f} {sig} "
              f"({time.time()-t_s:.0f}s)"
              if not np.isnan(p_val) else
              f"  [{name:25}] n={real['n_bets']:>6} wr={real['win_rate']:.3f} "
              f"cote={real['avg_odds']:.2f} roi={real['roi']:+.3f} p=N/A ({time.time()-t_s:.0f}s)")

    results_df = pd.DataFrame(results).sort_values('roi', ascending=False)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[{time.time()-t0:>6.1f}s] Saved: {OUTPUT_CSV}")
    print("\n=== TOP STRATEGIES BY ROI ===")
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
