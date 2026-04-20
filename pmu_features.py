#!/usr/bin/env python3
"""
Feature engineering avec correction point-in-time stricte.
Version optimisée — utilise uniquement des opérations pandas vectorisées.

Sécurité no-leak :
  Pour chaque feature historique, on calcule la stat sur les courses
  STRICTEMENT ANTÉRIEURES (via shift(1) dans chaque groupe).

Output: pmu_features.parquet
"""
import json
import os
import time
import numpy as np
import pandas as pd

INPUT_PARQUET = "pmu_dataset.parquet"
OUTPUT_PARQUET = "pmu_features.parquet"


# ── Helpers: fast point-in-time cumulative stats ─────────────────────────────

def cum_past_sum(df, group_col, value_col):
    """Somme cumulative STRICTE (passé uniquement) de value_col groupée par group_col.
    Utilise 2 opérations groupby vectorisées au lieu d'un lambda lent."""
    shifted = df.groupby(group_col, sort=False)[value_col].shift(1).fillna(0)
    return shifted.groupby(df[group_col].values, sort=False).cumsum()


def cum_past_sum_multi(df, group_cols, value_col):
    """Idem mais avec groupby multi-colonnes."""
    g = df.groupby(group_cols, sort=False)
    shifted = g[value_col].shift(1).fillna(0)
    # Pour le cumsum on re-groupe sur les mêmes clés
    return shifted.groupby([df[c].values for c in group_cols], sort=False).cumsum()


def cum_past_count(df, group_col):
    """Nombre de courses antérieures par groupe (0 au 1er apparition)."""
    return df.groupby(group_col, sort=False).cumcount()


def rolling_past_mean(df, group_col, value_col, window):
    """Moyenne rolling sur les N courses PASSÉES (exclut la course actuelle).
    Utilise cumsum diff pour vitesse."""
    # Shift(1) pour exclure la course actuelle
    shifted = df.groupby(group_col, sort=False)[value_col].shift(1).fillna(0)
    # Cumsum par groupe
    cum = shifted.groupby(df[group_col].values, sort=False).cumsum()
    # Cumsum décalé de `window` positions dans chaque groupe
    cum_lag = cum.groupby(df[group_col].values, sort=False).shift(window).fillna(0)
    # Rolling sum = cum_now - cum_(window ago)
    rolling_sum = cum - cum_lag
    # Nombre réel de courses dans la fenêtre (max = window, min = idx+1)
    idx = df.groupby(group_col, sort=False).cumcount()
    n_in_window = np.minimum(idx, window).clip(lower=1)
    return rolling_sum / n_in_window


def parse_music_json(js):
    try:
        return json.loads(js) if js else []
    except Exception:
        return []


def discipline_to_music_code(disc):
    return {'MONTE': 'm', 'ATTELE': 'a', 'HAIE': 'h', 'STEEPLECHASE': 's', 'PLAT': 'p'}.get(disc, '')


def music_features_vectorized(music_json_series, discipline_code_series):
    """Parse musique pour chaque ligne et retourne un DataFrame de features."""
    avg3, avg5, avg10 = [], [], []
    wins5, podium5, lens, disc_cons = [], [], [], []

    for js, dc in zip(music_json_series, discipline_code_series):
        parsed = parse_music_json(js)
        if not parsed:
            avg3.append(np.nan); avg5.append(np.nan); avg10.append(np.nan)
            wins5.append(0); podium5.append(0); lens.append(0)
            disc_cons.append(np.nan)
            continue
        positions = [(p if 1 <= p <= 9 else 10) for p, d in parsed]
        avg3.append(np.mean(positions[:3]) if positions else np.nan)
        avg5.append(np.mean(positions[:5]) if positions else np.nan)
        avg10.append(np.mean(positions[:10]) if positions else np.nan)
        wins5.append(sum(1 for p in positions[:5] if p == 1))
        podium5.append(sum(1 for p in positions[:5] if 1 <= p <= 3))
        lens.append(len(parsed))
        recent = [d for p, d in parsed[:5]]
        disc_cons.append(recent.count(dc) / len(recent) if recent else np.nan)

    return pd.DataFrame({
        'music_avg_last_3': avg3, 'music_avg_last_5': avg5, 'music_avg_last_10': avg10,
        'music_wins_last_5': wins5, 'music_podium_last_5': podium5,
        'music_len_musique': lens, 'music_discipline_consistency': disc_cons,
    })


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:>6.1f}s] Loading dataset...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"          {len(df):,} rows")

    df = df.sort_values(['date_dt', 'reunion', 'course', 'num_pmu']).reset_index(drop=True)

    # ───────── Race-level features (context + market) ─────────
    print(f"[{time.time()-t0:>6.1f}s] Context features...")
    df['distance_bin'] = (df['distance'] // 200 * 200).fillna(0).astype(int)
    df['day_of_week'] = df['date_dt'].dt.dayofweek
    df['month'] = df['date_dt'].dt.month
    df['year'] = df['date_dt'].dt.year

    print(f"[{time.time()-t0:>6.1f}s] Market features (drift, ranks, anomaly)...")
    cp = df['cote_probable'].replace(0, np.nan)
    cd = df['cote_direct'].replace(0, np.nan)
    df['drift_pct'] = ((cd - cp) / cp).fillna(0)
    df['odds_ratio'] = (cd / cp).fillna(1.0)
    df['inv_cote_direct'] = (1.0 / cd).fillna(0)
    df['inv_cote_probable'] = (1.0 / cp).fillna(0)

    df['odds_rank_probable'] = df.groupby('race_id')['cote_probable'].rank(method='min')
    df['odds_rank_direct'] = df.groupby('race_id')['cote_direct'].rank(method='min')

    # Cote anomaly (écart à la médiane du field / MAD)
    g_race_cd = df.groupby('race_id')['cote_direct']
    med = g_race_cd.transform('median')
    mad = (df['cote_direct'] - med).abs().groupby(df['race_id']).transform('median')
    df['cote_anomaly'] = np.where(mad > 0, (df['cote_direct'] - med) / mad, 0)

    tend_map = {'U': 1, 'D': -1, '=': 0, '': 0}
    df['tendance_enc'] = df['tendance'].map(tend_map).fillna(0)
    sexe_map = {'MALES': 0, 'FEMELLES': 1, 'HONGRES': 2}
    df['sexe_enc'] = df['sexe'].map(sexe_map).fillna(-1)
    df['deferre_enc'] = pd.factorize(df['deferre'].fillna(''))[0]

    # ───────── Musique (parse row by row, ~2.5M rows, ~30-60s) ─────────
    print(f"[{time.time()-t0:>6.1f}s] Music features (parsing 2.5M rows)...")
    df['discipline_code'] = df['discipline'].map(discipline_to_music_code).fillna('')
    mf = music_features_vectorized(df['musique_parsed_json'].values, df['discipline_code'].values)
    for c in mf.columns:
        df[c] = mf[c].values

    # ───────── Horse history (point-in-time) ─────────
    print(f"[{time.time()-t0:>6.1f}s] Horse history...")
    df['horse_n_races'] = cum_past_count(df, 'horse_key')
    df['horse_wins_cum'] = cum_past_sum(df, 'horse_key', 'won')
    df['horse_podiums_cum'] = cum_past_sum(df, 'horse_key', 'placed')
    df['horse_wr'] = np.where(df['horse_n_races'] > 0,
                               df['horse_wins_cum'] / df['horse_n_races'], np.nan)
    df['horse_podium_rate'] = np.where(df['horse_n_races'] > 0,
                                        df['horse_podiums_cum'] / df['horse_n_races'], np.nan)

    # Jours depuis dernière course
    df['horse_last_race_date'] = df.groupby('horse_key', sort=False)['date_dt'].shift(1)
    df['horse_days_since_last'] = (df['date_dt'] - df['horse_last_race_date']).dt.days

    # Gains cumulés (past only)
    df['horse_gains_cum'] = cum_past_sum(df, 'horse_key', 'gains')
    df['horse_gain_per_race'] = np.where(df['horse_n_races'] > 0,
                                          df['horse_gains_cum'] / df['horse_n_races'], 0)

    # ───────── Horse x context ─────────
    print(f"[{time.time()-t0:>6.1f}s] Horse x discipline/hippodrome/distance...")

    def cum_wr_multi(df, keys):
        idx = df.groupby(keys, sort=False).cumcount()
        w = cum_past_sum_multi(df, keys, 'won')
        return np.where(idx > 0, w / idx, np.nan), idx

    df['horse_wr_discipline'], _ = cum_wr_multi(df, ['horse_key', 'discipline'])
    df['horse_wr_hippodrome'], df['horse_n_hippodrome'] = cum_wr_multi(df, ['horse_key', 'hippodrome'])
    df['horse_wr_distance_bin'], _ = cum_wr_multi(df, ['horse_key', 'distance_bin'])

    # ───────── Jockey history ─────────
    print(f"[{time.time()-t0:>6.1f}s] Jockey history...")
    df['jockey_n_races'] = cum_past_count(df, 'jockey')
    df['jockey_wins_cum'] = cum_past_sum(df, 'jockey', 'won')
    df['jockey_wr'] = np.where(df['jockey_n_races'] > 0,
                                df['jockey_wins_cum'] / df['jockey_n_races'], np.nan)
    df['jockey_wr_last15'] = rolling_past_mean(df, 'jockey', 'won', 15)
    df['jockey_wr_last50'] = rolling_past_mean(df, 'jockey', 'won', 50)
    df['jockey_wr_last200'] = rolling_past_mean(df, 'jockey', 'won', 200)
    df['jockey_wr_hippodrome'], df['jockey_n_hippodrome'] = cum_wr_multi(df, ['jockey', 'hippodrome'])

    # ───────── Trainer history ─────────
    print(f"[{time.time()-t0:>6.1f}s] Trainer history...")
    df['trainer_n_races'] = cum_past_count(df, 'entraineur')
    df['trainer_wins_cum'] = cum_past_sum(df, 'entraineur', 'won')
    df['trainer_wr'] = np.where(df['trainer_n_races'] > 0,
                                  df['trainer_wins_cum'] / df['trainer_n_races'], np.nan)
    df['trainer_wr_last50'] = rolling_past_mean(df, 'entraineur', 'won', 50)
    df['trainer_wr_last200'] = rolling_past_mean(df, 'entraineur', 'won', 200)

    # ───────── Duos ─────────
    print(f"[{time.time()-t0:>6.1f}s] Duos (jockey+trainer, jockey+horse)...")
    df['duo_key'] = df['jockey'].astype(str) + '||' + df['entraineur'].astype(str)
    df['duo_wr'], df['duo_n_races'] = cum_wr_multi(df, ['duo_key'])

    df['jh_key'] = df['jockey'].astype(str) + '||' + df['horse_key'].astype(str)
    df['jh_wr'], df['jh_n_races'] = cum_wr_multi(df, ['jh_key'])

    # ───────── Unexpected ─────────
    print(f"[{time.time()-t0:>6.1f}s] Unexpected features...")
    df['horse_quick_return'] = (df['horse_days_since_last'] < 14).astype(int)
    df['horse_long_break'] = (df['horse_days_since_last'] > 90).astype(int)
    df['horse_very_long_break'] = (df['horse_days_since_last'] > 180).astype(int)

    df['last_jockey_on_horse'] = df.groupby('horse_key', sort=False)['jockey'].shift(1)
    df['jockey_changed'] = (df['jockey'] != df['last_jockey_on_horse']).astype(int)
    df.loc[df['last_jockey_on_horse'].isna(), 'jockey_changed'] = 0

    df['first_at_hippodrome'] = (df['horse_n_hippodrome'] == 0).astype(int)
    df['first_duo'] = (df['duo_n_races'] == 0).astype(int)

    df['is_favorite'] = (df['odds_rank_direct'] == 1).astype(int)
    max_rank = df.groupby('race_id')['odds_rank_direct'].transform('max')
    df['is_outsider'] = (df['odds_rank_direct'] >= max_rank * 0.7).astype(int)

    df['gains_vs_field_median'] = df['gains'] - df.groupby('race_id')['gains'].transform('median')

    # ───────── Save ─────────
    keep_cols = [
        'race_id', 'date_dt', 'reunion', 'course', 'num_pmu', 'horse_key', 'nom',
        'jockey', 'entraineur',
        'won', 'placed', 'finish_position',
        'dividende_gagnant', 'dividende_place', 'cote_direct', 'cote_probable',
        'hippodrome', 'discipline', 'distance', 'distance_bin', 'nb_partants',
        'age', 'sexe_enc', 'deferre_enc', 'handicap_poids',
        'day_of_week', 'month', 'year',
        'drift_pct', 'odds_ratio', 'inv_cote_direct', 'inv_cote_probable',
        'odds_rank_probable', 'odds_rank_direct', 'cote_anomaly', 'tendance_enc',
        'horse_n_races', 'horse_wr', 'horse_podium_rate',
        'horse_days_since_last', 'horse_gain_per_race',
        'horse_wr_discipline', 'horse_wr_hippodrome', 'horse_wr_distance_bin',
        'horse_n_hippodrome',
        'music_avg_last_3', 'music_avg_last_5', 'music_avg_last_10',
        'music_wins_last_5', 'music_podium_last_5', 'music_len_musique',
        'music_discipline_consistency',
        'jockey_n_races', 'jockey_wr',
        'jockey_wr_last15', 'jockey_wr_last50', 'jockey_wr_last200',
        'jockey_wr_hippodrome', 'jockey_n_hippodrome',
        'trainer_n_races', 'trainer_wr',
        'trainer_wr_last50', 'trainer_wr_last200',
        'duo_n_races', 'duo_wr', 'jh_n_races', 'jh_wr',
        'horse_quick_return', 'horse_long_break', 'horse_very_long_break',
        'jockey_changed', 'first_at_hippodrome', 'first_duo',
        'is_favorite', 'is_outsider', 'gains_vs_field_median',
        'nb_courses', 'nb_victoires', 'gains',
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_out = df[keep_cols].copy()

    print(f"[{time.time()-t0:>6.1f}s] Saving...")
    print(f"          Shape: {df_out.shape}")
    df_out.to_parquet(OUTPUT_PARQUET, compression='snappy', index=False)
    size_mb = os.path.getsize(OUTPUT_PARQUET) / 1e6
    print(f"[{time.time()-t0:>6.1f}s] Saved: {OUTPUT_PARQUET} ({size_mb:.1f} MB)")
    print(f"\nFeature columns ({len(keep_cols)}):")
    for c in keep_cols:
        print(f"  {c}")


if __name__ == '__main__':
    main()
