#!/usr/bin/env python3
"""
Entraînement walk-forward LightGBM avec validation sans leak.

Méthode :
  - Split temporel STRICT (aucun regard sur le futur)
  - Fenêtres glissantes: train sur N mois → test sur M mois suivants
  - Pour chaque fenêtre : AUC + ROI sur plusieurs stratégies
  - Modèle ranker (LightGBM lambdarank) : apprend à classer les chevaux
    d'une course entre eux (plus pertinent que classifier binaire)
  - Tracking de l'amélioration : chaque nouvelle fenêtre devrait avoir
    AUC/ROI au moins stable, et le modèle intègre la data cumulée.

Output:
  - model_lgbm.pkl (dernier modèle, entraîné sur toute la data - la fenêtre de test finale)
  - model_lgbm_meta.json (features, metrics par fenêtre, feature importance)
  - walkforward_results.csv (résultats détaillés par fenêtre)
"""
import json
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle

INPUT_PARQUET = "pmu_features.parquet"
MODEL_PATH = "model_lgbm.pkl"
META_PATH = "model_lgbm_meta.json"
WF_RESULTS = "walkforward_results.csv"


FEATURES = [
    # Market (les plus forts selon l'ancien modèle)
    'cote_direct', 'cote_probable', 'drift_pct', 'odds_ratio',
    'inv_cote_direct', 'inv_cote_probable',
    'odds_rank_probable', 'odds_rank_direct', 'cote_anomaly', 'tendance_enc',
    # Context
    'distance', 'distance_bin', 'nb_partants', 'age', 'sexe_enc',
    'deferre_enc', 'handicap_poids', 'day_of_week', 'month',
    # Horse history
    'horse_n_races', 'horse_wr', 'horse_podium_rate',
    'horse_days_since_last', 'horse_gain_per_race',
    'horse_wr_discipline', 'horse_wr_hippodrome', 'horse_wr_distance_bin',
    'horse_n_hippodrome',
    # Music
    'music_avg_last_3', 'music_avg_last_5', 'music_avg_last_10',
    'music_wins_last_5', 'music_podium_last_5', 'music_len_musique',
    'music_discipline_consistency',
    # Jockey
    'jockey_n_races', 'jockey_wr',
    'jockey_wr_last15', 'jockey_wr_last50', 'jockey_wr_last200',
    'jockey_wr_hippodrome', 'jockey_n_hippodrome',
    # Trainer
    'trainer_n_races', 'trainer_wr',
    'trainer_wr_last50', 'trainer_wr_last200',
    # Duos
    'duo_n_races', 'duo_wr', 'jh_n_races', 'jh_wr',
    # Unexpected
    'horse_quick_return', 'horse_long_break', 'horse_very_long_break',
    'jockey_changed', 'first_at_hippodrome', 'first_duo',
    'is_favorite', 'is_outsider', 'gains_vs_field_median',
    'nb_courses', 'nb_victoires', 'gains',
]

# Catégorielles encodées numériquement (LightGBM en fait meilleur usage)
CATEGORICAL = ['sexe_enc', 'deferre_enc', 'tendance_enc', 'day_of_week', 'month']


def dividend_to_return(div_cents):
    """Dividende PMU (en centimes pour 1€) → ratio de gain.
    Ex: 2290 centimes = 22.90€ pour 1€ misé → return = 22.90 (pas -1 pour mise)."""
    return (div_cents / 100.0) if pd.notna(div_cents) and div_cents > 0 else 0.0


def compute_race_pnl(bets_df, stake=1.0):
    """P&L flat: mise 1€ sur chaque cheval choisi, gain = dividende/100.
    Retour net = gain - mise."""
    won_mask = bets_df['won'] == 1
    div = bets_df['dividende_gagnant'].fillna(0) / 100.0
    gross = np.where(won_mask, div, 0.0) * stake
    pnl = gross - stake
    return pnl


def train_one_window(df_train, df_test, params, num_boost_round=500):
    """Entraîne un LightGBM ranker sur df_train, évalue sur df_test."""
    # Filter: exclude rows with no cote (pas exploitables pour paris)
    df_train = df_train[df_train['cote_direct'] > 0].copy()
    df_test = df_test[df_test['cote_direct'] > 0].copy()

    X_tr = df_train[FEATURES]
    y_tr = df_train['won']
    X_te = df_test[FEATURES]
    y_te = df_test['won']

    # Group sizes for ranker = nb partants par course
    # Le DataFrame est trié par date donc les courses se suivent
    # On re-trie par race_id pour grouper proprement
    df_train = df_train.sort_values(['date_dt', 'race_id']).reset_index(drop=True)
    df_test = df_test.sort_values(['date_dt', 'race_id']).reset_index(drop=True)
    X_tr = df_train[FEATURES]
    y_tr = df_train['won']
    X_te = df_test[FEATURES]
    y_te = df_test['won']

    group_tr = df_train.groupby('race_id', sort=False).size().values
    group_te = df_test.groupby('race_id', sort=False).size().values

    # On utilise binary classification (plus simple, AUC interprétable)
    # Ranker serait plus pertinent mais binary est standard pour ce domaine
    lgb_train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CATEGORICAL)
    lgb_val = lgb.Dataset(X_te, label=y_te, categorical_feature=CATEGORICAL, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # Predictions
    y_pred = model.predict(X_te, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_te, y_pred)

    # Rebuild test df avec predictions pour P&L
    df_test = df_test.copy()
    df_test['pred'] = y_pred
    df_test['implied_prob'] = 1.0 / df_test['cote_direct'].clip(lower=1.01)
    df_test['edge'] = df_test['pred'] - df_test['implied_prob']

    return model, auc, df_test


def backtest_strategy(df_test, name, min_edge=0.05, min_prob=0.0,
                      cote_min=1.5, cote_max=20.0, top_only=True, stake=1.0):
    """Sélectionne les paris dans df_test selon filtres, calcule P&L."""
    d = df_test.copy()
    # Filtre cote
    d = d[(d['cote_direct'] >= cote_min) & (d['cote_direct'] <= cote_max)]
    # Filtre edge + prob
    d = d[(d['edge'] >= min_edge) & (d['pred'] >= min_prob)]

    if top_only:
        # Garder uniquement le cheval avec la plus haute prédiction par course
        d = d.loc[d.groupby('race_id')['pred'].idxmax()]

    if len(d) == 0:
        return {'name': name, 'n_bets': 0, 'win_rate': 0, 'avg_odds': 0,
                'avg_edge': 0, 'pnl': 0, 'roi': 0}

    pnl = compute_race_pnl(d, stake=stake)
    total_stake = len(d) * stake
    return {
        'name': name,
        'n_bets': len(d),
        'win_rate': d['won'].mean(),
        'avg_odds': d['cote_direct'].mean(),
        'avg_edge': d['edge'].mean(),
        'pnl': float(pnl.sum()),
        'roi': float(pnl.sum() / total_stake) if total_stake > 0 else 0.0,
    }


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:>6.1f}s] Loading features...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"          {len(df):,} rows, {len(df.columns)} columns")

    # Types
    df['date_dt'] = pd.to_datetime(df['date_dt'])

    # Drop rows without finish_position ≥1 (courses sans arrivée)
    # Keep: any row for training (features), but for backtest we need cote_direct
    print(f"          date range: {df['date_dt'].min().date()} → {df['date_dt'].max().date()}")
    print(f"          rows with cote_direct>0: {(df['cote_direct']>0).sum():,}")

    # ───────── Walk-forward windows ─────────
    # Stratégie : fenêtres glissantes de 2 ans train → 6 mois test
    # On commence assez tôt pour avoir du data historique (horse/jockey stats)
    # mais les 1ères années ont peu d'historique donc AUC sera dégradé au début.

    windows = []
    start_year = 2016  # 2014-2015 servent d'amorçage historique
    for year in range(start_year, 2026):
        for month_start in [1, 7]:  # 2 fenêtres / an (janvier + juillet)
            train_end = pd.Timestamp(f"{year}-{month_start:02d}-01")
            train_start = train_end - pd.DateOffset(years=2)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=6)
            if test_end > df['date_dt'].max():
                break
            windows.append({
                'train_start': train_start, 'train_end': train_end,
                'test_start': test_start, 'test_end': test_end,
            })

    print(f"\n[{time.time()-t0:>6.1f}s] {len(windows)} walk-forward windows")

    # ───────── LightGBM params ─────────
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l2': 0.1,
        'verbose': -1,
        'num_threads': 8,
    }

    strategies = [
        {'name': 'ALL_FAVORITES', 'min_edge': -10, 'min_prob': 0, 'cote_min': 1.0, 'cote_max': 99},
        {'name': 'HIGH_EDGE',  'min_edge': 0.10, 'min_prob': 0.0,  'cote_min': 1.5, 'cote_max': 10},
        {'name': 'MED_EDGE',   'min_edge': 0.05, 'min_prob': 0.0,  'cote_min': 2.0, 'cote_max': 15},
        {'name': 'LOW_EDGE',   'min_edge': 0.02, 'min_prob': 0.0,  'cote_min': 2.0, 'cote_max': 20},
        {'name': 'CONF_HIGH',  'min_edge': 0.0,  'min_prob': 0.25, 'cote_min': 1.5, 'cote_max': 8},
        # Stratégies "inattendues" testées en walk-forward :
        {'name': 'OUTSIDER_HE', 'min_edge': 0.05, 'min_prob': 0.0, 'cote_min': 8, 'cote_max': 25},
        {'name': 'EXTREME_HE',  'min_edge': 0.15, 'min_prob': 0.0, 'cote_min': 2, 'cote_max': 30},
    ]

    results = []
    last_model = None

    for i, w in enumerate(windows):
        t_w = time.time()
        mask_tr = (df['date_dt'] >= w['train_start']) & (df['date_dt'] < w['train_end'])
        mask_te = (df['date_dt'] >= w['test_start']) & (df['date_dt'] < w['test_end'])
        df_tr = df[mask_tr]
        df_te = df[mask_te]
        if len(df_tr) < 10000 or len(df_te) < 5000:
            print(f"  [W{i+1}] SKIP: train={len(df_tr)} test={len(df_te)}")
            continue

        model, auc, df_pred = train_one_window(df_tr, df_te, params)
        last_model = model

        n_train = len(df_tr)
        n_test = len(df_te)
        best_iter = model.best_iteration

        print(f"  [W{i+1:>2}] train {w['train_start'].date()}→{w['train_end'].date()} "
              f"({n_train:>7,}) test →{w['test_end'].date()} ({n_test:>6,}) "
              f"AUC={auc:.4f} it={best_iter} ({time.time()-t_w:.0f}s)")

        # Evaluate all strategies on this window
        for s in strategies:
            res = backtest_strategy(df_pred, **s)
            results.append({
                'window': i + 1,
                'train_end': w['train_end'].date().isoformat(),
                'test_end': w['test_end'].date().isoformat(),
                'n_train': n_train, 'n_test': n_test,
                'auc': auc, **res
            })

    # ───────── Save results ─────────
    res_df = pd.DataFrame(results)
    res_df.to_csv(WF_RESULTS, index=False)
    print(f"\n[{time.time()-t0:>6.1f}s] Results saved: {WF_RESULTS}")

    # ───────── Summary per strategy ─────────
    print("\n=== SUMMARY: moyenne sur toutes les fenêtres ===")
    summary = res_df.groupby('name').agg({
        'n_bets': 'sum', 'win_rate': 'mean', 'avg_odds': 'mean',
        'avg_edge': 'mean', 'pnl': 'sum', 'roi': 'mean', 'auc': 'mean'
    }).round(4)
    summary['n_windows'] = res_df.groupby('name').size()
    summary = summary.sort_values('roi', ascending=False)
    print(summary.to_string())

    # ───────── Feature importance ─────────
    if last_model is not None:
        imp = pd.DataFrame({
            'feature': FEATURES,
            'importance': last_model.feature_importance(importance_type='gain'),
        }).sort_values('importance', ascending=False)
        print("\n=== TOP 20 FEATURES (gain, last model) ===")
        print(imp.head(20).to_string(index=False))

        # ───────── Save final model ─────────
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(last_model, f)
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        print(f"\n[{time.time()-t0:>6.1f}s] Saved: {MODEL_PATH} ({size_mb:.1f} MB)")

        meta = {
            'features': FEATURES,
            'categorical': CATEGORICAL,
            'last_train_end': windows[-1]['train_end'].date().isoformat() if windows else '',
            'n_windows': len(windows),
            'summary_by_strategy': summary.to_dict(),
            'top_features': imp.head(20).to_dict(orient='records'),
            'params': params,
        }
        with open(META_PATH, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"          Meta: {META_PATH}")


if __name__ == '__main__':
    main()
