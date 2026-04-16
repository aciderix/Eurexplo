# Pipeline PMU v3 — Roadmap complète

Objectif : passer d'un modèle LightGBM simple (AUC 0.79, ROI +20 %) à un
système vraiment puissant qui exploite **tout** ce dont on dispose :
- blobs JSON non utilisés (cotes définitives, performances détaillées, pronostics),
- learn-to-rank au lieu d'une classif binaire par cheval,
- objectif reward-aware (optimise ROI, pas AUC),
- ELO contextuel, head-to-head, embeddings entité,
- stacking multi-modèles + calibration,
- conformal prediction (incertitude quantifiée),
- contextual bandit (apprend directement la politique de pari),
- walk-forward mensuel avec intervalles de confiance.

**Tout se lance depuis ton PC** — les scripts sont prêts, tu n'as qu'à suivre l'ordre.

---

## 0 · Prérequis

```bash
# Python 3.10+ conseillé
pip install -r requirements_v3.txt

# Récupérer les parquets (si pas déjà fait) — voir DATA_LINKS.md
# Tu dois avoir au minimum :
#   pmu_dataset_v2.parquet
#   pmu_course_raw_v2.parquet
```

Les étapes ci-dessous construisent des parquets `pmu_feat_*_v3.parquet`
puis les joignent dans `pmu_features_v3.parquet` (matrice finale).

## 1 · Feature engineering v3 (parallélisable)

| Ordre | Script                       | Sortie                              | Durée approx. |
|-------|------------------------------|-------------------------------------|---------------|
| 1.1   | `pmu_feat_blobs.py`          | `pmu_feat_blobs_v3.parquet`         | 10-20 min     |
| 1.2   | `pmu_feat_elo_ctx.py`        | `pmu_feat_elo_ctx_v3.parquet`       | 15-25 min     |
| 1.3   | `pmu_feat_h2h.py`            | `pmu_feat_h2h_v3.parquet`           | 30-60 min     |
| 1.4   | `pmu_feat_embeddings.py`     | `pmu_feat_emb_v3.parquet`           | 20-40 min     |
| 1.5   | `pmu_feat_musique_seq.py`    | `pmu_feat_musique_seq_v3.parquet`   | 5-10 min      |

Ces scripts sont indépendants — tu peux les lancer en parallèle si ton
PC a assez de RAM (~16 Go confortable, 32 Go idéal).

```bash
python pmu_feat_blobs.py       &
python pmu_feat_elo_ctx.py     &
python pmu_feat_h2h.py         &
python pmu_feat_embeddings.py  &
python pmu_feat_musique_seq.py &
wait
```

Puis agréger :

```bash
python pmu_feat_build_all_v3.py
```

Produit `pmu_features_v3.parquet` (~400-500 colonnes).

## 2 · Recherche d'hyperparamètres (Optuna)

```bash
# 200 trials, ~2-4 heures sur ton PC
python pmu_optuna_v3.py --n-trials 200 --study pmu_v3_main
```

Produit `pmu_optuna_study.db` (SQLite) + `pmu_best_params_v3.json`.
Tu peux reprendre la study à tout moment.

## 3 · Entraînement walk-forward (ranker + reward)

```bash
# 12 folds mensuels sur 2024-2026 + modèle final sur tout
python pmu_train_v3.py --params pmu_best_params_v3.json --walk-forward monthly
```

Produit :
- `pmu_model_v3.lgb` (modèle final ranker)
- `pmu_oof_v3.parquet` (prédictions out-of-fold pour calibration et stacking)
- `pmu_walkforward_v3.json` (AUC/ROI par mois avec IC 95 %)

## 4 · Stacking multi-modèles

```bash
# Entraîne LGBM + XGBoost + CatBoost + MLP + modèle favori-only,
# puis un meta-learner qui apprend à les combiner
python pmu_stack_v3.py --params pmu_best_params_v3.json
```

Produit `pmu_stack_v3.pkl` + `pmu_oof_stack_v3.parquet`.

## 5 · Calibration

```bash
# Isotonic + Beta calibration sur les predictions OOF du stack
python pmu_calibrate_v3.py --oof pmu_oof_stack_v3.parquet
```

Produit `pmu_calibrator_v3.pkl`.

## 6 · Conformal prediction (bornes d'incertitude)

```bash
python pmu_conformal_v3.py --oof pmu_oof_stack_v3.parquet --alpha 0.1
```

Produit `pmu_conformal_v3.json` (quantiles de couverture).

## 7 · Backtest multi-stratégies

```bash
# Simule flat, Kelly, Kelly fractional, edge-thresholded (2/3/5/10%),
# top-3 combi, dutching, etc. sur 12 folds mensuels.
# Sortie : tableau ROI par stratégie par mois + IC 95 % bootstrap.
python pmu_backtest_v3.py --folds 12 --bootstrap 1000
```

Produit `pmu_backtest_v3.csv` + `pmu_backtest_v3.png` (courbes de PnL).

## 8 · Contextual bandit (politique RL directe)

```bash
# Entraîne un contextual bandit (Thompson Sampling sur régression linéaire
# bayésienne) qui maximise directement le PnL.
# Utilisé en complément / validation du modèle principal.
python pmu_bandit_v3.py --train-years 2014-2024 --eval-year 2025
```

Produit `pmu_bandit_v3.pkl` + rapport ROI.

## 9 · Prédiction live

```bash
# Mise à jour du prédicteur pour intégrer le stack + calibration + conformal
python pmu_predict_today_v3.py --date DDMMYYYY
```

---

## Commande tout-en-un

```bash
./run_all_v3.sh
```

Lance étapes 1 → 8 dans l'ordre (skip les étapes dont la sortie existe déjà).

---

## Attentes réalistes

Sur la base des littératures paris hippiques + nos résultats actuels :

| Composant                          | Gain espéré (cumulé) |
|------------------------------------|-----------------------|
| Features blobs + ELO ctx + H2H     | +1 à 2 % AUC          |
| Ranker + reward weighting          | +0.5 à 1.5 % AUC      |
| Optuna 200 trials                  | +0.3 à 0.8 % AUC      |
| Stacking 4 modèles                 | +0.3 à 1 % AUC        |
| Calibration isotonic               | ROI +2 à 5 pts        |
| Conformal (skip si incertain)      | Réduction variance    |
| **TOTAL**                          | **AUC 0.82-0.84, ROI +30-45 %** |

Plafond théorique d'un marché efficient : on ne fera pas mieux que ~+50 %
ROI en régime durable. Si on dépasse 60-80 %, c'est suspect → re-tester
la fuite.
