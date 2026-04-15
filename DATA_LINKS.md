# Parquet data files (Google Drive)

Ces fichiers sont trop volumineux pour Git (>100 MB). Lien de téléchargement direct
(respecter l'ordre de dépendance pour rebuilder si besoin) :

| Fichier                           | Taille   | Rôle                                   | Lien |
|-----------------------------------|----------|----------------------------------------|------|
| `pmu_course_raw_v2.parquet`       | 1.66 GB  | JSON bruts par course (blobs)          | https://drive.google.com/uc?export=download&confirm=t&id=1KuJiQu0HcgqmlH1qSe6dEBt6jK_QKgco |
| `pmu_dataset_v2.parquet`          | 187.1 MB | Dataset flat (1 ligne / (course,cheval)) — **base** | https://drive.google.com/uc?export=download&confirm=t&id=1DzVoNqV8IZN7jSpeGjz9ZD-XMGCkLFx4 |
| `pmu_feat_elo.parquet`            | 63 MB    | Features ELO (cheval/driver/entraineur) | https://drive.google.com/uc?export=download&confirm=t&id=1GpHtg3IUG27BgNVXepi2mwBCM8DzLegx |
| `pmu_feat_market.parquet`         | 44.8 MB  | Features marché / cotes                 | https://drive.google.com/uc?export=download&confirm=t&id=1rqfFcvlBN9O18WZkks98T9dWhiq4CJZ8 |
| `pmu_feat_musique.parquet`        | 13.7 MB  | Features musique parsée                 | https://drive.google.com/uc?export=download&confirm=t&id=1fY0QFNKUbUacuMbV0OOSzh5Q5kPlMwu- |
| `pmu_feat_rolling.parquet`        | 154 MB   | Features rolling 30/90/365j             | https://drive.google.com/uc?export=download&confirm=t&id=1D4whskae8vcem4iYw2UKDnPbC5gI12VC |
| `pmu_feat_target_enc.parquet`     | 94.9 MB  | Target-encoding lissé (pere/mere/...)   | https://drive.google.com/uc?export=download&confirm=t&id=18jWU3R3VQToJaRfUdgTh9EagRhan_sRD |
| `pmu_features_v2.parquet`         | 515.2 MB | **Matrice finale jointe** (229 cols)    | https://drive.google.com/uc?export=download&confirm=t&id=1dBcCtzMFVeR5LHcrnvUKUGKQVGJ0YG0m |

## Téléchargement rapide avec `gdown`

```bash
pip install gdown
gdown "https://drive.google.com/uc?export=download&confirm=t&id=1DzVoNqV8IZN7jSpeGjz9ZD-XMGCkLFx4" -O pmu_dataset_v2.parquet
# ... idem pour les autres
```

## Pipeline de reconstruction (si parquets manquants)

1. `pmu_build_dataset_v2.py`  → `pmu_dataset_v2.parquet` + `pmu_course_raw_v2.parquet`
2. `pmu_feat_musique.py` / `pmu_feat_market.py` / `pmu_feat_rolling.py` /
   `pmu_feat_elo.py` / `pmu_feat_target_enc.py` → les 5 `pmu_feat_*.parquet`
3. `pmu_feat_build_all.py` → `pmu_features_v2.parquet`
4. `pmu_train_v2.py` → `pmu_model_v2.lgb` + `pmu_model_v2.features.json`

## Usage courant

- Prédiction live : `pmu_predict_today.py --date DDMMYYYY`
- Backtest du jour vs résultats réels : `pmu_backtest_today.py --date DDMMYYYY`
- Audit de fuite : `pmu_leakage_check.py`
