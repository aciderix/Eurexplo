# Euromillions — Recherche Exploratoire

Exploration systématique de jeux de hasard : Euromillions, Loto, Keno, PMU, Courses UK.

## Structure

```
Eurexplo/
├── draws.json              # 1 936 tirages Euromillions
├── loto.json              # 7 332 tirages Loto France
├── keno.json              # 19 133 tirages Keno FDJ
├── pmu_races.json         # 5 688 courses PMU
├── kaggle/                # Données UK avec cotes (38 314 partants)
├── RAPPORT_*.md           # Tous les rapports de recherche
├── SPEC.md                # Spécifications API et acc\u00e8s
└── pmu_app.py             # Application Python compl\u00e8te PMU
```

## Accès API

Voir `SPEC.md` pour la documentation complète des API disponibles.

## Installation

```bash
pip install kaggle numpy scikit-learn torch flask
```

## Scripts disponibles

| Script | Description |
|--------|-------------|
| `pmu_app.py` | Application PMU avec modèle ML, backtest, 3 stratégies |
| `pmu_phone.py` | Version téléphone (Termux) |
