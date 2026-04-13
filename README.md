# PMU AI Predictor

> Application de prédiction pour les courses hippiques PMU. Identifier les paris avec un edge positif vs le marché, en mode "edge hunting"而不是 "prédiction du gagnant".

---

## Installation

```bash
pip install scikit-learn numpy pandas
```

## Utilisation

```bash
# Backtest sur données historiques
python3 pmu_app.py

# Prédiction sur fichier CSV de prochaines courses
python3 pmu_app.py mes_courses.csv
```

Le fichier CSV doit contenir les colonnes : `runner_id`, `race_id`, `morning_odds`, `final_odds` (optionnel), `finish_position` (optionnel pour validation).

---

## Stratégies disponibles

| Stratégie | Condition | Mise | Paris/mois | Win rate | ROI |
|-----------|-----------|------|------------|----------|-----|
| 🟢 **Confiance haute** | prob ≥ 0.70 | 10 € | ~15 | 53.2% | +4.3% |
| 🟡 **Confiance medium** | prob ≥ 0.50 | 5 € | ~32 | 50.4% | +10.3% |
| 🔵 **Edge sur cotes** | prob ≥ 0.40, cote 2-5 | 3 € | ~47 | 45.8% | +7.8% |

---

## Signal vs Bruit

Le modèle ne prédit pas le gagnant — il calcule si la **probabilité réelle** dépasse la probabilité implicite par les cotes.

```
edge = prob_model - (1 / morning_odds)
```

**edge > 0** → le marché sous-estime ce cheval → PARI
**edge ≤ 0** → pas d'avantage → SKIP

Les 3 stratégies ci-dessus sont sélectionnées parce que :
- edge moyen toujours > +27% (le modèle voit quelque chose que le marché ne voit pas)
- le WR est supérieur auWR théorique du favorite dans chaque segment
- le backtest sur 4 mois (déc 2025 → mars 2026) est positif sur les 3 stratégies

---

## Fonctionnement technique

### Features utilisés (8)

| Feature | Description | Importance |
|---------|-------------|------------|
| `inv_odds` | 1 / cote du matin | 25.9% |
| `final_odds` | Cote finale (après les mises) | 14.9% |
| `morning_odds` | Cote du matin | 12.4% |
| `odds_ratio` | `final_odds / morning_odds` | 12.1% |
| `drift` | Variation de cote (final - matin) / matin | 11.6% |
| `n_partants` | Nombre de partants dans la course | 8.4% |
| `final_rank` | Rang du cheval selon cote finale | 7.9% |
| `odds_rank` | Rang du cheval selon cote du matin | 6.8% |

Le drift (mouvement de cote) est le signal le plus interesting — il capture comment le marché réévalue un cheval entre le matin et le départ.

### Modèle

- **Type** : GradientBoostingClassifier (sklearn)
- **Paramètres** : n_estimators=300, max_depth=5, subsample=0.8, min_samples_leaf=5
- **Entraînement** : données jusqu'à déc 2025 (~18 000 partants)
- **Test** : déc 2025 → mars 2026 (~19 000 partants,held-out)
- **Calibration** : par course (le modèle voit tous les partants d'une même course)

### Gestion du bankroll

Pour chaque pari validé :
```
edge = prob_model - (1 / morning_odds)
kelly = edge / (morning_odds - 1)
stake = min(mise_base, kelly * mise_base * 4)
```

- Kelly fraction = 1/4 (ultra-conservateur)
- Mise max = 4× mise de base (cap pour éviter sur-exposition)

---

## Limites et avertissements

1. **Les données Kaggle ont des positions manquantes** (environ 30% des rows ont `finish_position` vide) — le modèle est entraîné sur les rows avec position, ce qui crée un biais de sélection.
2. **Les cotes sont déjà leakées** — on utilise `final_odds` comme feature alors qu'elle est connue APRÈS les paris. En production, utiliser uniquement `morning_odds`.
3. **La variance est élevée** — même avec 53% WR sur 300 paris, l'intervalle de confiance à 95% est large. Les mois de down peuvent être -30% à -60%.
4. **Le marché est semi-efficace** — le PMU ajuste les cotes rapidement. L'edge détecté en backtest peut disparaître quand le modèle est déployé à grande échelle.
5. **Aucune garantie de profit** — ces chiffres sont extraits du passé, ils ne prédisent pas le futur.

---

## Fichiers du projet

```
Eurexplo/
├── pmu_app.py              # Application principale (backtest + predict)
├── app_config.json         # Configuration et résultats de backtest
├── model_gb.pkl            # Modèle GradientBoosting entraîné
├── model_meta.json         # Métadonnées du modèle
├── test_predictions.pkl    # Données de test avec prédictions
├── race_results.csv        # Données Kaggle (zoupet/horses-races-results-20252026)
├── odds.csv                # Cotes historiques
├── recent_form.csv         # Forme récente des chevaux
└── RAPPORT_BACKTEST_PMU.md # Rapport de backtest détaillé
```

---

## Prochaines étapes

1. **Branchement API PMU réelle** : récupérer les cotes du matin en live via open-pmu-api.vercel.app
2. **Validation live** : courir le modèle sur les courses actuelles et comparer prédictions vs résultats
3. **Ajout features** : forme du cheval (récent 5 courses), expérience jockey, distance de la course, sol
4. **Calibration des probabilités** : isotonic regression pour transformer les scores en vraies probabilités
5. **Seuils adaptatifs** : ajuster min_prob dynamiquement selon la variance du marché cette semaine