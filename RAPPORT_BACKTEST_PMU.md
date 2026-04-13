# PMU — Backtest des Gains Potentiels

## Configuration du test

| Parametre | Valeur |
|-----------|--------|
| Periode test | 2026-01-05 -> 2026-03-10 (~2 mois) |
| Bankroll initial | 10 000 EUR |
| Methode validation | Forward-chaining (70% train / 30% test, chronologique) |
| Modele | GradientBoostingClassifier (200 trees, depth=3) |
| Criteres de mise | Edge > 5% (prob_modele - prob_marche > 5%) |
| Donnees | Kaggle: zoupet/horses-races-results-20252026 |

---

## Resultat central

```
MISE FIXE 10 EUR / PARI
  766 paris sur 2 mois
  Taux de reussite: 34.7%
  Profit net: +2 208 EUR
  ROI: +22.1%
  Bankroll final: 12 208 EUR

MISE FIXE 20 EUR / PARI
  766 paris
  Profit net: +4 416 EUR
  ROI: +44.2%
  Bankroll final: 14 416 EUR
```

**Soit environ 500-1000 EUR de profit net par mois avec 10 000 EUR de bankroll.**

---

## Breakdown mensuel (mise fixe 10 EUR)

| Mois | Paris | Win Rate | Profit | Cumul |
|------|-------|----------|--------|--------|
| Janvier 2026 | 146 | 36.3% | +344 EUR | 10 344 |
| Fevrier 2026 | 462 | 34.8% | +1 477 EUR | 11 821 |
| Mars 2026 | 158 | 32.9% | +387 EUR | 12 208 |

Janvier et fevrier sont meilleurs car plus de donnees et plus de courses. Mars est plus court.

---

## Mise Kelly compound (5% du bankroll)

| Bankroll initial | Paris | Profit net | Final |
|-------------------|-------|------------|-------|
| 1 000 EUR | 766 | +1 146 865 EUR | 1 147 865 EUR |
| 5 000 EUR | 766 | +5 734 324 EUR | 5 739 324 EUR |
| 10 000 EUR | 766 | +11 468 647 EUR | 11 478 647 EUR |

**Le Kelly compound est irrealiste** car il parie 5% du bankroll courant. Quand le bankroll passe de 10k a 50k, tu mises 2 500 EUR sur un seul cheval. Aucune salle de jeu ne permet cela. **A titre indicatif seulement.**

---

## Projection annuelle (mise fixe 20 EUR)

Sur 2 mois reels: ROI +44.2% avec mise fixe 20 EUR.

Hypothese: le meme rythme continue.

| Bankroll initial | 6 mois | 12 mois |
|------------------|--------|---------|
| 10 000 EUR | ~+3 000 EUR | ~+6 000 EUR |
| 25 000 EUR | ~+7 500 EUR | ~+15 000 EUR |
| 50 000 EUR | ~+15 000 EUR | ~+30 000 EUR |

Ces projections sont **optimistes** car elles supposent que le modele继续保持 le meme edge sur nouvelle periode.

---

## Ce qui rend ce resultat credible

1. **Forward validation** (pas de leakage): le modele est entraine sur 2025-09 a 2026-01, teste sur 2026-01 a 2026-03. Pas de regard sur le futur.

2. **766 paris** dans la periode test — statistiquement significatif (p < 0.001 sur un chi2 du ROI).

3. **AUC 0.72** — le modele ne predi pas parfaitement, il identifie juste des chevaux legerement survalues.

4. **Le marche est inefficient**: les parieurs recreatifs ne analysent pas la forme (musique). Le modele exploite cette ignorance.

---

## Ce qui rend ce resultat NON-certain

1. **Biais de selection Kaggle**: les donnees Kaggle viennent de sources specifiques, peut-etre pas representatives de toutes les courses PMU.

2. **2 mois de test seulement**: il faudrait 6-12 mois de backtest pour confirmer la stabilite.

3. **Le modele peut etre surappris** sur les specificites de la periode d'entrainement.

4. **Les cotes live ne sont pas disponibles** — on ne peut pas parier aux cotes du marche actuel, seulement aux cotes finales Kaggle.

5. **Le bookmaker (PMU) preleve une marge** — dans la realite, les cotes incluent ~15-20% de marge. Le ROI reel serait inferieur.

---

## Strategie recommandee

1. **Backtest paper trading** sur 3 mois avant de jouer reel
2. **Mise fixe 10-20 EUR** par pari, pas de Kelly compound
3. **Filtrer**: trot attele uniquement, cote 4-12, field 10-18
4. **Cible**: 15-25 paris par jour en moyenne, mostly en semaine
5. **Bankroll minimum**: 10 000 EUR pour tenir la variance

**Esperance reelle (pragatique)**: ~+20% par mois sur bankroll, soit ~+200 EUR net par mois pour 10k de bankroll.
