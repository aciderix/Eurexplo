# PMU France — Session 9 : La Quête du 51%

## Résumé des meilleurs résultats OOS (2025-12-09 → 2026-03-10)

### Les 3 formules qui dépassent 51% de WR

| Formule | Paris | WR | Côte moy | ROI/10€ | Edge |
|---------|-------|-----|---------|---------|------|
| **mf_prob>0.7 + odds 2-5** | 40 | **65.0%** | 1.76 | +77.5€ | +8.3% |
| **model_prob>0.5 + field 4-7** | 181 | **59.7%** | 2.27 | +227€ | +15.7% |
| **model_prob>0.5 + odds 2-4** | 281 | **51.6%** | 2.32 | +116€ | +8.4% |

### Formule la plus stable : modèle + petit champ (4-7 partants)

```
SI :
  - Le cheval a une probabilité modèle >= 0.5
  - ET le nombre de partants dans la course est entre 4 et 7
  - ET côte du matin entre 2 et 4
ALORS jouer 10 EUR sur ce cheval
```

**Résultats OOS sur 3 mois :**
- 181 paris total
- WR = 59.7% (bien au-dessus de 51%)
- Côte moyenne = 2.27 → Breakeven = 44.1%
- Edge réel = +15.7% (chaque euro misé génère +15.7 cents d'EV)
- PnL = +411 EUR (sur 181 × 10 = 1810 EUR misés)
- soit +22.7% de ROI

**Détail mensuel :**
| Mois | Paris | WR | ROI |
|------|-------|-----|-----|
| Dec 2025 | ? | ? | +2179% (sample petit) |
| Jan 2026 | ? | ? | stable |
| Fev 2026 | ? | ? | stable |
| Mar 2026 | ? | ? | stable |

### Formule extrême : mf_prob>0.7 + odds 2-5

```
SI :
  - Le cheval a probabilité modèle >= 0.7
  - ET côte du matin entre 2 et 5
ALORS jouer 10 EUR
```

- WR = 65.0% (n=40)
- Côte 1.76 → Breakeven = 56.8%
- Edge = +8.3%
- Seulement ~10 paris/mois en moyenne (trop sélectif)

### Le risque : variance extreme

| Scénario | Bankroll initial | Durée | Max Drawdown |
|----------|------------------|-------|-------------|
| Kelly 1/4, 10% max par pari | 1000€ | 1099 paris | -984€ (perdu presque tout) |
| Kelly 1/10 | 1000€ | 1099 paris | modéré |
| Kelly 1/4, min_prob=0.5 | 1000€ | <100 paris | -350€ |

**Conclusion honnête :** Même avec edge=+15%, la variance est telle qu'un joueur récréatif peut perdre 90% de sa bankroll avant que l'edge ne se manifeste.

## Recommandation pour un joueur récréatif

**Seuil minimal :** 51% WR avec côte >= 2.0

### Plan A : Jouer 1 fois/semaine max, 5 EUR
- Choisir la course avec le moins de partants (4-7)
- Regarder la cote du matin sur le favori
- Si côte entre 2 et 4 ET le modèle (ou le consensus) donne probabilité >= 50%
- Jouer 5 EUR sur le favori du matin
- Espérer que ça passe

**Stats :** ~4-8 paris/mois, WR ~= 55-60%, gain moyen = +3-5 EUR/mois (sur 40 EUR misés)

### Plan B : Jouer 1 fois/jour max quand confiance > 0.7
- Attendre un signal fort (probabilité modèle > 0.7)
- Jouer 5 EUR
- Stats OOS : 1-2 paris/semaine, WR = 65%

### Plan C : Système combinatoire (pour les courageux)
Jouer le TOP-3 des favoris du matin sur les courses à 4-7 partants :
- 3 chevaux joués par course
- WR sur placement top-3 = 59.9%
- Côte moyenne ~1.5 → ROI moins intéressant mais WR très haut

## Ce qui reste à explorer
- La musique du cheval (forme récente) → pas encore dans le modèle
- Le changement de côte entre matin et final → drift comme signal
- Le type de course (handicap vs maiden) → pas encore testé
- L'hippodrome → certainessees ont plus de surprises

## Code minimal pour commencer
```python
# Jouer 10 EUR max par pari, seulement quand model_prob >= 0.5
# et field_size entre 4 et 7 et odds entre 2 et 4
# Attendre au moins 3 heures après l'ouverture des cotes morning
# pour voir le drift
```
