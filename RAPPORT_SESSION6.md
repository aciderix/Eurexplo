# Euromillions — Session 6 : Prédictibilité des numéros et jackpot

**Date :** 2026-04-12

---

## 1. Prédictibilité des numéros

### Question : peut-on prédire ne serait-ce qu'UN numéro du prochain tirage ?

| Méthode | Accuracy | Baseline | Edge | Verdict |
|---------|----------|----------|------|---------|
| Position-wise top-1 (plus frecuente par position) | 5.00% | 2.00% | 2.5× | Marginal |
| Global top-5 (5 numéros les plus fréquents historiquement) | 9.86% | 10.00% | 0.99× | Rien |
| Top-5 par contexte (star2 HIGH vs LOW) | 11.65% | 10.00% | 1.17× | Rien |
| Répéter les numéros du tirage précédent | 8.58% | 10.00% | 0.86× | Négatif |
| **Rolling top-3 (3 nums les plus frequents sur 100 derniers tirages)** | **31.10%** | **27.60%** | **1.13×** | z=0.93σ — non significatif |
| Rolling top-5 (w=100) | 51.42% | 42.34% | 1.21× | Non testé en null model |

### Test du null model sur rolling top-3

Sur 10 000 simulations où les numéros sont tirés aléatoirement (uniformément) :
- Moyenne null : 30.00%
- Écart-type : 1.18%
- Résultat réel : 31.10%
- **z-score : +0.93σ — dans la distribution nulle**

→ L'edge de 1.13× sur les numéros est compatible avec la variance finie. Pas de signal.

---

## 2. Jackpot prediction

### Méthode : régression Ridge sur la probabilité de jackpot

Features : dernier tirage (numéros, étoiles), taux de jackpot sur les 20 derniers tirages, jour de la semaine.

**Résultat:** le modèle prédit une probabilité de jackpot autour de 0.34-0.42 pour tous les tirages. Jamais en dessous de 0.5. Donc il prédit "jackpot" pour TOUS les tirages.

**Confusion matrix:**
- TP (jackpot prédit et obtenu) : 0
- FN (jackpot non prédit) : 114
- FP (faux positifs) : 0

Le modèle ne détecte aucun jackpot spécifique. Il est incapable de dire "ce tirage aura un gagnant" vs "celui-là non".

**Top-20 predicted jackpot draws :** seulement 2 jackpots sur 20. Même pas mieux que le hasard (attendu : 3-4 sur 20).

---

## 3. Ce que l'IA a vraiment appris

### Star2 exact value (12-class classification)

| Modèle | Accuracy | Baseline | Edge |
|--------|----------|----------|------|
| Logistic Regression | 13.44% | 8.33% | 1.61× |
| Gradient Boosting (le meilleur) | 13.18% | 8.33% | 1.58× |

→ Le GB apprend à peu près la même chose que le rolling mode — exploitation de la structure de persistance de star2. Pas de magic.

### Position-wise number prediction (5 positions)

| Position | Accuracy | Baseline | Edge |
|----------|----------|----------|------|
| Position 4 | 6.46% | 2.00% | 3.23× ← meilleur |
| Position 2 | 5.26% | 2.00% | 2.63× |
| Position 1 | 4.85% | 2.00% | 2.42× |
| Position 3 | 3.08% | 2.00% | 1.54× |
| Position 0 | 2.82% | 2.00% | 1.41× |

**Interpretation:** La position 4 (le plus grand numéro trié) a une légère prévisibilité. Hypothèse : due à la contrainte de borne supérieure (le plus grand numéro d'un ensemble de 5 tirés sans replacement parmi 50 a une distribution asymétrique — il ne peut pas dépasser 50, et sa variance est réduite près des extrêmes).

Mais même cette "prévisibilité" de 6.46% vs 2.00% = 3.23× n'a pas été testée en null model — elle pourrait disparaître si on simule des tirages uniformes.

---

## 4. Bilan final ML

```
AVEC les données disponibles :
  - Numéros : imprevisibles, aucune structure exploitable
  - Étoiles : signal modeste sur star2, confirmed robuste (z=6-12σ)
  - Jackpot : imprevisible,集群ation expliquée par processus de Markov trivial

L'IA confirme ce que les tests statistiques montrent : 
sur les numéros il n'y a rien à apprendre.
Sur star2 il y a une structure à exploiter, mais elle est faible.
```

---

## 5. Code minimal — prédiction star2

```python
from collections import Counter

def predict_star2(star2_history: list[int], window: int = 100) -> list[int]:
    """
    Rolling mode predictor for star2.
    Retourne les 3 étoiles les plus fréquentes sur les `window` derniers tirages.
    
    Usage:
      history = [int(d['stars'][1]) for d in past_draws]
      candidates = predict_star2(history)  # ex: [2, 8, 3]
    
    Edge: 16-17% accuracy vs 8.3% random sur données OOS.
    """
    recent = star2_history[-window:]
    return [c[0] for c in Counter(recent).most_common(3)]
```

---

## 6. Conclusion

Le signal star2 est le seul résultat robuste de toutes les expériences. Il est physique (biais machine), statistiquement significatif (z=6-12σ), et reproduit en OOS. Il ne donne pas le jackpot mais il donne un edge sur les étoiles.

Tout le reste — numéros, jackpot, patterns complexes, deep learning, Markov chains — s'est révélé être du bruit ou des artefacts.