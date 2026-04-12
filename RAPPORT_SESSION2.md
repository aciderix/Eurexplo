# Euromillions — Session 2 : Nouvelles explorations

**Date :** 2026-04-12
**Données :** 1936 tirages (inchangé)

---

## 1. Jackpot Clustering — z = -21.6σ

### Test des runs sur les jackpots

```python
# Répartition des intervalles entre jackpots consécutifs :
gap=1:  21.3%   # jackpot → jackpot au tirage suivant
gap=2:  19.0%
gap=3:  10.2%
gap=4:  14.3%
gap=5:   8.2%
gap>=6:  8.3%

Runs test :
  Observed = 348 clusters
  Expected = 683.2
  z = -21.629
  p < 0.0001
```

**Interprétation :** Les jackpots se regroupent en "tempêtes" (bursts). Après un jackpot, la probabilité de re-jackpot est significativement plus haute que la moyenne. La distribution des gaps est bimodale : soit le jackpot suivant arrive dans les 2 tirages (40%), soit il faut attendre beaucoup plus longtemps.

**Falsification :** Ce n'est pas un artifact du système de notation. Les jackpots sont déterminés par la combination 5+2, qui est indépendante d'un tirage à l'autre. Ce clustering est réel et statistiquement robuste (z = -21.6σ).

**Mécanisme suspecté :** Après un jackpot, le jackpot repart de zéro. Mais les combinaisons "faciles à jouer" (petits nombres, suites) reviennent en jeu. Les joueurs continuent de jouer ces combinaisons sur plusieurs tirages, augmentant la probabilité de partage en cas de victoire.

---

## 2. Distribution théorique des étoiles

### Découverte : star1 = minimum, star2 = maximum de 2 tirages

Les étoiles sont tirées avec replacement. Le système de notation les trie (petite d'abord). Donc :

- `star1` = valeur MINIMALE des 2 tirages
- `star2` = valeur MAXIMALE des 2 tirages

Cela change complètement l'analyse théorique.

```python
# Distribution théorique du minimum (star1) :
P(min = k) = [(13-k)/12]^2 - [(12-k)/12]^2

# Distribution théorique du maximum (star2) :
P(max = k) = [k/12]^2 - [(k-1)/12]^2

# Chi2 vs theory :
Star1 chi2(df=11) = 91.54  # Rejeté
Star2 chi2(df=11) = 135.06  # Rejeté encore plus fort
```

Les étoiles ne suivent même pas leur propre modèle théorique (minimum/maximum de deux uniformes). Elles sont significativement biaisées vers les valeurs extrêmes de chaque côté.

### star1 : déficit aux valeurs extrêmes hautes

| star1 | Observé | Théorique | z |
|-------|---------|-----------|---|
| 1 | 332 | 309 | +1.30 |
| 2 | 354 | 282 | +4.27 |
| 7 | 121 | 148 | -2.21 |
| 10 | 33 | 67 | -4.17 |
| 11 | 15 | 40 | -3.99 |
| 12 | 0 | 13 | -3.67 |

star1 = 12 n'apparaît JAMAIS. star1 = 11 seulement 15 fois vs 40 attendues. La machine refuse les deux étoiles égales ou proches du maximum.

### star2 : excès aux valeurs moyennes-hautes, déficit aux extrêmes

| star2 | Observé | Théorique | z |
|-------|---------|-----------|---|
| 1 | 0 | 13 | -3.67 |
| 6 | 181 | 148 | +2.72 |
| 7 | 223 | 175 | +3.65 |
| 8 | 252 | 202 | +3.54 |
| 9 | 285 | 229 | +3.73 |
| 12 | 170 | 309 | -7.92 |

star2 = 12 (maximum) apparaît 170 fois vs 309 attendues — déficit majeur. La valeur 12 est tirée 45% moins souvent que théorique.

**Conclusion :** La machine à étoiles ne tire pas deux nombres uniformes [1-12] de façon indépendante. Elle favorise les combinaisons où les deux étoiles ne sont ni trop basses ni trop hautes, avec une préférence pour les valeurs 6-9.

---

## 3. Couplage star1 ↔ star2

### Test d'indépendance (chi2 sur contingency table 12×12)

```python
Chi2 (df=121): très élevé, p < 0.0001
→ REJETÉ : star1 et star2 sont dépendants
```

Quand star1 est haute (10-11), star2 est presque toujours dans les valeurs extrêmes aussi. La médiane de star2 sachant star1=11 est 12.

**Matrice de transition des régimes :**

```
              → low-low  → mixed  → high-high
From low-low      28%      56%        16%
From mixed        24%      58%        19%
From high-high    24%      55%        21%
```

Pas de persistance anormale des régimes — le système est quasi-markovien. Mais le Jackpot rate diffère : low-low = 25.8%, high-high = 20.2%.

---

## 4. Formule star2 : Rolling Mode vsbiais marginal

### Test avec baseline corrigée

```python
# Comparaison : rolling mode vs sample FROM OBSERVED MARGINAL
window=50:  mode=15.6%, biased_random=10.9%, improvement=1.43×
window=100: mode=16.5%, biased_random=11.6%, improvement=1.42×
window=200: mode=16.0%, biased_random=9.7%,  improvement=1.65×
```

Même en comparant avec un random qui tire depuis la distribution marginale biaisée (pas uniforme), le rolling mode conserve un edge de 1.4-1.6×.

### Combinaison star1 → star2 via couplage

```python
Conditional (star1_mode -> star2): acc= ~0.13
Rolling mode star2:              acc= 0.165
→ Le rolling mode surpasse le prédicteur conditionnel
```

Tenter de，利用 star1 来 prédire star2 n'apporte rien. Le signal de rolling frequency domine.

### Prédicteur combiné weighted

```python
Score(s) = rolling_freq(s) × P(s | star1_mode)
Weighted: acc=0.163
Plain rolling mode: acc=0.165
→ La complexité ajoutée ne aide pas
```

---

## 5. Analyse géométrique des numéros

### Numéros sur un cercle [1-50]

```python
Mean angular gap (réel): 1.0466
Null model (random):     1.0344 ± 0.2538
z-score: 0.048
→ Géométriquement Indistinguible de aléatoire
```

Aucune structure géométrique. Les numéros sont uniformément distribués sur le cercle. Pas de clustering directionnel.

---

## 6. Analyse des prizes (5+1 et 5+0)

### Montant des gains selon longueur du roll

```
5+1 | roll court (<=3): mean=€412,928 (n=1273)
5+1 | roll long (>=10): mean=€947,972 (n=122)
     → x2.3× quand le jackpot a roulé longtemps

5+0 | roll court (<=3): mean=€81,494
5+0 | roll long (>=10): mean=€48,370
     → division par 1.7× (plus de gagnants se partagent)
```

Le 5+0 diminue quand le roll est long car plus de gens jouent (jackpot积累), augmentant le nombre de gagnants 5+0. Le 5+1 augmente car le jackpotaccumule.

---

## 7. Falsifications négatives (session 2)

| Test | Résultat | Verdict |
|------|----------|---------|
| star2 mode dans récent (batches de 50) | Très variable (0.11-0.58) | Instable — certains batches font mieux, d'autres moins |
| Géométrie des numéros | z = 0.05 | Pas de structure |
| Prédicteur star1 → star2 | 0.13 vs 0.165 | Rolling mode surpasse |
| Regime-aware predictor | 0.142 vs 0.156 | Complexity ne aide pas |
| (11,12) pair | z = -2.3 | Déficitaire, pas excessif |

---

## 8. Synthèse session 1 + 2

### Ce qui tient après double falsification

| Signal | Session 1 | Session 2 | Tient ? |
|--------|-----------|-----------|---------|
| star2 non-uniforme | Oui | Oui (chi2=135) | Oui, robuste |
| star2 autocorr lag-3 | Oui (z=6.85) | Confirmé (z=3.5 vs marginal) | Oui mais mécanisme différent |
| star2 rolling mode OOS | 16.6% vs 8.3% | 1.4-1.6× vs biased baseline | Oui |
| Jackpot clustering | Nouveau | z=-21.6, très fort | Oui, fort |
| star1 = min théorique | Non testé | Rejeté (chi2=91) | Oui |
| star12 rupture 2016 | Oui | Confirmé | Oui |
| Numéros uniforme | Oui | Oui + géométrie OK | Oui |

### Ce qui ne tient pas

- star1 autocorr (session 1): lag-1 = 0.02, z = 0.95 — non significatif
- Pair (11,12) excess:'était un calcul erroné — il est déficitaire
- Regime-aware predictor: ne bat pas rolling mode simple
- Couplage star1→star2 predictor: domination du rolling mode

---

## 9. Exploration recommandée pour session 3

1. **Jackpot clustering mécanisme :** analyzer si les combinaisons numérotiques sont differentes dans les "tempêtes de jackpot"
2. **Effet du montant du jackpot :** est-ce que les jackpots > €100M ont une dynamique différente ?
3. **Saisonnalité :** y a-t-il des périodes de l'année avec plus de jackpots ?
4. **Récupération après une tempête :** après un cluster de jackpots, y a-t-il une période creuse ?
5. **Couplage nombre-étoile :** les numéros 1-50 ont-ils des patterns selon les valeurs d'étoile ?
