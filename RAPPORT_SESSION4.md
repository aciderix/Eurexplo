# Euromillions — Session 4 : Nouvelles explorations

**Date :** 2026-04-12
**Données :** 1936 tirages

---

## 1. Hypothèses invalidées

### 1.1 Mirror symmetry (1↔50, 2↔49)
Les numéros miroir n'apparaissent pas plus souvent que le hasard.
```
Observé: 367 tirages avec au moins une paire miroir
Null model: mean=382, std=17.6
z-score = -0.87σ → Non significatif
```

### 1.2 Position bias (chi2 corrigé vs order-stat null)
Toutes les positions sont compatibles avec l'uniforme. Les chi2 élevés venaient du artifact de tri (les order statistics ne sont pas uniforms sur [1,50]).
```
pos0: chi2(df=29)=19.1, p=0.92
pos1: chi2(df=37)=47.8, p=0.11
pos2: chi2(df=40)=43.9, p=0.31
pos3: chi2(df=37)=25.3, p=0.93
pos4: chi2(df=28)=27.1, p=0.51
→ Toutes les positions: CONFORMES à l'uniforme
```

### 1.3 Number entropy vs star2
L'entropie des numéros (concentration de la distribution) ne dépend pas de la valeur de star2.
L'hypothèse "quand star2 est haute, les numéros sont plus dispersés" est fausse.

### 1.4 Number range → star2
La range des numéros (max-min) ne corrèle pas avec star2 (r=0.028, p=0.22).

### 1.5 Last digit co-occurrence
z=1.12σ → Non significatif. Les numéros partageant le même dernier chiffre ne sont pas favorisés.

---

## 2. Découvertes mineures

### 2.1 Star12 : apparition
star12 est absent jusqu'en septembre 2016 (sur 968 premiers tirages). Puis apparition soudaine avec fréquence ~9%. Hypothèse : changement de machine à étoiles.

### 2.2 Pair entropy
Les deux étoiles sont toujours différentes (entropie = log(2) = 0.693 pour 100% des tirages). star1 ≠ star2 toujours — rule of the game.

### 2.3 Large gap → mean reversion (gap >= 8)
```
Après gap >= 8:  mean_next_star2 = 7.33
Après gap < 8:   mean_next_star2 = 8.17
t-test: t = -2.44, p = 0.015 *
```
Quand star2 fait un grand saut (gap >= 8), le tirage suivant tend à revenir vers la moyenne (mean reversion). Mais l'OOS test donne +0.23pp vs rolling mode — marginal.

### 2.4 Triangle numbers
4 triangle numbers dans un tirage → jp_rate = 42.9% (n=7).样本 trop petit pour conclure.

### 2.5 Primes
4+ primes → jp_rate = 33.3% (n=39). Là aussi sample petit.

---

## 3. Structure découverte : Star2 transition matrix

Matrice de transition star2[t] → star2[t+1] (probabilités conditionnelles en %) :

```
       1    2    3    4    5    6    7    8    9   10   11   12
  2:   0    7   10    3    3    3   24   14   17    3   10    3
  3:   0    0    4    6    6   14    9   10   16   13   14    9
  4:   0    1    1    5    7   11   12   14   19   11   11    9
  5:   0    2    4    7    7   11   13   14   14   13    9    7
  6:   0    1    4    7    8    8   15   14   13   12   10    8
  7:   0    2    4    6   10    8   11   13   18   12    8    8
  8:   0    1    3    3    8    8   11   16   19   13   12    6
  9:   0    3    4    6    8    8   12   14   12   11   13    8
 10:   0    0    3    3    6    9   10   12   15   16   18    9
 11:   0    1    4    4    5   12    9   13   10   15   17   10
 12:   0    2    3    4    8   10   12    7   13   11   14   16
```

Note : **star2=1 n'apparaît JAMAIS** dans tout le dataset (1936 tirages, 3872 étoiles). star1=1 apparaît 332 fois mais star2=1 = 0.

Structure visible :
- star2=2 → jump à 7 (24%), 9 (17%)
- star2=[3-9] → forte probabilité vers 9 et 8
- star2=10/11 → forte probabilité vers 11
- star2=12 → sticky (16% de rester à 12)

---

## 4. Bi-level Markov (s2[t-1], s2[t]) → s2[t+1]

Hypothèse : utiliser les 2 derniers star2 pour prédire le suivant.
Résultat : **négatif en strict OOS**.
```
Bilevel Markov (lookup only, strict OOS): 15.40%
Rolling mode:                               16.78%
Improvement:                                -1.38pp
```
Cause : sparsity. Chaque clé (s2[t-1], s2[t]) n'apparaît que ~13 fois en moyenne. Les prédictions dérivées sont bruitantes.

Le rolling mode reste supérieur car il agglomère toute l'information sans la fragmenter.

---

## 5. Joint state (s1, s2)[t-1] → s2[t] — résutat mixte

```
Fixed joint OOS: 15.40% (67/435)
Rolling mode:    16.78%
```
Le joint state n'améliore pas le rolling mode. Cependant, il y a une structure claire dans la distribution de star1 conditionnée par star2 : star1 est systématiquement BAS quand star2 est BAS.

```
star2=2  → mean_star1=1.00 (star1 toujours = 1)
star2=3  → mean_star1=1.54
star2=12 → mean_star1=5.71
```
star2 BAS force star1 BAS. star2 HAUT permet plus de variance.

---

## 6. Cross-prediction contemporary (s1[t] vs s2[t])

Corrélation star1 vs star2 contemporaine : r=0.036, p=0.11 — Non significative.

Pour certainess étoiles 1 spécifiques, la distribution de star2 diffère significativement (KS test, p<0.05) :
- star1=1 → star2 plus bas (mean=2.58 vs 8.15 global)
- star1=12 → star2 plus haut (mean=9.67 vs 8.15 global)

---

## 7. Entropie comme signal de confiance (validé)

```
w=50  Q1 (low entropy):  acc=16.45%
w=100 Q1 (low entropy):  acc=17.98%
w=200 Q1 (low entropy): acc=17.93%

w=100 Q5 (high entropy): acc=14.17%
Delta: +3.81pp
```
**Confirmé.** L'entropie de Shannon de la fenêtre glissante star2 est un signal de confiance robuste : quand les étoiles sont concentrées (low entropy), le rolling mode est plus précis.

---

## 8. Prime/Jackpot correlation (faible)

| N primes | n | jp_rate |
|----------|---|---------|
| 0 | 311 | 21.5% |
| 2 | 696 | 23.9% |
| 3 | 237 | 24.5% |
| 4 | 39 | **33.3%** |

Les tirages avec 4+ nombres premiers ont un taux de jackpot plus élevé. sample très petit (n=39) → pas de claim fort.

---

## 9. Synthèse session 4

### Invalidé (falsifié proprement)
- Mirror symmetry numbers
- Position bias (chi2 artifact corrigé)
- Number entropy → star2 coupling
- Number range → star2 correlation
- Last digit co-occurrence
- Bi-level Markov (star2 only)
- Joint state (s1,s2) → star2

### Confirmé
- **star2=1 absent** de tout le dataset (anomalie)
- **Large gap → mean reversion** (p=0.015, mais improvement marginale OOS)
- **Entropie = signal confiance** (Δ=3.8pp, confirmé)
- star12 apparition en 2016 (changement de machine probable)
- star2=12 sticky (16% de rester à 12)

### Mécanisme star2 : résumé
star2 est un processus autorégressif avec :
1. **Biais systématique** : mean=8.15 (vs 6.5 attendu pour uniforme [1,12])
2. **Autocorrélation persistante** : r=0.0775 (lag-1), r=0.1548 (lag-3)
3. **Mean reversion bornée** : contrainte [1,12] force les extrêmes à revenir
4. **Sticky states** : star2=10/11/12 ont forte probabilité de rester
5. **Régime changes** : introduction de star12 en 2016

