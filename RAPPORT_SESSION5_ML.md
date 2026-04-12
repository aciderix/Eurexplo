# Euromillions — Session 5 : Apprentissage Profond

**Date :** 2026-04-12
**Données :** 1936 tirages

---

## 1. Test de fuite (leak detection)

### 1.1 Leak detection —star2 comme target

Tentative initiale : utiliser star2 comme feature ET comme target.

```
TARGET A: star2 value — AVEC leakage (star2 contemporain dans features)
  LogReg:  acc=0.812  edge=4.57x
  RF-200: acc=0.918  edge=5.16x
  GB-200: acc=1.000  edge=5.62x ← 100% — absurde

TARGET B: star2 >= mode — AVEC leakage
  GB-200: acc=1.000  AUC=1.000 ← leak total
```

→ **GB prédit star2 en utilisant star2 lui-même.** Ce n'est PAS un modèle prédictif. Corrigé.

---

## 2. Configuration leak-free

**Règle stricte :** pour prédire le tirage N, on n'utilise que les données des tirages 1..N-1.

Features construites uniquement à partir de `star2[:t]` pour prédire `star2[t]` :

| Feature | Description |
|---------|-------------|
| `s2_lag2` | star2[t-1] |
| `s2_lag3` | star2[t-2] |
| `s1_lag1` | star1[t-1] |
| `s2_mode50` | mode star2 des 50 derniers tirages passés |
| `s2_mode100` | mode star2 des 100 derniers tirages passés |
| `s2_mode200` | mode star2 des 200 derniers tirages passés |
| `s2_ent50` | entropie de Shannon fenêtre 50 |
| `s2_ent100` | entropie de Shannon fenêtre 100 |
| `s2_mean50`, `s2_mean100` | moyenne |
| `s2_std50`, `s2_std100` | écart-type |

Split : 80% train (tirages 2-1549), 20% test (tirages 1550-1936).

---

## 3. GB exact value — star2 (12-class)

```
Model: GradientBoostingClassifier(n_estimators=500, max_depth=5)

Décomposition par valeur de star2 (test set):
  star2= 2 (n=  2): correct=0/2   (trop rare pour prédire)
  star2= 3 (n= 15): correct=0/15
  star2= 4 (n= 13): correct=0/13
  star2= 5 (n= 19): correct=2/19
  star2= 6 (n= 28): correct=5/28   acc=17.9%
  star2= 7 (n= 37): correct=10/37 acc=27.0% ← BEST
  star2= 8 (n= 44): correct=3/44   acc= 6.8%
  star2= 9 (n= 52): correct=4/52   acc= 7.7%
  star2=10 (n= 59): correct=9/59   acc=15.3%
  star2=11 (n= 49): correct=11/49  acc=22.4%
  star2=12 (n= 69): correct=8/69   acc=11.6%

Overall:  acc=0.1344  baseline(random)=0.0833  edge=1.61x
Rolling mode baseline: 0.1628  edge=0.83x ← GB SOUS le rolling mode
```

**Observation critique :** Le GB prédit principalement 7 et 11 (les modes récents). Il est incapable de capturer les transitions. Il sous-performe le simple rolling mode sur ce target.

---

## 4. KNN — alternative non-paramétrique

KNN ne fait pas d'hypothèses paramétriques. Il trouve les voisins les plus proches dans l'espace des features.

```
KNN k=  3: acc=0.1137  edge=1.36x
KNN k=  5: acc=0.1189  edge=1.43x
KNN k= 10: acc=0.1318  edge=1.58x
KNN k= 20: acc=0.1525  edge=1.83x ← BEST KNN
KNN k= 50: acc=0.1499  edge=1.80x
```

KNN avec k=20 trouve une structure locale utile. Cependant toujours sous le rolling mode (0.1628).

---

## 5. Feature importance (GB, target exact star2)

```
s2_std100      : 0.1501  ← Variance locale = signal le plus fort
s2_ent50       : 0.1449  ← Concentration des étoiles récentes
s2_std50       : 0.1448
s2_ent100      : 0.1405
s2_mean100     : 0.1152
s2_mean50      : 0.1035
s2_lag3        : 0.0579
s2_lag2        : 0.0561
s1_lag1        : 0.0513
s2_mode50      : 0.0213  ← Le mode lui-même est peu informatif
s2_mode100     : 0.0143
s2_mode200     : 0.0093
```

**Résultat contre-intuitif :** La volatilité locale (std, entropy) est plus prédictive que le mode. Cela suggère que les **régimes de haute variance** dans star2 sont plus prévisibles — ou que GB utilise la variance comme proxy pour la distribution.

---

## 6. Hyperparameter sweep (GB, target exact star2)

```
depth=3  n=100: acc=0.1189  edge=1.43x
depth=3  n=300: acc=0.1111  edge=1.33x
depth=3  n=500: acc=0.0982  edge=1.18x
depth=5  n=100: acc=0.1214  edge=1.46x
depth=5  n=300: acc=0.1292  edge=1.55x
depth=5  n=500: acc=0.1344  edge=1.61x ← best
depth=8  n=100: acc=0.1189  edge=1.43x
depth=8  n=300: acc=0.1266  edge=1.52x
depth=8  n=500: acc=0.1318  edge=1.58x
```

Plus de profondeur et plus d'estimateurs améliorent légèrement. Pas de surapprentissage visible (le test set reste stable).

---

## 7. Prédiction des positions de numéros

使用 les mêmes features (étoiles passées) pour prédire les numéros du tirage suivant.

```
Position 0 (minimum): acc=0.0672  baseline=0.0200  edge=3.36x ***
Position 1:           acc=0.0388  baseline=0.0200  edge=1.94x *
Position 2 (médiane):  acc=0.0258  baseline=0.0200  edge=1.29x
Position 3:           acc=0.0284  baseline=0.0200  edge=1.42x
Position 4 (maximum):  acc=0.0646  baseline=0.0200  edge=3.23x ***
```

**Résultat inattendu :** Les positions 0 et 4 (min et max des 5 numéros) sont prévisibles avec un edge de 3x à partir des seules features d'étoiles. Ce n'est PAS un artifact de fuite (features ne contiennent pas les numéros contemporains) — les étoiles et numéros partagent peut-être un facteur causal non observé (configuration machine, environnement).

---

## 8. Synthèse ML

### Ce qui ne marche PAS (en strict OOS)

| Approche | Accuracy | vs Random | vs Rolling Mode |
|----------|----------|-----------|-----------------|
| GB exact star2 | 13.4% | 1.61× | 0.83× |
| KNN-20 exact star2 | 15.3% | 1.83× | 0.94× |
| GB top-3 star2 | leak (circulaire) | — | — |

### Ce qui marche (avec réserve)

| Approche | Accuracy | vs Random | Notes |
|----------|----------|-----------|-------|
| Position 0/4 prediction | 6.5% | **3.3×** | Edge significatif, mécanisme unclear |
| KNN k=20 | 15.3% | 1.83× | Sous rolling mode |

### Diagnostic

Le GB et KNN échouent à battre le rolling mode car :

1. **Le rolling mode est déjà optimal** : pour un processus autorégressif stationnaire, la mode de la fenêtre glissante est le prédicteur optimal sans modèle paramétrique
2. **Les features ne portent pas de signal supplémentaire** : std et entropy sont des transformations de la même information que mode+mean, sans signal orthogonal
3. **Le nombre de classes (12) fragment le signal** : chaque étoile n'a que ~160 observations dans le test set, insuffisant pour entraîner un GB avec 500 estimateurs

### Piste non-explorée

La prévisibilité des positions 0 et 4 (min/max) avec un edge de 3× via les seules features d'étoiles est le résultat le plus intriguing. Cela mérite une investigation dédiée :

- Les features d'étoiles capturent un facteur latent (température machine ?)
- Ce facteur affecte aussi la distribution des numéros
- Il faudrait vérifier si c'est robuste sur les derniers tirages uniquement

