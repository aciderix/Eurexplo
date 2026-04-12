# Euromillions — Rapport de Recherche Exploratoire (v2)

**Date :** 2026-04-12
**Source :** `euromillions-api` (pedro-mealha) via API `https://euromillions.api.pedromealha.dev`
**Données :** 1936 tirages, 2004-02-13 → 2026-04-10
**Pipeline :** générer → mesurer → détecter → falsifier → recommencer

---

## 1. Données

| Champ | Valeur |
|-------|--------|
| Tirages totaux | 1936 |
| Période | 2004-02-13 → 2026-04-10 |
| Numéros | 5 par tirage, range 1–50 |
| Étoiles | 2 par tirage, range 1–12 |
| Winners | 442 / 1936 (22.8%) |
| Jours | Vendredi uniquement (2004-2011), Mardi+Vendredi (2011+) |

---

## 2. Résultats Négatifs (Numéros) — Aucun Signal

Tous les tests sur les 5 numéros principaux (1-50) confirment un tirage parfaitement aléatoire.

| Test | Résultat | Verdict |
|------|----------|---------|
| Chi2 uniformité | 53.7 < 67.5 (df=49) | Uniforme |
| Autocorrélation (lags 1-10) | tous \|r\| < 0.013 | Aucune mémoire |
| Hot/Cold numbers | écart < 0.24% | Pas de signal |
| Co-occurrence paires | chi2=26.3 < 30.1 | Conforme au nul |
| FFT périodicité | Aucun pic | Pas de cycle |
| ML (LogisticRegression) | AUC=0.56 | ≈ hasard |
| Sommes autocorr | tous \|r\| < 0.03 | Nul |
| Spread autocorr | tous \|r\| < 0.02 | Nul |
| Gaps autocorr | tous \|r\| < 0.04 | Nul |
| Even/odd autocorr | r=-0.018 | Nul |
| PCA variance | z ∈ [-2.4, +1.8] | Normal |
| Graphe spectral gap | z=+1.82 | Normal |
| Wavelets (Haar) | z ∈ [-1.8, -0.02] | Normal |
| Compression (Kolmogorov) | z=-0.38 | Normal |
| Hamming distance consécutifs | mean=9.01 (théo=9.0) | Parfait |
| Clustering (K-means) | Aucune prédiction star2 | Nul |
| Embedding MDS | Clusters sans structure temporelle | Nul |
| Repeat patterns | Obs ≈ attendu hypergeom | Nul |

**Conclusion définitive :** Les numéros sont tirés par un processus indistinguable du hasard uniforme. Aucun edge exploitable.

---

## 3. Résultats Positifs — Étoiles

### 3.1 Non-uniformité MASSIVE des étoiles

**Star1 (min)** — chi2 = 1048.6 (critique = 19.7)

| Étoile | Observé | z-score |
|--------|---------|---------|
| 1 | 332 | **+13.4** |
| 2 | 354 | **+15.2** |
| 3 | 313 | **+11.9** |
| 4 | 212 | **+4.0** |
| 5 | 202 | **+3.2** |
| 6 | 167 | +0.5 |
| 7 | 121 | **-3.2** |
| 8 | 115 | **-3.7** |
| 9 | 72 | **-7.0** |
| 10 | 33 | **-10.1** |
| 11 | 15 | **-11.5** |
| 12 | 0 | **-12.7** |

**Star2 (max)** — chi2 = 128.3 (critique = 19.7)

| Étoile | Observé | z-score |
|--------|---------|---------|
| 2 | 29 | -7.2 |
| 3 | 70 | -4.3 |
| 4 | 94 | -2.3 |
| 5 | 141 | +0.8 |
| 6 | 181 | +2.2 |
| 7 | 223 | +4.1 |
| 8 | 252 | +5.1 |
| 9 | 285 | +6.4 |
| 10 | 245 | +4.8 |
| 11 | 246 | +4.9 |
| 12 | 170 | -1.5 |

**Découverte v2 :** Star1 est ENCORE PLUS biaisée que star2 (chi2=1048 vs 128). Edge plus fort sur star1.

### 3.2 Changement de régime

| Période | Caractéristique |
|---------|----------------|
| 2004-2016 (tirages 0-939) | Star12 absente (0 apparitions) |
| 2016+ (tirages 940+) | Star12 ≈ 9% des tirages |
| 2004-2011 | Vendredi uniquement |
| 2011+ | Mardi + Vendredi |

Premier tirage avec star12 : 2016-09-27 (tirage #940).
Hypothèse : changement de machine/format des étoiles.

### 3.3 Autocorrélation star2

Confirmée après contrôle pour : order statistics, changement de régime, non-uniformité marginale.

| Test | lag 1 | lag 3 | lag 6 |
|------|-------|-------|-------|
| Real | +0.078 | +0.155 | +0.113 |
| Null (permutation) z | +3.4σ | +6.8σ | +5.0σ |
| Null (order stats) z | +3.5σ | +7.0σ | +5.0σ |
| Pre-2016 (permut) z | +2.7σ | +4.5σ | +4.9σ |
| Post-2016 (permut) z | -0.6σ | +2.6σ | -0.1σ |

**Signal pré-2016 :** très fort. **Post-2016 :** plus faible, lag 3 seulement.

### 3.4 Compression — Structure temporelle confirmée

```
Star2 real compression: 0.4892
Null (même marginale, iid): 0.5040 ± 0.0024
z = -6.18σ
```

La compression détecte une structure temporelle AU-DELÀ de la simple non-uniformité. Le signal est dans l'ORDRE des tirages.

### 3.5 Jour de la semaine

| Période | Mardi star2 | Vendredi star2 | p-value |
|---------|-------------|----------------|---------|
| Pre-2016 | 8.15 | 7.29 | 0.000 |
| Post-2016 | 8.60 | 8.83 | 0.179 |
| Post-2011 | 8.44 | 8.53 | 0.478 |

**Verdict :** L'effet Mardi/Vendredi pré-2016 est un artefact temporel (seul le vendredi existe avant 2011). Post-2011, aucune différence significative.

### 3.6 Paires d'étoiles — Forte dépendance

Les paires voisines sont massivement surreprésentées (les 2 étoiles sont corrélées) :

| Paire | Obs | Expected | z |
|-------|-----|----------|---|
| (11,12) | 15 | 1.3 | **+11.9** |
| (1,2) | 29 | 5.0 | **+10.8** |
| (10,11) | 23 | 4.2 | **+9.2** |
| (2,3) | 38 | 12.8 | **+7.0** |
| (3,4) | 40 | 15.2 | **+6.4** |

Ceci est cohérent : les étoiles voisines numériquement sont physiquement proches dans la machine → plus de chance d'être tirées ensemble.

---

## 4. Formules Prédictives

### 4.1 Rolling Mode — Star2

| Fenêtre | Accuracy | Baseline | z-score | Edge |
|---------|----------|----------|---------|------|
| w=20 | **16.91%** | 8.33% | +13.6σ | 2.03× |
| w=50 | 15.59% | 8.33% | +11.4σ | 1.87× |
| w=100 | 16.45% | 8.33% | +12.6σ | 1.97× |
| w=150 | **16.80%** | 8.33% | +12.9σ | 2.02× |

### 4.2 Rolling Mode — Star1 (NOUVEAU)

| Métrique | Accuracy | Baseline | z-score | Edge |
|----------|----------|----------|---------|------|
| top-1 (w=100) | **18.30%** | 8.33% | +15.5σ | **2.20×** |
| top-2 | 33.88% | 16.67% | +19.8σ | 2.03× |
| top-3 | **49.13%** | 25.00% | +23.9σ | **1.97×** |
| top-5 | 70.81% | 41.67% | +25.3σ | 1.70× |

**Star1 est PLUS prédictible que star2** — c'est une découverte nouvelle.

### 4.3 Prédiction combinée (les 2 étoiles)

| Stratégie | Accuracy | Baseline | z-score |
|-----------|----------|----------|---------|
| Les 2 modes exacts | 1.80% | 0.69% | +5.7σ |
| Les 2 dans top-3 | **16.94%** | 6.25% | **+18.9σ** |

### 4.4 Validation temporelle (5 folds)

```
Fold 0 (2004-2011): rolling=19.9%
Fold 1 (2011-2015): rolling=15.5%
Fold 2 (2015-2018): rolling=18.6%
Fold 3 (2018-2022): rolling=12.9%
Fold 4 (2022-2026): rolling=16.2%
Moyenne: 16.63% (baseline 8.33%)
```

95% Bootstrap CI: [14.76%, 18.25%] — entièrement au-dessus du baseline.

---

## 5. Falsifications Détaillées

### Signaux Détruits

| Signal | Pourquoi détruit |
|--------|-----------------|
| Wavelets (Haar) | Énergie ≈ null model (z=-1.8, -0.02) |
| Graphe co-occurrence | Spectral gap z=+1.8, normal |
| PCA numéros | Variance ratios z ∈ [-2.4, +1.8] |
| Clustering → stars | Aucun pouvoir prédictif |
| Compression numéros | z=-0.38, normal |
| Transition matrix chi2 | 107.7 < 148.8 → indépendant (iid) |
| Higher-order Markov | Pas de gain vs rolling mode |
| RF/GB vs rolling mode | RF=16.4% ≈ rolling=15.8% → pas de gain |
| Jour semaine (post-2011) | p=0.478, pas de différence |
| Bounded random walk | star2 ≠ random walk (autocorr lag-1 trop faible : 0.08 vs 0.59) |

### Signaux Partiellement Confirmés

| Signal | Statut |
|--------|--------|
| Modular arithmetic (mod k) | z=+2.5 à +4.7 vs même marginale → signal faible réel |
| Compression temporelle star2 | z=-6.18 vs iid même marginale → structure temporelle réelle |
| Star pair proximity bias | z>+10 pour voisins → biais mécanique probable |

### Signaux Confirmés Robustes

| Signal | Evidence |
|--------|----------|
| Non-uniformité étoiles | chi2>100, z>8σ |
| Régime change 2016 | star12 absente puis 9% |
| Star1 rolling mode | 18.3%, z=+15.5σ, 5 folds |
| Star2 rolling mode | 16.5%, z=+12.6σ, 5 folds, bootstrap CI |
| Combined star prediction top-3 | 16.9%, z=+18.9σ |

---

## 6. Analyse des Mécanismes

### Pourquoi les étoiles sont-elles non-uniformes ?

1. **Pool réduit (12 vs 50)** : moins de mixing → biais mécaniques plus visibles
2. **Proximité physique** : les paires voisines (1,2), (10,11), (11,12) sont massivement surreprésentées → les boules voisines numériquement sont probablement voisines dans la machine
3. **Changement de matériel en 2016** : star12 apparaît soudainement → remplacement de machine probable
4. **Distribution asymétrique** : star1 concentrée sur 1-5, star2 sur 7-12 → confirme un biais mécanique systématique

### Pourquoi les numéros sont-ils uniformes ?

1. **Pool large (50)** : les biais mécaniques se diluent
2. **Mixing plus efficace** : machine à boules plus grande → meilleur brassage
3. **Tous les tests convergent** : chi2, autocorr, FFT, compression, PCA — tout conforme

---

## 7. Stratégie Optimale

```python
from collections import Counter

def predict_stars(star1_history: list[int], star2_history: list[int],
                  window: int = 100, top_n: int = 3) -> tuple[list[int], list[int]]:
    """
    Prédit les top_n étoiles candidates pour chaque position.

    Returns:
        (top_n star1 candidates, top_n star2 candidates)
    """
    recent_s1 = star1_history[-window:]
    recent_s2 = star2_history[-window:]
    top_s1 = [c[0] for c in Counter(recent_s1).most_common(top_n)]
    top_s2 = [c[0] for c in Counter(recent_s2).most_common(top_n)]
    return top_s1, top_s2


def generate_star_pairs(top_s1: list[int], top_s2: list[int]) -> list[tuple[int, int]]:
    """Génère toutes les paires valides (s1 < s2) depuis les candidates."""
    pairs = []
    for a in top_s1:
        for b in top_s2:
            if a < b:
                pairs.append((a, b))
    return sorted(pairs)
```

**Performance attendue :**
- Chaque étoile dans top-3 : ~49% (star1), ~46% (star2)
- Paire dans top-3×top-3 : ~17% (baseline 6.25%, edge 2.7×)
- Numéros : aléatoire (aucun edge)

---

## 8. Pistes Futures Non Explorées

1. **Machine-specific analysis** : identifier les changements de machine via changepoint detection
2. **Étoiles lucky stars vs regular** : certains pays ont des tirages légèrement différents
3. **Analyse des prix** : le montant des gains influence-t-il le tirage suivant ? (indirectement via nombre de joueurs)
4. **Analyse fréquentielle fine** : les pré-2011 vendredi seulement → machine identique mais cadence différente
5. **Prédiction conditionnelle** : si star1=X, quelle star2 est la plus probable ?
6. **Transfer learning** : appliquer le même framework à d'autres loteries pour vérifier si le biais étoiles est universel

---

## 9. Limites

- **Numéros non prédictibles** — l'edge est UNIQUEMENT sur les étoiles
- **Edge limité** — même avec 2× le baseline, la probabilité reste faible
- **Risque de changement** — un nouveau matériel peut modifier la distribution
- **Pas de jackpot** — pour gagner le jackpot (5+2), il faut les 5 numéros justes (non prédictibles)
- **Utilité pratique** — l'edge sur les étoiles améliore les rangs de gain inférieurs, pas le jackpot
