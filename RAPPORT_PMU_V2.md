# PMU Courses Hippiques — Rapport d'Exploration Exhaustive V2

**Données :** 5 620 courses nettoyées, 2004-2026
**Source :** `pmu_races.json` (open-pmu-api)
**Méthode :** Pipeline `générer → mesurer → détecter → falsifier → recommencer`

---

## Résumé Exécutif

**Un signal structurel existe** : le numéro gagnant n'est pas distribué uniformément — il dépend du contexte de la course (type, distance, hippodrome). Ce signal survit à la falsification walk-forward avec **+2.99pp en Top-5** au-dessus du hasard pur.

**Mais ce signal n'est pas monétisable** : la marge du PMU (~30%) écrase complètement le faible edge statistique. ROI simulé = **-18.5%**.

---

## Phase 1 : Exploration Statistique

### 1.1 Distribution des gagnants

| Numéro | Victoires | Taux normalisé | Ratio vs attendu |
|--------|-----------|----------------|------------------|
| N°9 | 388 | 6.90% | 1.124 ↑ |
| N°6 | 379 | 6.74% | 1.098 |
| N°3 | 375 | 6.67% | 1.087 |
| N°8 | 370 | 6.58% | 1.072 |
| N°1 | 300 | 5.34% | 0.869 ↓ |
| N°15 | 274 | 4.84% | 0.913 ↓ |

**Falsification (permutation test) :** p = 1.0 → Les biais par numéro sont un **artefact de max-picking**. En tirant 18 numéros aléatoires, on obtient naturellement un max ratio ~1.12. **FAUX SIGNAL.**

### 1.2 Test Chi-2 par nombre de partants

| Partants | Courses | Chi2 | p-value | Verdict |
|----------|---------|------|---------|---------|
| 13 | 155 | 18.0 | 0.115 | Uniforme |
| 14 | 455 | 15.6 | 0.271 | Uniforme |
| 15 | 610 | 23.2 | 0.056 | Uniforme |
| **16** | **2231** | **34.7** | **0.003** | **BIAIS** |
| 17 | 411 | 22.9 | 0.118 | Uniforme |
| 18 | 1457 | 17.9 | 0.398 | Uniforme |

Biais significatif à 16 partants : N°1 (-20.4%), N°15 (-24.7%), N°6 (+19.0%), N°2 (+16.9%).
**Falsification temporelle :** Instable sur 5 périodes → **NON EXPLOITABLE** tel quel.

### 1.3 Parité et Modulo

| Test | Résultat | p-value |
|------|----------|---------|
| Pair/Impair | 49.4% / 50.6% | Non significatif |
| Mod 3 | Uniforme | 0.318 |
| Mod 4 | Biais léger | 0.006 |
| **Mod 5** | **Biais fort** | **<0.0001** |
| **Mod 6** | **Biais fort** | **<0.0001** |

Les biais modulo 5 et 6 reflètent la distribution non-uniforme globale (les petits numéros gagnent légèrement plus car ils participent à plus de courses), pas un pattern exploitable.

---

## Phase 2 : Analyse Temporelle et Séquentielle

### 2.1 Autocorrélation

| Lag | r | p-value |
|-----|---|---------|
| 1 | +0.010 | 0.439 |
| 6 | +0.028 | 0.033 |
| 9 | +0.028 | 0.035 |

Autocorrélation quasi-nulle → **pas de mémoire séquentielle exploitable**.

### 2.2 Exposant de Hurst

- **H = 0.589** (R² = 0.991)
- Légèrement persistant (>0.5), mais insuffisant pour prédire.

### 2.3 Runs Test

- Z = -1.57, p = 0.115
- **Séquence compatible avec l'aléatoire.**

### 2.4 Entropie

| Mesure | Valeur (bits) |
|--------|---------------|
| H(W_n) inconditionnel | 4.142 |
| H(W_n \| W_{n-1}) | 4.080 |
| H(W_n \| W_{n-1}, W_{n-2}) | 3.322 |

Le gagnant précédent fournit **1.51%** d'information. Avec 2 gagnants précédents, ça monte mais c'est probablement du bruit (sparse data : 324 états, ~17 obs/état).

### 2.5 Complexité de Kolmogorov

- Ratio compression séquence gagnants : 0.5875
- Ratio séquence aléatoire : 0.5884
- Différence : **0.15%** → **Pas de structure détectable** par compression.

---

## Phase 3 : Approches Non Conventionnelles

### 3.1 Hypothèses Absurdes (toutes falsifiées)

| Hypothèse | Résultat |
|-----------|----------|
| Somme chiffres date → gagnant | r = -0.013, p = 0.32 → Aucun signal |
| XOR positions → suivant | r = 0.009, p = 0.48 → Aucun signal |
| Distance divisible par gagnant | 32.9% observé vs 33.0% attendu → Aucun signal |
| Fibonacci gagnant | 36.7% vs 37.5% attendu → Aucun signal |
| Nombres premiers gagnants | 39.2% vs 38.9% attendu → Aucun signal |
| (jour % partants) + 1 = gagnant | 6.69% vs 6.25% → Marginal |
| Collatz(prev) mod N + 1 | 6.23% → Aucun signal |
| prev - 1 mod N + 1 | 6.61% walk-forward → **+0.36pp marginal** |

### 3.2 Chaîne de Markov

- Distribution stationnaire cohérente avec les fréquences empiriques
- Backtest : Top-1 = 6.32%, Top-5 = 31.85% → **Pas d'edge significatif**

### 3.3 Clustering Spectral

- 4 clusters : {1-9}, {10-15}, {16}, {17-18}
- Reflète uniquement l'effet de participation (petits numéros toujours éligibles)
- **Aucune structure cachée**

### 3.4 Wavelet

- Échelle dominante : 2 courses (pas de période significative)
- Énergie top-10 fréquences : 2.55% du total → **Pas de composante périodique**

---

## Phase 4 : Machine Learning

### 4.1 Gradient Boosting (53 features)

| Métrique | Score | Baseline | Edge |
|----------|-------|----------|------|
| Top-1 | 8.08% | 6.25% | +1.83pp |
| Top-3 | 22.89% | 18.75% | +4.14pp |
| Top-5 | 36.09% | 31.25% | +4.84pp |

**Falsification walk-forward :** Top-1 = 6.20%, Top-5 = 31.76% → **OVERFITTING.** L'edge disparaît en validation temporelle.

### 4.2 Formule Explicite (α,β,γ,δ grid search)

- Meilleurs params après grid search : Top-3 = 20.02% (+1.27pp)
- **Falsification walk-forward** : Edge détruit → **OVERFITTING confirmé**

---

## Phase 5 : Le Signal Survivant

### Contexte Conditionnel (Type × Distance × Lieu)

Le seul signal qui **survit à la falsification walk-forward** :

| Contexte | Top-1 | Edge-1 | Top-5 | Edge-5 | Périodes gagnantes |
|----------|-------|--------|-------|--------|--------------------|
| Type seul | 6.77% | +0.52 | 32.64% | +1.39 | 12/23 |
| Type+Lieu | 6.79% | +0.54 | 32.11% | +0.86 | 15/23 |
| Type+Dist | 6.23% | -0.02 | 32.44% | +1.19 | 16/23 |
| **Type+Dist+Lieu** | **7.30%** | **+1.05** | **33.49%** | **+2.24** | **16/23** |
| Type+Lieu+DOW | 7.25% | +1.00 | 30.97% | -0.28 | 9/23 |

### Formule Finale Optimisée

```
Score(k, course) = 0.3 × ContextFreq(k) + 0.7 × Uniform(k)
```

**où :**
- `ContextFreq(k)` = fréquence de k parmi les gagnants passés de courses de même (type, distance/500, lieu)
- `Uniform(k)` = 1 / nb_partants
- Lookback : 3000 dernières courses
- Min 5 courses de même contexte requises

**Résultat walk-forward :**
- **Top-1 : 7.22% (edge +0.97pp)**
- **Top-5 : 34.24% (edge +2.99pp)**

### Interprétation

Le signal est **structurel** : dans un Attelé à Vincennes sur 2700m, certains numéros de départ ont un léger avantage (probablement lié à la corde, au couloir de départ, à la piste). Ce n'est pas de la prédiction au sens fort, c'est une distribution conditionnelle non-uniforme.

---

## Phase 6 : Simulation de Paris

| Stratégie | Mise/course | Nb paris | ROI |
|-----------|-------------|----------|-----|
| Top-1 avec seuil confiance >1.3 | 10€ | 744 | **-29.2%** |
| Top-1 sans seuil | 10€ | 2 810 | **-18.5%** |

**La marge PMU (~30%) rend toute stratégie déficitaire** même avec un edge statistique de +3pp.

---

## Conclusion Finale

### Ce qui a été détecté
1. **Distribution conditionnelle non-uniforme** : le numéro gagnant dépend du contexte (type + distance + lieu), avec un edge walk-forward validé de +2.99pp en Top-5.
2. **Exposant de Hurst = 0.59** : légère persistance, compatible avec le signal contextuel.
3. **Information mutuelle faible** (1.51%) entre gagnants consécutifs.

### Ce qui a été falsifié et détruit
- Biais par numéro (N°9, etc.) → max-picking bias
- ML (Gradient Boosting, Random Forest) → overfitting
- Autocorrélation séquentielle → bruit
- Patterns modulo, Fibonacci, premiers → bruit
- Complexité de Kolmogorov → pas de structure
- Wavelets, Fourier → pas de périodicité
- Chaîne de Markov → pas d'edge
- Automates cellulaires (triplets) → pas d'edge

### Verdict

**Il existe un micro-signal structurel réel** (~3pp au-dessus du hasard) lié au contexte de la course. Mais :

1. Ce signal est **trop faible** pour surmonter la marge du PMU (~30%)
2. Les données manquent de **features critiques** (identité cheval, jockey, cotes, forme récente) qui sont la vraie source de prédictibilité dans les courses hippiques
3. Le numéro de dossard seul ne contient pas assez d'information pour prédire le résultat

### Pour aller plus loin

Pour obtenir un véritable edge exploitable, il faudrait :
1. **Les cotes de départ** — c'est LE facteur déterminant pour le value betting
2. **L'historique individuel des chevaux** — forme, performances, gains
3. **Les conditions de course** — terrain, météo, catégorie
4. **Données de musique** — classements récents de chaque cheval

Le problème n'est pas mathématique — il est dans les données.
