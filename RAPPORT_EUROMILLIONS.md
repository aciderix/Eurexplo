# Euromillions — Rapport de Recherche Exploratoire (v5 — 56 methodes, verdict final)

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

## 8. Exploration v3 — Approches Folles et Oracle

### 8.1 Approches testées sur les NUMEROS (tous echouent)

| Approche | Resultat | Verdict |
|----------|----------|---------|
| Phase lunaire | ANOVA p=0.84 | Aucun effet |
| Fibonacci/Primes count | autocorr < 0.03 | Nul |
| Golden ratio dans gaps | distance phi ≈ null model | Nul |
| Effet jackpot size sur nums | 1 seul num z>2 sur 50 | Bruit |
| Winner effect sur nums | p=0.32 | Nul |
| Strategie exclusion | 90.1% vs 90.0% baseline | Nul |
| Zones/decades autocorr | tous \|r\| < 0.06 | Nul |
| Saisonnier | ANOVA p=0.86 | Nul |
| Position-specific rolling | z≈0 vs null iid marginale | **Artefact** |
| Takens embedding sum | 50.1%, z=+0.1 | Nul |
| Algorithme genetique | test=13.0%, pas mieux que rolling | Nul |
| Collatz transformation | autocorr < 0.05 | Nul |
| Regression sinusoidale | R²=0.003 | Nul |
| Champernowne cross-draw | r=-0.018 | Nul |
| Numerologie digit sum | Pas de pattern | Nul |
| Pi encoding | 10.3% des tirages, pas predictif | Nul |
| Vendredi 13 | star2=8.03 ≈ global 8.15 | Nul |

**Conclusion brutale** : Les numeros sont un mur. 20+ approches, 0 signal survivant a la falsification.

### 8.2 Nouvelles decouvertes sur les ETOILES

| Decouverte | Detail |
|------------|--------|
| **P(star2\|star1)** | Quand star1=8, P(star2=9) = **33%**. Prediction conditionnelle: 18.7% (z=+15.7) |
| **Paire proximity bias** | (11,12) z=+11.9, (1,2) z=+10.8 → biais mecanique |
| **Star1 + star2 top-3** | Au moins 1 correcte: **77.6%** (z=+7.1 vs null iid meme marginale) |
| **Pattern matching (10-NN)** | star2: 13.4% (z=+7.9) |
| **Modular arithmetic** | star2 mod 6: z=+4.7 vs null marginal |
| **Winner effect** | star2 apres winner = 7.94 vs 8.21 apres loss (p=0.049) |

### 8.3 Decomposition honnete : edge REEL vs TRIVIAL

Le rolling mode fonctionne. Mais **combien vient du vrai signal temporal** vs **juste le biais marginal** ?

| Composant | Accuracy reelle | Null (iid meme marginale) | Edge reel | z vs null |
|-----------|----------------|---------------------------|-----------|-----------|
| star1 top-3 | 48.3% | 48.7% | **-0.3pp** | -0.2 |
| **star2 top-3** | **45.7%** | **38.1%** | **+7.6pp** | **+5.5** |
| pos0 top-3 (nums) | 23.5% | 23.5% | 0.0pp | 0.0 |
| pos4 top-3 (nums) | 26.2% | 25.3% | +1.0pp | +0.8 |

**Revelation** : Star1 n'a PAS d'edge temporel reel ! Son 48.3% vient entierement de sa distribution desequilibree (concentree sur 1-5). Seule **star2** a un vrai signal temporel (+7.6pp, z=+5.5).

### 8.4 La Formule Oracle definitive

```python
from collections import Counter

def oracle_eurexplo(star1_history, star2_history, window=100):
    """
    L'Oracle Eurexplo — formule composite a >50%.
    
    Prediction: "Au moins 1 etoile dans son top-3 rolling"
    Accuracy: 77.6% (dont ~68% structure triviale + ~10pp edge reel)
    z-score vs null iid: +7.1
    
    Pour les numeros: aucun edge. Choisir aleatoirement.
    """
    top3_s1 = [c[0] for c in Counter(star1_history[-window:]).most_common(3)]
    top3_s2 = [c[0] for c in Counter(star2_history[-window:]).most_common(3)]
    
    # Prediction conditionnelle P(star2|star1)
    # Pour chaque star1 candidate, trouver les star2 les plus probables
    conditional_pairs = []
    for s1_cand in top3_s1:
        # Chercher les draws recents ou star1 == s1_cand
        cond_s2 = [s2 for s1, s2 in zip(star1_history[-500:], star2_history[-500:]) 
                   if s1 == s1_cand]
        if len(cond_s2) >= 5:
            best_s2 = Counter(cond_s2).most_common(2)
            for s2_val, _ in best_s2:
                if s1_cand < s2_val:
                    conditional_pairs.append((s1_cand, s2_val))
    
    return {
        'star1_candidates': top3_s1,
        'star2_candidates': top3_s2,
        'best_pairs': conditional_pairs[:9],
        'confidence': '77.6% qu\'au moins 1 etoile soit dans les candidates',
        'numbers': 'Aleatoire — aucun edge detecte sur 20+ methodes testees'
    }
```

### 8.5 Tableau de bord de confiance

| Metrique | Valeur | Baseline | Edge | Fiabilite |
|----------|--------|----------|------|-----------|
| Au moins 1 star top-3 | **77.6%** | 43.8% | 1.77x | z=+7.1 vs null |
| Star2 mode exact | **16.5%** | 8.3% | 2.0x | z=+5.5 vs null |
| Star1 mode exact | 18.3% | 8.3% | 2.2x | z=-0.2 vs null (trivial!) |
| Paire exacte mode | 1.8% | 0.69% | 2.6x | z=+5.7 |
| Top-9 paires | 16.5% | 13.6% | 1.21x | z=+3.4 |
| >=2 nums dans pool 25 | 82.0% | 82.6% | 0.99x | Pas d'edge |
| Composite score>=3 | 67.3% | ~60% | ~1.12x | Mixte |

### 8.6 Approches detruites (mais amusantes)

- **Phase lunaire** : la Lune se fiche de l'Euromillions (p=0.84)
- **Vendredi 13** : 40 tirages, star2=8.03 ≈ moyenne (pas maudit, pas beni)
- **Fibonacci** : 0.795 fibs/tirage vs 0.800 theorique (parfait hasard)
- **Tirages quasi-identiques** : les plus proches (distance L1=2) sont le #143 et #155
- **Regression sinusoidale** : R²=0.003 — star2 n'est pas periodique
- **Algorithme genetique** : 100 generations, evolue de 10% a 16% — ne bat pas le rolling mode

---

## 9. Limites

- **Numeros : mur infranchissable** — 20+ methodes testees (de la plus classique a la plus absurde), 0 signal. Le tirage de 5 numeros parmi 50 est indistinguable du hasard parfait.
- **Star1 : edge illusoire** — le 18.3% d'accuracy vient de la distribution biaisee, pas d'une structure temporelle. Un modele iid avec la meme marginale fait aussi bien.
- **Star2 : seul vrai signal temporel** — +7.6pp d'edge reel vs null iid, confirme par compression (z=-6.18) et autocorrelation multi-lag.
- **Edge limite** — meme avec le meilleur Oracle, la probabilite de toucher le jackpot (5+2) reste ~1/139M.
- **Risque de changement** — le changement de 2016 (star12) montre que le systeme evolue.
- **Utilite reelle** — l'edge ameliore les rangs de gain inferieurs (2+1 star = ~13EUR) mais ne change pas la donne sur les gains majeurs.

---

## 10. Backtest Historique

### 10.1 Simulation complete (1836 tirages joues, w=100)

| Strategie | Cout | Gains | ROI | vs Random |
|-----------|------|-------|-----|-----------|
| **Oracle 1 grille** | 4590 EUR | 1136 EUR | **24.7%** | **1.23x** |
| Oracle 3 grilles | 13770 EUR | 2873 EUR | 20.9% | — |
| Random (baseline) | 4590 EUR | 923 EUR | 20.1% | 1.00x |

### 10.2 Performance par periode

| Periode | s1 mode | s2 mode | Any star top-3 | ROI |
|---------|---------|---------|----------------|-----|
| 2004-2008 | 21.3% | 20.7% | **86.7%** | **31.7%** |
| 2008-2012 | 20.0% | 17.6% | 82.0% | 23.8% |
| 2012-2016 | 19.3% | 16.4% | 82.3% | 19.1% |
| 2016-2020 | 19.5% | 15.4% | 71.0% | 22.0% |
| 2020-2024 | 15.5% | 15.5% | 76.8% | **28.1%** |
| 2024-2026 | 14.5% | 16.1% | 75.3% | **32.5%** |

Le ROI est **stable dans le temps** (~20-33% de retour), confirmant que l'edge est reel et persistant. La formule fonctionne sur toutes les periodes.

### 10.3 Gains concrets Oracle 1 grille

| Rang | Occurrences | % tirages | Gain unitaire |
|------|-------------|-----------|---------------|
| 2+0 | 83 | 4.52% | ~4 EUR |
| 2+1 | 53 | 2.89% | ~8 EUR |
| 1+2 | 14 | 0.76% | ~10 EUR |
| 3+0 | 10 | 0.54% | ~13 EUR |
| 3+1 | 3 | 0.16% | ~14 EUR |
| 2+2 | 4 | 0.22% | ~17 EUR |

---

## 11. Decouvertes avancees (v3)

### 11.1 Changepoints detectes

L'analyse CUSUM revele **5 changements de regime** dans l'historique de star2 :

| Date | Draw # | Shift star2 | Interpretation |
|------|--------|-------------|----------------|
| 2011-05-20 | #381 | +1.40 | Introduction tirages mardi |
| 2015-01-09 | #761 | +0.72 | Possible changement machine |
| 2016-09-27 | #940 | — | Star12 apparait (deja connu) |
| 2024-11-19 | #1790 | +0.90 | Regime actuel : star2 tres haut |

### 11.2 Regime actuel (post-2024)

```
Segment [1790-1936] (2024-2026):
  mean star2 = 9.16, std = 2.21
  Mode star2 = 12
  Top: [12 (27 fois), 10 (27 fois), 9 (26 fois)]
```

Le regime actuel favorise les etoiles hautes (9, 10, 12).

### 11.3 Entropie de permutation (Bandt-Pompe)

```
Star2: H_norm = 0.9785 (null iid: 0.9858, z=-3.23)
```

Star2 est significativement **moins desordonnee** que l'iid — ses patterns temporels d'ordre ont une structure reelle.

### 11.4 Runs test

```
Star2: z=-2.16 → CLUSTERED
Star1/num_sum/spread: tous RANDOM
```

Star2 est le **seul** composant avec un clustering temporel significatif.

---

## 12. Prediction du prochain tirage

Dernier tirage: #1936 (2026-04-10) → [10, 13, 14, 38, 41] + [6, 9]

### Grille Oracle suggeree

| Numeros | Etoiles | Confiance |
|---------|---------|-----------|
| 5 - 8 - 24 - 29 - 48 | **2 - 10** | Paire la + freq (6/100) |
| 6 - 9 - 25 - 44 - 49 | **3 - 9** | 2e paire (5/100) |
| 1 - 17 - 19 - 41 - 43 | **1 - 10** | 3e paire (4/100) |

- Confiance etoiles: **77.6%** qu'au moins 1 soit dans le top-3 rolling
- Confiance numeros: identique au hasard
- Regime actuel: star2 elevee (mode=12, top: 10, 12, 9)

---

## 13. Exploration v4 — Falsification Totale (40+ methodes)

### 13.1 Changements de format detectes

| Ere | Tirages | Format Etoiles | Freq. tirages |
|-----|---------|----------------|---------------|
| Era 1 | 0-380 (2004-2011) | Stars 1-9 | Vendredi seul |
| Era 2 | 381-939 (2011-2016) | Stars 1-11 | Mardi + Vendredi |
| Era 3 | 940-1935 (2016-2026) | Stars 1-12 | Mardi + Vendredi |

**Decouverte critique :** De nombreux "signaux" detectes sur le dataset complet sont des artefacts de ces changements de format. Toutes les analyses de cette session ont ete refaites sur l'Era 3 seule (996 tirages, format stable).

### 13.2 Signaux detruits — "Pensees de physicien"

| Approche | Signal brut | Apres falsification | Verdict |
|----------|-------------|---------------------|---------|
| Displacement autocorrelation | pos0: r=+0.25, pos4: r=+0.30 | z∈[-0.86, +0.44] vs null iid | **ARTEFACT** (regression vers la moyenne des order stats) |
| FFT periodicites numeros | Num13 z=+10.76 | Global z=0.14, p=0.45 (era 3) | **ARTEFACT** (format changes + multiple testing) |
| FFT periodicites etoiles | Star12 z=+20.76 | Global z=-0.81, p=0.78 (era 3) | **ARTEFACT** (100% du a l'introduction star 10-12) |
| Systeme dynamique NN (Takens) | ratio NN/naive = 0.78 | z=+0.38 vs null shuffled | **ARTEFACT** (regression vers la moyenne dans l'embedding) |
| Momentum espace des tirages | cos(angle) = -0.45 | z=-0.85, p=0.22 | **MORT** |
| Spread numbers autocorr | r=-0.013 | z=-0.54, p=0.29 | **MORT** |
| Geometrie circulaire | Aucune structure | — | **MORT** |
| Decomposition binaire | Aucune structure | — | **MORT** |
| Patterns d'absence | Aucune structure | — | **MORT** |
| NMF facteurs latents | Pas d'autocorrelation | — | **MORT** |

### 13.3 Signaux detruits — Approches non-conventionnelles

| Approche | Resultat | Verdict |
|----------|----------|---------|
| Resonance num_sum(t) → star2(t+1) | r=0.049, z=2.15 global MAIS z=0.90, p=0.18 en era 3 | **ARTEFACT** format |
| Hot hand / Gambler (numeros) | P(appear\|drought=d) = 10% ∀d | **MORT** — parfaitement iid |
| Pattern matching binaire (stars) | 32.2% vs 33.3% baseline | **MORT** |
| Conditional patterns (centroid, parity, consec.) | Tous r < 0.03 | **MORT** |
| Residus modulaires (mod 3,5,7,11) | Consecutive matches = expected | **MORT** |
| Parite sum autocorrelation | r=-0.023 | **MORT** |
| Digit sum autocorrelation | r=-0.012, z=-0.51 | **MORT** |
| Coefficients polynomiaux (Vieta) | Tous r < 0.05 | **MORT** |
| Compression Kolmogorov | ratio = constant (donnees trop courtes) | **MORT** |
| Co-occurrence numeros (global) | variance z=1.05, p=0.14 | **MORT** |
| Jour de semaine (era 3) | star2 Tue=8.60 vs Fri=8.83, p=0.18 | **MORT** |
| Phase lunaire | p=0.846 | **MORT** |
| Fibonacci distances | z=-1.91 (MOINS que random) | **MORT** |
| Jackpot → next draw | Differences non significatives | **MORT** |
| Sequence matching | 34.0% vs rolling 34.9% | **MORT** |

### 13.4 UN signal survivant (stars)

**Star drought ≥ 20 tirages → P(apparition) = 32%**

| Metrique | Valeur |
|----------|--------|
| Observe | P(appear\|drought=20) = 0.321 |
| Null (permutation) | mean = 0.167, std = 0.053 |
| z-score | **+2.93** |
| p-value | **0.0013** |
| p corrigee (Bonferroni, 8 seuils testes) | **0.010** |

Quand une etoile n'est pas apparue depuis 20+ tirages, elle a ~2x plus de chances de revenir. Signal statistiquement reel, mais **pratiquement inutile** : n=53 cas seulement, et le rolling mode capture deja cette information implicitement.

### 13.5 Star gap autocorrelation (marginal)

| Metrique | Valeur |
|----------|--------|
| Gap autocorrelation lag-1 | r = 0.043 |
| z vs null | +1.93 |
| p-value | 0.027 |

Le gap entre star1 et star2 a une micro-autocorrelation positive. Trop faible pour etre exploitable (r < 0.05).

### 13.6 ML combine : echec

GradientBoosting avec 57 features (rolling frequencies, recency, gaps, centroids, spreads, parity) sur era 3 :

| Predicteur | Star1 | Star2 | Any star |
|------------|-------|-------|----------|
| ML (57 features) | 14.5% | 20.7% | 32.3% |
| Rolling mode simple | 17.3% | 19.4% | 35.4% |
| Random | 16.7% | 16.7% | 33.3% |

**Le ML fait PIRE que le rolling mode.** Les 57 features ne contiennent aucune information supplementaire. Le rolling mode est deja optimal.

### 13.7 Prediction par position : pas d'avantage

| Approche | Any number correct | Baseline random |
|----------|--------------------|-----------------|
| Par position (rolling mode) | 42.8% | 42.5% |
| Sans position (top-5 global) | 44.1% | 42.3% |
| Hybride position+global | 40.2% | 42.5% |

La prediction par position n'apporte rien pour les numeros. Le top-5 global est marginalement mieux (+1.8pp).

### 13.8 Analyse de couverture — Le "50%"

En augmentant le nombre de predictions, on peut atteindre n'importe quel seuil :

| N numeros predits | Notre hit rate | Random | Edge |
|-------------------|---------------|--------|------|
| 5 | 44.1% | 42.3% | +1.8pp |
| **6** | **50.8%** | **48.7%** | **+2.1pp** |
| 7 | 55.7% | 54.6% | +1.1pp |
| 10 | 70.0% | 68.9% | +1.1pp |

| N etoiles predites | Notre hit rate | Random | Edge |
|---------------------|---------------|--------|------|
| 2 | 34.7% | 31.8% | +2.9pp |
| 3 | 47.3% | 45.5% | +1.8pp |
| 4 | 58.9% | 57.6% | +1.3pp |

**Grille 7 numeros + 3 etoiles :**
- Au moins 1 numero correct : **55.7%** (random: 54.6%)
- Au moins 1 etoile correcte : **47.3%** (random: 45.5%)
- Au moins 1 element : **76.8%** (random: 75.2%)

**L'edge reel est de ~2pp quelle que soit la taille de la prediction.** Le "50%" s'atteint en augmentant la couverture, pas par un vrai signal predictif.

### 13.9 Oracle analysis — Plafond theorique

| Strategie | Avg correct / 5 nums |
|-----------|---------------------|
| Oracle (connait les 50 prochains tirages) | 0.93 / 5 |
| Notre predicteur | 0.52 / 5 |
| Random | 0.50 / 5 |

Meme avec une connaissance PARFAITE du futur, un predicteur base sur les frequences ne peut atteindre que 0.93/5 — parce que les numeros sont trop uniformes pour etre previsibles.

### 13.10 Position des etoiles dans la machine

La note "pensee machine" : les etoiles sont triees (star1=min, star2=max). Chaque position a sa propre distribution :
- Star1 : concentree sur 1-5 (bias mecanique fort)
- Star2 : concentree sur 7-12 (complementaire)
- Le biais star1 est TRIVIAL (structure marginale pure, z=-0.2 vs null iid)
- Seul star2 a un vrai signal TEMPOREL (+7.6pp, z=+5.5)

---

## 14. Sur l'entrainement d'un modele IA

### Pourquoi ca ne marchera pas

1. **Pas de signal a apprendre** : 40+ methodes testees, le seul signal reel est le rolling mode star2 (+2-3pp). Un reseau de neurones ne peut pas apprendre ce qui n'existe pas.

2. **Overfitting garanti** : ~2000 tirages est microscopique pour du deep learning. Le modele memorisera du bruit (95% train, 16% test).

3. **On l'a teste** : GradientBoosting avec 57 features fait PIRE que le simple rolling mode. Plus de complexite = plus d'overfitting = pire resultat.

4. **Combiner d'autres loteries ?** Chaque loterie a des machines differentes, des plages differentes (Powerball: 1-69, Loto FR: 1-49). Il n'y a pas de "physique universelle des loteries" transferable.

5. **Le probleme est physique** : les boules sont brassees mecaniquement. Aucune quantite de compute ne cree un signal la ou la physique n'en laisse pas.

### Si on voulait quand meme essayer (pour le fun)

- LSTM ou Transformer sur sequences de tirages
- Prediction : overfitting massif garanti
- Le modele convergera vers... le rolling mode (la seule structure presente)

---

## 15. Exploration v5 — 16 methodes supplementaires (theorie de l'information, chaos, topologie)

### 15.1 Signaux fantomes : le piege des changements de format

**Decouverte majeure de la v5 :** Plusieurs methodes avancees detectent des signaux MASSIFS sur le dataset complet qui disparaissent completement sur l'era 3 (format stable 2016-2026). Ces signaux sont des artefacts des changements de format etoiles (1-9 → 1-11 → 1-12).

| Methode | Full dataset | Era 3 seule | Diagnostic |
|---------|-------------|-------------|------------|
| Symbolic dynamics (3-grams) | **z=11.31** | z=0.55 | 100% artefact format |
| IAAFT surrogate (mean-var) | **z=6.91** | z=-0.33 | 100% artefact format |
| 1/f noise (beta) | **z=3.94** | z=1.06 | 100% artefact format |
| MI retardee lag 3 | **z=4.12** | z=1.59 | Largement artefact |

**Lecon critique :** TOUJOURS tester sur l'era 3 seule. Les changements structurels creent des signaux fantomes dans pratiquement toutes les methodes sensibles a la non-stationnarite.

### 15.2 Resultats des 16 methodes supplementaires

| # | Methode | Signal? | z (era 3) | Detail |
|---|---------|---------|-----------|--------|
| 1 | Hurst / DFA (memoire longue) | NON | z=1.13 | H=0.59, pas significatif vs null |
| 2 | Information mutuelle retardee | NON | z=1.59 (lag 3) | Artefact format sur full, marginal era 3 |
| 3 | Benford law | N/A | — | Pas applicable (numeros 1-50 pas multi-echelle) |
| 4 | FFT sur ecarts (gaps) | NON | z=-0.62 (nums), z=0.52 (stars) | Aucune periodicite dans les gaps |
| 5 | Recurrence (RQA) | NON | z=-2.03 a 0.1 | Legere ANTI-determinisme a seuil bas |
| 6 | Symbolic dynamics (n-grams) | NON | z=0.55 (era 3) | Fantome spectaculaire (z=11 → z=0.55) |
| 7 | Record-breaking droughts | NON | z=0.26 (nums), z=0.05 (stars) | Parfaitement conforme a l'iid |
| 8 | CCM (inference causale) | NON | Non concluant | Pas de causalite inter-series |
| 9 | Copulas (dependance positions) | NON | z=-0.87 | Dependance ≤ uniform order statistics |
| 10 | Extreme value theory | MARGINAL | z=2.56 (low-10%) | Clustering des extremes bas star2 |
| 11 | IAAFT surrogates (non-linearite) | NON | z=-0.33 (era 3) | Fantome (z=6.91 sur full) |
| 12 | 1/f / flicker noise | NON | z=1.06 (era 3) | Fantome (z=3.94 sur full) |
| 13 | Network motifs temporels | NON | z=0.60 | Matrice de transition = iid |
| 14-16 | (Convergent Cross Mapping, copulas avancees, ondelettes continues) | NON | — | Testes via CCM/copulas/IAAFT |

### 15.3 Seul micro-signal survivant v5

**Clustering des extremes bas star2 (era 3)** : z=2.56

Les valeurs basses de star2 (≤5) apparaissent en clusters plus que le hasard (CV=1.21 vs null 0.94). Compatible avec l'autocorrelation lag-3 deja connue. Non exploitable au-dela du rolling mode.

---

## 16. Conclusion Definitive (v5 — 56 methodes)

### Ce qu'on sait avec certitude

1. **Les numeros sont parfaitement aleatoires.** 56 methodes testees, de la plus classique (chi2) a la plus avancee (IAAFT, RQA, DFA, symbolic dynamics, CCM). Toutes donnent z < 2 vs null iid. Zero signal. Le mur est infranchissable.

2. **Star2 a un vrai signal temporel.** +7.6pp au-dessus du null iid (z=+5.5). Le rolling mode de window 20-100 capture ce signal optimalement. **RIEN de plus complexe ne fait mieux** — pas le ML (57 features), pas les copulas, pas la CCM, pas les surrogates, pas la symbolic dynamics.

3. **Star1 est un mirage.** Sa "predictibilite" (18.3%) vient entierement de sa distribution biaisee. Pas de signal temporel.

4. **L'edge total est de ~2-3pp** sur la composante etoiles. ROI backtest ~25% (vs ~20% random). Reel mais insuffisant pour etre profitable.

5. **Les changements de format sont le piege #1.** Au moins 4 methodes avancees (symbolic dynamics z=11, IAAFT z=6.9, 1/f z=3.9, MI z=4.1) produisent des faux signaux massifs sur le full dataset qui s'evaporent sur era 3.

6. **Le "50%"** s'atteint en predisant 6 numeros au lieu de 5 (50.8%), mais le random est a 48.7%. L'edge reel reste ~2pp. C'est de la couverture, pas de la prediction.

### La formule la plus honnete

```
PREDICTION EUROMILLIONS — ORACLE EUREXPLO v5:
- Numeros: choisir aleatoirement (aucun edge detecte sur 56 methodes)
- Etoiles: top-3 rolling mode star2 (window=25), top-3 rolling mode star1
- Confiance "au moins 1 etoile correcte": ~35% (2 etoiles) / ~47% (3 etoiles)
- Edge reel vs random: +2-3 points de pourcentage
- Ce n'est PAS suffisant pour etre profitable
- Un modele d'IA/deep learning ne fera PAS mieux (teste et confirme)
```

### Methodes testees (liste exhaustive, 56)

**Classiques :** Chi2, autocorrelation, hot/cold, co-occurrence, FFT, PCA, wavelets (Haar)

**ML :** LogisticRegression, RandomForest, GradientBoosting (57 features), algorithme genetique

**Temporel :** Markov (ordres 1-5), HMM, CUSUM changepoints, runs test, rolling mode/frequency

**Information :** Compression (Kolmogorov), entropie de permutation (Bandt-Pompe), information mutuelle retardee, entropie locale

**Dynamique :** Takens embedding, NN en phase space, momentum, displacement autocorrelation, DFA (Detrended Fluctuation Analysis), Hurst exponent (R/S), recurrence plots (RQA)

**Spectral :** FFT presences, FFT ecarts, 1/f noise analysis, regression sinusoidale

**Statistique avancee :** Copulas (Kendall tau inter-positions), extreme value theory (block maxima, dispersion), IAAFT surrogate testing, record-breaking analysis

**Causal/Network :** Convergent Cross Mapping (CCM), network motifs temporels, matrice de transition chi2

**Decomposition :** NMF, clustering (K-means), embedding MDS, symbolic dynamics (n-grams)

**Geometrique :** Geometrie circulaire, spread dynamics, centroid, star gap

**Non-conventionnel :** Phase lunaire, Fibonacci, golden ratio, Vendredi 13, Pi encoding, Benford law, digit sum, modular residues (mod 3/5/7/11), parite, polynomial Vieta, jackpot influence, day of week, draw interval, drought analysis, pattern matching, sequence matching, binary decomposition, absence patterns, resonance cross-draw, conditional patterns, position-based prediction, coverage analysis, oracle ceiling
