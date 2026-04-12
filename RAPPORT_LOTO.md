# Loto France — Rapport de Recherche Exploratoire

**Date :** 2026-04-13  
**Source :** FDJ opendata (`media.fdj.fr/static/csv/loto/`)  
**Données :** 7332 tirages, 1976-05-19 → 2024-07-24

---

## 1. Structure des données

### 1.1 Deux époques distinctes

| Époque | Tirages | Boules | Pool chance | Description |
|--------|---------|--------|-------------|-------------|
| **Ancienne** (1976–2008) | 4858 | 6 | 1–49 | 5 numéros + 1 complémentaire (tirée du même pool) |
| **Nouvelle** (2008+) | 2474 | 5 | 1–10 | 5 numéros + 1 "numéro chance" (pool séparé 1–10) |

Les données officielles FDJ sont en 2 formats avec des changements de structure (colonnes, dates, pool).

---

## 2. Tests sur les numéros

### 2.1 Numéros — Uniformité (nouvelle formule)

```python
# 2474 tirages × 5 boules / 49 numéros
Chi2 (df=48) = 34.5  # Critique (α=0.05) = 67.5
→ Accepté : compatible avec uniforme
```

### 2.2 Numéros — Autocorrélation

```python
# Test sur nouvelle formule uniquement (pool 1-49)
Corrélation lag-1 (complémentaire): r = -0.0095
Corrélation lag-1..10 (numéro chance): tous |r| < 0.024
→ Aucune mémoire temporelle détectée
```

### 2.3 Numéros — Hot/Cold (OOS, w=100)

```python
# Pour chaque numéro: prédire "apparaît" si freq(window) > seuil
Mean hits par tirage (top-5 mode): 0.51 / 5
Baseline attendu (5/49): 0.102 / 5
Edge: 0.04x → AUCUNE structure exploitable
```

**Résultat :** Les numéros du Loto sont parfaitement compatibles avec un processus aléatoire uniforme. Aucun pattern, aucune mémoire, aucun signal détectable.

---

## 3. Tests sur le numéro chance

### 3.1 Ancienne formule (pool 1-49)

```python
# 4858 tirages, complémentaire tirée du même pool 1-49
Chi2 (df=48) = 46.9  # Critique = 67.5
→ Accepté : la complémentaire est uniforme
Autocorrélation lag-1: r = -0.0095 (nulle)
```

### 3.2 Nouvelle formule (pool 1-10)

```python
# 2474 tirages, numéro chance tirée du pool 1-10
Chi2 (df=9) = 9.5  # Critique = 16.9
→ Accepté : le numéro chance est compatible avec uniforme
```

Fréquences observées vs attendues (2474/10 = 247.4) :

| Valeur | Observé | Attendu | z |
|--------|---------|---------|---|
| 1 | 260 | 247 | +0.80 |
| 2 | 238 | 247 | -0.60 |
| 3 | 238 | 247 | -0.60 |
| 4 | 247 | 247 | -0.03 |
| 5 | 260 | 247 | +0.80 |
| 6 | 223 | 247 | -1.55 |
| 7 | 274 | 247 | +1.69 |
| 8 | 226 | 247 | -1.36 |
| 9 | 255 | 247 | +0.48 |
| 10 | 253 | 247 | +0.36 |

**Aucune valeur n'est significativement différente (tous |z| < 2).**

### 3.3 Rolling mode prédiction sur numéro chance (OOS)

```python
Window | Accuracy | Baseline | Edge
-------|----------|----------|-----
  30   |  7.88%   |  10.00%  | 0.79x
  50   |  9.70%   |  10.00%  | 0.97x
 100   |  9.29%   |  10.00%  | 0.93x
 200   | 12.12%   |  10.00%  | 1.21x
```

→ **Pas de signal robuste.** L'accuracy oscille autour du random, aucun edge stable.

---

## 4. Comparaison Euromillions vs Loto

| Propriété | Euromillions | Loto |
|----------|-------------|------|
| Tirages analysés | 1936 | 7332 |
| Numéros | 5 / 50 (uniforme) | 5 / 49 (uniforme) |
| Étoiles/Chance | 2 / 12 (**biaisée**, autocorr) | 1 / 10 (uniforme) |
| Signal sur étoiles | **OUI** (rolling mode, z=12σ) | **NON** |
| Autocorrélation étoiles | star2 lag-3 : r=+0.155*** | chance : r<0.024 |
| Regime change | star12 apparaît en 2016 | Non détecté |

### Hypothèse expliquant la différence

À l'EuroMillions, la machine à étoiles est **physiquement différente** de la machine à boules (étoiles = boules plus petites et légères). Le pool d'étoiles (12) est petit, ce qui amplifie les biais machine. La machine à numéros (50 boules) est plus stable et compense mieux les imperfections physiques.

Au Loto, le numéro chance (1-10) est tiré depuis une machine séparée mais le pool很小 (10), et le pool de numéros (49) est plus grand, ce qui dilue les biais. Le changement de formule en 2008 (pool chance 1-10 au lieu de 1-49) a encore réduit les biais potentiels.

---

## 5. Conclusion

**Loto = proceso aléatoire puro.**

- Numéros : parfaitement compatibles avec uniforme, zéro signal
- Numéro chance : parfaitement compatible avec uniforme (nouvelle formule), zéro signal
- Ancienne complémentaire : également uniforme
- Rolling mode sur chance : aucune edge stable
- Les biais machine visibles sur EuroMillions étoiles ne se retrouvent pas sur Loto

**Recommandation :** Le Loto français n'offre aucune capacité prédictive exploitable. Seuls les tirages EuroMillions étoiles montrent un signal statistiquement significatif — mais avec une edge limitée (16% vs 8% random).

---

## 6. Données sources

Fichiers CSV FDJ :
- `loto_197605.zip` → `loto.csv` (4858 tirages, 1976–2008)
- `loto_200810.zip` → `nouveau_loto.csv` (1317 tirages, oct 2008–mars 2017)
- `loto_201703.zip` → `loto2017.csv` (310 tirages, mars 2017–fév 2019)
- `loto_201902.zip` → `loto_201902.csv` (107 tirages, fév–nov 2019)
- `loto_201911.zip` → `loto_201911.csv` (740 tirages, nov 2019–juil 2024)

Données brutes consolidées : `loto.json`