# Euromillions — Session 8 : KENO break-through

**Date :** 2026-04-13  
**Source :** FDJ Keno opendata (`sto.api.fdj.fr`)  
**Données :** 19,133 tirages, 1993-09-16 → 2025-11-02  
**Structure :** 20 boules tirées sans remise parmi 70 (pool 1-70)

---

## 1. Données

| Champ | Valeur |
|-------|--------|
| Tirages | 19 133 |
| Boules par tirage | 20 |
| Pool | 1–70 |
| Périodicité | 2 tirages/jour (midi + soir) |
| Multiplicateur | disponible depuis ~2018 (3 933 tirages) |
| Emplacement | 70 colonnes (historique complet) |

---

## 2. Test d'uniformité — ACCEPTÉ

```python
Chi2(df=69) = 51.0, p = 0.9491
Max |z| = 1.96σ (dans l'enveloppe normale)
→ Accepté : les 70 numéros sont compatibles avec uniforme
```

Les boules du Keno sont parfaitement uniformes. Même pas besoin de chercher des biais.

---

## 3. Position bias — SIGNAL FORT DÉTECTÉ

### Anomalie découverte

Les boules ne sont pas tirées dans un ordre aléatoire. Elles semblent être **ordonnées par valeur** dans le résultat :

- **Boule 1** : apparaît massivement en position basse (positions 1-5 très sur-représentées)
- **Boule 70** : apparaît massivement en position haute (positions 16-20 très sur-représentées)

```python
Ball  1: n=5480, chi2(df=19)=104120, std_pos=21.8%  ← REJET FORT
Ball 35: n=5520, chi2(df=19)=11249,  std_pos=7.1%
Ball 70: n=5374, chi2(df=19)=102106, std_pos=21.8%  ← REJET FORT
```

### Interprétation

La boule 1 est **systématiquement placée dans les premières positions** (triées ASC). La boule 70 est **systématiquement placée dans les dernières positions**. 

Cela signifie que :
1. Soit les boules sont tirées et **ensuite triées** (comportement normal d'une machine de tirage)
2. Soit la machine a un biais positionnel systématique par valeur

Pour les boules de milieu de gamme (ex: boule 35), le biais positionnel est beaucoup plus faible (std=7.1%), ce qui est attendu si le tri est la cause.

### Hypothèse : les boules ne sont pas tirées random mais排序ées

Les 20 boules sont visiblement stockées dans un ordre précis et tirées dans cet ordre. Le pool 1-70 a peut-être un **ordre de stockage physique** qui cause ce biais.

---

## 4. Prédictibilité — Résultats OOS

| Stratégie | Accuracy | Baseline | Edge |
|-----------|---------|---------|------|
| Rolling mode (w=50, top-3) | 28.43% | 85.71% | **0.33x** ← pire que random |
| Due ball (gap le + long) | 29.12% | 28.57% | 1.02x ← aucun edge |
| Random (top-3 aléatoire) | ~28.57% | 28.57% | 1.00x |

### Interpretation

La rolling mode est **pire** que le random — parce que la fenêtre glissante inclut des boules qui ne sont plus favorites. Le système n'a pas de mémoire courte exploitable.

**Conclusion :** Même si les boules ont un biais positionnel structurel, **la valeur de la boule n'a aucun lien avec sa probabilité d'apparition**. Chaque boule a la même probabilité, le biais est uniquement dans l'ordre de tri.

---

## 5. Multiplicateur

Distribution (n=3933 tirages) :

| Multiplicateur | Comptage | % |
|---------------|---------|---|
| x1 | 966 | 24.6% |
| x2 | 1376 | 35.0% |
| x3 | 989 | 25.1% |
| x4 | 415 | 10.6% |
| x5 | 150 | 3.8% |
| x10 | 37 | 0.9% |

**Observation :** x2 est le plus fréquent (35%), x1 et x3 sont proches (~25%). Le x10 est rare (0.9%). Cette distribution est publique FDJ et constitue la base de la plupart des stratégies multiplicateur.

---

## 6. Comparaison des 3 jeux

| | EuroMillions | Loto | KENO |
|--|--|--|--|
| Tirages | 1936 | 7332 | **19133** |
| Signal编号 | AUCUN | AUCUN | AUCUN (uniforme) |
| Signal étoile/chance | **OUI** (star2 autocorr) | AUCUN | **POSITION BIAS** (fort) |
| Meilleure stratégie | Rolling mode (w=100) | Aucune | Aucune (edge <1.02x) |
| Edge max | ~2x | 1.00x | 1.02x |
| Anomalie | Biais machine étoiles | Régime change 2008 | Tri positionnel fort |

---

## 7. Conclusion KENO

**Les boules sont parfaitement uniformes** — aucune anomalie de fréquence. 

**MAIS** il y a un biais positionnel Extrêmement fort pour les boules 1 et 70. Ce biais est probablement dû au **tri ASC** des résultats par la machine FDJ, pas à un défaut du processus de tirage.

**Pas de prédictibilité exploitable sur les valeurs.** Le multiplicateur x2 arrive ~35% du temps — c'est la seule information structurelle utile.

---

## 8. Recommandation exploration

Le break-through sur KENO serait d'explorer :
1. **Le multiplicateur comme variable cible** — peut-on prédire si le prochain tirage aura mult > x3 ?
2. **Les "emplacements"** — les données contiennent 70 colonnes d'emplacements (les positions de chaque boule dans le historique). C'est une riche source d'analyse de corrélation entre positions.
3. **L'ordre de stockage hypothéqué** — si la machine stocke les boules dans un ordre fixe, il y a peut-être un effet de position d retrieval.

Données brutes : `keno.json` (19 133 tirages)