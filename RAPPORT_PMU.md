# PMU Courses Hippiques — Rapport de Recherche Exploratoire

**Données :** 5 688 courses, 2004-2025, via `open-pmu-api` (nanaelie/open-pmu-api)
**Source :** `https://open-pmu-api.vercel.app/api/arrivees`

---

## Données disponibles

| Champ | Description | Utilisable pour prédiction |
|-------|-------------|------------------------|
| `date` | JJ/MM/AAAA | Oui |
| `lieu` | Hippodrome | Partiellement |
| `type` | Attelé / Monté | Partiellement |
| `distance` | En mètres | Partiellement |
| `partants` | Nombre de partants | Oui |
| `arrivee` | Liste ordinale des positions (1er, 2e, 3e...) | Limité |
| `prix` | Nom de l'épreuve | Non |
| `montant` | Montant dotation | Non |

**Manquant critique :**
- Numéro de départ (post position) par cheval
- Cotes de départ
- Historique individuel des chevaux
- Jockey / driver
- Gains, forme récente du cheval

---

## Résultat principal : données insuffisantes pour la prédiction

L'arrivée ne contient que les 5-20 premiers chevaux classés. Elle ne contient **pas** :
- La liste complète des partants au départ
- Le numéro de starting stall (position au départ)
- Les chevaux éliminés ou non-partants dans cette course

Cela rend impossible :
- Le calcul de vraie fréquence de victoire (victoires / courses courues)
- La comparaison fair entre chevaux
- La construction de features de forme individuelle

---

## Tests effectués

### 1. Biais de numéro de cheval en 1ère position

**Hypothèse :** certains numéros de cheval (numéro de départ) ont une probabilité plus haute de finir 1er.

**Résultat :** L'hypothèse est **non testable** avec ces données seules. Les numéros dans `arrivee` sont les numéros de cheval (numéro d'identification de l'animal, pas sa position de départ). Sans la liste complète des partants, on ne peut pas savoir si le cheval 9 gagne plus parce qu'il est meilleur ou parce qu'il court plus souvent.

### 2. Distribution des 1ères places par numéro de cheval

| Cheval | 1ères places | Fréquence |
|--------|-------------|-----------|
| Cheval 9 | 388 | 6.9% |
| Cheval 6 | 379 | 6.7% |
| Cheval 3 | 375 | 6.7% |
| Cheval 8 | 370 | 6.6% |
| ... | ... | ... |
| Cheval 18 | 102 | 1.8% |
| Cheval 17 | 119 | 2.1% |

**Interpretation :** Ces chiffres reflètent la fréquence de participation de chaque cheval dans les courses PMU (un cheval peut courir des centaines de courses). Ils ne constituent PAS un biais de prédiction.

### 3. Effet hippodrome

| Hippodrome | Courses | Cheval le plus fréquent en 1ère |
|------------|---------|-------------------------------|
| Vincennes | 2 486 | Cheval 11 (99x) |
| Auteuil | 585 | Cheval 11 (53x) |
| Paris-Longchamp | 518 | Cheval 12 (31x) |
| Deauville | 503 | Cheval 7 (44x) |

Pas de signal exploitable détecté.

### 4. Effet distance

Les courses sont réparties de 2000m à 3500m. Pas de relation simple détectée entre distance et probabilité de victoire d'un cheval spécifique — les sample sizes par bin deviennent trop petits.

---

## Comparaison avec EuroMillions

| | EuroMillions | PMU |
|--|--|--|
| Tirages | 1 936 | 5 688 |
| Espace | 5/50 + 2/12 | 1 à 20 partants |
| Données disponibles | Complètes (numeros + etoiles) | Partielles (arrivee top-N) |
| Biais détecté | Oui — étoiles non uniformes | Non testable |
| Prédictibilité | Low (numéros) / Medium (étoiles) | Indéterminable |

---

## Conclusion

**Les courses hippiques sont.mathématiquement exploitables** — les vraisadvantages players utilisent l'historique individuel des chevaux, les cotes, et les conditions de course pour construire des modèles prédictifs. Mais les données d'arrivée seules (sans cotes, sans historique cheval) sont **insuffisantes** pour reproduire cet edge.

Pour aller plus loin, il faudrait :
1. **Les cotes de départ** (données officielles PMU ou sites comme Turf.fr)
2. **L'historique individuel de chaque cheval** (gains, victoires, Performances récentes)
3. **Les conditions de course** (genre de course, age des chevaux, gains maxima)

Ces données sont disponibles via des APIs ou scraping de sites spécialisés (Turf BZH, Zeturf, etc.) mais nécessitent un travail de collecte plus important.
