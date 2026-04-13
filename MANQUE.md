# Ce qui manque — PMU AI Predictor

## Ce qu'on a déjà

| Élément | Status | Source |
|---------|--------|--------|
| Historique courses PMU France (2004→2026) | ✅ | Open PMU API |
| Caractéristiques chevaux/jockeys/entraineurs | ✅ | Open PMU API |
| Résultats détaillés UK avec cotes | ✅ | Kaggle (`zoupet/horses-races-results`) |
| Cotes finales UK | ✅ | Kaggle (`arnopub/courses-hippiques/odds.csv`) |
| Modèle ML gradient boosting | ✅ | `pmu_app.py` |
| Backtests documentés | ✅ | `RAPPORT_BACKTEST_PMU.md` |
| Interface web (Zo Space) | ✅ | `https://kka.zo.space/` |

## Ce qui manque

### 1. Cotes PMU françaises en temps réel — BLOQUANT

**Pourquoi c'est bloquant :** sans cotes, on ne peut pas calculer le Value Betting (equity vs cote offerte). Tous nos signaux sont relatifs (tel cheval est "meilleur" que tel autre) mais sans cote on ne sait pas si le marché offre une value.

**Pistes par ordre de faisabilité :**

#### A. Scraping turf-fr.com (faisable, rapide)
- turf-fr.com affiche les cotes PMU pour chaque cheval
- URL par course : `https://www.turf-fr.com/courses-pmu/cotes/evolution-cotes-pmu/{reunion}-{prix}`
- Contenu dans le HTML (pas de JS complexe)
- **À faire :** écrire un scraper BeautifulSoup qui récupère les cotes pour les courses du jour
- Limite : site public, scraping modéré accepté, mais scraping intensif peut être bloqué

#### B. API GenyBet (nécessite inscription)
- GenyBet propose une API ou un flux de données courses
- À vérifier : `https://api.genybet.fr` ou équivalent
- Necessite un compte et potentiellement une partnership

#### C. Contacter l'auteur d'Open PMU API
- L'API existe et sert les arrivées (`/api/arrivees`)
- Lui demander d'ajouter les cotes (`/api/cotes?date=JJ/MM/AAAA`)
- C'est open source : `https://github.com/nicogue/open-pmu-api`

### 2. Courses du jour en temps réel — BLOQUANT

L'API Open PMU ne donne que les **arrivées passées**. Pour les courses à venir :
- Vérifier `/api/arrivees?date=AUJOURDHUIT` chaque matin
- Ou utiliser `https://www.pmu.fr` directement

### 3. Tracker de paris personnel — RECOMMANDÉ

Pour comparer prédit vs réel et calculer le ROI réel :
- Enregistrer tes paris quelque part (pas encore fait)
- Comparer avec les résultats une fois la course terminée
- Utiliser l'API Open PMU pour vérifier les résultats automatiquement

### 4. Pipeline de mise à jour quotidienne — OPTIONNEL

- Télécharger les nouvelles courses via Open PMU API chaque jour
- Retraîner le modèle tous les 1-2 mois
- Mettre à jour les top-entraineurs/jockeys weekly

---

## Plan d'action prioritaire

1. **Scraper turf-fr.com** pour les cotes → 2-3h de travail
2. **Vérifier si pmu.fr a une API** (compte requis ?) → 30 min
3. **Construire le tracker de paris** → 1h
4. **Combiner cotes + prédictions dans l'app** → 2h

---
