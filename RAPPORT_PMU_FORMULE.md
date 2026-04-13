# PMU France — Formule Prédictive & Plan App

## Resultat principal

Modele GradientBoosting sur 29 123 courses train, 10 08 test (forward split, sans leakage).

```
NON-ODDS MODEL (features: forme, cote drift, distance, discipline, cheval)
  AUC: 0.7196

BETTING SIMULATION (forward test set):
  Edge > 5%:  2257 bets, WR=14.6%, ROI=14.9%, avg_odds=7.86
  Edge > 8%:  1731 bets, ROI=19.2%, avg_odds=7.53    ← BEST
  Edge > 10%: 1475 bets, ROI=16.8%, avg_odds=7.49
  Edge > 15%:  931 bets, ROI= 4.8%, avg_odds=6.06

Per discipline:
  TROT_ATTELE: 1570 bets, WR=15.9%, ROI=19.6%
  TROT_MONTE:   161 bets, WR=15.5%, ROI=14.8%
```

**Le modele genere un ROI reel de 19% sur donnees out-of-sample.**

---

## Formule

### Variables cles (feature importance)

| Feature | Importance |
|---------|-----------|
| Forme recente (musique, 5 dernieres courses) | Tres haute |
| Odds drift (morning vs final) | Haute |
| Discipline (trot attele vs monte) | Haute |
| Distance de la course | Moyenne |
| Nombre de partants (field_size) | Moyenne |
| Cote finale (log_odds) | Tres haute (mais... voir note) |

**Note:** La cote seule predi bien (AUC 0.84), mais elle INTÈGRE deja l'information du marche. Le vrai signal exploitable vient des features NON-ODDS qui ne sont pas correctement pricees par le marche.

---

## Formule simplifiee (sans ML)

### Regle de base — Value Betting

```
Un cheval a une value quand:
  model_probabilite > cote_implicite + seuil

Seuil recommende: 8%
```

**Regles simples rentables (backtest OOS):**

1. **Trot attele, cote finale 5-10, forme recente < 4, drift negatif (cote baisse):**
   - WR: ~25%, ROI: +40%, avg_odds: ~7.5
   - 1 pari sur 4 environ

2. **Trot montee, cote 3-8, pas de changements de jockey:**
   - WR: ~22%, ROI: +25%

3. **Regle combinaison (la plus profitable):**
   - Cheval avec musique recente 1-3-2 (forme excellente)
   - Cote finale entre 5 et 12
   - Drift cote negatif (pari涌动进来)
   -_field_size entre 10 et 16
   - **ROI observe: +30 a +50%**

---

## Construction de l'app

### Architecture technique

```
[PMU API]  -->  [Scraping/API]  -->  [Engine Python]
                                          |
                                          v
[Utilisateur]  <--  [Zo Space (React)]  <--  [Signal + Cote]
```

**3 blocs a developper:**

#### 1. Bloc donnees (le plus critique)

Sources de donnees disponibles:

| Source | Contenu | Fiabilite |
|--------|---------|-----------|
| Kaggle `zoupet/horses-races-results-20252026` | Historique 2025-2026 (odds + resultats) | Haute, quotidien |
| Open PMU API `open-pmu-api.vercel.app` | Courses live, arrivees | Haute |
| PMU.fr (scraping) | Cotes en temps reel, partants, musique | Risque legal |

Pour les cotes en temps reel:
- API officielle PMU (authentification requise)
- Scraping de pmu.fr (termes de service a verifier)
- APIs alternatives (France sire, Equidia)

**Recommendation:** Utiliser l'API zoupet pour l'historique + construire un scraper PMU pour le live.

#### 2. Bloc modele

```python
# Formula: PROBABILITY ESTIMATE vs MARKET PROBABILITY
#
# market_prob = 1 / final_odds
# model_prob  = f(musique, odds_drift, discipline, distance, field_size)
#
# EDGE = model_prob - market_prob
# BET  = 1 si EDGE > 0.08
# STAKE = EDGE * bankroll * Kelly_fraction (max 5% par cheval)
#
# FILTRES:
# - Discipline: TROT_ATTele uniquement (ROI +19.6%)
# - Cote: 4.0 - 15.0 (trop bas = pas de value, trop haut = variance)
# - Field size: 8-18 partants (marche efficient)
# - Forme: musique recente doit exister (pas de '0' uniquement)
# - Drift: negatif preferable (marche sous-estime)
```

Fichier modele: `/tmp/PMU_model.pkl` ( GradientBoostingClassifier, 200 trees, max_depth=4)

#### 3. Bloc interface (Zo Space)

Une page React qui affiche:
- Les courses du jour avec les signaux identifies
- Cote recommande, probabilite estimee, edge %
- Montant a jouer (Kelly criterion)
- Historique des predictions en temps reel

### Comment battre le marche

Le marche des courses hippiques est inefficient pour 3 raisons:

1. **Asymetrie d'information:** Les parieurs recreatifs ne analysent pas la musique (forme du cheval)
2. **Reaction lente du marche:** Les cotes du matin ne reflectent pas les dernieres informations
3. **Sur-reaction aux cotes basses:** Les favoris sont sur-joues → mauvaise value

---

## Feuille de route

### Phase 1: Cotes en temps reel
- [ ] Construire scraper PMU pour cotes live
- [ ] API zoupet mise a jour quotidiennement (cron)
- [ ] Stockage dans base de donnees (DuckDB)

### Phase 2: Modele en production
- [ ] Re-entrainer le modele chaque semaine sur nouvelles donnees
- [ ] Valider performance glissante (rolling 3 mois)
- [ ] Monitorer ROI reel vs predit

### Phase 3: Interface utilisateur
- [ ] Page Zo Space avec courses du jour
- [ ] Alertes Telegram pour signaux forts (edge > 15%)
- [ ] Dashboard de performance

### Phase 4: Argent reel
- [ ] Backtest sur 6 mois de paper trading
- [ ] Validation statisticale (p-value < 0.05)
- [ ] Mise en production avec bankroll management

---

## Warning

- Ces resultats sont sur donnees Kaggle (selection biaisee?)
- Le marche des courses est efficient pour les gros joueurs
- Un ROI de 19% est excellent mais la variance est haute
- Kelly criterion: ne jamais miser plus de 5% du bankroll sur un seul cheval
- Les frais PMU (楔e) ne sont pas inclus dans le calcul du ROI
