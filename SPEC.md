# Accès API — Eurexplo

**Dernière mise à jour :** 2026-04-13
**Repo :** https://github.com/aciderix/Eurexplo

---

## Source temps réel : PMU Open API

**Base URL :** `https://open-pmu-api.vercel.app/api`

| Méthode | Endpoint | Description | Réponse |
|---------|----------|-------------|---------|
| GET | `/courses` | Courses du jour | `{id, date, nom, hippodrome, type_course, distance}` |
| GET | `/courses/[id]` | Détail d'une course | `{...partants: [{cheval, jockey, costaud, stats}]}` |
| GET | `/runs` | Tous les partants | Liste complète des chevaux partants |
| GET | `/runs/[id]` | Fiche partant | `{cheval, cote, position, historique}` |
| GET | `/chevaux` | Index chevaux | `{id, nom}` |
| GET | `/chevaux/[id]` | Fiche cheval (historique 50 courses) | `{nom, age, courses[], victories, places}` |
| GET | `/jockeys` | Index jockeys | `{id, nom}` |
| GET | `/jockeys/[id]` | Fiche jockey | `{nom, courses, victoires}` |

**Exemple récup courses du jour :**
```python
import requests
r = requests.get('https://open-pmu-api.vercel.app/api/courses')
courses = r.json()  # [{id, date, hippodrome, nom, partants:[{cheval, jockey}]}]
```

**Exemple fiche cheval :**
```python
r = requests.get('https://open-pmu-api.vercel.app/api/chevaux/ID_CHEVAL')
cheval = r.json()  # → {nom, age, historique 50 courses, % win, % place}
```

**Limites :** API gratuite, aucune clé requise.

---

## Données historiques stockées (fichiers JSON/CSV)

| Donnée | Nb lignes | Source | Fichier |
|--------|-----------|--------|---------|
| Courses PMU France | 5 688 courses | PMU Open API + scraping | `pmu_races.json` |
| Partants UK/IRE avec cotes | 38 314 | Kaggle `zoupet/horses-races-results-20252026` | `kaggle/race_results.csv` |
| Cotes UK/IRE | 49 842 | Kaggle `arnopub/horse-racing-odds` | `kaggle/odds.csv` |
| Historique forme UK | 203 340 rows | Kaggle `deltaromeo/horse-racing-uk-ireland-2015-2025` | `kaggle/recent_form.csv` |
| Tirages Keno FDJ | 19 133 | FDJ ZIP `media.fdj.fr` | `keno.json` |
| Tirages Loto FDJ | 7 332 | FDJ ZIP `media.fdj.fr` | `loto.json` |
| Tirages Euromillions | 1 936 | API `euromillions.api.pedromealha.dev` | `draws.json` |

---

## FDJ — Tirages (historique)

### Euromillions
```bash
# API publique (aucune clé requise)
curl https://euromillions.api.pedromealha.dev/v1/draws        # tous les tirages
curl https://euromillions.api.pedromealha.dev/v1/draws/1936   # tirage précis
```

### Loto / Keno — Fichiers ZIP annuels
```bash
# Loto
curl -L "https://media.fdj.fr/static/csv/loto/loto_2024.zip" -o loto.zip
curl -L "https://media.fdj.fr/static/csv/loto/loto_2023.zip" -o loto.zip

# Keno
curl -L "https://media.fdj.fr/static/csv/keno/keno_2025.zip" -o keno.zip
curl -L "https://media.fdj.fr/static/csv/keno/keno_2024.zip" -o keno.zip
```

---

## Kaggle — Téléchargement des datasets

```bash
# Installer kaggle
pip install kaggle

# Configurer credentials
mkdir -p ~/.kaggle
# Copier kaggle.json dans ~/.kaggle/kaggle.json (depuis kaggle.com/account)

# Télécharger les datasets UK/FR
kaggle datasets download -d arnopub/courses-hippiques
kaggle datasets download -d zoupet/horses-races-results-20252026 -f race_results.csv
kaggle datasets download -d zoupet/horses-races-results-20252026 -f odds.csv
kaggle datasets download -d deltaromeo/horse-racing-uk-ireland-2015-2025
kaggle datasets download -d arnopub/horse-racing-odds
```

**Structure fichiers CSV :**
- `race_results.csv` : `runner_id, race_id, horse_name, result_position, odds, ...`
- `odds.csv` : `odds_id, runner_id, european_odds, win_probability, ...`
- `recent_form.csv` : `date, course_name, horse_name, result, SP, ...`

---

## Zo Space — Routes de l'app

| Route | Type | Description |
|-------|------|-------------|
| `/` | page | Dashboard React complet |
| `/api/courses` | api | Courses du jour + stats |
| `/api/dashboard` | api | KPIs (WR, ROI, bankroll) |
| `/api/predict` | api | Prédictions par cheval |

---

## Variables d'environnement (Zo Space / production)

| Variable | Description |
|----------|-------------|
| `KAGGLE_USERNAME` | Username Kaggle |
| `KAGGLE_KEY` | Clé API Kaggle |
| `STRIPE_SECRET_KEY` | Clé Stripe (optionnel, pour version premium) |
| `ZO_API_KEY` | Clé API Zo (optionnel, pour notifications) |

---

## Configuration locale (scripts Python)

Créer `config.json` à la racine du projet :

```json
{
  "kaggle": {
    "username": "votre_username",
    "key": "votre_cle_api"
  }
}
```

**Ne JAMAIS commit ce fichier.**

---

## Axes non implémentés (TODOs)

- [ ] Mise à jour automatique des cotes en temps réel (H-1 course)
- [ ] Récupération des résultats de chaque course en direct
- [ ] Historique personnel des paris utilisateur (base SQLite)
- [ ] Notifications push (Telegram / email) sur bets du jour
- [ ] Intégration Stripe pour version premium
- [ ] Données UK temps réel (open-pmu-api ne couvre que FR)