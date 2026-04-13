# PMU AI Predictor — Application Web

## 1. Concept & Vision

Application de prédiction pour les courses hippiques françaises (PMU). L'interface présente les courses du jour avec les prédictions du modèle, les stratégies recommandées, et les performances historiques. L'objectif est de donner à l'utilisateur un outil sérieux, documenté, avec des chiffres vérifiables — pas un꾸미기 de promesses exagérées.

Philosophy : transparence totale sur les performances, les limites, et les risques.

---

## 2. Design Language

**Aesthetic:** Dark data-driven, inspired by Bloomberg Terminal + modern fintech. Dense information, clear hierarchy, zero fluff.

**Color palette:**
- Background: #0a0a0f (near-black)
- Card surface: #141420
- Border: #2a2a3a
- Primary accent: #3b82f6 (blue — confiance haute)
- Warning accent: #f59e0b (amber — confiance medium)
- Danger: #ef4444 (red)
- Success: #22c55e (green)
- Text primary: #f1f5f9
- Text muted: #94a3b8

**Typography:**
- Headings: Inter (bold), fallback sans-serif
- Body/data: JetBrains Mono (monospace for numbers), fallback monospace
- Sizes: 12px base for dense data, 14px for readable content

**Motion:**
- Minimal — only skeleton loaders and subtle progress indicators
- No decorative animations

---

## 3. Layout & Structure

### Page principale — Dashboard

```
[HEADER] Logo + Nav (Dashboard | Recherche | Stats | A propos)
[COURSE FILTER] Date picker + Hipodrome filter
[COURSE GRID] Cards pour chaque course du jour
  [RACE CARD] Hour, hipodrome, distance, discipline
    [RUNNER TABLE] Classement par probabilité predite
      | # | Cheval | Cote | Probabilité | Edge | Strategie | Kelly |
```

### Pages secondaires

- **Recherche** — explorer un cheval / driver / entraineur par nom
- **Stats** — performance du modèle (WR, ROI, distribution par segment)
- **A propos** — documentation, methodology, limitations, risks

---

## 4. Features & Interactions

### Course du jour
- Affiche les courses du jour depuis open-pmu-api
- Chaque cheval prédit avec probabilité, cote FDJ, edge (valeur vs probabilité)
- Recommandation stratégique (Confiance haute/Médium/Edge)
- Indicateur Kelly (mise recommandée)

### Backtest interactif
- Sélectionner période, stratégie, niveau de mise
- VoirWR, ROI, distribution des gains/pertes
- Graphique évoluant du bankroll

### Recherche
- Recherche par nom de cheval, driver, entraineur
- Historique de performances

### Données temps réel
- Courses: open-pmu-api (Vercel)
- Cotes: open-pmu-api ou scraping FDJ
- Résultats: open-pmu-api après course

---

## 5. Component Inventory

### RaceCard
- Header: heure, hippodrome, distance, discipline (Obstacle/Plat/Trot)
- Badge statut: à venir / en cours / terminé
- Runner table: trié par probabilité predite
- Statut: skeleton loader / data / no races

### RunnerRow
- Position dans le pronostic (1-5)
- Nom cheval (lien vers fiche)
- Cote FDJ
- Barre de probabilité (%)
- Badge edge: vert si >5%, rouge si <0
- Badge strategie: HIGH/MED/EDGE
- Mise Kelly (en euros)

### StatsPanel
- 4 KPI en ligne: WR global, ROI, Paris joués, Bankroll simulé
- Graphique ligne: évolution bankroll (pari fixe)
- Heatmap: WR par type de course (discipline, distance, nm partants)

### RiskDisclaimer
- Toujours visible en bas: "Les performances passées ne garantissent pas les résultats futurs. Jouer comporte des risques. Ne jouez pas plus que ce que vous pouvez perdre."

---

## 6. Technical Approach

**Stack:** Zo Space (Hono + React) — zero-config, instant deploy

**Backend:**
- API routes pour charger les données en temps réel
- Cache des résultats en mémoire (1h TTL)
- Modèle pickle loaded au démarrage

**Data sources:**
- Courses/chevaux: `https://open-pmu-api.vercel.app/api/courses/daily/{date}`
- Cotes: à déterminer (open-pmu-api ou scraping)
- Résultats: open-pmu-api après course

**Model:**
- Pre-loaded au démarrage du server (pickle)
- Prediction via features déjà calculées dans le script
- Pour l'app web: on recalcule les features côté serveur à partir des données temps réel

**API endpoints:**
- `GET /api/courses?date=YYYY-MM-DD` — courses du jour
- `GET /api/predictions/:race_id` — predictions pour une course
- `GET /api/stats` — performance du modèle
- `GET /api/search?q=cheval` — recherche

**Fichiers:**
- `SPEC.md` — ce document
- `api/` — routes backend
- `pages/` — composants React
- `scripts/` — scripts de processing (pré-calculation des predictions)