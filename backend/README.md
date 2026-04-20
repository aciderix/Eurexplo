# Backend PMU — Supabase

Pipeline: scraper PMU → DB Supabase → edge functions → app Android.

## Setup local (une fois)

```bash
# 1. Python deps
cd backend
pip install -r requirements.txt

# 2. Env vars (copier .env.example à la racine → .env)
cd ..
cp .env.example .env
$EDITOR .env    # Coller SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY
```

`.env` est gitignoré — ne jamais le commiter.

## Déploiement du schéma

Les migrations sont dans `supabase/migrations/`. Deux options pour les appliquer :

### Option A — Supabase CLI (recommandé)

```bash
# 1. Créer un Personal Access Token (PAT) ici :
#    https://supabase.com/dashboard/account/tokens
#    Le copier dans SUPABASE_ACCESS_TOKEN (une fois, dans ta shell).

export SUPABASE_ACCESS_TOKEN=sbp_xxxxx

# 2. Lier le projet (une fois)
~/.local/bin/supabase link --project-ref aomsraqmueugzzozjyiz
#   → demande le mot de passe DB (depuis Settings → Database → Connection string)

# 3. Pousser les migrations
~/.local/bin/supabase db push

# Pour voir ce que ça va faire sans l'appliquer :
~/.local/bin/supabase db push --dry-run
```

### Option B — Dashboard (manuel)

Ouvrir `https://supabase.com/dashboard/project/aomsraqmueugzzozjyiz/sql/new`,
coller le contenu de chaque fichier `supabase/migrations/*.sql` **dans l'ordre
alphabétique**, et exécuter.

## Scripts backend

```bash
# Uploader le modèle entraîné localement
python backend/upload_model.py \
    --model model_lgbm.pkl \
    --meta model_lgbm_meta.json \
    --version 2026-04-14 \
    --activate

# Ingérer les courses du jour dans la DB
python backend/ingest_daily.py --date 14042026
```

## Cron serveur (à mettre en place plus tard)

```cron
# /etc/cron.d/pmu
0 6  * * *  /usr/bin/python3 /srv/pmu/pmu_scraper_turbo.py --resume
30 6 * * *  /usr/bin/python3 /srv/pmu/backend/ingest_daily.py --date $(date +\%d\%m\%Y)
45 6 * * *  /usr/bin/python3 /srv/pmu/backend/predict_today.py
0 3 * * 1   /usr/bin/python3 /srv/pmu/pmu_train_walkforward.py --last-2y
```

## Architecture

```
PMU API → pmu_scraper_turbo.py → pmu_daily_json/*.json
                                         │
                                         ▼
                           backend/ingest_daily.py
                                         │
                                         ▼
                    ┌─────── Supabase Postgres ───────┐
                    │  races, participants,            │
                    │  predictions, bets_log,          │
                    │  model_versions                  │
                    └──────────────────────────────────┘
                                         │
                          Edge functions /predict-today
                                         │
                                         ▼
                                  Android app
```

## Sécurité

- **Service role key** / **`sb_secret_*`** : à garder dans `.env`. JAMAIS
  dans l'app Android, JAMAIS commitée.
- **Anon / `sb_publishable_*`** : safe à embarquer dans l'app Android. Les
  policies RLS empêchent tout accès non autorisé.
- Les policies RLS sur `bets_log` isolent chaque utilisateur.
```
