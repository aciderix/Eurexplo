# Accès API — Eurexplo

## Kaggle (pour télécharger les données UK/FR)

1. Créer un compte sur [kaggle.com](https://kaggle.com)
2. Aller dans Account → Create New API Token → télécharge `kaggle.json`
3. Stocker le fichier dans `~/.kaggle/kaggle.json` sur la machine qui lance les scripts

```bash
mkdir -p ~/.kaggle
cp /chemin/vers/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle
```

**Attention :** ne jamais commit ce fichier dans Git.

---

## PMU France — Open PMU API

**Base URL :** `https://open-pmu-api.vercel.app/api`

**Endpoints disponibles :**

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/runs` | Liste de tous les partants |
| GET | `/runs/[id]` | Détail d'un partant |
| GET | `/courses` | Liste des courses |
| GET | `/courses/[id]` | Détail d'une course |
| GET | `/chevaux` | Liste de tous les chevaux |
| GET | `/chevaux/[id]` | Fiche complète d'un cheval |
| GET | `/jockeys` | Liste des jockeys |
| GET | `/jockeys/[id]` | Fiche d'un jockey |

**Limites :** endpoint gratuit, aucune clé requise (pour l'instant).

---

## FDJ (Française des Jeux)

### Tirages Euromillions / Loto

Source : `https://github.com/pedro-mealha/euromillions-api`

API publique : `https://euromillions.api.pedromealha.dev`

```bash
# Tous les tirages
curl https://euromillions.api.pedromealha.dev/v1/draws

# Un tirage précis
curl https://euromillions.api.pedromealha.dev/v1/draws/1936
```

### Keno — Fichiers ZIP FDJ

```bash
# Historique par année
curl -L "https://media.fdj.fr/static/csv/keno/keno_2025.zip" -o keno.zip
curl -L "https://media.fdj.fr/static/csv/keno/keno_2024.zip" -o keno.zip
curl -L "https://media.fdj.fr/static/csv/keno/keno_2023.zip" -o keno.zip
```

---

## Configuration pour les scripts

Créer `config.json` (jamais commit) :

```json
{
  "kaggle": {
    "username": "votre_username",
    "key": "votre_api_key"
  }
}
```

---

## Variables d'environnement (pour Zo Space / production)

| Variable | Description |
|----------|-------------|
| `KAGGLE_USERNAME` | Username Kaggle |
| `KAGGLE_KEY` | Clé API Kaggle |
| `STRIPE_SECRET_KEY` | Clé Stripe (si vente premium) |
| `ZO_API_KEY` | Clé API Zo (pour accès bot) |
