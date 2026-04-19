# Handoff — pipeline PMU v3 sur Kaggle

Doc de reprise pour un agent qui prend la suite. À jour au 19 avril 2026.

## TL;DR

Le pipeline ML PMU v3 tourne sur un Kaggle notebook qui clone ce repo, installe
les requirements, restaure l'état depuis un dataset Kaggle, lance
l'orchestrateur, snapshot tout et re-push le dataset avant de rendre la main.
Chaque session Kaggle dure max 9 h ; l'orchestrateur s'arrête proprement à
8 h (`PMU_BUDGET_MIN=480`) pour laisser 1 h de snapshot + upload.

Au moment de cette note :
- **v15 en cours** (ou fini au moment où tu lis) — doit juste exécuter
  `backtest` + `bandit`, les 7 autres stages sont déjà en cache sur le dataset.
- Kernel : https://www.kaggle.com/code/aciiderixx/pmu-v3-pipeline
- State dataset : https://www.kaggle.com/datasets/aciiderixx/pmu-v3-state

## Credentials

```bash
# Kaggle (KGAT_ token, PAS un kaggle.json)
export KAGGLE_API_TOKEN=KGAT_55fb107012d69bbd3762aa70a7c35f5e
# Tous les appels CLI doivent passer ce header.
# KAGGLE_USERNAME / KAGGLE_KEY donnent 401 avec ce token.
```

Le repo GitHub `aciderix/Eurexplo` est **public** (sinon le kernel ne peut pas
cloner). La branche de travail est `claude/setup-kaggle-pmu-pipeline-DIXU0`.

## Structure

| Fichier | Rôle |
|---|---|
| `kaggle_runner/notebook.ipynb` | Cellule unique, lance `kernel_entrypoint`. **Sans creds** (committée). |
| `/tmp/pmu-kernel/notebook.ipynb` | Copie avec creds `KGAT_` injectées, poussée sur Kaggle. Pas dans git. |
| `/tmp/pmu-kernel/kernel-metadata.json` | `dataset_sources: ["aciiderixx/pmu-v3-state"]`. Pas dans git. |
| `kaggle_runner/kernel_entrypoint.py` | Clone repo → install → restore state → lance orchestrator → snapshot → push dataset. Thread périodique (5 min) + SIGTERM handler pour survivre aux hard-kills. |
| `kaggle_runner/orchestrator.py` | Définit les 9 stages + boucle avec budget wall-clock. |
| `kaggle_runner/train_v3_resumable.py` | Wrapper train avec checkpoint par fold mensuel. |
| `kaggle_runner/stack_v3_resumable.py` | Wrapper stack avec checkpoint par fold + flags `--no-mlp`, `--max-estimators`. |
| `kaggle_runner/download_data.py` | Fetch 8 parquets v2 depuis Google Drive. |
| `kaggle_runner/heartbeat.py` | Écrit `state/heartbeat.json` toutes les 30s. |
| `kaggle_runner/gpu_env.py` | Détecte T4 et ajoute les kwargs GPU aux constructeurs LGB/XGB/Cat. |

## Stages du pipeline

L'orchestrateur skippe un stage si ses outputs sont présents (sentinel).
Ordre d'exécution :

1. `feat_blobs` → `pmu_feat_blobs_v3.parquet` (streaming parquet, cf commit `fd69eb7`)
2. `feat_elo_ctx` → `pmu_feat_elo_ctx_v3.parquet`
3. `feat_h2h` → `pmu_feat_h2h_v3.parquet`
4. `feat_musique_seq` → `pmu_feat_musique_seq_v3.parquet`
5. `build_all` → `pmu_features_v3.parquet` (matrice complète, 664 MB)
6. `optuna` → `pmu_best_params_v3.json` (80 trials, subsample 0.3)
7. `train` → `pmu_model_v3.lgb` + `pmu_oof_v3.parquet` (walk-forward mensuel 2018→2026, checkpoints par mois)
8. `stack` → `pmu_stack_v3.pkl` + `pmu_oof_stack_v3.parquet` (4 base learners + meta LogReg)
9. `calibrate` → `pmu_calibrator_v3.pkl`
10. `conformal` → `pmu_conformal_v3.json`
11. `backtest` → `pmu_backtest_v3.csv` (1000 bootstrap)
12. `bandit` → `pmu_bandit_v3.pkl` (`required=False` — peut échouer sans arrêter le pipeline)

## Env vars (définies dans `notebook.ipynb` cellule 1)

| Var | Défaut | Commentaire |
|---|---|---|
| `PMU_BRANCH` | `claude/setup-kaggle-pmu-pipeline-DIXU0` | Branche à cloner |
| `PMU_REPO_URL` | `https://github.com/aciderix/Eurexplo.git` | Repo public |
| `PMU_GPU` | `1` | Active les kwargs GPU pour LGB/XGB/Cat si T4 dispo |
| `PMU_BUDGET_MIN` | `480` | Budget 8 h ; l'orchestrator SIGTERM ses subprocess à la deadline |
| `PMU_OPTUNA_TRIALS` | `80` | Nombre de trials TPE |
| `PMU_OPTUNA_SAMPLE_FRAC` | `0.3` | Sous-échantillon stratifié par mois pour les trials (les best params sont appliqués sur full data au stage train) |
| `PMU_STATE_DATASET` | `aciiderixx/pmu-v3-state` | Dataset où snapshot + push à la fin |
| `PMU_STACK_FROM_YM` | `2023-01` | Stack OOF seulement sur 39 mois (au lieu de 75 à partir de 2020-01) |
| `PMU_STACK_NO_MLP` | `1` | Skip MLPClassifier (lent, faible apport) |
| `PMU_STACK_MAX_EST` | `600` | Cap `n_estimators` pour LGB binary/ranker + `iterations` pour XGB/CatBoost. Sans cap, Optuna donne 2632 → fold de 57 min. |

## Commandes essentielles

```bash
# Status du kernel en cours
KAGGLE_API_TOKEN=KGAT_... kaggle kernels status aciiderixx/pmu-v3-pipeline

# Fetch des outputs (ne marche qu'une fois le kernel TERMINÉ, pas running)
mkdir -p /tmp/vN-log
KAGGLE_API_TOKEN=KGAT_... kaggle kernels output aciiderixx/pmu-v3-pipeline -p /tmp/vN-log

# Inspecter l'état du pipeline après un run
cat /tmp/vN-log/outputs/state/stages.json      # ok / cached / timed_out / error_N
cat /tmp/vN-log/outputs/state/heartbeat.json   # stage en cours + dernière vie

# Lister les checkpoints par fold
ls /tmp/vN-log/outputs/state/train_v3_ckpt/ | wc -l
ls /tmp/vN-log/outputs/state/stack_v3_ckpt/ | wc -l

# Lister les fichiers du state dataset (source de vérité inter-session)
KAGGLE_API_TOKEN=KGAT_... kaggle datasets files aciiderixx/pmu-v3-state --page-size 500
```

## Pousser une nouvelle version (vN+1)

1. Modifier le code, commit, `git push -u origin claude/setup-kaggle-pmu-pipeline-DIXU0`.
2. **Ne pas modifier** `kaggle_runner/notebook.ipynb` pour les creds — elles sont
   injectées uniquement dans la copie `/tmp/`.
3. Si la cellule 1 du notebook a changé (nouveau env var), resync + injection :

```bash
python3 <<'PY'
import json, pathlib
tmp = pathlib.Path("/tmp/pmu-kernel/notebook.ipynb")
nb = json.loads(tmp.read_text())
repo_nb = json.loads(pathlib.Path("kaggle_runner/notebook.ipynb").read_text())
repo_src = "".join([c["source"] if isinstance(c["source"], str) else "".join(c["source"])
                    for c in repo_nb["cells"] if c["cell_type"] == "code"][:1])
creds = """import os, json, pathlib
os.environ['KAGGLE_USERNAME'] = 'aciiderixx'
os.environ['KAGGLE_KEY']      = 'KGAT_55fb107012d69bbd3762aa70a7c35f5e'
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_55fb107012d69bbd3762aa70a7c35f5e'
_kd = pathlib.Path.home() / '.kaggle'
_kd.mkdir(exist_ok=True)
(_kd / 'kaggle.json').write_text(json.dumps({'username': os.environ['KAGGLE_USERNAME'],
                                             'key':      os.environ['KAGGLE_KEY']}))
(_kd / 'kaggle.json').chmod(0o600)

"""
code_cell = [c for c in nb["cells"] if c["cell_type"] == "code"][0]
code_cell["source"] = creds + repo_src
tmp.write_text(json.dumps(nb, indent=1))
PY

# Push (retry sur 503)
cd /tmp/pmu-kernel
until KAGGLE_API_TOKEN=KGAT_... kaggle kernels push 2>&1 | grep -q "successfully pushed"; do sleep 20; done
```

## Pièges rencontrés

- **`kaggle` CLI 401** si on utilise `kaggle.json` avec le token KGAT_ —
  **toujours** passer par `KAGGLE_API_TOKEN=KGAT_...` en env var.
- **503 Service Unavailable** sur `kaggle kernels push` / `datasets version` —
  intermittent, faire une boucle `until ... success`.
- **`kaggle kernels output` renvoie vide** quand le kernel est `RUNNING`.
  Les outputs sont finalisés seulement en fin de run.
- **OOM sur `pmu_course_raw_v2.parquet` (1.66 GB)** si on fait
  `pq.read_table(...).to_pandas()` → déjà fixé par streaming
  `iter_batches(batch_size=2000, columns=[...])` (commit `fd69eb7`).
- **Papermill affiche `CalledProcessError` sur exit rc=2** — c'est le signal
  budget_timeout propre de l'orchestrateur, PAS un crash. Vérifier
  `state/stages.json` pour la liste réelle des stages finis.
- **Zip de Kaggle à la restauration** : `kaggle datasets version --dir-mode zip`
  zippe le dossier mais ne l'extrait pas au mount. `restore_from_input` de
  `kernel_entrypoint.py` fait le unzip.
- **Stack lent sans les flags** : sans `PMU_STACK_MAX_EST=600` et
  `PMU_STACK_NO_MLP=1`, un fold de 1.5M lignes prend ~57 min au lieu de ~18 min.

## Historique des kernel versions

| vN | Résultat | Ce qui a été fait |
|---|---|---|
| v1-v9 | setup / debug | credentials, branche, repo public, features |
| v10 | OOM sur feat_blobs | a servi à identifier le bug parquet → fd69eb7 |
| v11 | `budget_timeout` sur train | optuna complet (0.7917), train à 2024-02/26 mois restants |
| v12 | `budget_timeout` sur stack | train fini, stack à 2020-05 (5 folds /75) |
| v13 | `budget_timeout` sur stack | stack à 2023-09 (9 folds de v13 + 5 anciens v12) |
| v14 | `budget_timeout` sur backtest | stack+calibrate+conformal OK ; backtest démarré mais pas fini |
| v15 | (en cours) | backtest + bandit |

## Commits clés (branche `claude/setup-kaggle-pmu-pipeline-DIXU0`)

```
c2f25f7 stack: cap base-learner n_estimators via PMU_STACK_MAX_EST (default 600)
2c454f7 stack: env-var knobs for from-ym + no-mlp
44f72de snapshot_outputs: also persist v2 parquets + handle zipped state
d60b273 kernel: periodic snapshot + SIGTERM handler to survive hard-kill
0fc077d Speed up Optuna without quality hit + enable state-dataset push
3bf9f5d orch: download all 5 feat parquets, not just the 2 required
fd69eb7 feat_blobs: stream pmu_course_raw_v2 parquet to fit in Kaggle RAM
b95fde5 Point kernel notebook at the kaggle_runner branch so clone finds it
893eecb Add kaggle_runner: resumable GPU orchestration for v3 pipeline
```

## Si v15 a fini avec succès

Tous les artefacts sont dans le state dataset. Pour les récupérer localement :

```bash
KAGGLE_API_TOKEN=KGAT_... kaggle datasets download aciiderixx/pmu-v3-state -p /tmp/final
unzip -d /tmp/final /tmp/final/pmu-v3-state.zip
ls /tmp/final/outputs/pmu_*.lgb /tmp/final/outputs/pmu_*.pkl /tmp/final/outputs/pmu_*.json
```

Les artefacts prêts pour prédiction :
- `pmu_model_v3.lgb` + `pmu_model_v3.rk.lgb` (train binary + ranker)
- `pmu_stack_v3.pkl` (stack)
- `pmu_calibrator_v3.pkl` (calibration isotonique/platt)
- `pmu_conformal_v3.json` (intervalles de confiance 90 %)
- `pmu_backtest_v3.csv` (ROI historique bootstrap)
- `pmu_bandit_v3.pkl` (RL bandit, si v15 a pu finir ce stage)

Le script `pmu_predict_today_v3.py` est là pour l'inférence quotidienne.

## Si v15 a encore timeout

Probable scénario : backtest + bandit ne tiennent pas dans le budget.
Checklist :
1. Lire `state/stages.json` et `state/logs/backtest.log`.
2. Si backtest = ok mais bandit = timed_out → relancer v16 (bandit seul,
   rapide normalement).
3. Si backtest = timed_out → regarder si `pmu_backtest_v3.py` supporte un
   checkpoint / sample plus petit (actuellement `--bootstrap 1000` hardcodé
   dans orchestrator.py ligne 118).

## Prochaines étapes suggérées

- Si pipeline complet : valider ROI du backtest, déployer `predict_today_v3`.
- Si encore bloqué : réduire `--bootstrap 500` dans l'orchestrator pour
  backtest ; ajouter `required=False` sur backtest comme bandit si l'on ne veut
  pas que ça bloque bandit.
- Il y aurait moyen d'ajouter un vrai checkpoint dans `pmu_backtest_v3.py`
  (stocker les bootstrap samples déjà calculés) mais honnêtement 1000
  bootstrap sur 45 MB d'OOF devrait tenir en 30 min max.
