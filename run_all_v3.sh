#!/usr/bin/env bash
#
# run_all_v3.sh — orchestration tout-en-un du pipeline PMU v3.
#
# - Skip les étapes dont la sortie existe déjà (idempotent).
# - Logue chaque étape dans logs_v3/.
# - Continue en cas d'erreur sur une feature (on fera un rapport à la fin).
#
# Usage :
#   chmod +x run_all_v3.sh
#   ./run_all_v3.sh
#   ./run_all_v3.sh --force            # force toutes les étapes
#   ./run_all_v3.sh --skip-optuna      # skip la recherche d'hyperparams
#   ./run_all_v3.sh --skip-stack       # skip le stacking
#
# Prérequis :
#   - Python >=3.10 avec requirements_v3.txt installé
#   - pmu_dataset_v2.parquet + pmu_course_raw_v2.parquet présents
#     (voir DATA_LINKS.md pour les liens de téléchargement)

set -euo pipefail

PY=${PY:-python3}
FORCE=0
SKIP_OPTUNA=0
SKIP_STACK=0
SKIP_BANDIT=0

for arg in "$@"; do
    case "$arg" in
        --force)        FORCE=1 ;;
        --skip-optuna)  SKIP_OPTUNA=1 ;;
        --skip-stack)   SKIP_STACK=1 ;;
        --skip-bandit)  SKIP_BANDIT=1 ;;
        -h|--help)
            echo "Usage: $0 [--force] [--skip-optuna] [--skip-stack] [--skip-bandit]"
            exit 0
            ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

LOG_DIR="logs_v3"
mkdir -p "$LOG_DIR"

log() { echo -e "\n\033[1;36m[v3] $*\033[0m"; }

need() {
    local out="$1"; shift
    if [[ $FORCE -eq 0 && -f "$out" ]]; then
        log "SKIP  $out  (exists — use --force to rebuild)"
        return 1
    fi
    return 0
}

step() {
    local name="$1"; shift
    local cmd="$*"
    log "RUN   $name"
    local log_file="$LOG_DIR/${name}.log"
    if ! bash -c "$cmd" 2>&1 | tee "$log_file"; then
        log "ERR   $name — see $log_file"
        return 1
    fi
}

# ── 0 · Pré-flight ──
log "pré-flight"
for f in pmu_dataset_v2.parquet pmu_course_raw_v2.parquet; do
    if [[ ! -f "$f" ]]; then
        log "MANQUE $f — voir DATA_LINKS.md pour le télécharger"
        exit 1
    fi
done

# ── 1 · Feature engineering v3 (parallélisable) ──
log "Etape 1 · Feature engineering v3"
declare -A FEATS=(
    [blobs]="pmu_feat_blobs_v3.parquet"
    [elo_ctx]="pmu_feat_elo_ctx_v3.parquet"
    [h2h]="pmu_feat_h2h_v3.parquet"
    [emb]="pmu_feat_emb_v3.parquet"
    [mus_seq]="pmu_feat_musique_seq_v3.parquet"
)
declare -A SCRIPTS=(
    [blobs]="pmu_feat_blobs.py"
    [elo_ctx]="pmu_feat_elo_ctx.py"
    [h2h]="pmu_feat_h2h.py"
    [emb]="pmu_feat_embeddings.py"
    [mus_seq]="pmu_feat_musique_seq.py"
)

# Décommente pour run PARALLÈLE (demande ~16+ Go RAM) :
# for k in "${!FEATS[@]}"; do
#     if need "${FEATS[$k]}"; then
#         ( step "feat_$k" "$PY ${SCRIPTS[$k]}" ) &
#     fi
# done
# wait

# Séquentiel (plus safe) :
for k in blobs elo_ctx h2h mus_seq; do
    if need "${FEATS[$k]}"; then
        step "feat_$k" "$PY ${SCRIPTS[$k]}"
    fi
done
# Embeddings = optionnel (long) — décommente si tu veux :
# if need "${FEATS[emb]}"; then step "feat_emb" "$PY ${SCRIPTS[emb]}"; fi

# ── 2 · Join feature matrix ──
if need pmu_features_v3.parquet; then
    step build_all "$PY pmu_feat_build_all_v3.py"
fi

# ── 3 · Optuna (skipable) ──
if [[ $SKIP_OPTUNA -eq 0 ]]; then
    if need pmu_best_params_v3.json; then
        step optuna "$PY pmu_optuna_v3.py --n-trials 200 --study pmu_v3_main"
    fi
else
    log "SKIP  Optuna"
fi

# ── 4 · Walk-forward training ──
if need pmu_model_v3.lgb; then
    if [[ -f pmu_best_params_v3.json ]]; then
        step train "$PY pmu_train_v3.py --params pmu_best_params_v3.json --walk-forward monthly"
    else
        step train "$PY pmu_train_v3.py --walk-forward monthly"
    fi
fi

# ── 5 · Stacking ──
if [[ $SKIP_STACK -eq 0 ]]; then
    if need pmu_stack_v3.pkl; then
        step stack "$PY pmu_stack_v3.py"
    fi
else
    log "SKIP  stack"
fi

# ── 6 · Calibration ──
if [[ -f pmu_oof_stack_v3.parquet ]]; then
    if need pmu_calibrator_v3.pkl; then
        step calibrate "$PY pmu_calibrate_v3.py --oof pmu_oof_stack_v3.parquet"
    fi
fi

# ── 7 · Conformal prediction ──
if [[ -f pmu_oof_stack_v3.parquet ]]; then
    if need pmu_conformal_v3.json; then
        step conformal "$PY pmu_conformal_v3.py --oof pmu_oof_stack_v3.parquet --alpha 0.1"
    fi
fi

# ── 8 · Backtest ──
if [[ -f pmu_oof_stack_v3.parquet ]]; then
    if need pmu_backtest_v3.csv; then
        step backtest "$PY pmu_backtest_v3.py --bootstrap 1000"
    fi
fi

# ── 9 · Bandit (skipable) ──
if [[ $SKIP_BANDIT -eq 0 ]]; then
    if need pmu_bandit_v3.pkl; then
        step bandit "$PY pmu_bandit_v3.py --train-end 2024-12 --eval-start 2025-01"
    fi
else
    log "SKIP  bandit"
fi

log "✔ Pipeline v3 terminé"
log "Artefacts :"
ls -la pmu_*v3* 2>/dev/null || true
