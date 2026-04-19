# PMU v3 pipeline on Kaggle

This directory wraps the v3 pipeline so it can run on Kaggle GPU notebooks:

- **Resumable**: every stage is idempotent, and the two long stages (`train`,
  `stack`) checkpoint per-fold so nothing is recomputed after a session ends.
- **GPU-ready**: XGBoost and CatBoost automatically use GPU on Kaggle (LGBM
  stays on CPU unless you set `PMU_GPU_LGB=1` and install a GPU wheel).
- **Monitorable**: a heartbeat file + per-stage logs are written in
  `/kaggle/working/state/`; a local `monitor.py` polls the kernel via the
  Kaggle API and surfaces status + tail of the logs.

Nothing in the existing `pmu_*_v3.py` scripts was modified. Everything lives
under `kaggle_runner/` and is imported as a regular python package.

## Files

```
kaggle_runner/
├── __init__.py                   # package marker
├── orchestrator.py               # runs stages, respects budget, writes heartbeat
├── download_data.py              # fetches parquets from Drive (DATA_LINKS.md)
├── gpu_env.py                    # PMU_GPU / PMU_GPU_XGB / PMU_GPU_CAT flags
├── heartbeat.py                  # 30 s heartbeat writer + stage-log tee
├── train_v3_resumable.py         # walk-forward LGBM train with per-fold cache
├── stack_v3_resumable.py         # stack (LGBM + XGB + CatBoost + MLP) with cache
├── kernel_entrypoint.py          # runs inside the Kaggle kernel (clone + run)
├── notebook.ipynb                # single-cell notebook that calls the entrypoint
├── kernel-metadata.json          # `kaggle kernels push` metadata
├── push_kernel.sh                # helper to push the kernel from your laptop
├── monitor.py                    # local watchdog: kaggle CLI + heartbeat tail
└── README.md                     # this file
```

## Pipeline stages

Ordered as in `run_all_v3.sh`; orchestrator skips any stage whose outputs
already exist on disk.

| # | Stage name         | Outputs                                           | Resume mechanism                 |
|---|--------------------|---------------------------------------------------|----------------------------------|
| 1 | feat_blobs         | `pmu_feat_blobs_v3.parquet`                       | output file sentinel             |
| 2 | feat_elo_ctx       | `pmu_feat_elo_ctx_v3.parquet`                     | output file sentinel             |
| 3 | feat_h2h           | `pmu_feat_h2h_v3.parquet`                         | output file sentinel             |
| 4 | feat_musique_seq   | `pmu_feat_musique_seq_v3.parquet`                 | output file sentinel             |
| 5 | build_all          | `pmu_features_v3.parquet`                         | output file sentinel             |
| 6 | optuna             | `pmu_best_params_v3.json`                         | SQLite study (`--resume`)        |
| 7 | train              | `pmu_model_v3.lgb`, `pmu_oof_v3.parquet`          | per-fold parquet cache           |
| 8 | stack              | `pmu_stack_v3.pkl`, `pmu_oof_stack_v3.parquet`    | per-month parquet cache          |
| 9 | calibrate          | `pmu_calibrator_v3.pkl`                           | output file sentinel             |
|10 | conformal          | `pmu_conformal_v3.json`                           | output file sentinel             |
|11 | backtest           | `pmu_backtest_v3.csv`                             | output file sentinel             |
|12 | bandit (optional)  | `pmu_bandit_v3.pkl`                               | output file sentinel             |

## Workflow

### 0. ⚠ Rotate your API token

The token you pasted in chat is burned — anyone with the transcript can use
it. Go to https://www.kaggle.com/settings → *API Tokens* → delete it and
generate a new one. Paste it into `~/.kaggle/kaggle.json`, never into code.

### 1. Set up the Kaggle CLI on your laptop

```bash
pip install "kaggle>=1.8" gdown
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions list    # smoke test
```

### 2. Edit `kernel-metadata.json`

Replace `REPLACE_WITH_YOUR_USERNAME` with your Kaggle username. For example:

```json
{ "id": "jdoe/pmu-v3-pipeline", "title": "PMU v3 pipeline", ... }
```

### 3. Push the kernel

```bash
mkdir -p /tmp/pmu-kernel
cp kaggle_runner/kernel-metadata.json /tmp/pmu-kernel/
cp kaggle_runner/notebook.ipynb       /tmp/pmu-kernel/
# edit the metadata id first
bash kaggle_runner/push_kernel.sh /tmp/pmu-kernel
```

The kernel clones the branch, installs requirements, downloads the base
parquets from Drive, then runs the orchestrator.

### 4. Let it run (first session)

A Kaggle GPU notebook runs up to ~9 h. `PMU_BUDGET_MIN=480` makes the
orchestrator stop cleanly 8 minutes before that, flush all checkpoints, and
exit. All artifacts land under `/kaggle/working/outputs/` and stay attached to
the notebook version as downloadable files.

If Optuna takes too long, set `PMU_SKIP_OPTUNA=1` in the notebook cell —
defaults will be used for training.

### 5. Resume in a second session

Two options:

**(a) Attach the previous notebook's outputs as input** (easiest): on the
Kaggle UI, *Add Data → Notebook Output → select your previous run*. Re-run
the notebook. The entrypoint copies everything under `/kaggle/input/*/state/`
back to `/kaggle/working/state/` and the orchestrator skips every stage whose
sentinel exists.

**(b) Push everything as a private dataset**. In the notebook cell, set

```python
os.environ["PMU_STATE_DATASET"] = "jdoe/pmu-v3-state"
```

The entrypoint will `kaggle datasets create` (first run) or `kaggle datasets
version` (subsequent) at the end of each kernel session. On the next run,
attach that dataset as input and the restore step picks it up automatically.

### 6. Monitor from your laptop

```bash
python -m kaggle_runner.monitor --slug jdoe/pmu-v3-pipeline --interval 300
```

Every 5 minutes it prints:

- Kernel state (running/queued/complete/error)
- Current stage + last heartbeat timestamp
- Progress counters (fold K/N, last AUC…)
- Tail of the current stage log
- ⚠ warning if the heartbeat hasn't moved for 15 min (stuck kernel)

Add `--once` for a one-shot check (nice in cron/launchd).

## GPU controls

```bash
# master switch (auto-on if nvidia-smi is present)
export PMU_GPU=1

# LightGBM GPU — disabled by default because the Kaggle wheel is CPU-only.
# Flip on only if you rebuild the wheel; otherwise leave at 0.
export PMU_GPU_LGB=0

# XGBoost — enabled by default; uses tree_method=hist + device=cuda:0
export PMU_GPU_XGB=1

# CatBoost — enabled by default; uses task_type=GPU
export PMU_GPU_CAT=1

# Which GPU (T4 x2 gives you 0,1; default = 0)
export PMU_GPU_DEVICES=0
```

## Running locally (same code path)

```bash
pip install -r requirements_v3.txt gdown
python -m kaggle_runner.orchestrator \
    --repo-dir . --state-dir ./state --budget-minutes 600
```

## Budget / safety knobs

| Env var              | Default | Effect                                                 |
|----------------------|---------|--------------------------------------------------------|
| `PMU_BUDGET_MIN`     | 480     | Soft wall-clock minutes before the orchestrator stops |
| `PMU_OPTUNA_TRIALS`  | 200     | Optuna trials (SQLite-resumable)                      |
| `PMU_SKIP_OPTUNA`    | 0       | Skip the 200-trial search, use defaults               |
| `PMU_SKIP_STACK`     | 0       | Skip the 5-model stack                                |
| `PMU_SKIP_BANDIT`    | 0       | Skip the Thompson-sampling bandit                     |

## What's not automated yet

- `pmu_feat_embeddings.py` (Word2Vec) is intentionally left out of the default
  stage list — it's slow and optional. Add it back to `default_stages()` in
  `orchestrator.py` if you want it.
- Hashes aren't checked on the Drive downloads; `download_data.py` only
  validates file size to detect truncation. If you want stronger integrity
  checks, add SHA sums to `DATA_LINKS.md` and compare in `_download_one`.
