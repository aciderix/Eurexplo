#!/usr/bin/env python3
"""Resumable wrapper around pmu_stack_v3.

Per-month OOF chunks are cached as parquet files; on restart, completed months
are loaded from disk instead of retrained. Final meta learner is refit from the
concatenated OOF so we get identical outputs to the one-shot script.

GPU flags for XGBoost / CatBoost come from kaggle.gpu_env.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pmu_stack_v3 as v3  # noqa: E402

from kaggle_runner import heartbeat as hb  # noqa: E402
from kaggle_runner import gpu_env  # noqa: E402

warnings.filterwarnings("ignore")


def _patch_gpu_params(p_json: dict) -> dict:
    out = {}
    for k, v in p_json.items():
        out[k] = gpu_env.apply_lgb(v) if isinstance(v, dict) else v
    return out


def _patched_train_fold(tr_df, va_df, feat_cols, params, use_mlp, xgb_gpu, cb_gpu):
    """Call v3.train_stack_fold after swapping the XGB / CatBoost branches
    to add GPU kwargs. We monkey-patch locally to avoid forking the whole fn."""
    import lightgbm as lgb  # noqa: F401
    from sklearn.neural_network import MLPClassifier  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401

    # Push GPU kwargs for xgb / catboost via constructor monkey-patching.
    orig_xgb = None
    orig_cb = None
    if v3.HAS_XGB and xgb_gpu:
        import xgboost as xgb_mod

        class _GpuXGB(xgb_mod.XGBClassifier):
            def __init__(self, **kw):
                kw = gpu_env.apply_xgb(kw)
                super().__init__(**kw)

        orig_xgb = xgb_mod.XGBClassifier
        xgb_mod.XGBClassifier = _GpuXGB
        v3.xgb = xgb_mod

    if v3.HAS_CAT and cb_gpu:
        import catboost as cb_mod

        class _GpuCat(cb_mod.CatBoostClassifier):
            def __init__(self, **kw):
                kw.update(gpu_env.catboost_kwargs())
                super().__init__(**kw)

        orig_cb = cb_mod.CatBoostClassifier
        cb_mod.CatBoostClassifier = _GpuCat
        v3.cb = cb_mod

    try:
        return v3.train_stack_fold(tr_df, va_df, feat_cols, params, use_mlp=use_mlp)
    finally:
        if orig_xgb is not None:
            import xgboost as xgb_mod
            xgb_mod.XGBClassifier = orig_xgb
            v3.xgb = xgb_mod
        if orig_cb is not None:
            import catboost as cb_mod
            cb_mod.CatBoostClassifier = orig_cb
            v3.cb = cb_mod


def run(args) -> int:
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if Path(args.params).exists():
        p_json = json.loads(Path(args.params).read_text())
    else:
        # Fallback defaults, same as v3.main()
        p_json = {
            "binary": {"objective": "binary", "metric": "auc", "n_estimators": 2000,
                       "learning_rate": 0.03, "num_leaves": 255, "min_child_samples": 40,
                       "feature_fraction": 0.8, "bagging_fraction": 0.85, "bagging_freq": 5,
                       "lambda_l1": 0.5, "lambda_l2": 0.5, "verbose": -1, "n_jobs": -1,
                       "random_state": 42},
            "ranker": {"objective": "lambdarank", "metric": "ndcg", "n_estimators": 2000,
                       "learning_rate": 0.03, "num_leaves": 255, "min_child_samples": 40,
                       "feature_fraction": 0.8, "bagging_fraction": 0.85, "bagging_freq": 5,
                       "lambda_l1": 0.5, "lambda_l2": 0.5, "verbose": -1, "n_jobs": -1,
                       "random_state": 42},
        }
    p_json = _patch_gpu_params(p_json)
    print(f"[stack] gpu_env: {gpu_env.summary()}")

    t0 = time.time()
    print(f"[stack] loading {args.features}…")
    df = pq.read_table(args.features).to_pandas()
    df = df[df["won"].notna()].copy()
    df["won"] = df["won"].astype(int)
    df["year_month"] = df["file_date"].str[:7]
    feat_cols = v3._select_features(df)
    print(f"[stack]   rows={len(df):,} feats={len(feat_cols)} in {time.time()-t0:.1f}s")

    heart: hb.Heartbeat | None = None
    if args.heartbeat:
        heart = hb.Heartbeat(Path(args.heartbeat))
        heart.start()
        heart.set_stage("stack_v3", features=args.features)

    months = list(v3._iter_months(df, args.from_ym, args.fast))
    print(f"[stack] {len(months)} monthly folds")

    oof_parts: list[pd.DataFrame] = []
    done = 0
    for tr, va, ym in months:
        ckpt = ckpt_dir / f"oof_{ym}.parquet"
        if ckpt.exists() and not args.force:
            try:
                oof_parts.append(pd.read_parquet(ckpt))
                done += 1
                print(f"[stack]   SKIP {ym} (cached)")
                if heart:
                    heart.progress(month=ym, status="cached",
                                   done=done, total=len(months))
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[stack]   re-running {ym} (cache unreadable: {exc})")
                ckpt.unlink()

        tr_df = df[tr].sort_values(["file_date", "race_id", "num_pmu"])
        va_df = df[va].sort_values(["file_date", "race_id", "num_pmu"])
        t1 = time.time()
        fold = _patched_train_fold(
            tr_df, va_df, feat_cols, p_json,
            use_mlp=not args.no_mlp,
            xgb_gpu=gpu_env.use_gpu_xgb(),
            cb_gpu=gpu_env.use_gpu_cat(),
        )
        meta_df = v3.build_meta_features(va_df, fold["base_preds"])
        meta_df.to_parquet(ckpt, compression="zstd", index=False)
        oof_parts.append(meta_df)
        done += 1
        elapsed = time.time() - t1
        print(f"[stack]   DONE {ym}  n_tr={tr.sum():,}  n_va={va.sum():,}  {elapsed:.0f}s")
        if heart:
            heart.progress(month=ym, status="done", done=done,
                           total=len(months), last_fold_seconds=round(elapsed, 1))

    if not oof_parts:
        print("[stack] no folds processed.")
        if heart:
            heart.stop(ok=False, error="no folds processed")
        return 1

    oof = pd.concat(oof_parts, ignore_index=True)
    oof["log_drd"] = np.log1p(oof["drd_rapport"].clip(lower=0))
    meta_cols = [c for c in oof.columns if c.startswith(("p_", "rank_", "pct_"))] + ["log_drd"]
    meta_X = oof[meta_cols].fillna(0).values
    meta_y = oof["won"].astype(int).values

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    meta_base = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500,
                                    class_weight="balanced", n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof["p_stack"] = cross_val_predict(
        meta_base, meta_X, meta_y, cv=kf, method="predict_proba", n_jobs=-1
    )[:, 1]
    meta = meta_base.fit(meta_X, meta_y)
    auc_stack = roc_auc_score(meta_y, oof["p_stack"])
    print(f"[stack] stack OOF AUC = {auc_stack:.4f}")

    oof.to_parquet(args.out_oof, compression="zstd", index=False)
    with open(args.out_pkl, "wb") as fh:
        pickle.dump({"meta": meta, "meta_cols": meta_cols,
                     "feat_cols": feat_cols, "params": p_json}, fh)
    print(f"[stack] wrote {args.out_oof} + {args.out_pkl}")

    if heart:
        heart.stop(ok=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",       default="pmu_features_v3.parquet")
    ap.add_argument("--params",         default="pmu_best_params_v3.json")
    ap.add_argument("--checkpoint-dir", default="state/stack_v3_ckpt")
    ap.add_argument("--from-ym",        default="2020-01")
    ap.add_argument("--fast",           action="store_true")
    ap.add_argument("--no-mlp",         action="store_true")
    ap.add_argument("--force",          action="store_true")
    ap.add_argument("--out-oof",        default="pmu_oof_stack_v3.parquet")
    ap.add_argument("--out-pkl",        default="pmu_stack_v3.pkl")
    ap.add_argument("--heartbeat",      default=None)
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
