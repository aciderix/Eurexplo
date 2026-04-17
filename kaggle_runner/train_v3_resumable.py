#!/usr/bin/env python3
"""Resumable wrapper around pmu_train_v3.

Re-uses the fold training function from the vendored v3 script, but persists
OOF predictions **per fold** under a checkpoint dir. If the process is killed
and restarted, completed folds are skipped (the parquet chunks are loaded and
concatenated at the end).

Usage:
  python -m kaggle_runner.train_v3_resumable \
      --features pmu_features_v3.parquet \
      --params pmu_best_params_v3.json \
      --checkpoint-dir state/train_v3_ckpt \
      --walk-forward monthly \
      --from-year 2018
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pmu_train_v3 as v3  # noqa: E402

from kaggle_runner import heartbeat as hb  # noqa: E402
from kaggle_runner import gpu_env  # noqa: E402


def _fold_key(name: str) -> str:
    return name.replace("=", "_").replace(" ", "")


def run(args) -> int:
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load params
    p_bin = gpu_env.apply_lgb(v3.DEFAULT_LGBM)
    p_rk  = gpu_env.apply_lgb(v3.DEFAULT_RANKER)
    if args.params and Path(args.params).exists():
        cfg = json.loads(Path(args.params).read_text())
        p_bin.update(gpu_env.apply_lgb(cfg.get("binary", {})))
        p_rk.update(gpu_env.apply_lgb(cfg.get("ranker", {})))
        print(f"[train] loaded params from {args.params}")
    print(f"[train] gpu_env: {gpu_env.summary()}")

    t0 = time.time()
    print(f"[train] loading {args.features}…")
    df = pq.read_table(args.features).to_pandas()
    df = df[df["won"].notna()].copy()
    df["won"] = df["won"].astype(int)
    feat_cols = v3.select_features(df)
    print(f"[train]   rows={len(df):,} feats={len(feat_cols)} in {time.time()-t0:.1f}s")

    heart: hb.Heartbeat | None = None
    if args.heartbeat:
        heart = hb.Heartbeat(Path(args.heartbeat))
        heart.start()
        heart.set_stage("train_v3", features=args.features)

    fold_iter = list(v3.iter_folds(df, args.walk_forward, args.from_year, args.fast))
    print(f"[train] {len(fold_iter)} folds total")

    oof_parts: list[pd.DataFrame] = []
    done = 0
    for tr, va, name in fold_iter:
        key = _fold_key(name)
        ckpt = ckpt_dir / f"oof_{key}.parquet"
        if ckpt.exists() and not args.force:
            try:
                oof_parts.append(pd.read_parquet(ckpt))
                done += 1
                print(f"[train]   SKIP {name} (cached at {ckpt.name})")
                if heart:
                    heart.progress(fold=name, status="cached",
                                   done=done, total=len(fold_iter))
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[train]   re-running {name} (cache unreadable: {exc})")
                ckpt.unlink()

        tr_df = df[tr].sort_values(["file_date", "race_id", "num_pmu"])
        va_df = df[va].sort_values(["file_date", "race_id", "num_pmu"])
        if tr_df.empty or va_df.empty:
            continue

        X_tr = v3.to_float32(tr_df, feat_cols).values
        y_tr = tr_df["won"].astype(int).values
        w_tr = np.log1p(tr_df["drd_rapport"].fillna(0).clip(lower=0)).astype(np.float32).values
        w_tr = np.where(w_tr <= 0, 1.0, w_tr)
        g_tr = tr_df.groupby("race_id", sort=False).size().values

        X_va = v3.to_float32(va_df, feat_cols).values
        y_va = va_df["won"].astype(int).values
        g_va = va_df.groupby("race_id", sort=False).size().values

        t1 = time.time()
        out = v3.train_fold(X_tr, y_tr, w_tr, g_tr, X_va, y_va, g_va, p_bin, p_rk)

        va_out = va_df.copy()
        va_out["p_bin"] = out["p_bin"]
        va_out["s_rk"] = out["s_rk"]
        va_out["s_rk_norm"] = va_out.groupby("race_id")["s_rk"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
        va_out["meta"] = (
            args.alpha_meta * va_out["s_rk_norm"] + (1 - args.alpha_meta) * va_out["p_bin"]
        )
        fold_out = va_out[["race_id", "num_pmu", "file_date", "won", "drd_rapport",
                           "p_bin", "s_rk", "s_rk_norm", "meta"]]
        fold_out.to_parquet(ckpt, compression="zstd", index=False)

        try:
            auc = roc_auc_score(y_va, out["p_bin"])
        except ValueError:
            auc = float("nan")
        oof_parts.append(fold_out)
        done += 1
        elapsed = time.time() - t1
        print(f"[train]   DONE {name}  AUC={auc:.4f}  {elapsed:.0f}s  → {ckpt.name}")
        if heart:
            heart.progress(fold=name, status="done", done=done,
                           total=len(fold_iter), last_auc=round(float(auc), 4),
                           last_fold_seconds=round(elapsed, 1))

    if oof_parts:
        oof = pd.concat(oof_parts, ignore_index=True)
        out_path = Path(args.out_oof)
        oof.to_parquet(out_path, compression="zstd", index=False)
        print(f"[train] wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    else:
        print("[train] no folds processed.")

    # ── Final full-train model ──
    if not args.no_final:
        print("[train] final full-train…")
        full = df.sort_values(["file_date", "race_id", "num_pmu"]).copy()
        X = v3.to_float32(full, feat_cols).values
        y = full["won"].astype(int).values
        w = np.log1p(full["drd_rapport"].fillna(0).clip(lower=0)).astype(np.float32).values
        w = np.where(w <= 0, 1.0, w)
        g = full.groupby("race_id", sort=False).size().values

        import lightgbm as lgb
        m_bin = lgb.LGBMClassifier(**p_bin)
        m_bin.fit(X, y, sample_weight=w, callbacks=[lgb.log_evaluation(-1)])
        m_rk = lgb.LGBMRanker(**p_rk)
        m_rk.fit(X, y, group=g, callbacks=[lgb.log_evaluation(-1)])

        out_model = Path(args.out_model)
        m_bin.booster_.save_model(str(out_model))
        m_rk.booster_.save_model(str(out_model.with_suffix(".rk.lgb")))
        Path(args.out_feats).write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")
        print(f"[train] wrote {out_model} + .rk.lgb + {args.out_feats}")

    if heart:
        heart.stop(ok=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",      default="pmu_features_v3.parquet")
    ap.add_argument("--params",        default="pmu_best_params_v3.json")
    ap.add_argument("--checkpoint-dir", default="state/train_v3_ckpt")
    ap.add_argument("--walk-forward",  choices=["monthly", "yearly"], default="monthly")
    ap.add_argument("--from-year",     type=int, default=2018)
    ap.add_argument("--alpha-meta",    type=float, default=0.5)
    ap.add_argument("--fast",          action="store_true")
    ap.add_argument("--force",         action="store_true",
                    help="re-run all folds, ignore cache")
    ap.add_argument("--no-final",      action="store_true")
    ap.add_argument("--out-oof",       default="pmu_oof_v3.parquet")
    ap.add_argument("--out-model",     default="pmu_model_v3.lgb")
    ap.add_argument("--out-feats",     default="pmu_model_v3.features.json")
    ap.add_argument("--heartbeat",     default=None,
                    help="path to heartbeat.json to update during training")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
