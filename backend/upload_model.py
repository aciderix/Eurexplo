"""Upload a trained LightGBM model + metadata to Supabase.

Usage:
  python backend/upload_model.py \
      --model model_lgbm.pkl \
      --meta model_lgbm_meta.json \
      --version 2026-04-14 \
      --activate
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from supabase_client import get_client


def upload(model_path: str, meta_path: str, version: str, activate: bool) -> None:
    client = get_client()

    model_bytes = pathlib.Path(model_path).read_bytes()
    meta = json.loads(pathlib.Path(meta_path).read_text())

    storage_path = f"model_lgbm_{version}.pkl"
    print(f"Uploading {len(model_bytes)/1e6:.2f} MB to models/{storage_path} ...")
    client.storage.from_("models").upload(
        storage_path,
        model_bytes,
        {"content-type": "application/octet-stream", "upsert": "true"},
    )

    row = {
        "version": version,
        "train_start": meta.get("train_start"),
        "train_end": meta.get("train_end"),
        "auc_cv": meta.get("auc_cv"),
        "n_train_rows": meta.get("n_train_rows"),
        "hyperparams": meta.get("hyperparams"),
        "feature_importance": meta.get("feature_importance"),
        "storage_path": f"models/{storage_path}",
        "is_active": activate,
    }

    if activate:
        # Only one active model at a time.
        client.table("model_versions").update({"is_active": False}).neq("version", version).execute()

    print(f"Inserting model_versions row (version={version}, active={activate}) ...")
    client.table("model_versions").upsert(row, on_conflict="version").execute()

    print("Done.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model_lgbm.pkl")
    p.add_argument("--meta", required=True, help="Path to model_lgbm_meta.json")
    p.add_argument("--version", required=True, help='Version string, e.g. "2026-04-14"')
    p.add_argument("--activate", action="store_true", help="Mark as active model")
    args = p.parse_args()
    upload(args.model, args.meta, args.version, args.activate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
