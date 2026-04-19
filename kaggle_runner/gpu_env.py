"""GPU/env helpers for the v3 pipeline on Kaggle.

Kaggle pre-installed LightGBM is CPU-only. Swapping to GPU requires a custom
wheel rebuild at kernel start; we don't force it — CPU LGB is already fast
enough on Kaggle's 4-core VMs, and the walk-forward is bottlenecked by IO.

XGBoost and CatBoost wheels on Kaggle are GPU-aware out of the box, so the
*stack* stage benefits from GPU without any extra install.

Flags (via env):
  PMU_GPU=1            master switch, default 1 when CUDA is detected.
  PMU_GPU_LGB=0|1      try lightgbm device="gpu" (requires GPU wheel). default 0
  PMU_GPU_XGB=0|1      use tree_method="hist", device="cuda". default = PMU_GPU
  PMU_GPU_CAT=0|1      use task_type="GPU". default = PMU_GPU
  PMU_GPU_DEVICES      "0" or "0,1" (T4 x2 on Kaggle). default "0"
"""
from __future__ import annotations

import os
import shutil
import subprocess


def _cuda_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, timeout=5
        )
        return True
    except Exception:  # noqa: BLE001
        return False


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def gpu_master_on() -> bool:
    return _flag("PMU_GPU", "1" if _cuda_available() else "0")


def use_gpu_lgb() -> bool:
    return _flag("PMU_GPU_LGB", "0") and gpu_master_on()


def use_gpu_xgb() -> bool:
    return _flag("PMU_GPU_XGB", "1" if gpu_master_on() else "0")


def use_gpu_cat() -> bool:
    return _flag("PMU_GPU_CAT", "1" if gpu_master_on() else "0")


def apply_lgb(params: dict) -> dict:
    """Return a copy of an LGBM params dict with GPU flags if enabled."""
    p = dict(params)
    if use_gpu_lgb():
        p.setdefault("device", "gpu")
        p.setdefault("gpu_platform_id", 0)
        p.setdefault("gpu_device_id", int(os.environ.get("PMU_GPU_DEVICES", "0").split(",")[0]))
        p["verbose"] = -1
    return p


def apply_xgb(params: dict) -> dict:
    p = dict(params)
    if use_gpu_xgb():
        p["tree_method"] = "hist"
        p["device"] = f"cuda:{os.environ.get('PMU_GPU_DEVICES', '0').split(',')[0]}"
    return p


def catboost_kwargs() -> dict:
    if use_gpu_cat():
        return {"task_type": "GPU",
                "devices": os.environ.get("PMU_GPU_DEVICES", "0")}
    return {}


def summary() -> str:
    return (
        f"gpu_master={gpu_master_on()} "
        f"lgb={use_gpu_lgb()} xgb={use_gpu_xgb()} cat={use_gpu_cat()} "
        f"devices={os.environ.get('PMU_GPU_DEVICES', '0')}"
    )
