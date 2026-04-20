#!/usr/bin/env python3
"""Kaggle-kernel entrypoint.

Copy/paste this into a Kaggle Notebook's first code cell, or push it with
`push_kernel.sh` (see this directory). It:

  1. Clones the v3 branch of the Eurexplo repo into /kaggle/working/repo.
  2. Installs requirements_v3.txt.
  3. Restores previous checkpoints from an attached Kaggle dataset (if any).
  4. Runs kaggle_runner.orchestrator with a wall-clock budget.
  5. Snapshots /kaggle/working/state to /kaggle/working/ outputs so you can
     download / re-attach them on the next run.

Assumes kernel settings:
  - Internet: ON (needed to clone + gdown)
  - Accelerator: GPU T4 x2 (or P100)
  - Secrets (optional):
      KAGGLE_USERNAME, KAGGLE_KEY  — only needed if you want the kernel to
                                    push a new dataset version automatically.

Env knobs (set via notebook metadata or `os.environ[...] = ...` above):
  PMU_BRANCH=claude/review-pmu-scraper-nXJcB
  PMU_GPU=1
  PMU_BUDGET_MIN=480
  PMU_SKIP_OPTUNA=0|1
  PMU_SKIP_STACK=0|1
  PMU_OPTUNA_TRIALS=200
  PMU_STATE_DATASET=<user>/pmu-v3-state   (optional, triggers dataset push)
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

REPO_URL = os.environ.get(
    "PMU_REPO_URL",
    "https://github.com/aciderix/Eurexplo.git",
)
BRANCH = os.environ.get("PMU_BRANCH", "claude/review-pmu-scraper-nXJcB")

WORKING = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
REPO_DIR = WORKING / "repo"
STATE_DIR = WORKING / "state"
INPUTS = Path("/kaggle/input")


def _sh(cmd: list[str], **kw) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, **kw)


def clone_repo() -> None:
    if REPO_DIR.exists():
        print(f"[kernel] repo already present at {REPO_DIR}, fetching…")
        _sh(["git", "-C", str(REPO_DIR), "fetch", "--depth=1", "origin", BRANCH])
        _sh(["git", "-C", str(REPO_DIR), "checkout", "FETCH_HEAD"])
        return
    _sh(["git", "clone", "--depth=1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)])


def install_requirements() -> None:
    req = REPO_DIR / "requirements_v3.txt"
    _sh([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    _sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])
    _sh([sys.executable, "-m", "pip", "install", "-q", "gdown>=5.2"])


def _unzip_into(zip_path: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    print(f"[kernel] extracted {zip_path.name} → {target}")


def restore_from_input() -> None:
    """Copy checkpoints/artifacts from any attached input dataset into the
    repo's working directory. Attach a dataset to your kernel whose files are:
      - pmu_dataset_v2.parquet / pmu_course_raw_v2.parquet / pmu_feat_*.parquet
      - state.zip (or state/... if unzipped) with heartbeat, stages.json,
        train_v3_ckpt/*, stack_v3_ckpt/*, logs/*
      - any pmu_*.parquet / pmu_*.pkl / pmu_*.json / *.lgb artifacts that
        previous sessions produced.
    """
    if not INPUTS.exists():
        return
    for entry in INPUTS.iterdir():
        if not entry.is_dir():
            continue
        print(f"[kernel] scanning input dataset {entry.name}")
        # Kaggle CLI with --dir-mode zip uploads each subdir as a single zip;
        # Kaggle does NOT auto-extract on attach, so we do it here.
        for z in entry.glob("*.zip"):
            if z.stem == "state":
                _unzip_into(z, STATE_DIR)
            else:
                _unzip_into(z, REPO_DIR / z.stem)
        for p in entry.rglob("*"):
            if not p.is_file() or p.suffix == ".zip":
                continue
            rel = p.relative_to(entry)
            # state/* lands under STATE_DIR; everything else under REPO_DIR
            if rel.parts and rel.parts[0] == "state":
                dest = STATE_DIR / Path(*rel.parts[1:])
            else:
                dest = REPO_DIR / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(p, dest)
            except OSError:
                shutil.copy2(p, dest)


def run_pipeline() -> int:
    env = os.environ.copy()
    env.setdefault("PMU_GPU", "1")
    env.setdefault("PMU_BUDGET_MIN",
                   os.environ.get("PMU_BUDGET_MIN", "480"))
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_DIR), env.get("PYTHONPATH", "")])
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable, "-m", "kaggle_runner.orchestrator",
        "--repo-dir", str(REPO_DIR),
        "--state-dir", str(STATE_DIR),
        "--budget-minutes", env["PMU_BUDGET_MIN"],
    ]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env, cwd=str(REPO_DIR))
    return proc.wait()


def snapshot_outputs() -> None:
    """Copy checkpoints + key artifacts under /kaggle/working/ so they are
    kept by Kaggle once the kernel finishes (Kaggle keeps the contents of
    /kaggle/working/ as the notebook's outputs)."""
    dest = WORKING / "outputs"
    dest.mkdir(parents=True, exist_ok=True)
    if STATE_DIR.exists():
        (dest / "state").mkdir(exist_ok=True)
        for p in STATE_DIR.rglob("*"):
            if p.is_file():
                rel = p.relative_to(STATE_DIR)
                out = dest / "state" / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)
    # Parquets / models / configs don't mutate once written — hard-link to
    # avoid re-copying multi-GB files on every periodic snapshot. Includes
    # both v2 source parquets (so the next session skips the Drive download)
    # and v3 derived artifacts.
    for pat in ("pmu_*.parquet", "pmu_*.lgb", "pmu_*.json", "pmu_*.pkl",
                "pmu_*.csv"):
        for p in REPO_DIR.glob(pat):
            if not p.is_file():
                continue
            out = dest / p.name
            if out.exists():
                continue
            try:
                os.link(p, out)
            except OSError:
                shutil.copy2(p, out)
    print(f"[kernel] wrote outputs snapshot to {dest}")


def push_state_dataset() -> None:
    slug = os.environ.get("PMU_STATE_DATASET")
    if not slug:
        return
    kaggle_dir = Path.home() / ".kaggle"
    if not (kaggle_dir / "kaggle.json").exists():
        print("[kernel] PMU_STATE_DATASET set but no kaggle.json — skipping push")
        return
    outputs = WORKING / "outputs"
    meta = outputs / "dataset-metadata.json"
    title = slug.split("/")[-1]
    meta.write_text(
        '{"id": "%s", "title": "%s", "licenses": [{"name": "CC0-1.0"}]}' % (slug, title),
        encoding="utf-8",
    )
    try:
        _sh(["kaggle", "datasets", "version",
             "-p", str(outputs), "-m",
             f"auto-push from kernel {os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '?')}",
             "--dir-mode", "zip"])
    except subprocess.CalledProcessError:
        # First push: dataset doesn't exist yet → create it
        _sh(["kaggle", "datasets", "create",
             "-p", str(outputs), "--dir-mode", "zip"])


_STOP_SNAPSHOT = threading.Event()


def _periodic_snapshot(interval_s: int = 300) -> None:
    """Background loop that re-runs snapshot_outputs every interval_s so that
    /kaggle/working/outputs/ always reflects the most recent state on disk.

    Kaggle preserves /kaggle/working/ as the notebook version's artifacts even
    if the kernel is hard-killed (9 h wall-clock, pre-empt, crash), so the
    incremental snapshot is our last line of defence against lost work.
    """
    while not _STOP_SNAPSHOT.wait(interval_s):
        try:
            snapshot_outputs()
        except Exception as exc:  # noqa: BLE001
            print(f"[kernel] periodic snapshot failed: {exc}", flush=True)


def _install_sigterm_handler() -> None:
    def _handler(signum, frame):  # noqa: ARG001
        print(f"[kernel] caught signal {signum} — flushing snapshot + state dataset",
              flush=True)
        _STOP_SNAPSHOT.set()
        try:
            snapshot_outputs()
        finally:
            try:
                push_state_dataset()
            except Exception as exc:  # noqa: BLE001
                print(f"[kernel] state-dataset push in signal handler failed: {exc}",
                      flush=True)
        sys.exit(143 if signum == signal.SIGTERM else 130)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def main() -> int:
    clone_repo()
    install_requirements()
    restore_from_input()
    _install_sigterm_handler()
    t = threading.Thread(target=_periodic_snapshot, args=(300,), daemon=True)
    t.start()
    try:
        rc = run_pipeline()
    finally:
        _STOP_SNAPSHOT.set()
        snapshot_outputs()
        try:
            push_state_dataset()
        except Exception as exc:  # noqa: BLE001
            print(f"[kernel] final state-dataset push failed: {exc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
