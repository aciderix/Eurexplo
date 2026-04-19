#!/usr/bin/env python3
"""Kaggle-friendly orchestrator for the PMU v3 pipeline.

Replaces `run_all_v3.sh`. Each stage:
  - is skipped if its sentinel output file already exists (idempotent);
  - has its own per-stage log file in <state>/logs/<stage>.log;
  - updates a heartbeat JSON every 30s so an external monitor can check status;
  - is stopped cleanly if the wall-clock budget is about to expire — on the
    next session, the stage resumes from its per-fold checkpoints (train/stack)
    or from its SQLite store (optuna).

Example:
  python -m kaggle_runner.orchestrator \
      --state-dir /kaggle/working/state \
      --repo-dir  /kaggle/working/repo \
      --budget-minutes 480

Stages can be skipped with --skip stage1,stage2 or a single stage can be run
with --only stage1.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kaggle_runner.heartbeat import Heartbeat


@dataclass
class Stage:
    name: str
    outputs: list[str]           # skip if all exist
    command: list[str]           # executed with cwd=repo_dir
    required: bool = True        # fail pipeline if this stage errors


def default_stages(repo: Path, state: Path, skip_optuna: bool, skip_stack: bool,
                   skip_bandit: bool, heartbeat: Path) -> list[Stage]:
    def hb(arg: str = "--heartbeat") -> list[str]:
        return [arg, str(heartbeat)]

    stages: list[Stage] = []

    # ── 1 feature engineering ──
    stages += [
        Stage("feat_blobs",    ["pmu_feat_blobs_v3.parquet"],
              ["python", "pmu_feat_blobs.py"]),
        Stage("feat_elo_ctx",  ["pmu_feat_elo_ctx_v3.parquet"],
              ["python", "pmu_feat_elo_ctx.py"]),
        Stage("feat_h2h",      ["pmu_feat_h2h_v3.parquet"],
              ["python", "pmu_feat_h2h.py"]),
        Stage("feat_musique_seq", ["pmu_feat_musique_seq_v3.parquet"],
              ["python", "pmu_feat_musique_seq.py"]),
        # feat_emb is long and optional — skip by default
    ]

    # ── 2 build feature matrix ──
    stages.append(
        Stage("build_all", ["pmu_features_v3.parquet"],
              ["python", "pmu_feat_build_all_v3.py"])
    )

    # ── 3 optuna (SQLite-resumable via its own --resume) ──
    if not skip_optuna:
        optuna_cmd = ["python", "pmu_optuna_v3.py",
                      "--n-trials", os.environ.get("PMU_OPTUNA_TRIALS", "200"),
                      "--study", "pmu_v3_main", "--resume"]
        sample_frac = os.environ.get("PMU_OPTUNA_SAMPLE_FRAC", "")
        if sample_frac:
            optuna_cmd += ["--sample-frac", sample_frac]
        stages.append(
            Stage("optuna", ["pmu_best_params_v3.json"], optuna_cmd)
        )

    # ── 4 train (resumable wrapper) ──
    stages.append(
        Stage("train",
              ["pmu_model_v3.lgb", "pmu_oof_v3.parquet"],
              ["python", "-m", "kaggle_runner.train_v3_resumable",
               "--checkpoint-dir", str(state / "train_v3_ckpt"),
               *hb()])
    )

    # ── 5 stack (resumable wrapper) ──
    if not skip_stack:
        stack_cmd = ["python", "-m", "kaggle_runner.stack_v3_resumable",
                     "--checkpoint-dir", str(state / "stack_v3_ckpt"),
                     *hb()]
        from_ym = os.environ.get("PMU_STACK_FROM_YM", "")
        if from_ym:
            stack_cmd += ["--from-ym", from_ym]
        if os.environ.get("PMU_STACK_NO_MLP", "0") == "1":
            stack_cmd += ["--no-mlp"]
        stack_max_est = os.environ.get("PMU_STACK_MAX_EST", "")
        if stack_max_est:
            stack_cmd += ["--max-estimators", stack_max_est]
        stages.append(
            Stage("stack",
                  ["pmu_stack_v3.pkl", "pmu_oof_stack_v3.parquet"],
                  stack_cmd)
        )

    # ── 6 calibration ──
    stages.append(
        Stage("calibrate", ["pmu_calibrator_v3.pkl"],
              ["python", "pmu_calibrate_v3.py", "--oof", "pmu_oof_stack_v3.parquet"])
    )

    # ── 7 conformal ──
    stages.append(
        Stage("conformal", ["pmu_conformal_v3.json"],
              ["python", "pmu_conformal_v3.py",
               "--oof", "pmu_oof_stack_v3.parquet", "--alpha", "0.1"])
    )

    # ── 8 backtest ──
    stages.append(
        Stage("backtest", ["pmu_backtest_v3.csv"],
              ["python", "pmu_backtest_v3.py", "--bootstrap", "1000"])
    )

    # ── 9 bandit ──
    if not skip_bandit:
        stages.append(
            Stage("bandit", ["pmu_bandit_v3.pkl"],
                  ["python", "pmu_bandit_v3.py",
                   "--train-end", "2024-12", "--eval-start", "2025-01"],
                  required=False)
        )

    return stages


def all_outputs_exist(repo: Path, stage: Stage) -> bool:
    return all((repo / o).exists() for o in stage.outputs)


def run_stage(stage: Stage, repo: Path, state: Path, budget_deadline: float) -> int:
    """Run a stage as a subprocess, streaming to a per-stage log file.

    If the budget deadline is reached, sends SIGTERM and returns 124.
    """
    logs_dir = state / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{stage.name}.log"

    with log_path.open("a", encoding="utf-8", buffering=1) as fh:
        fh.write(f"\n==== {stage.name} @ {datetime.now(timezone.utc).isoformat()} ====\n")
        fh.write(f"cmd: {' '.join(stage.command)}\n")
        fh.flush()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTHONPATH",
                       os.pathsep.join([str(repo), env.get("PYTHONPATH", "")]))

        proc = subprocess.Popen(
            stage.command, cwd=str(repo),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, env=env,
        )

        timed_out = False
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                fh.write(line)
                sys.stdout.write(f"[{stage.name}] {line}")
                if time.time() > budget_deadline:
                    timed_out = True
                    break
            if timed_out:
                print(f"[orch] budget reached — stopping {stage.name}")
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return 124
            return proc.wait()
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()


def download_data_if_missing(repo: Path) -> None:
    needed = [
        "pmu_dataset_v2.parquet", "pmu_course_raw_v2.parquet",
        "pmu_feat_elo.parquet", "pmu_feat_market.parquet",
        "pmu_feat_musique.parquet", "pmu_feat_rolling.parquet",
        "pmu_feat_target_enc.parquet",
    ]
    missing = [f for f in needed if not (repo / f).exists()]
    if not missing:
        print("[orch] datasets present, skipping download")
        return
    print(f"[orch] fetching {len(missing)} parquets from Google Drive: {missing}")
    subprocess.check_call([
        sys.executable, "-m", "kaggle_runner.download_data",
        "--dest", str(repo),
    ], cwd=str(repo))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir",  default=".")
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--budget-minutes", type=int,
                    default=int(os.environ.get("PMU_BUDGET_MIN", "480")),
                    help="soft wall-clock budget before clean stop (minutes)")
    ap.add_argument("--skip", default="",
                    help="comma-separated stage names to skip")
    ap.add_argument("--only", default="",
                    help="comma-separated stage names to run exclusively")
    ap.add_argument("--skip-optuna", action="store_true",
                    default=os.environ.get("PMU_SKIP_OPTUNA", "0") == "1")
    ap.add_argument("--skip-stack",  action="store_true",
                    default=os.environ.get("PMU_SKIP_STACK", "0") == "1")
    ap.add_argument("--skip-bandit", action="store_true",
                    default=os.environ.get("PMU_SKIP_BANDIT", "0") == "1")
    args = ap.parse_args()

    repo = Path(args.repo_dir).resolve()
    state = Path(args.state_dir).resolve()
    state.mkdir(parents=True, exist_ok=True)

    heartbeat_path = state / "heartbeat.json"
    heart = Heartbeat(heartbeat_path, interval_s=30)
    heart.start()
    heart.set_stage("orchestrator_boot",
                    repo=str(repo), state=str(state),
                    budget_minutes=args.budget_minutes)

    try:
        download_data_if_missing(repo)
    except Exception as exc:  # noqa: BLE001
        heart.stop(ok=False, error=f"download failed: {exc}")
        raise

    stages = default_stages(
        repo, state,
        skip_optuna=args.skip_optuna,
        skip_stack=args.skip_stack,
        skip_bandit=args.skip_bandit,
        heartbeat=heartbeat_path,
    )

    skip_set = {s.strip() for s in args.skip.split(",") if s.strip()}
    only_set = {s.strip() for s in args.only.split(",") if s.strip()}

    deadline = time.time() + args.budget_minutes * 60
    status: dict[str, str] = {}
    status_path = state / "stages.json"

    for st in stages:
        if only_set and st.name not in only_set:
            status[st.name] = "not-selected"
            continue
        if st.name in skip_set:
            print(f"[orch] SKIP {st.name} (user skip)")
            status[st.name] = "skipped"
            continue
        if all_outputs_exist(repo, st):
            print(f"[orch] SKIP {st.name} (outputs present)")
            status[st.name] = "cached"
            continue

        heart.set_stage(st.name)
        print(f"\n[orch] ━━ {st.name} ━━")
        rc = run_stage(st, repo, state, deadline)
        status[st.name] = {
            0: "ok",
            124: "timed_out",
        }.get(rc, f"error_{rc}")
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

        if rc == 124:
            print("[orch] budget reached — exiting cleanly (resume later).")
            heart.stop(ok=False, error="budget_timeout")
            return 2
        if rc != 0:
            if st.required:
                heart.stop(ok=False, error=f"{st.name} failed with rc={rc}")
                return rc
            print(f"[orch] {st.name} failed (rc={rc}) but non-required, continuing")

    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    heart.stop(ok=True)
    print("[orch] pipeline finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
