#!/usr/bin/env python3
"""Poll a Kaggle kernel to see if it's still healthy.

Runs on your laptop. Every N seconds it:
  1. queries `kaggle kernels status <slug>` → prints state (running/queued/…);
  2. downloads the kernel's output files (heartbeat.json + log tails) into
     ./monitor_out/<slug>/ ;
  3. prints the current stage, last heartbeat, and the tail of every stage log;
  4. warns if the heartbeat hasn't moved for `--stale-minutes`.

Usage:
  export KAGGLE_USERNAME=you
  export KAGGLE_KEY=xxxx            # or put kaggle.json in ~/.kaggle/
  python -m kaggle_runner.monitor --slug you/pmu-v3-pipeline --interval 300

Add --once to do a single check.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=capture, text=True)


def check_cli() -> None:
    if shutil.which("kaggle") is None:
        print("ERR: kaggle CLI not found. pip install kaggle", file=sys.stderr)
        sys.exit(1)


def kernel_status(slug: str) -> str:
    r = _run(["kaggle", "kernels", "status", slug])
    if r.returncode != 0:
        return f"CLI error: {r.stderr.strip()}"
    return r.stdout.strip()


def fetch_outputs(slug: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    r = _run(["kaggle", "kernels", "output", slug, "-p", str(dest), "-w"])
    if r.returncode != 0:
        print(f"[monitor] kernel output fetch failed: {r.stderr.strip()}",
              file=sys.stderr)
        return False
    return True


def read_heartbeat(dest: Path) -> dict | None:
    candidates = list(dest.rglob("heartbeat.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    try:
        return json.loads(candidates[0].read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[monitor] heartbeat unreadable: {exc}", file=sys.stderr)
        return None


def tail_logs(dest: Path, n_lines: int = 10) -> None:
    log_files = sorted(dest.rglob("logs/*.log"), key=lambda p: p.stat().st_mtime)
    if not log_files:
        return
    last = log_files[-1]
    try:
        lines = last.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:  # noqa: BLE001
        return
    print(f"[monitor] tail({last.name}):")
    for line in lines[-n_lines:]:
        print(f"   {line}")


def staleness_seconds(hb: dict) -> float:
    ts = hb.get("last_beat") or hb.get("started_at")
    if not ts:
        return -1.0
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return -1.0
    return (datetime.now(timezone.utc) - t).total_seconds()


def one_check(slug: str, dest: Path, stale_minutes: int) -> bool:
    print(f"\n===== {datetime.now().isoformat(timespec='seconds')} =====")
    status = kernel_status(slug)
    print(f"[monitor] kaggle status: {status}")
    if not fetch_outputs(slug, dest):
        return False
    hb = read_heartbeat(dest)
    if hb is None:
        print("[monitor] no heartbeat.json yet")
    else:
        stale = staleness_seconds(hb)
        print(f"[monitor] stage={hb.get('stage')}  "
              f"started_at={hb.get('started_at')}  "
              f"last_beat={hb.get('last_beat')}  "
              f"stale={stale:.0f}s  done={hb.get('done')}")
        if hb.get("progress"):
            print(f"[monitor] progress: {hb['progress']}")
        if hb.get("error"):
            print(f"[monitor] ERROR reported: {hb['error']}")
        if stale > stale_minutes * 60 and not hb.get("done"):
            print(f"[monitor] ⚠ heartbeat stale for {stale/60:.0f} min — kernel "
                  "may be frozen. Consider restarting it.")
    tail_logs(dest)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True,
                    help="owner/kernel-slug (e.g. yourname/pmu-v3-pipeline)")
    ap.add_argument("--dest", default="monitor_out",
                    help="local directory for downloaded outputs")
    ap.add_argument("--interval", type=int, default=300,
                    help="polling interval in seconds")
    ap.add_argument("--stale-minutes", type=int, default=15,
                    help="warn if heartbeat unchanged for N minutes")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    check_cli()
    dest = Path(args.dest) / args.slug.replace("/", "__")
    while True:
        one_check(args.slug, dest, args.stale_minutes)
        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
