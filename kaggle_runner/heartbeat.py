"""Heartbeat + stage-log helpers.

Writes a small JSON file every `interval` seconds while the pipeline runs.
The remote monitor (see `monitor.py`) polls this file via the Kaggle API.

Also exposes a StageLog context manager that tees stdout/stderr to a per-stage
log file — used by the orchestrator so each stage has its own tail-able log.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class Heartbeat:
    def __init__(self, path: Path, interval_s: int = 30):
        self.path = Path(path)
        self.interval = interval_s
        self._state: dict = {
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stage": "boot",
            "stage_started_at": datetime.now(timezone.utc).isoformat(),
            "progress": {},
            "last_beat": datetime.now(timezone.utc).isoformat(),
            "done": False,
            "ok": None,
            "error": None,
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush()

    # ── public API ─────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, ok: bool = True, error: str | None = None) -> None:
        with self._lock:
            self._state["done"] = True
            self._state["ok"] = ok
            self._state["error"] = error
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._flush()

    def set_stage(self, stage: str, **extra) -> None:
        with self._lock:
            self._state["stage"] = stage
            self._state["stage_started_at"] = datetime.now(timezone.utc).isoformat()
            self._state["progress"] = {}
            self._state.update(extra)
        self._flush()

    def progress(self, **kv) -> None:
        with self._lock:
            self._state["progress"].update(kv)
        self._flush()

    # ── internals ──────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                self._state["last_beat"] = datetime.now(timezone.utc).isoformat()
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            payload = json.dumps(self._state, indent=2, ensure_ascii=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:  # noqa: BLE001
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass


@contextmanager
def stage_log(log_dir: Path, stage: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}.log"
    fh = log_path.open("a", encoding="utf-8", buffering=1)
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, fh)
    sys.stderr = _Tee(orig_err, fh)
    fh.write(f"\n==== {stage} @ {datetime.now(timezone.utc).isoformat()} ====\n")
    try:
        yield log_path
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        fh.close()
