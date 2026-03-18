#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import itertools
import json
import math
import os
import random
import signal
import shutil
import smtplib
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from path_utils import resolve_runs_path


ROOT = Path(__file__).resolve().parents[1]
_STOP_EVENT = threading.Event()
_ACTIVE_PROCS_LOCK = threading.Lock()
_ACTIVE_PROCS: set[subprocess.Popen[Any]] = set()


def _register_proc(proc: subprocess.Popen[Any]) -> None:
    with _ACTIVE_PROCS_LOCK:
        _ACTIVE_PROCS.add(proc)


def _unregister_proc(proc: subprocess.Popen[Any]) -> None:
    with _ACTIVE_PROCS_LOCK:
        _ACTIVE_PROCS.discard(proc)


def _snapshot_active_procs() -> List[subprocess.Popen[Any]]:
    with _ACTIVE_PROCS_LOCK:
        return [p for p in _ACTIVE_PROCS if p.poll() is None]


def _terminate_active_processes(phase_name: str, grace_seconds: float = 2.0) -> None:
    procs = _snapshot_active_procs()
    if not procs:
        return
    print(f"[{phase_name}] interrupt: stopping {len(procs)} active process(es)")

    # Jobs run in their own process groups; signal the whole group first.
    for proc in procs:
        if proc.poll() is not None:
            continue
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            continue
        except Exception:
            pass

    deadline = time.time() + max(0.0, float(grace_seconds))
    while time.time() < deadline:
        alive = [p for p in procs if p.poll() is None]
        if not alive:
            return
        time.sleep(0.1)

    for proc in procs:
        if proc.poll() is not None:
            continue
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except Exception:
            pass

    deadline = time.time() + 1.0
    while time.time() < deadline:
        alive = [p for p in procs if p.poll() is None]
        if not alive:
            return
        time.sleep(0.1)

    for proc in procs:
        if proc.poll() is not None:
            continue
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except Exception:
            pass


_MEMORY_GROUP_KEYS: Tuple[str, ...] = (
    "method.name",
    "train.batch_size",
    "train.device_batch_size",
    "train.micro_batch_size",
    "method.ours.micro_batch_size",
    "model.backbone_backend",
    "model.backbone",
    "model.hf_pretrained_name",
    "model.hf_load_pretrained",
    "model.hidden_dim",
    "model.seq_len",
    "model.max_seq_len",
    "model.num_experts",
    "model.expert_type",
    "model.lora_rank",
    "model.lora_alpha",
    "model.ffn_hidden_dim",
    "model.routing_mode",
    "model.top_k",
    "data.source",
    "data.datasets",
    "data.train_size",
    "data.val_size",
)


def _ts_now() -> float:
    return float(time.time())


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True))
        f.write("\n")


class ProgressTracker:
    def __init__(self, status_root: Path, *, notifier: "PipelineNotifier | None" = None) -> None:
        self._lock = threading.Lock()
        self.status_root = status_root
        self.notifier = notifier
        self.progress_path = status_root / "progress.json"
        self.events_path = status_root / "progress_events.jsonl"
        self.failures_path = status_root / "worker_failures.jsonl"
        self._started_at = _ts_now()
        self._total_jobs = 0
        self._completed_jobs = 0
        self._failed_jobs = 0
        self._current_phase = ""
        self._flush(event="init", extra={})

    def _snapshot(self) -> Dict[str, Any]:
        elapsed = max(0.0, _ts_now() - self._started_at)
        pct = 0.0
        if self._total_jobs > 0:
            pct = 100.0 * float(self._completed_jobs) / float(self._total_jobs)
        eta = None
        if self._completed_jobs > 0 and self._total_jobs > self._completed_jobs:
            rate = float(self._completed_jobs) / max(elapsed, 1.0e-9)
            rem = float(self._total_jobs - self._completed_jobs)
            eta = float(rem / max(rate, 1.0e-9))
        return {
            "started_at_epoch": float(self._started_at),
            "updated_at_epoch": float(_ts_now()),
            "elapsed_sec": float(elapsed),
            "current_phase": self._current_phase,
            "total_jobs": int(self._total_jobs),
            "completed_jobs": int(self._completed_jobs),
            "failed_jobs": int(self._failed_jobs),
            "percent": float(min(100.0, max(0.0, pct))),
            "eta_sec": None if eta is None else float(max(0.0, eta)),
        }

    def _flush(self, *, event: str, extra: Dict[str, Any]) -> None:
        self.status_root.mkdir(parents=True, exist_ok=True)
        snap = self._snapshot()
        self.progress_path.write_text(json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8")
        evt = {
            "ts_epoch": float(_ts_now()),
            "event": str(event),
            **extra,
            "progress": {
                "completed_jobs": int(snap["completed_jobs"]),
                "total_jobs": int(snap["total_jobs"]),
                "percent": float(snap["percent"]),
                "phase": str(snap["current_phase"]),
            },
        }
        _append_jsonl(self.events_path, evt)

    def note(self, event: str, extra: Dict[str, Any]) -> None:
        with self._lock:
            self._flush(event=event, extra=dict(extra))

    def set_phase(self, phase_name: str) -> None:
        with self._lock:
            self._current_phase = str(phase_name)
            self._flush(event="phase_start", extra={"phase": str(phase_name)})
            if self.notifier is not None:
                self.notifier.notify(
                    "phase_start",
                    subject=f"[moe-pipeline] phase start: {phase_name}",
                    lines=[
                        f"phase={phase_name}",
                        f"status_root={self.status_root}",
                    ],
                )

    def clear_phase(self, phase_name: str) -> None:
        with self._lock:
            self._current_phase = ""
            self._flush(event="phase_end", extra={"phase": str(phase_name)})
            if self.notifier is not None:
                self.notifier.notify(
                    "phase_end",
                    subject=f"[moe-pipeline] phase end: {phase_name}",
                    lines=[
                        f"phase={phase_name}",
                        f"status_root={self.status_root}",
                    ],
                )

    def add_planned_jobs(self, phase_name: str, n: int) -> None:
        if n <= 0:
            return
        with self._lock:
            self._total_jobs += int(n)
            self._flush(event="plan_add", extra={"phase": str(phase_name), "planned_add": int(n)})

    def job_done(self, *, phase_name: str, job: "RunJob", gpu: int, reused: bool) -> None:
        with self._lock:
            self._completed_jobs += 1
            self._flush(
                event="job_done",
                extra={
                    "phase": str(phase_name),
                    "method": str(job.method_alias),
                    "candidate": str(job.candidate_id),
                    "seed": int(job.seed),
                    "gpu": int(gpu),
                    "reused": bool(reused),
                },
            )

    def job_retry(
        self,
        *,
        phase_name: str,
        job: "RunJob",
        gpu: int,
        attempt: int,
        retries: int,
        err: str,
        log_path: Path,
    ) -> None:
        payload = {
            "ts_epoch": float(_ts_now()),
            "phase": str(phase_name),
            "method": str(job.method_alias),
            "candidate": str(job.candidate_id),
            "seed": int(job.seed),
            "gpu": int(gpu),
            "attempt": int(attempt),
            "max_retries": int(retries),
            "error": str(err),
            "log_path": str(log_path),
        }
        with self._lock:
            _append_jsonl(self.failures_path, payload)
            self._flush(event="job_retry", extra=payload)

    def job_failed(self, *, phase_name: str, job: "RunJob", gpu: int, err: str, log_path: Path) -> int:
        payload = {
            "phase": str(phase_name),
            "method": str(job.method_alias),
            "candidate": str(job.candidate_id),
            "seed": int(job.seed),
            "gpu": int(gpu),
            "error": str(err),
            "log_path": str(log_path),
        }
        with self._lock:
            self._failed_jobs += 1
            _append_jsonl(self.failures_path, {"ts_epoch": float(_ts_now()), **payload})
            self._flush(event="job_failed", extra=payload)
            if self.notifier is not None:
                self.notifier.notify(
                    "job_failed",
                    subject=f"[moe-pipeline] job failed: {phase_name}",
                    lines=[
                        f"phase={phase_name}",
                        f"method={job.method_alias}",
                        f"candidate={job.candidate_id}",
                        f"seed={job.seed}",
                        f"gpu={gpu}",
                        f"log_path={log_path}",
                        f"error={err}",
                        f"failed_jobs={self._failed_jobs}",
                    ],
                )
            return int(self._failed_jobs)

    def failed_jobs(self) -> int:
        with self._lock:
            return int(self._failed_jobs)


class PipelineNotifier:
    def __init__(self, emails: Sequence[str], events: Sequence[str], context: Dict[str, Any]) -> None:
        self.emails = [str(x).strip() for x in emails if str(x).strip()]
        self.events = {str(x).strip() for x in events if str(x).strip()}
        self.context = dict(context)

    def enabled_for(self, event: str) -> bool:
        return bool(self.emails) and (str(event) in self.events)

    def notify(self, event: str, *, subject: str, lines: Sequence[str]) -> None:
        if not self.enabled_for(event):
            return
        _send_mail_notification(self.emails, subject=subject, lines=list(lines), context=self.context)


def _send_mail_notification(
    emails: Sequence[str],
    *,
    subject: str,
    lines: Sequence[str],
    context: Dict[str, Any],
) -> None:
    recipients = [str(x).strip() for x in emails if str(x).strip()]
    if not recipients:
        return

    host = socket.gethostname()
    body_lines = [
        f"host={host}",
        f"cwd={ROOT}",
    ]
    for k, v in sorted(context.items(), key=lambda x: x[0]):
        body_lines.append(f"{k}={v}")
    body_lines.append("")
    body_lines.extend(str(x) for x in lines)
    body = "\n".join(body_lines) + "\n"

    mail_bin = shutil.which("mail")
    if mail_bin is not None:
        try:
            subprocess.run([mail_bin, "-s", str(subject), *recipients], input=body, text=True, check=True)
            return
        except Exception:
            pass

    sendmail_bin = shutil.which("sendmail")
    if sendmail_bin is not None:
        msg = [
            f"Subject: {subject}",
            "Content-Type: text/plain; charset=utf-8",
            f"To: {', '.join(recipients)}",
            "",
            body,
        ]
        try:
            subprocess.run([sendmail_bin, "-t"], input="\n".join(msg), text=True, check=True)
            return
        except Exception:
            pass

    smtp_host = str(os.environ.get("PIPELINE_SMTP_HOST", "")).strip()
    smtp_user = str(os.environ.get("PIPELINE_SMTP_USER", "")).strip()
    smtp_password = str(os.environ.get("PIPELINE_SMTP_PASSWORD", "")).strip()
    smtp_from = str(os.environ.get("PIPELINE_SMTP_FROM", smtp_user)).strip()
    smtp_port_raw = str(os.environ.get("PIPELINE_SMTP_PORT", "465")).strip() or "465"
    smtp_ssl = str(os.environ.get("PIPELINE_SMTP_SSL", "1")).strip().lower() not in {"0", "false", "no"}
    if smtp_host and smtp_user and smtp_password and smtp_from:
        try:
            smtp_port = int(smtp_port_raw)
        except Exception:
            smtp_port = 465
        msg = [
            f"Subject: {subject}",
            f"From: {smtp_from}",
            f"To: {', '.join(recipients)}",
            "Content-Type: text/plain; charset=utf-8",
            "",
            body,
        ]
        try:
            if smtp_ssl:
                with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
                    server.login(smtp_user, smtp_password)
                    server.sendmail(smtp_from, recipients, "\n".join(msg))
            else:
                with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(smtp_user, smtp_password)
                    server.sendmail(smtp_from, recipients, "\n".join(msg))
            return
        except Exception as e:
            print(f"[notify] smtp send failed subject={subject} recipients={recipients} error={type(e).__name__}: {e}")
            return

    print(f"[notify] skipped subject={subject} recipients={recipients} (mail/sendmail/smtp unavailable)")


def _safe_token(text: str) -> str:
    out: List[str] = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s or "x"


def _parse_scalar(raw: str) -> Any:
    t = str(raw).strip()
    if not t:
        return ""
    lo = t.lower()
    if lo == "true":
        return True
    if lo == "false":
        return False
    if lo == "null" or lo == "none":
        return None
    try:
        if t.startswith("0") and len(t) > 1 and t[1].isdigit():
            raise ValueError
        return int(t)
    except Exception:
        pass
    try:
        return float(t)
    except Exception:
        pass
    if (t.startswith("[") and t.endswith("]")) or (t.startswith("{") and t.endswith("}")):
        try:
            return json.loads(t)
        except Exception:
            return t
    return t


def _set_kv_to_map(set_kv: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for kv in set_kv:
        if "=" not in str(kv):
            continue
        k, v = str(kv).split("=", 1)
        key = k.strip()
        if not key:
            continue
        out[key] = v.strip()
    return out


def _normalize_json_value(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _normalize_json_value(v[k]) for k in sorted(v.keys(), key=lambda x: str(x))}
    if isinstance(v, (list, tuple)):
        return [_normalize_json_value(x) for x in v]
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return str(v)


def _effective_cfg_value(base_cfg: Dict[str, Any], kv_map: Dict[str, str], key: str) -> Any:
    if key in kv_map:
        return _parse_scalar(kv_map[key])
    return _get_by_path(base_cfg, key)


def _memory_group_payload(
    *,
    phase_name: str,
    job: "RunJob",
    base_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    kv_map = _set_kv_to_map(job.set_kv)
    features: Dict[str, Any] = {}
    for k in _MEMORY_GROUP_KEYS:
        features[k] = _normalize_json_value(_effective_cfg_value(base_cfg, kv_map, k))
    return {
        "phase": str(phase_name),
        "method_alias": str(job.method_alias),
        "features": features,
    }


def _memory_group_key(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _load_probe_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _save_probe_cache(path: Path, cache: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _query_gpu_total_memory_mb(gpu: int) -> float:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
        "-i",
        str(int(gpu)),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return 0.0
    for line in str(out).splitlines():
        t = line.strip()
        if not t:
            continue
        try:
            return float(t)
        except Exception:
            continue
    return 0.0


def _probe_group_peak_mem_mb(
    *,
    phase_name: str,
    rep_job: "RunJob",
    config: str,
    gpu: int,
    probe_steps: int,
    logs_root: Path,
    probe_timeout_sec: float,
) -> float:
    probe_set_map = _set_kv_to_map(rep_job.set_kv)
    probe_set_map["train.steps"] = str(max(1, int(probe_steps)))
    probe_set_map["train.eval_every_steps"] = "0"
    probe_set = [f"{k}={v}" for k, v in probe_set_map.items()]

    with tempfile.TemporaryDirectory(prefix="moe_probe_") as td:
        td_path = Path(td)
        out_json = td_path / "probe_summary.json"
        out_curve = td_path / "probe_curve.csv"
        log_path = logs_root / "probe" / f"{_safe_token(phase_name)}__gpu{int(gpu)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [
            sys.executable,
            str(ROOT / "scripts" / "run.py"),
            "--config",
            str(config),
        ]
        for kv in probe_set:
            cmd.extend(["--set", kv])
        cmd.extend(["--out", str(out_json), "--curve_out", str(out_curve)])

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))

        with log_path.open("w", encoding="utf-8") as lf:
            lf.write(f"[probe] phase={phase_name} gpu={gpu}\n")
            lf.write("[probe] cmd=\n")
            lf.write(" ".join(cmd) + "\n")
            lf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _register_proc(proc)
            try:
                rc = proc.wait(timeout=max(30.0, float(probe_timeout_sec)))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5.0)
                except Exception:
                    pass
                raise RuntimeError(f"probe timeout phase={phase_name} gpu={gpu}")
            finally:
                _unregister_proc(proc)

        if int(rc) != 0:
            raise RuntimeError(f"probe failed rc={rc} phase={phase_name} gpu={gpu} log={log_path}")

        payload = _read_summary(out_json)
        overhead = dict(payload.get("overhead", {}))
        peak = float(overhead.get("peak_cuda_memory_mb", 0.0))
        if peak <= 0.0:
            peak = 1.0
        return float(peak)


@contextmanager
def _file_lock(path: Path, *, tracker: ProgressTracker | None, label: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tracker is not None:
        tracker.note("lock_wait", {"lock": str(label), "path": str(path)})
    with path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        if tracker is not None:
            tracker.note("lock_acquired", {"lock": str(label), "path": str(path)})
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            if tracker is not None:
                tracker.note("lock_released", {"lock": str(label), "path": str(path)})


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for tok in str(s).replace(" ", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def _parse_csv_strs(s: str) -> List[str]:
    out: List[str] = []
    for tok in str(s).replace(" ", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(t)
    return out


def _fmt_f(v: float) -> str:
    return f"{float(v):.10g}"


def _canonical_method_token(tok: str) -> str:
    t = str(tok).strip().lower()
    alias = {
        "baseline": "baseline",
        "baseline_avg": "baseline",
        "avg": "baseline",
        "cagrad": "cagrad",
        "baseline_cagrad": "cagrad",
        "ours": "ours",
        "ablation": "ablation",
        "ablation_nonorm": "ablation",
        "ablation_non_normalized": "ablation",
        "ours_nonorm": "ablation",
        "ours_non_normalized": "ablation",
        "non_normalized": "ablation",
    }
    if t not in alias:
        raise RuntimeError(f"unsupported method token: {tok!r}")
    return alias[t]


@dataclass(frozen=True)
class KnobSpec:
    key: str
    scale: str  # "log" | "linear"
    lo: float
    hi: float
    local_factor: float = 2.0
    local_span: float = 0.15

    def clip(self, x: float) -> float:
        return float(min(self.hi, max(self.lo, x)))

    def sample(self, rng: random.Random) -> float:
        if self.scale == "log":
            lo = math.log10(self.lo)
            hi = math.log10(self.hi)
            return float(10.0 ** rng.uniform(lo, hi))
        if self.scale == "linear":
            return float(rng.uniform(self.lo, self.hi))
        raise RuntimeError(f"unknown scale={self.scale!r}")

    def local_points(self, center: float, points: int) -> List[float]:
        if points <= 1:
            return [self.clip(center)]
        half = points // 2
        idxs = list(range(-half, half + 1))
        if len(idxs) > points:
            idxs = idxs[:points]
        vals: List[float] = []
        for i in idxs:
            if self.scale == "log":
                v = center * (self.local_factor ** i)
            else:
                delta = max((self.hi - self.lo) * self.local_span, 1.0e-12)
                v = center + float(i) * delta
            vals.append(self.clip(float(v)))
        uniq = sorted({round(v, 12) for v in vals})
        return [float(v) for v in uniq]


@dataclass(frozen=True)
class MethodSpec:
    alias: str
    method_name: str
    fixed_overrides: Dict[str, str]
    knobs: Tuple[KnobSpec, KnobSpec, KnobSpec]


@dataclass
class Candidate:
    cid: str
    stage: str
    params: Dict[str, float]


@dataclass
class RunJob:
    order: int
    method_alias: str
    candidate_id: str
    seed: int
    steps: int
    eval_every: int
    out_json: Path
    out_curve: Path
    set_kv: List[str]


@dataclass
class RunResult:
    method_alias: str
    candidate_id: str
    seed: int
    best_val_acc: float
    final_val_acc: float
    score_05_05: float
    out_json: str
    out_curve: str
    reused: bool


def _method_specs() -> Dict[str, MethodSpec]:
    lr_knob = KnobSpec("train.lr", "log", 2.0e-6, 5.0e-3, local_factor=2.0)

    return {
        "baseline": MethodSpec(
            alias="baseline",
            method_name="baseline_avg",
            fixed_overrides={},
            knobs=(
                lr_knob,
                KnobSpec("train.weight_decay", "linear", 0.0, 0.05, local_span=0.20),
                KnobSpec("train.grad_clip", "linear", 0.5, 2.0, local_span=0.20),
            ),
        ),
        "cagrad": MethodSpec(
            alias="cagrad",
            method_name="baseline_cagrad",
            fixed_overrides={},
            knobs=(
                lr_knob,
                KnobSpec("method.baseline_cagrad.c", "linear", 0.1, 1.0, local_span=0.20),
                KnobSpec("method.baseline_cagrad.inner_lr", "log", 1.0e-2, 5.0e-1, local_factor=2.0),
            ),
        ),
        "ours": MethodSpec(
            alias="ours",
            method_name="ours",
            fixed_overrides={"method.ours.use_load_norm": "true"},
            knobs=(
                lr_knob,
                KnobSpec("method.ours.lambda_align", "log", 1.0e-5, 3.0e-2, local_factor=2.0),
                KnobSpec("method.ours.eps", "log", 1.0e-9, 1.0e-6, local_factor=3.0),
            ),
        ),
        "ablation": MethodSpec(
            alias="ablation",
            method_name="ours",
            fixed_overrides={"method.ours.use_load_norm": "false"},
            knobs=(
                lr_knob,
                KnobSpec("method.ours.lambda_align", "log", 1.0e-5, 3.0e-2, local_factor=2.0),
                KnobSpec("method.ours.eps", "log", 1.0e-9, 1.0e-6, local_factor=3.0),
            ),
        ),
    }


def _load_base_config(config_path: str, cli_set: Sequence[str]) -> Dict[str, Any]:
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from moe_gc import load_config  # type: ignore

    return load_config(config_path, list(cli_set))


def _get_by_path(d: Dict[str, Any], key: str) -> Any:
    cur: Any = d
    for tok in key.split("."):
        if not isinstance(cur, dict) or tok not in cur:
            return None
        cur = cur[tok]
    return cur


def _candidate_fingerprint(params: Dict[str, float]) -> str:
    items = sorted((str(k), float(v)) for k, v in params.items())
    return "|".join(f"{k}={v:.12g}" for k, v in items)


def _default_params(
    *,
    spec: MethodSpec,
    base_cfg: Dict[str, Any],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for kb in spec.knobs:
        cur = _get_by_path(base_cfg, kb.key)
        if isinstance(cur, (int, float)):
            out[kb.key] = float(kb.clip(float(cur)))
            continue
        if kb.scale == "log":
            out[kb.key] = float(kb.clip(math.sqrt(kb.lo * kb.hi)))
        else:
            out[kb.key] = float(kb.clip(0.5 * (kb.lo + kb.hi)))
    return out


def _split_coordinate_trials(total_trials: int, knob_count: int) -> List[int]:
    if knob_count <= 0:
        raise RuntimeError("knob_count must be > 0")
    # Smoke compatibility: allow ultra-tiny runs (e.g., hpo_trials=1),
    # interpreting it as one anchor-only candidate per knob.
    if total_trials <= 1:
        return [1] * int(knob_count)

    # If user passes a very small trial budget, distribute at least one trial per knob.
    if total_trials < int(knob_count):
        return [1] * int(knob_count)

    # Normal path: split trials across knobs.
    base = int(total_trials) // int(knob_count)
    rem = int(total_trials) % int(knob_count)
    out = [base + (1 if i < rem else 0) for i in range(knob_count)]
    if any(x < 1 for x in out):
        raise RuntimeError(f"invalid coordinate trial split: {out}")
    return out


def _san_key(k: str) -> str:
    return str(k).replace(".", "_")


def _build_coordinate_candidates(
    *,
    spec: MethodSpec,
    base_params: Dict[str, float],
    knob_idx: int,
    trials_this_round: int,
    seed: int,
) -> List[Candidate]:
    if knob_idx < 0 or knob_idx >= len(spec.knobs):
        raise RuntimeError(f"invalid knob_idx={knob_idx}")
    if trials_this_round <= 0:
        raise RuntimeError("trials_this_round must be > 0")

    kb = spec.knobs[knob_idx]
    stage = f"coord:{kb.key}"
    rng = random.Random(int(seed))
    out: List[Candidate] = []
    seen: set[str] = set()

    anchor = {k: float(v) for k, v in base_params.items()}
    fp0 = _candidate_fingerprint(anchor)
    seen.add(fp0)
    out.append(Candidate(cid=f"coord_{knob_idx+1:02d}_anchor", stage=stage, params=anchor))

    idx = 0
    while len(out) < trials_this_round:
        cand = {k: float(v) for k, v in base_params.items()}
        cand[kb.key] = float(kb.sample(rng))
        fp = _candidate_fingerprint(cand)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(Candidate(cid=f"coord_{knob_idx+1:02d}_{_san_key(kb.key)}_{idx:03d}", stage=stage, params=cand))
        idx += 1
    return out


def _build_local_variance_topk_candidates(
    *,
    spec: MethodSpec,
    center_params: Dict[str, float],
    topk_knob_keys: Sequence[str],
    already_seen: set[str],
    grid_points: int,
) -> List[Candidate]:
    if grid_points <= 0:
        raise RuntimeError("grid_points must be > 0")
    if not topk_knob_keys:
        return []

    kb_by_key = {kb.key: kb for kb in spec.knobs}
    chosen_keys = [k for k in topk_knob_keys if k in kb_by_key]
    if not chosen_keys:
        return []

    grids: List[List[float]] = []
    chosen_specs: List[KnobSpec] = []
    for key in chosen_keys:
        kb = kb_by_key[key]
        chosen_specs.append(kb)
        grids.append(kb.local_points(center=float(center_params[key]), points=grid_points))

    out: List[Candidate] = []
    idx = 0
    for vals in itertools.product(*grids):
        cand = {k: float(v) for k, v in center_params.items()}
        for i, kb in enumerate(chosen_specs):
            cand[kb.key] = float(vals[i])
        fp = _candidate_fingerprint(cand)
        if fp in already_seen:
            continue
        already_seen.add(fp)
        out.append(Candidate(cid=f"local_var_topk_{idx:03d}", stage="local_var_topk", params=cand))
        idx += 1
    return out


def _to_set_args(
    *,
    method: MethodSpec,
    candidate: Candidate,
    cli_set: Sequence[str],
    seed: int,
    steps: int,
    eval_every: int,
) -> List[str]:
    kvs: List[str] = []
    kvs.extend(list(cli_set))
    kvs.append(f"seed={int(seed)}")
    kvs.append(f"method.name={method.method_name}")
    kvs.append(f"train.steps={int(steps)}")
    kvs.append(f"train.eval_every_steps={int(eval_every)}")
    for k, v in method.fixed_overrides.items():
        kvs.append(f"{k}={v}")
    for k, v in sorted(candidate.params.items()):
        kvs.append(f"{k}={_fmt_f(v)}")
    return kvs


def _read_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_is_valid(path: Path, curve_path: Path) -> bool:
    if (not path.exists()) or (not curve_path.exists()):
        return False
    try:
        payload = _read_summary(path)
    except Exception:
        return False
    if "best_val_acc" not in payload or "final_val_acc" not in payload:
        return False
    return True


def _job_log_path(logs_root: Path, phase_name: str, job: RunJob, gpu: int, attempt: int) -> Path:
    phase_tok = _safe_token(phase_name)
    cand_tok = _safe_token(job.candidate_id)
    name = f"{job.method_alias}__{cand_tok}__s{int(job.seed)}__a{int(attempt)}.log"
    return logs_root / f"gpu{int(gpu)}" / phase_tok / name


def _run_job_once(job: RunJob, config: str, gpu: int, *, log_path: Path) -> None:
    if _STOP_EVENT.is_set():
        raise RuntimeError("interrupted")

    cmd: List[str] = [
        sys.executable,
        str(ROOT / "scripts" / "run.py"),
        "--config",
        str(config),
    ]
    for kv in job.set_kv:
        cmd.extend(["--set", kv])
    cmd.extend(["--out", str(job.out_json), "--curve_out", str(job.out_curve)])

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(
            f"[run] method={job.method_alias} candidate={job.candidate_id} seed={job.seed} "
            f"steps={job.steps} gpu={gpu}\n"
        )
        lf.write("[run] cmd=\n")
        lf.write(" ".join(cmd) + "\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _register_proc(proc)
        try:
            rc = proc.wait()
        finally:
            _unregister_proc(proc)

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _run_job_with_retry(
    job: RunJob,
    config: str,
    gpu: int,
    retries: int,
    *,
    phase_name: str,
    logs_root: Path,
    progress: ProgressTracker | None,
) -> RunResult:
    if _STOP_EVENT.is_set():
        raise RuntimeError("interrupted")

    reused = False
    if _summary_is_valid(job.out_json, job.out_curve):
        reused = True
    else:
        job.out_json.parent.mkdir(parents=True, exist_ok=True)
        job.out_curve.parent.mkdir(parents=True, exist_ok=True)
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            log_path = _job_log_path(logs_root, phase_name, job, gpu, attempt)
            try:
                _run_job_once(job, config=config, gpu=gpu, log_path=log_path)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                err_text = f"{type(e).__name__}: {e}"
                if progress is not None:
                    progress.job_retry(
                        phase_name=phase_name,
                        job=job,
                        gpu=gpu,
                        attempt=attempt + 1,
                        retries=retries,
                        err=err_text,
                        log_path=log_path,
                    )
                if _STOP_EVENT.is_set():
                    break
                if attempt >= retries:
                    break
                print(
                    f"[retry] method={job.method_alias} cand={job.candidate_id} seed={job.seed} "
                    f"gpu={gpu} attempt={attempt+1}/{retries}"
                )
                time.sleep(2.0)
        if last_err is not None:
            raise last_err
        if not _summary_is_valid(job.out_json, job.out_curve):
            raise RuntimeError(f"run output invalid: {job.out_json}")

    payload = _read_summary(job.out_json)
    best = float(payload["best_val_acc"])
    final = float(payload["final_val_acc"])
    score = 0.5 * best + 0.5 * final
    return RunResult(
        method_alias=job.method_alias,
        candidate_id=job.candidate_id,
        seed=int(job.seed),
        best_val_acc=best,
        final_val_acc=final,
        score_05_05=score,
        out_json=str(job.out_json),
        out_curve=str(job.out_curve),
        reused=bool(reused),
    )


def _resolve_group_slots(
    *,
    phase_name: str,
    payload: Dict[str, Any],
    jobs: Sequence[RunJob],
    config: str,
    gpus: Sequence[int],
    logs_root: Path,
    progress: ProgressTracker | None,
    probe_cache: Dict[str, Any],
    probe_cache_path: Path,
    gpu_mem_util_ratio: float,
    probe_steps: int,
    probe_timeout_sec: float,
    disable_mem_probe: bool,
    max_workers_per_gpu: int,
) -> Dict[int, int]:
    key = _memory_group_key(payload)
    ratio = max(0.10, min(0.98, float(gpu_mem_util_ratio)))

    cached = probe_cache.get(key)
    peak_mb = 0.0
    if isinstance(cached, dict):
        try:
            peak_mb = float(cached.get("peak_mem_mb", 0.0))
        except Exception:
            peak_mb = 0.0

    need_exec = any(not _summary_is_valid(j.out_json, j.out_curve) for j in jobs)

    if peak_mb <= 0.0 and need_exec and (not disable_mem_probe):
        rep = jobs[0]
        probe_gpu = int(gpus[0])
        try:
            peak_mb = _probe_group_peak_mem_mb(
                phase_name=phase_name,
                rep_job=rep,
                config=config,
                gpu=probe_gpu,
                probe_steps=max(1, int(probe_steps)),
                logs_root=logs_root,
                probe_timeout_sec=float(probe_timeout_sec),
            )
        except Exception as e:  # noqa: BLE001
            peak_mb = 0.0
            if progress is not None:
                progress.note(
                    "probe_fallback",
                    {
                        "phase": str(phase_name),
                        "error": f"{type(e).__name__}: {e}",
                        "fallback_slots": 1,
                    },
                )
        if peak_mb > 0.0:
            probe_cache[key] = {
                "ts_epoch": float(_ts_now()),
                "phase": str(phase_name),
                "payload": payload,
                "peak_mem_mb": float(peak_mb),
            }
            _save_probe_cache(probe_cache_path, probe_cache)

    slots: Dict[int, int] = {}
    for g in gpus:
        total_mb = _query_gpu_total_memory_mb(int(g))
        if peak_mb > 0.0 and total_mb > 0.0:
            val = int((float(total_mb) * ratio) // max(1.0, float(peak_mb)))
            slots[int(g)] = max(1, val)
        else:
            slots[int(g)] = 1
        if max_workers_per_gpu > 0:
            slots[int(g)] = max(1, min(int(max_workers_per_gpu), int(slots[int(g)])))

    if progress is not None:
        progress.note(
            "group_slots",
            {
                "phase": str(phase_name),
                "group_key": key,
                "peak_mem_mb": float(peak_mb),
                "slots": {str(k): int(v) for k, v in sorted(slots.items())},
                "need_exec": bool(need_exec),
            },
        )
    return slots


def _run_jobs_group(
    *,
    indexed_jobs: Sequence[Tuple[int, RunJob]],
    config: str,
    gpus: Sequence[int],
    slots_by_gpu: Dict[int, int],
    retries: int,
    phase_name: str,
    logs_root: Path,
    progress: ProgressTracker | None,
    out: List[RunResult | None],
    max_failed_jobs: int,
) -> None:
    if not indexed_jobs:
        return

    caps = {int(g): max(1, int(slots_by_gpu.get(int(g), 1))) for g in gpus}
    max_workers = max(1, int(sum(caps.values())))
    print(f"[{phase_name}] group launch jobs={len(indexed_jobs)} slots={caps}")

    ex = ThreadPoolExecutor(max_workers=max_workers)
    pending: deque[Tuple[int, RunJob]] = deque(indexed_jobs)
    in_flight_by_gpu = {int(g): 0 for g in gpus}
    dispatched_by_gpu = {int(g): 0 for g in gpus}
    fut_meta: Dict[Any, Tuple[int, RunJob, int, Path]] = {}
    wait_shutdown = True

    def _pick_gpu() -> int | None:
        available = [int(g) for g in gpus if int(in_flight_by_gpu[int(g)]) < int(caps[int(g)])]
        if not available:
            return None
        return min(
            available,
            key=lambda gi: (
                float(in_flight_by_gpu[gi]) / float(max(1, caps[gi])),
                int(dispatched_by_gpu[gi]),
                int(gi),
            ),
        )

    try:
        while pending or fut_meta:
            while pending:
                chosen_gpu = _pick_gpu()
                if chosen_gpu is None:
                    break
                idx, job = pending.popleft()
                in_flight_by_gpu[chosen_gpu] += 1
                dispatched_by_gpu[chosen_gpu] += 1
                fut = ex.submit(
                    _run_job_with_retry,
                    job,
                    config,
                    chosen_gpu,
                    retries,
                    phase_name=phase_name,
                    logs_root=logs_root,
                    progress=progress,
                )
                fut_meta[fut] = (idx, job, chosen_gpu, _job_log_path(logs_root, phase_name, job, chosen_gpu, retries))

            if not fut_meta:
                continue

            done_set, _ = wait(list(fut_meta.keys()), return_when=FIRST_COMPLETED)
            for fut in done_set:
                idx, job, gpu, log_path = fut_meta.pop(fut)
                in_flight_by_gpu[gpu] = max(0, int(in_flight_by_gpu[gpu]) - 1)
                try:
                    rr = fut.result()
                except Exception as e:  # noqa: BLE001
                    failed_jobs = 1
                    if progress is not None:
                        failed_jobs = progress.job_failed(
                            phase_name=phase_name,
                            job=job,
                            gpu=gpu,
                            err=f"{type(e).__name__}: {e}",
                            log_path=log_path,
                        )
                    if failed_jobs >= max(1, int(max_failed_jobs)):
                        if progress is not None and progress.notifier is not None:
                            progress.notifier.notify(
                                "failure_limit_reached",
                                subject=f"[moe-pipeline] failure limit reached: {phase_name}",
                                lines=[
                                    f"phase={phase_name}",
                                    f"failed_jobs={failed_jobs}",
                                    f"max_failed_jobs={max_failed_jobs}",
                                    f"last_job={job.method_alias}/{job.candidate_id}/s{job.seed}",
                                    f"log_path={log_path}",
                                    f"error={type(e).__name__}: {e}",
                                ],
                            )
                        _STOP_EVENT.set()
                        wait_shutdown = False
                        for ff in fut_meta:
                            ff.cancel()
                        _terminate_active_processes(phase_name=phase_name)
                        raise RuntimeError(
                            f"failure limit reached in phase={phase_name}: "
                            f"failed_jobs={failed_jobs} max_failed_jobs={max_failed_jobs}"
                        ) from e
                    print(
                        f"[warn] phase={phase_name} method={job.method_alias} cand={job.candidate_id} "
                        f"seed={job.seed} gpu={gpu} failed_jobs={failed_jobs}/{max_failed_jobs}; continuing"
                    )
                    continue
                out[idx] = rr
                if progress is not None:
                    progress.job_done(phase_name=phase_name, job=job, gpu=gpu, reused=bool(rr.reused))
    except KeyboardInterrupt:
        _STOP_EVENT.set()
        wait_shutdown = False
        for ff in fut_meta:
            ff.cancel()
        _terminate_active_processes(phase_name=phase_name)
        raise
    finally:
        ex.shutdown(wait=wait_shutdown, cancel_futures=True)


def _run_jobs_parallel(
    *,
    jobs: Sequence[RunJob],
    config: str,
    gpus: Sequence[int],
    retries: int,
    phase_name: str,
    base_cfg: Dict[str, Any],
    logs_root: Path,
    progress: ProgressTracker | None,
    probe_cache: Dict[str, Any],
    probe_cache_path: Path,
    gpu_mem_util_ratio: float,
    probe_steps: int,
    probe_timeout_sec: float,
    disable_mem_probe: bool,
    max_workers_per_gpu: int,
    max_failed_jobs: int,
) -> List[RunResult]:
    if not jobs:
        return []
    if not gpus:
        raise RuntimeError("empty gpus list")

    print(f"[{phase_name}] launch jobs={len(jobs)} gpus={list(gpus)}")
    if progress is not None:
        progress.add_planned_jobs(phase_name, len(jobs))
        progress.set_phase(phase_name)

    out: List[RunResult | None] = [None] * len(jobs)

    groups: Dict[str, Dict[str, Any]] = {}
    for i, job in enumerate(jobs):
        payload = _memory_group_payload(phase_name=phase_name, job=job, base_cfg=base_cfg)
        key = _memory_group_key(payload)
        rec = groups.setdefault(key, {"payload": payload, "items": []})
        rec["items"].append((i, job))

    try:
        for key, rec in groups.items():
            indexed_jobs = list(rec["items"])
            payload = dict(rec["payload"])
            slots_by_gpu = _resolve_group_slots(
                phase_name=phase_name,
                payload=payload,
                jobs=[x[1] for x in indexed_jobs],
                config=config,
                gpus=gpus,
                logs_root=logs_root,
                progress=progress,
                probe_cache=probe_cache,
                probe_cache_path=probe_cache_path,
                gpu_mem_util_ratio=float(gpu_mem_util_ratio),
                probe_steps=int(probe_steps),
                probe_timeout_sec=float(probe_timeout_sec),
                disable_mem_probe=bool(disable_mem_probe),
                max_workers_per_gpu=int(max_workers_per_gpu),
            )
            if progress is not None:
                progress.note(
                    "group_start",
                    {
                        "phase": str(phase_name),
                        "group_key": str(key),
                        "group_jobs": int(len(indexed_jobs)),
                        "slots": {str(k): int(v) for k, v in sorted(slots_by_gpu.items())},
                    },
                )
            _run_jobs_group(
                indexed_jobs=indexed_jobs,
                config=config,
                gpus=gpus,
                slots_by_gpu=slots_by_gpu,
                retries=int(retries),
                phase_name=phase_name,
                logs_root=logs_root,
                progress=progress,
                out=out,
                max_failed_jobs=int(max_failed_jobs),
            )
            if progress is not None:
                progress.note(
                    "group_end",
                    {
                        "phase": str(phase_name),
                        "group_key": str(key),
                        "group_jobs": int(len(indexed_jobs)),
                    },
                )
    finally:
        if progress is not None:
            progress.clear_phase(phase_name)

    result = [x for x in out if x is not None]
    if len(result) != len(jobs):
        raise RuntimeError(
            f"phase={phase_name} expected {len(jobs)} results but got {len(result)}; aborting to avoid bad aggregation"
        )
    return result


def _agg_candidate_scores(
    results: Sequence[RunResult],
    candidates: Sequence[Candidate],
    method_alias: str,
) -> List[Dict[str, Any]]:
    c_map = {c.cid: c for c in candidates}
    bucket: Dict[str, List[RunResult]] = {}
    for r in results:
        bucket.setdefault(r.candidate_id, []).append(r)

    rows: List[Dict[str, Any]] = []
    for cid, rs in bucket.items():
        bests = [r.best_val_acc for r in rs]
        finals = [r.final_val_acc for r in rs]
        scores = [r.score_05_05 for r in rs]
        row = {
            "method": method_alias,
            "candidate_id": cid,
            "stage": c_map[cid].stage if cid in c_map else "",
            "n_seeds": len(rs),
            "score_mean": float(statistics.fmean(scores)),
            "score_std": float(statistics.pstdev(scores) if len(scores) > 1 else 0.0),
            "best_mean": float(statistics.fmean(bests)),
            "final_mean": float(statistics.fmean(finals)),
        }
        if cid in c_map:
            for k, v in sorted(c_map[cid].params.items()):
                row[k] = float(v)
        rows.append(row)
    rows.sort(key=lambda x: float(x["score_mean"]), reverse=True)
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k in seen:
                continue
            seen.add(k)
            fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _ensure_manifest(path: Path, expected: Dict[str, Any]) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")
        return
    cur = json.loads(path.read_text(encoding="utf-8"))
    if cur != expected:
        raise RuntimeError(
            "pipeline manifest mismatch, refusing resume.\n"
            + json.dumps({"expected": expected, "found": cur}, indent=2, sort_keys=True)
        )


def _run_plotters(
    *,
    final_dir: Path,
    methods: Sequence[str],
    final_seeds: Sequence[int],
    skip_mvp: bool,
) -> None:
    methods_csv = ",".join(methods)
    seeds_csv = ",".join(str(int(x)) for x in final_seeds)

    cmd_plot = [
        sys.executable,
        str(ROOT / "scripts" / "plot_seed_mean_band.py"),
        "--runs_dir",
        str(final_dir),
        "--methods",
        methods_csv,
        "--seeds",
        seeds_csv,
        "--band",
        "std",
        "--out",
        str(final_dir / "seed_mean_band_std.png"),
        "--summary_out",
        str(final_dir / "seed_mean_band_std_summary.json"),
        "--val_table_out",
        str(final_dir / "seed_mean_band_val_last.csv"),
    ]
    subprocess.run(cmd_plot, cwd=str(ROOT), check=True)

    cmd_load = [
        sys.executable,
        str(ROOT / "scripts" / "summarize_router_load.py"),
        "--runs_dir",
        str(final_dir),
        "--methods",
        methods_csv,
        "--seeds",
        seeds_csv,
        "--out_csv",
        str(final_dir / "router_load_summary.csv"),
    ]
    subprocess.run(cmd_load, cwd=str(ROOT), check=True)

    cmd_paper = [
        sys.executable,
        str(ROOT / "scripts" / "plot_paper_metrics.py"),
        "--final_dir",
        str(final_dir),
        "--methods",
        methods_csv,
        "--seeds",
        seeds_csv,
        "--band",
        "std",
        "--out_dir",
        str(final_dir),
    ]
    subprocess.run(cmd_paper, cwd=str(ROOT), check=True)

    if skip_mvp:
        return
    cmd_mvp = [
        sys.executable,
        str(ROOT / "scripts" / "plot_mvp_12pack.py"),
        "--runs_dir",
        str(final_dir),
        "--methods",
        methods_csv,
        "--seeds",
        seeds_csv,
        "--band",
        "std",
        "--out_dir",
        str(final_dir),
    ]
    subprocess.run(cmd_mvp, cwd=str(ROOT), check=True)


def _run_hpo_exports(hpo_root: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_hpo_best_params.py"),
        "--hpo_dir",
        str(hpo_root),
        "--out_csv",
        str(hpo_root / "hpo_best_params.csv"),
        "--out_png",
        str(hpo_root / "hpo_best_scores.png"),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    _STOP_EVENT.clear()

    p = argparse.ArgumentParser(description="Generic pipeline: coordinate HPO + variance-topk local grid -> final -> plot.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True, help="relative path will be placed under runs/")
    p.add_argument("--methods", type=str, default="baseline,cagrad,ours")
    p.add_argument("--gpus", type=str, default="0,1")
    p.add_argument("--hpo_seeds", type=str, default="2,3")
    p.add_argument("--final_seeds", type=str, default="2,3,5,7,11")
    p.add_argument("--hpo_trials", type=int, default=96, help="total coordinate trials per method (split across knobs)")
    p.add_argument("--hpo_steps", type=int, default=50)
    p.add_argument("--final_steps", type=int, default=600)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--local_topk", type=int, default=3)
    p.add_argument("--local_grid_points", type=int, default=3)
    p.add_argument("--hpo_seed_base", type=int, default=20260311)
    p.add_argument("--retries", type=int, default=2, help="max retry count after a failed worker run")
    p.add_argument("--gpu_mem_util_ratio", type=float, default=0.80, help="target gpu memory occupancy ratio for slot sizing")
    p.add_argument("--probe_steps", type=int, default=2, help="quick probe steps per memory group")
    p.add_argument("--probe_timeout_sec", type=float, default=600.0)
    p.add_argument("--disable_mem_probe", action="store_true")
    p.add_argument("--max_workers_per_gpu", type=int, default=4, help="0 means no explicit cap")
    p.add_argument("--max_failed_jobs", type=int, default=3, help="stop the whole pipeline after this many final job failures")
    p.add_argument("--notify_emails", type=str, default=os.environ.get("PIPELINE_NOTIFY_EMAILS", ""))
    p.add_argument(
        "--notify_events",
        type=str,
        default=os.environ.get(
            "PIPELINE_NOTIFY_EVENTS",
            "pipeline_done,pipeline_failed,failure_limit_reached",
        ),
    )
    p.add_argument("--skip_mvp", action="store_true")
    p.add_argument("--set", action="append", default=[], help="extra key=value overrides passed to run.py")
    args = p.parse_args()

    method_tokens = [_canonical_method_token(x) for x in _parse_csv_strs(args.methods)]
    method_tokens = list(dict.fromkeys(method_tokens))
    specs_all = _method_specs()
    specs = [specs_all[m] for m in method_tokens]

    gpus = _parse_csv_ints(args.gpus)
    if not gpus:
        raise RuntimeError("empty --gpus")
    hpo_seeds = _parse_csv_ints(args.hpo_seeds)
    final_seeds = _parse_csv_ints(args.final_seeds)
    if not hpo_seeds:
        raise RuntimeError("empty --hpo_seeds")
    if not final_seeds:
        raise RuntimeError("empty --final_seeds")

    out_dir = resolve_runs_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    status_root = out_dir / "status"
    logs_root = out_dir / "logs" / "gpu"
    locks_root = out_dir / "locks"
    status_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    locks_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": str(args.config),
        "methods": [s.alias for s in specs],
        "hpo_seeds": [int(x) for x in hpo_seeds],
        "final_seeds": [int(x) for x in final_seeds],
        "hpo_trials": int(args.hpo_trials),
        "hpo_steps": int(args.hpo_steps),
        "final_steps": int(args.final_steps),
        "eval_every": int(args.eval_every),
        "local_topk": int(args.local_topk),
        "local_grid_points": int(args.local_grid_points),
        "hpo_seed_base": int(args.hpo_seed_base),
        "hpo_strategy": "coordinate_then_variance_topk_local_grid",
        "extra_set": list(args.set),
        "retries": int(args.retries),
        "gpu_mem_util_ratio": float(args.gpu_mem_util_ratio),
        "probe_steps": int(args.probe_steps),
        "probe_timeout_sec": float(args.probe_timeout_sec),
        "disable_mem_probe": bool(args.disable_mem_probe),
        "max_workers_per_gpu": int(args.max_workers_per_gpu),
        "max_failed_jobs": int(args.max_failed_jobs),
        "notify_emails": [x for x in str(args.notify_emails).split(",") if x.strip()],
        "notify_events": [x for x in str(args.notify_events).split(",") if x.strip()],
    }
    _ensure_manifest(out_dir / "pipeline_manifest.json", manifest)

    base_cfg = _load_base_config(args.config, args.set)
    hpo_root = out_dir / "hpo"
    final_root = out_dir / "final"
    hpo_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    notifier = PipelineNotifier(
        emails=[x.strip() for x in str(args.notify_emails).split(",") if x.strip()],
        events=[x.strip() for x in str(args.notify_events).split(",") if x.strip()],
        context={
            "out_dir": str(out_dir),
            "config": str(args.config),
            "gpus": ",".join(str(x) for x in gpus),
        },
    )
    progress = ProgressTracker(status_root, notifier=notifier)
    probe_cache_path = status_root / "memory_probe_cache.json"
    probe_cache = _load_probe_cache(probe_cache_path)

    progress.note(
        "pipeline_start",
        {
            "out_dir": str(out_dir),
            "status_root": str(status_root),
            "logs_root": str(logs_root),
            "locks_root": str(locks_root),
            "methods": [s.alias for s in specs],
            "gpus": [int(x) for x in gpus],
        },
    )

    all_best_cfg: Dict[str, Dict[str, Any]] = {}
    global_hpo_rows: List[Dict[str, Any]] = []

    try:
        for mi, spec in enumerate(specs):
            print(f"\n[method] {spec.alias} -> {spec.method_name}")
            progress.note("method_start", {"method": spec.alias, "phase": "hpo"})

            method_dir = hpo_root / spec.alias
            method_dir.mkdir(parents=True, exist_ok=True)

            with _file_lock(
                locks_root / f"hpo_method_{_safe_token(spec.alias)}.lock",
                tracker=progress,
                label=f"hpo_method:{spec.alias}",
            ):
                trial_alloc = _split_coordinate_trials(int(args.hpo_trials), len(spec.knobs))
                current_params = _default_params(spec=spec, base_cfg=base_cfg)

            all_candidates: List[Candidate] = []
            all_results: List[RunResult] = []
            coordinate_rounds: List[Dict[str, Any]] = []
            sensitivity_rows: List[Dict[str, Any]] = []

            order = 0
            for ki, kb in enumerate(spec.knobs):
                round_trials = int(trial_alloc[ki])
                coord_candidates = _build_coordinate_candidates(
                    spec=spec,
                    base_params=current_params,
                    knob_idx=ki,
                    trials_this_round=round_trials,
                    seed=int(args.hpo_seed_base) + mi * 10007 + ki * 7919,
                )

                jobs_round: List[RunJob] = []
                for cand in coord_candidates:
                    for sd in hpo_seeds:
                        run_dir = method_dir / "trial_runs" / cand.cid
                        out_json = run_dir / f"{spec.alias}_s{sd}.json"
                        out_curve = run_dir / f"{spec.alias}_s{sd}_curve.csv"
                        jobs_round.append(
                            RunJob(
                                order=order,
                                method_alias=spec.alias,
                                candidate_id=cand.cid,
                                seed=int(sd),
                                steps=int(args.hpo_steps),
                                eval_every=int(args.eval_every),
                                out_json=out_json,
                                out_curve=out_curve,
                                set_kv=_to_set_args(
                                    method=spec,
                                    candidate=cand,
                                    cli_set=args.set,
                                    seed=int(sd),
                                    steps=int(args.hpo_steps),
                                    eval_every=int(args.eval_every),
                                ),
                            )
                        )
                        order += 1

                with _file_lock(
                    locks_root / f"hpo_method_{_safe_token(spec.alias)}__coord_{_safe_token(kb.key)}.lock",
                    tracker=progress,
                    label=f"hpo_coord:{spec.alias}:{kb.key}",
                ):
                    round_results = _run_jobs_parallel(
                        jobs=jobs_round,
                        config=args.config,
                        gpus=gpus,
                        retries=max(0, int(args.retries)),
                        phase_name=f"{spec.alias}:coord:{kb.key}",
                        base_cfg=base_cfg,
                        logs_root=logs_root,
                        progress=progress,
                        probe_cache=probe_cache,
                        probe_cache_path=probe_cache_path,
                        gpu_mem_util_ratio=float(args.gpu_mem_util_ratio),
                        probe_steps=max(1, int(args.probe_steps)),
                        probe_timeout_sec=float(args.probe_timeout_sec),
                        disable_mem_probe=bool(args.disable_mem_probe),
                        max_workers_per_gpu=max(0, int(args.max_workers_per_gpu)),
                        max_failed_jobs=max(1, int(args.max_failed_jobs)),
                    )
                round_agg = _agg_candidate_scores(round_results, coord_candidates, spec.alias)
                if not round_agg:
                    raise RuntimeError(f"no coordinate rows for method={spec.alias} knob={kb.key}")

                best_row = round_agg[0]
                best_cid = str(best_row["candidate_id"])
                best_cand = next((c for c in coord_candidates if c.cid == best_cid), None)
                if best_cand is None:
                    raise RuntimeError(f"coordinate best candidate not found: {best_cid}")

                current_params = {k: float(v) for k, v in best_cand.params.items()}
                score_means = [float(r["score_mean"]) for r in round_agg]
                score_var = float(statistics.pvariance(score_means) if len(score_means) > 1 else 0.0)

                coordinate_rounds.append(
                    {
                        "round": int(ki + 1),
                        "knob": kb.key,
                        "trial_budget": int(round_trials),
                        "selected_candidate_id": best_cid,
                        "selected_params": {k: float(v) for k, v in sorted(current_params.items())},
                        "score_var": score_var,
                        "candidates": [c.__dict__ for c in coord_candidates],
                    }
                )
                sensitivity_rows.append(
                    {
                        "round": int(ki + 1),
                        "knob": kb.key,
                        "score_var": score_var,
                        "score_mean_best": float(best_row["score_mean"]),
                        "score_std_best": float(best_row["score_std"]),
                        "selected_candidate_id": best_cid,
                    }
                )

                all_candidates.extend(coord_candidates)
                all_results.extend(round_results)

            sensitivity_rows.sort(key=lambda x: float(x["score_var"]), reverse=True)
            top_n = min(max(1, int(args.local_topk)), len(spec.knobs))
            topk_knobs = [str(x["knob"]) for x in sensitivity_rows[:top_n]]

            seen = {_candidate_fingerprint(c.params) for c in all_candidates}
            local = _build_local_variance_topk_candidates(
                spec=spec,
                center_params=current_params,
                topk_knob_keys=topk_knobs,
                already_seen=seen,
                grid_points=max(1, int(args.local_grid_points)),
            )

            local_results: List[RunResult] = []
            if local:
                jobs_local: List[RunJob] = []
                for cand in local:
                    for sd in hpo_seeds:
                        run_dir = method_dir / "trial_runs" / cand.cid
                        out_json = run_dir / f"{spec.alias}_s{sd}.json"
                        out_curve = run_dir / f"{spec.alias}_s{sd}_curve.csv"
                        jobs_local.append(
                            RunJob(
                                order=order,
                                method_alias=spec.alias,
                                candidate_id=cand.cid,
                                seed=int(sd),
                                steps=int(args.hpo_steps),
                                eval_every=int(args.eval_every),
                                out_json=out_json,
                                out_curve=out_curve,
                                set_kv=_to_set_args(
                                    method=spec,
                                    candidate=cand,
                                    cli_set=args.set,
                                    seed=int(sd),
                                    steps=int(args.hpo_steps),
                                    eval_every=int(args.eval_every),
                                ),
                            )
                        )
                        order += 1

                with _file_lock(
                    locks_root / f"hpo_method_{_safe_token(spec.alias)}__local_var_topk.lock",
                    tracker=progress,
                    label=f"hpo_local_topk:{spec.alias}",
                ):
                    local_results = _run_jobs_parallel(
                        jobs=jobs_local,
                        config=args.config,
                        gpus=gpus,
                        retries=max(0, int(args.retries)),
                        phase_name=f"{spec.alias}:local_var_topk",
                        base_cfg=base_cfg,
                        logs_root=logs_root,
                        progress=progress,
                        probe_cache=probe_cache,
                        probe_cache_path=probe_cache_path,
                        gpu_mem_util_ratio=float(args.gpu_mem_util_ratio),
                        probe_steps=max(1, int(args.probe_steps)),
                        probe_timeout_sec=float(args.probe_timeout_sec),
                        disable_mem_probe=bool(args.disable_mem_probe),
                        max_workers_per_gpu=max(0, int(args.max_workers_per_gpu)),
                        max_failed_jobs=max(1, int(args.max_failed_jobs)),
                    )

            all_candidates = list(all_candidates) + list(local)
            all_results = list(all_results) + list(local_results)
            all_agg = _agg_candidate_scores(all_results, all_candidates, spec.alias)
            if not all_agg:
                raise RuntimeError(f"no merged hpo rows for method={spec.alias}")

            best_row = all_agg[0]
            best_cid = str(best_row["candidate_id"])
            best_cand = next((c for c in all_candidates if c.cid == best_cid), None)
            if best_cand is None:
                raise RuntimeError(f"best candidate not found: {best_cid}")

            all_best_cfg[spec.alias] = {
                "method_alias": spec.alias,
                "method_name": spec.method_name,
                "candidate_id": best_cid,
                "score_mean": float(best_row["score_mean"]),
                "score_std": float(best_row["score_std"]),
                "params": {k: float(v) for k, v in sorted(best_cand.params.items())},
                "fixed_overrides": dict(spec.fixed_overrides),
                "search_strategy": "coordinate_fix_others_then_variance_topk_local_grid",
                "coordinate_trials_by_knob": [int(x) for x in trial_alloc],
                "topk_var_knobs": list(topk_knobs),
                "top_candidates_from_coarse": [],
            }

            (method_dir / "best_config.json").write_text(
                json.dumps(all_best_cfg[spec.alias], indent=2, sort_keys=True),
                encoding="utf-8",
            )

            rows_per_run: List[Dict[str, Any]] = []
            for rr in all_results:
                rows_per_run.append(
                    {
                        "method": rr.method_alias,
                        "candidate_id": rr.candidate_id,
                        "seed": rr.seed,
                        "best_val_acc": rr.best_val_acc,
                        "final_val_acc": rr.final_val_acc,
                        "score_05_05": rr.score_05_05,
                        "reused": rr.reused,
                        "summary_json": rr.out_json,
                        "curve_csv": rr.out_curve,
                    }
                )
            _write_csv(method_dir / "hpo_per_run.csv", rows_per_run)
            _write_csv(method_dir / "hpo_agg.csv", all_agg)
            (method_dir / "candidate_manifest.json").write_text(
                json.dumps(
                    {
                        "coordinate_rounds": coordinate_rounds,
                        "sensitivity": sensitivity_rows,
                        "selected_knobs_for_local": topk_knobs,
                        "local_candidates": [c.__dict__ for c in local],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            global_hpo_rows.extend(all_agg)

            progress.note("method_done", {"method": spec.alias, "phase": "hpo"})

        _write_csv(hpo_root / "hpo_agg_all_methods.csv", global_hpo_rows)
        (hpo_root / "best_configs.json").write_text(json.dumps(all_best_cfg, indent=2, sort_keys=True), encoding="utf-8")
        _run_hpo_exports(hpo_root)

        final_results: List[RunResult] = []
        for spec in specs:
            progress.note("method_start", {"method": spec.alias, "phase": "final"})
            best = all_best_cfg[spec.alias]
            best_cand = Candidate(
                cid=str(best["candidate_id"]),
                stage="best",
                params={str(k): float(v) for k, v in dict(best["params"]).items()},
            )

        method_final_jobs: List[RunJob] = []
        order = 0
        for sd in final_seeds:
            out_json = final_root / f"{spec.alias}_s{sd}.json"
            out_curve = final_root / f"{spec.alias}_s{sd}_curve.csv"
            method_final_jobs.append(
                RunJob(
                    order=order,
                    method_alias=spec.alias,
                    candidate_id=best_cand.cid,
                    seed=int(sd),
                    steps=int(args.final_steps),
                    eval_every=int(args.eval_every),
                    out_json=out_json,
                    out_curve=out_curve,
                    set_kv=_to_set_args(
                        method=spec,
                        candidate=best_cand,
                        cli_set=args.set,
                        seed=int(sd),
                        steps=int(args.final_steps),
                        eval_every=int(args.eval_every),
                    ),
                )
            )
            order += 1

            with _file_lock(
                locks_root / f"final_method_{_safe_token(spec.alias)}.lock",
                tracker=progress,
                label=f"final_method:{spec.alias}",
            ):
                method_results = _run_jobs_parallel(
                    jobs=method_final_jobs,
                    config=args.config,
                    gpus=gpus,
                    retries=max(0, int(args.retries)),
                    phase_name=f"final:{spec.alias}",
                    base_cfg=base_cfg,
                    logs_root=logs_root,
                    progress=progress,
                    probe_cache=probe_cache,
                    probe_cache_path=probe_cache_path,
                    gpu_mem_util_ratio=float(args.gpu_mem_util_ratio),
                    probe_steps=max(1, int(args.probe_steps)),
                    probe_timeout_sec=float(args.probe_timeout_sec),
                    disable_mem_probe=bool(args.disable_mem_probe),
                    max_workers_per_gpu=max(0, int(args.max_workers_per_gpu)),
                    max_failed_jobs=max(1, int(args.max_failed_jobs)),
                )
            final_results.extend(method_results)
            progress.note("method_done", {"method": spec.alias, "phase": "final"})

        final_per_run: List[Dict[str, Any]] = []
        for rr in final_results:
            final_per_run.append(
                {
                    "method": rr.method_alias,
                    "seed": rr.seed,
                    "best_val_acc": rr.best_val_acc,
                    "final_val_acc": rr.final_val_acc,
                    "score_05_05": rr.score_05_05,
                    "reused": rr.reused,
                    "summary_json": rr.out_json,
                    "curve_csv": rr.out_curve,
                }
            )
        _write_csv(final_root / "final_per_run.csv", final_per_run)

        final_agg_rows: List[Dict[str, Any]] = []
        by_method: Dict[str, List[RunResult]] = {}
        for rr in final_results:
            by_method.setdefault(rr.method_alias, []).append(rr)
        for m, rs in sorted(by_method.items(), key=lambda x: x[0]):
            bests = [x.best_val_acc for x in rs]
            finals = [x.final_val_acc for x in rs]
            scores = [x.score_05_05 for x in rs]
            final_agg_rows.append(
                {
                    "method": m,
                    "n_seeds": len(rs),
                    "best_mean": float(statistics.fmean(bests)),
                    "best_std": float(statistics.pstdev(bests) if len(bests) > 1 else 0.0),
                    "final_mean": float(statistics.fmean(finals)),
                    "final_std": float(statistics.pstdev(finals) if len(finals) > 1 else 0.0),
                    "score_mean": float(statistics.fmean(scores)),
                    "score_std": float(statistics.pstdev(scores) if len(scores) > 1 else 0.0),
                }
            )
        final_agg_rows.sort(key=lambda x: float(x["score_mean"]), reverse=True)
        _write_csv(final_root / "final_agg.csv", final_agg_rows)
        (final_root / "final_best_configs_snapshot.json").write_text(
            json.dumps(all_best_cfg, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        _run_plotters(
            final_dir=final_root,
            methods=[s.alias for s in specs],
            final_seeds=final_seeds,
            skip_mvp=bool(args.skip_mvp),
        )

        progress.note(
            "pipeline_done",
            {
                "out_dir": str(out_dir),
                "hpo_best": str(hpo_root / "best_configs.json"),
                "final_agg": str(final_root / "final_agg.csv"),
                "progress_file": str(status_root / "progress.json"),
                "gpu_logs_root": str(logs_root),
                "failure_log": str(status_root / "worker_failures.jsonl"),
                "probe_cache": str(probe_cache_path),
            },
        )
        notifier.notify(
            "pipeline_done",
            subject=f"[moe-pipeline] done: {out_dir.name}",
            lines=[
                f"out_dir={out_dir}",
                f"hpo_best={hpo_root / 'best_configs.json'}",
                f"final_agg={final_root / 'final_agg.csv'}",
                f"progress_file={status_root / 'progress.json'}",
                f"failure_log={status_root / 'worker_failures.jsonl'}",
            ],
        )
    except Exception as e:
        progress.note(
            "pipeline_failed",
            {
                "out_dir": str(out_dir),
                "error": f"{type(e).__name__}: {e}",
                "failed_jobs": int(progress.failed_jobs()),
                "failure_log": str(status_root / "worker_failures.jsonl"),
            },
        )
        notifier.notify(
            "pipeline_failed",
            subject=f"[moe-pipeline] failed: {out_dir.name}",
            lines=[
                f"out_dir={out_dir}",
                f"error={type(e).__name__}: {e}",
                f"failed_jobs={progress.failed_jobs()}",
                f"failure_log={status_root / 'worker_failures.jsonl'}",
                "",
                traceback.format_exc(),
            ],
        )
        raise

    print(f"\n[pipeline done] out_dir={out_dir}")
    print(f"[hpo best] {(hpo_root / 'best_configs.json')}")
    print(f"[final agg] {(final_root / 'final_agg.csv')}")
    print(f"[progress] {(status_root / 'progress.json')}")
    print(f"[gpu logs] {logs_root}")
    print(f"[worker failures] {(status_root / 'worker_failures.jsonl')}")


if __name__ == "__main__":
    main()
