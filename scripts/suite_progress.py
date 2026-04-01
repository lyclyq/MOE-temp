#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _now() -> float:
    return float(time.time())


def _safe_token(text: str) -> str:
    out: List[str] = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s or "x"


def _read_json(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else dict(default)
    if isinstance(data, dict):
        return data
    return {} if default is None else dict(default)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_csv_row(path: Path, row: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if need_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _bar(percent: float, width: int = 24) -> str:
    pct = max(0.0, min(100.0, float(percent)))
    done = int(round((pct / 100.0) * width))
    done = max(0, min(width, done))
    return "[" + "#" * done + "-" * (width - done) + f"] {pct:6.2f}%"


def _fmt_eta(sec: Any) -> str:
    if sec is None:
        return "n/a"
    try:
        rem = max(0, int(float(sec)))
    except Exception:
        return "n/a"
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _root_paths(root: Path) -> Dict[str, Path]:
    return {
        "groups_dir": root / "groups",
        "group_events_csv": root / "group_events.csv",
        "overall_status_json": root / "overall_status.json",
        "groups_status_csv": root / "groups_status.csv",
        "overall_history_csv": root / "overall_progress_history.csv",
        "monitor_stop": root / "monitor.stop",
    }


def _group_path(root: Path, group: str) -> Path:
    return _root_paths(root)["groups_dir"] / f"{_safe_token(group)}.json"


def _base_group_payload(group: str, total_runs: int) -> Dict[str, Any]:
    return {
        "group": str(group),
        "total_runs": max(0, int(total_runs)),
        "completed_runs": 0,
        "failed_runs": 0,
        "current_index": 0,
        "current_state": "idle",
        "current_label": "",
        "current_out_dir": "",
        "current_message": "",
        "started_at_epoch": None,
        "last_update_epoch": _now(),
        "last_event": "init",
    }


def cmd_init_group(args: argparse.Namespace) -> int:
    root = Path(args.root)
    paths = _root_paths(root)
    paths["groups_dir"].mkdir(parents=True, exist_ok=True)
    payload = _base_group_payload(group=args.group, total_runs=int(args.total_runs))
    _write_json(_group_path(root, args.group), payload)
    _append_csv_row(
        paths["group_events_csv"],
        {
            "ts_epoch": _now(),
            "group": args.group,
            "event": "init",
            "total_runs": int(args.total_runs),
            "completed_runs": 0,
            "failed_runs": 0,
            "current_index": 0,
            "current_state": "idle",
            "current_label": "",
            "current_out_dir": "",
            "message": "",
        },
        fieldnames=[
            "ts_epoch",
            "group",
            "event",
            "total_runs",
            "completed_runs",
            "failed_runs",
            "current_index",
            "current_state",
            "current_label",
            "current_out_dir",
            "message",
        ],
    )
    return 0


def cmd_update_group(args: argparse.Namespace) -> int:
    root = Path(args.root)
    path = _group_path(root, args.group)
    payload = _read_json(path, _base_group_payload(group=args.group, total_runs=0))
    payload["group"] = str(args.group)
    payload["total_runs"] = max(0, int(payload.get("total_runs", 0)))
    payload["last_update_epoch"] = _now()
    payload["last_event"] = str(args.state)

    if args.total_runs is not None:
        payload["total_runs"] = max(0, int(args.total_runs))
    if args.index is not None:
        payload["current_index"] = max(0, int(args.index))
    if args.label is not None:
        payload["current_label"] = str(args.label)
    if args.out_dir is not None:
        payload["current_out_dir"] = str(args.out_dir)
    if args.message is not None:
        payload["current_message"] = str(args.message)

    state = str(args.state)
    if state == "running":
        payload["current_state"] = "running"
        payload["started_at_epoch"] = _now()
    elif state == "done":
        payload["completed_runs"] = min(
            max(0, int(payload.get("total_runs", 0))),
            int(payload.get("completed_runs", 0)) + 1,
        )
        payload["current_state"] = "idle"
    elif state == "failed":
        payload["failed_runs"] = min(
            max(0, int(payload.get("total_runs", 0))),
            int(payload.get("failed_runs", 0)) + 1,
        )
        payload["current_state"] = "failed"
    elif state == "idle":
        payload["current_state"] = "idle"
    else:
        raise RuntimeError(f"unsupported state={state!r}")

    _write_json(path, payload)
    _append_csv_row(
        _root_paths(root)["group_events_csv"],
        {
            "ts_epoch": _now(),
            "group": str(args.group),
            "event": state,
            "total_runs": int(payload.get("total_runs", 0)),
            "completed_runs": int(payload.get("completed_runs", 0)),
            "failed_runs": int(payload.get("failed_runs", 0)),
            "current_index": int(payload.get("current_index", 0)),
            "current_state": str(payload.get("current_state", "")),
            "current_label": str(payload.get("current_label", "")),
            "current_out_dir": str(payload.get("current_out_dir", "")),
            "message": str(payload.get("current_message", "")),
        },
        fieldnames=[
            "ts_epoch",
            "group",
            "event",
            "total_runs",
            "completed_runs",
            "failed_runs",
            "current_index",
            "current_state",
            "current_label",
            "current_out_dir",
            "message",
        ],
    )
    return 0


def _load_pipeline_progress(out_dir: str) -> Dict[str, Any]:
    if not out_dir:
        return {}
    status_path = Path(out_dir) / "status" / "progress.json"
    data = _read_json(status_path)
    if not data:
        return {}
    return {
        "phase": str(data.get("current_phase", "")),
        "percent": float(data.get("percent", 0.0) or 0.0),
        "completed_jobs": int(data.get("completed_jobs", 0) or 0),
        "total_jobs": int(data.get("total_jobs", 0) or 0),
        "failed_jobs": int(data.get("failed_jobs", 0) or 0),
        "eta_sec": data.get("eta_sec"),
        "progress_file": str(status_path),
        "gpu_logs_root": str(Path(out_dir) / "logs" / "gpu"),
    }


def _collect(root: Path, expected_groups: Sequence[str]) -> Dict[str, Any]:
    group_names: List[str] = [str(x) for x in expected_groups if str(x).strip()]
    if not group_names:
        groups_dir = _root_paths(root)["groups_dir"]
        if groups_dir.exists():
            group_names = sorted(p.stem for p in groups_dir.glob("*.json"))

    rows: List[Dict[str, Any]] = []
    total_runs = 0
    completed_runs = 0
    failed_runs = 0
    running_groups = 0
    overall_equiv_done = 0.0

    for group in group_names:
        payload = _read_json(_group_path(root, group), _base_group_payload(group, 0))
        total = max(0, int(payload.get("total_runs", 0)))
        done = max(0, int(payload.get("completed_runs", 0)))
        failed = max(0, int(payload.get("failed_runs", 0)))
        current_state = str(payload.get("current_state", "idle"))
        current_out_dir = str(payload.get("current_out_dir", ""))
        pipe = _load_pipeline_progress(current_out_dir)

        local_frac = 0.0
        if current_state == "running":
            running_groups += 1
            try:
                local_frac = max(0.0, min(1.0, float(pipe.get("percent", 0.0)) / 100.0))
            except Exception:
                local_frac = 0.0

        group_equiv_done = min(float(total), float(done + failed) + local_frac)
        group_percent = (100.0 * group_equiv_done / float(total)) if total > 0 else 0.0

        row = {
            "group": group,
            "total_runs": total,
            "completed_runs": done,
            "failed_runs": failed,
            "current_index": int(payload.get("current_index", 0) or 0),
            "current_state": current_state,
            "current_label": str(payload.get("current_label", "")),
            "current_out_dir": current_out_dir,
            "current_message": str(payload.get("current_message", "")),
            "group_percent": float(group_percent),
            "group_bar": _bar(group_percent),
            "pipeline_phase": str(pipe.get("phase", "")),
            "pipeline_percent": float(pipe.get("percent", 0.0) or 0.0),
            "pipeline_bar": _bar(float(pipe.get("percent", 0.0) or 0.0)),
            "pipeline_completed_jobs": int(pipe.get("completed_jobs", 0) or 0),
            "pipeline_total_jobs": int(pipe.get("total_jobs", 0) or 0),
            "pipeline_failed_jobs": int(pipe.get("failed_jobs", 0) or 0),
            "pipeline_eta_sec": pipe.get("eta_sec"),
            "pipeline_progress_file": str(pipe.get("progress_file", "")),
            "pipeline_gpu_logs_root": str(pipe.get("gpu_logs_root", "")),
            "last_update_epoch": payload.get("last_update_epoch"),
        }
        rows.append(row)

        total_runs += total
        completed_runs += done
        failed_runs += failed
        overall_equiv_done += group_equiv_done

    suite_percent = (100.0 * overall_equiv_done / float(total_runs)) if total_runs > 0 else 0.0
    terminal_groups = [
        r
        for r in rows
        if int(r["completed_runs"]) + int(r["failed_runs"]) >= int(r["total_runs"]) and int(r["total_runs"]) > 0
    ]
    summary = {
        "updated_at_epoch": _now(),
        "expected_groups": group_names,
        "suite": {
            "group_count": len(group_names),
            "terminal_group_count": len(terminal_groups),
            "running_group_count": running_groups,
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "percent": float(suite_percent),
            "bar": _bar(suite_percent),
        },
        "groups": rows,
    }
    return summary


def _write_summary(root: Path, summary: Dict[str, Any]) -> None:
    paths = _root_paths(root)
    _write_json(paths["overall_status_json"], summary)

    group_fields = [
        "group",
        "total_runs",
        "completed_runs",
        "failed_runs",
        "current_index",
        "current_state",
        "current_label",
        "current_out_dir",
        "current_message",
        "group_percent",
        "group_bar",
        "pipeline_phase",
        "pipeline_percent",
        "pipeline_bar",
        "pipeline_completed_jobs",
        "pipeline_total_jobs",
        "pipeline_failed_jobs",
        "pipeline_eta_sec",
        "pipeline_progress_file",
        "pipeline_gpu_logs_root",
        "last_update_epoch",
    ]
    _write_csv(paths["groups_status_csv"], list(summary.get("groups", [])), fieldnames=group_fields)

    suite = dict(summary.get("suite", {}))
    _append_csv_row(
        paths["overall_history_csv"],
        {
            "ts_epoch": summary.get("updated_at_epoch"),
            "group_count": suite.get("group_count", 0),
            "terminal_group_count": suite.get("terminal_group_count", 0),
            "running_group_count": suite.get("running_group_count", 0),
            "total_runs": suite.get("total_runs", 0),
            "completed_runs": suite.get("completed_runs", 0),
            "failed_runs": suite.get("failed_runs", 0),
            "percent": suite.get("percent", 0.0),
            "bar": suite.get("bar", ""),
        },
        fieldnames=[
            "ts_epoch",
            "group_count",
            "terminal_group_count",
            "running_group_count",
            "total_runs",
            "completed_runs",
            "failed_runs",
            "percent",
            "bar",
        ],
    )


def _render_lines(summary: Dict[str, Any]) -> List[str]:
    suite = dict(summary.get("suite", {}))
    lines = [
        (
            "[suite] "
            f"{suite.get('bar', _bar(0.0))} "
            f"runs={suite.get('completed_runs', 0) + suite.get('failed_runs', 0)}/{suite.get('total_runs', 0)} "
            f"done={suite.get('completed_runs', 0)} failed={suite.get('failed_runs', 0)} "
            f"running_groups={suite.get('running_group_count', 0)}"
        )
    ]
    for row in summary.get("groups", []):
        lines.append(
            (
                f"[group:{row.get('group')}] "
                f"{row.get('group_bar')} "
                f"runs={int(row.get('completed_runs', 0)) + int(row.get('failed_runs', 0))}/{row.get('total_runs', 0)} "
                f"state={row.get('current_state', '')} "
                f"phase={row.get('pipeline_phase', '') or '-'} "
                f"local={row.get('pipeline_bar', _bar(0.0))} "
                f"jobs={row.get('pipeline_completed_jobs', 0)}/{row.get('pipeline_total_jobs', 0)} "
                f"eta={_fmt_eta(row.get('pipeline_eta_sec'))} "
                f"label={row.get('current_label', '')}"
            )
        )
    return lines


def cmd_render(args: argparse.Namespace) -> int:
    root = Path(args.root)
    summary = _collect(root, expected_groups=[x for x in str(args.groups).split(",") if x.strip()])
    _write_summary(root, summary)
    for line in _render_lines(summary):
        print(line, flush=True)
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    expected_groups = [x for x in str(args.groups).split(",") if x.strip()]
    interval = max(2.0, float(args.interval))
    stop_path = _root_paths(root)["monitor_stop"]

    while True:
        summary = _collect(root, expected_groups=expected_groups)
        _write_summary(root, summary)
        for line in _render_lines(summary):
            print(line, flush=True)

        suite = dict(summary.get("suite", {}))
        group_count = int(suite.get("group_count", 0) or 0)
        terminal_group_count = int(suite.get("terminal_group_count", 0) or 0)
        if stop_path.exists():
            return 0
        if group_count > 0 and terminal_group_count >= group_count:
            return 0
        time.sleep(interval)


def cmd_stop(args: argparse.Namespace) -> int:
    stop_path = _root_paths(Path(args.root))["monitor_stop"]
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.write_text("stop\n", encoding="utf-8")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Suite-level progress tracker and monitor.")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_init = sp.add_parser("init-group")
    p_init.add_argument("--root", type=str, required=True)
    p_init.add_argument("--group", type=str, required=True)
    p_init.add_argument("--total-runs", type=int, required=True)
    p_init.set_defaults(func=cmd_init_group)

    p_up = sp.add_parser("update-group")
    p_up.add_argument("--root", type=str, required=True)
    p_up.add_argument("--group", type=str, required=True)
    p_up.add_argument("--state", type=str, required=True, choices=["running", "done", "failed", "idle"])
    p_up.add_argument("--index", type=int, default=None)
    p_up.add_argument("--label", type=str, default=None)
    p_up.add_argument("--out-dir", type=str, default=None)
    p_up.add_argument("--message", type=str, default=None)
    p_up.add_argument("--total-runs", type=int, default=None)
    p_up.set_defaults(func=cmd_update_group)

    p_render = sp.add_parser("render")
    p_render.add_argument("--root", type=str, required=True)
    p_render.add_argument("--groups", type=str, default="")
    p_render.set_defaults(func=cmd_render)

    p_mon = sp.add_parser("monitor")
    p_mon.add_argument("--root", type=str, required=True)
    p_mon.add_argument("--groups", type=str, default="")
    p_mon.add_argument("--interval", type=float, default=15.0)
    p_mon.set_defaults(func=cmd_monitor)

    p_stop = sp.add_parser("stop")
    p_stop.add_argument("--root", type=str, required=True)
    p_stop.set_defaults(func=cmd_stop)

    args = p.parse_args()
    rc = int(args.func(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
