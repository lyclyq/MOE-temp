#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def list_gpu_snapshot() -> list[dict[str, Any]]:
    rc, stdout, _ = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "utilization_gpu_pct": int(parts[4]),
            }
        )
    return rows


def pane_records(session: str) -> list[dict[str, Any]]:
    rc, stdout, _ = run_command(
        [
            "tmux",
            "list-panes",
            "-t",
            session,
            "-F",
            "#{session_name}\t#{window_index}\t#{pane_index}\t#{pane_pid}\t#{pane_current_command}\t#{pane_current_path}",
        ]
    )
    if rc != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 6:
            continue
        rows.append(
            {
                "session_name": parts[0],
                "window_index": int(parts[1]),
                "pane_index": int(parts[2]),
                "pane_pid": int(parts[3]),
                "pane_current_command": parts[4],
                "pane_current_path": parts[5],
            }
        )
    return rows


def capture_pane_tail(session: str, lines: int) -> list[str]:
    rc, stdout, _ = run_command(
        ["tmux", "capture-pane", "-p", "-t", f"{session}:0", "-S", f"-{lines}"]
    )
    if rc != 0:
        return []
    return [line.rstrip() for line in stdout.splitlines() if line.strip()]


def find_current_out_dir(lines: list[str]) -> str | None:
    matches: list[str] = []
    for line in lines:
        matches.extend(re.findall(r"out=([^\s]+)", line))
    return matches[-1] if matches else None


def find_current_label(lines: list[str]) -> str | None:
    matches: list[str] = []
    for line in lines:
        found = re.findall(r"\[group:[^\]]+\]\[\d+\]\s+([^\r\n]+)", line)
        matches.extend(found)
    return matches[-1] if matches else None


def select_suite_progress_root(session: str, suite_root: Path) -> Path | None:
    candidates = {
        "experiment": [
            suite_root / "_suite_progress_experiment",
            suite_root / "_suite_progress",
        ],
        "experiment2": [
            suite_root / "_suite_progress_experiment2",
            suite_root / "_suite_progress",
        ],
        "experiment3": [
            suite_root / "_suite_progress_experiment3",
            suite_root / "_suite_progress",
        ],
    }.get(session, [suite_root / "_suite_progress"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def collect_suite_progress(session: str, suite_root: Path) -> dict[str, Any] | None:
    progress_root = select_suite_progress_root(session, suite_root)
    if progress_root is None:
        return None
    payload = {
        "root": str(progress_root),
        "overall_status": read_json(progress_root / "overall_status.json"),
    }
    groups_dir = progress_root / "groups"
    if groups_dir.exists():
        group_payloads: dict[str, Any] = {}
        for group_file in sorted(groups_dir.glob("*.json")):
            group_payload = read_json(group_file)
            if group_payload is not None:
                group_payloads[group_file.stem] = group_payload
        if group_payloads:
            payload["groups"] = group_payloads
    group_rows = read_csv_rows(progress_root / "groups_status.csv")
    if group_rows:
        payload["groups_status"] = group_rows
    return payload


def suite_progress_current_group(suite_progress: dict[str, Any] | None) -> dict[str, Any] | None:
    if not suite_progress:
        return None
    overall = suite_progress.get("overall_status") or {}
    groups = overall.get("groups") or []
    if groups:
        return groups[0]
    group_payloads = suite_progress.get("groups") or {}
    if group_payloads:
        first_key = sorted(group_payloads)[0]
        return group_payloads[first_key]
    group_rows = suite_progress.get("groups_status") or []
    if group_rows:
        return group_rows[0]
    return None


def collect_session(session: str, repo_root: Path, suite_root: Path) -> dict[str, Any]:
    rc, _, _ = run_command(["tmux", "has-session", "-t", session])
    exists = rc == 0
    result: dict[str, Any] = {
        "session": session,
        "exists": exists,
        "repo_root": str(repo_root),
    }

    suite_progress = collect_suite_progress(session, suite_root)
    if suite_progress is not None:
        result["suite_progress"] = suite_progress
    current_group = suite_progress_current_group(suite_progress)

    if not exists:
        return result

    panes = pane_records(session)
    capture_tail = capture_pane_tail(session, 120)
    current_out_dir = find_current_out_dir(capture_tail)
    current_label = find_current_label(capture_tail)
    if current_group:
        current_out_dir_path = repo_root / current_out_dir if current_out_dir else None
        if not current_out_dir or (current_out_dir_path is not None and not current_out_dir_path.exists()):
            current_out_dir = current_group.get("current_out_dir")
        current_label = current_label or current_group.get("current_label")
    plan_lines = [
        line
        for line in capture_tail
        if "[plan]" in line or "[multi]" in line or "[group:" in line or line.startswith("done:")
    ][-40:]

    result["panes"] = panes
    result["tmux_plan"] = {
        "plan_lines": plan_lines,
        "capture_tail": capture_tail[-80:],
        "current_label": current_label,
    }

    if current_out_dir:
        out_path = repo_root / current_out_dir
        progress_path = out_path / "status" / "progress.json"
        result["directories"] = {
            "current_out_dir": current_out_dir,
            "status_dir": str(out_path / "status"),
            "logs_dir": str(out_path / "logs"),
        }
        progress_payload = read_json(progress_path)
        if progress_payload is None and current_group:
            progress_file = current_group.get("pipeline_progress_file")
            if progress_file:
                progress_path = repo_root / progress_file
                progress_payload = read_json(progress_path)
        if progress_payload is not None:
            result["progress"] = {
                "progress_json_path": str(progress_path),
                "payload": progress_payload,
            }
        manifest = read_json(out_path / "pipeline_manifest.json")
        if manifest is not None:
            result["manifest"] = {
                "path": str(out_path / "pipeline_manifest.json"),
                "payload": manifest,
            }

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--suite-root", default="runs/paper_suite/single_gpu")
    parser.add_argument("--sessions", nargs="+", default=["experiment", "experiment2"])
    parser.add_argument("--output")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    suite_root = (repo_root / args.suite_root).resolve()

    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "hostname": subprocess.run(["hostname"], text=True, capture_output=True).stdout.strip(),
        "repo_root": str(repo_root),
        "suite_root": str(suite_root),
        "gpu_snapshot": list_gpu_snapshot(),
        "sessions": {
            session: collect_session(session, repo_root, suite_root) for session in args.sessions
        },
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
