#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pexpect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--suite-root", default="runs/paper_suite/single_gpu")
    parser.add_argument("--remote-output", required=True)
    parser.add_argument("--local-output", required=True)
    parser.add_argument("--sessions", nargs="+", default=["experiment", "experiment2"])
    args = parser.parse_args()
    sessions = args.sessions
    if len(sessions) == 1 and " " in sessions[0]:
        sessions = [item for item in sessions[0].split() if item]

    remote_cmd = "cd {root} && python3 scripts/export_active_tmux_status_json.py --repo-root {root} --suite-root {suite_root} --output {output} --sessions {sessions}".format(
        root=shlex.quote(args.remote_root),
        suite_root=shlex.quote(args.suite_root),
        output=shlex.quote(args.remote_output),
        sessions=" ".join(shlex.quote(item) for item in sessions),
    )
    ssh_cmd = (
        f"ssh -o StrictHostKeyChecking=no -p {shlex.quote(args.port)} "
        f"{shlex.quote(args.user)}@{shlex.quote(args.host)} {shlex.quote(remote_cmd)}"
    )

    child = pexpect.spawn(ssh_cmd, encoding="utf-8", timeout=300)
    child.expect("password:")
    child.sendline(args.password)
    child.expect(pexpect.EOF)
    stdout = child.before
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"failed to locate JSON payload in ssh output: {stdout!r}")
    payload = json.loads(stdout[start : end + 1])
    payload["synced_at_local"] = datetime.now(timezone.utc).astimezone().isoformat()

    local_output = Path(args.local_output)
    local_output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=local_output.parent) as handle:
        handle.write(text)
        temp_name = handle.name
    Path(temp_name).replace(local_output)


if __name__ == "__main__":
    main()
