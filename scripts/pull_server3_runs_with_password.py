#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import pexpect


def run_with_password(cmd: str, password: str, timeout: int) -> str:
    child = pexpect.spawn(cmd, encoding="utf-8", timeout=timeout)
    idx = child.expect(["password:", pexpect.EOF, pexpect.TIMEOUT])
    if idx == 0:
        child.sendline(password)
        child.expect(pexpect.EOF)
    return child.before


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="1.95.193.128")
    ap.add_argument("--port", default="32394")
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default="VyCJkiwD0R")
    ap.add_argument("--repo-root", default="/root/Optimization/MOE-grad-conflict-routing-appendix2")
    ap.add_argument("--local-root", default="runs_server/server3")
    args = ap.parse_args()

    local_root = pathlib.Path(args.local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    ssh_base = f"ssh -o StrictHostKeyChecking=accept-new -p {args.port} {args.user}@{args.host}"

    snapshot_cmd = (
        f"{ssh_base} "
        f"\"hostname; echo ---TMUX---; tmux ls 2>/dev/null || true; "
        f"echo ---GPU---; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
        f"--format=csv,noheader,nounits 2>/dev/null || true\""
    )
    snapshot = run_with_password(snapshot_cmd, args.password, 120)
    (local_root / "_remote_snapshot.txt").write_text(snapshot)

    rsync_targets = [
        f"{args.user}@{args.host}:{args.repo_root}/runs/paper_suite_supplement/",
        f"{args.user}@{args.host}:{args.repo_root}/runs/paper_suite_server3/",
    ]
    for target in rsync_targets:
        cmd = (
            "rsync -az --partial "
            f"-e 'ssh -o StrictHostKeyChecking=accept-new -p {args.port}' "
            f"{target} {local_root}/"
        )
        run_with_password(cmd, args.password, 3600)


if __name__ == "__main__":
    main()
