#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import pexpect


def parse_paths(items: list[str]) -> list[str]:
    if len(items) == 1 and ";" in items[0]:
        return [part.strip() for part in items[0].split(";") if part.strip()]
    return [item.strip() for item in items if item.strip()]


def run_rsync(
    *,
    host: str,
    port: str,
    user: str,
    password: str,
    remote_path: Path,
    local_path: Path,
    remote_kind: str,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)

    source = (
        f"{user}@{host}:{shlex.quote(str(remote_path))}/"
        if remote_kind == "dir"
        else f"{user}@{host}:{shlex.quote(str(remote_path))}"
    )
    destination = f"{local_path}/" if remote_kind == "dir" else str(local_path)
    ssh_rsh = f"ssh -o StrictHostKeyChecking=no -p {shlex.quote(port)}"
    cmd = (
        f"rsync -a --delete -e {shlex.quote(ssh_rsh)} "
        f"{source} {shlex.quote(destination)}"
    )

    child = pexpect.spawn(cmd, encoding="utf-8", timeout=3600)
    index = child.expect(["password:", pexpect.EOF, pexpect.TIMEOUT])
    if index == 0:
        child.sendline(password)
        child.expect(pexpect.EOF)
    elif index == 2:
        raise RuntimeError(f"rsync timed out: {cmd}")
    if child.exitstatus not in (0, None) and child.signalstatus is None:
        raise RuntimeError(f"rsync failed rc={child.exitstatus}: {cmd}\n{child.before}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("--local-base", required=True)
    parser.add_argument("--paths", nargs="+", required=True)
    args = parser.parse_args()

    paths = parse_paths(args.paths)
    remote_base = Path(args.remote_base)
    local_base = Path(args.local_base)
    local_base.mkdir(parents=True, exist_ok=True)

    for rel in paths:
        rel_path = Path(rel)
        remote_path = remote_base / rel_path
        local_path = local_base / rel_path

        remote_kind_cmd = (
            "if test -d {path}; then echo dir; "
            "elif test -f {path}; then echo file; "
            "else echo missing; fi"
        ).format(path=shlex.quote(str(remote_path)))
        check_cmd = (
            f"ssh -o StrictHostKeyChecking=no -p {shlex.quote(args.port)} "
            f"{shlex.quote(args.user)}@{shlex.quote(args.host)} "
            f"{shlex.quote(remote_kind_cmd)}"
        )
        child = pexpect.spawn(check_cmd, encoding="utf-8", timeout=120)
        index = child.expect(["password:", pexpect.EOF, pexpect.TIMEOUT])
        if index == 0:
            child.sendline(args.password)
            child.expect(pexpect.EOF)
        elif index == 2:
            raise RuntimeError(f"remote existence check timed out for {remote_path}")
        if child.exitstatus not in (0, None):
            raise RuntimeError(f"remote path not found: {remote_path}")
        remote_kind = (child.before or "").strip().splitlines()[-1].strip()
        if remote_kind not in {"dir", "file"}:
            raise RuntimeError(f"remote path not found: {remote_path}")

        run_rsync(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            remote_path=remote_path,
            local_path=local_path,
            remote_kind=remote_kind,
        )

    subprocess.run(["find", str(local_base), "-maxdepth", "3", "-type", "d"], check=False)


if __name__ == "__main__":
    main()
