#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def _csv_tokens(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Run one pipeline process per method, then merge outputs and plot once.")
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--gpus", default="0")
    p.add_argument("--hpo_seeds", default="2,3")
    p.add_argument("--final_seeds", default="2,3,5,7,11")
    p.add_argument("--hpo_trials", type=int, default=96)
    p.add_argument("--coord_trials_per_knob", type=int, default=0)
    p.add_argument("--hpo_steps", type=int, default=200)
    p.add_argument("--final_steps", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--hpo_eval_every", type=int, default=100)
    p.add_argument("--hpo_eval_val_fraction", type=float, default=0.2)
    p.add_argument("--hpo_skip_train_eval", action="store_true")
    p.add_argument("--local_topk", type=int, default=3)
    p.add_argument("--local_grid_points", type=int, default=3)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--gpu_mem_util_ratio", type=float, default=0.7)
    p.add_argument("--probe_steps", type=int, default=100)
    p.add_argument("--probe_timeout_sec", type=float, default=600.0)
    p.add_argument("--disable_mem_probe", action="store_true")
    p.add_argument("--max_workers_per_gpu", type=int, default=1)
    p.add_argument("--max_failed_jobs", type=int, default=3)
    p.add_argument("--notify_emails", type=str, default="")
    p.add_argument("--notify_events", type=str, default="")
    p.add_argument("--skip_mvp", action="store_true")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()

    methods = _csv_tokens(args.methods)
    out_dir = Path(args.out_dir)
    worker_root = out_dir / "_method_workers"
    log_root = worker_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    procs: List[tuple[str, subprocess.Popen[str], Path]] = []
    base_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "pipeline_hpo_final_plot.py"),
        "--config",
        args.config,
        "--gpus",
        args.gpus,
        "--hpo_seeds",
        args.hpo_seeds,
        "--final_seeds",
        args.final_seeds,
        "--hpo_trials",
        str(int(args.hpo_trials)),
        "--coord_trials_per_knob",
        str(int(args.coord_trials_per_knob)),
        "--hpo_steps",
        str(int(args.hpo_steps)),
        "--final_steps",
        str(int(args.final_steps)),
        "--eval_every",
        str(int(args.eval_every)),
        "--hpo_eval_every",
        str(int(args.hpo_eval_every)),
        "--hpo_eval_val_fraction",
        str(float(args.hpo_eval_val_fraction)),
        "--local_topk",
        str(int(args.local_topk)),
        "--local_grid_points",
        str(int(args.local_grid_points)),
        "--retries",
        str(int(args.retries)),
        "--gpu_mem_util_ratio",
        str(float(args.gpu_mem_util_ratio)),
        "--probe_steps",
        str(int(args.probe_steps)),
        "--probe_timeout_sec",
        str(float(args.probe_timeout_sec)),
        "--max_workers_per_gpu",
        str(int(args.max_workers_per_gpu)),
        "--max_failed_jobs",
        str(int(args.max_failed_jobs)),
        "--notify_emails",
        args.notify_emails,
        "--notify_events",
        args.notify_events,
        "--skip_hpo_exports",
        "--skip_final_plotters",
    ]
    if args.hpo_skip_train_eval:
        base_cmd.append("--hpo_skip_train_eval")
    if args.disable_mem_probe:
        base_cmd.append("--disable_mem_probe")
    if args.skip_mvp:
        base_cmd.append("--skip_mvp")
    for item in args.set:
        base_cmd.extend(["--set", item])

    for method in methods:
        method_out = worker_root / method
        method_out.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{method}.log"
        cmd = list(base_cmd) + ["--out_dir", str(method_out), "--methods", method]
        log_f = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_f, stderr=subprocess.STDOUT, text=True)
        procs.append((method, proc, log_path))
        print(f"[method-worker] launch method={method} pid={proc.pid} out={method_out} log={log_path}")

    failed: List[str] = []
    for method, proc, log_path in procs:
        rc = proc.wait()
        if rc != 0:
            failed.append(f"{method}(rc={rc},log={log_path})")
        else:
            print(f"[method-worker] done method={method} log={log_path}")

    if failed:
        raise SystemExit("method worker failure: " + ", ".join(failed))

    merge_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "merge_method_worker_outputs.py"),
        "--out_dir",
        str(out_dir),
        "--worker_root",
        str(worker_root),
        "--methods",
        ",".join(methods),
        "--final_seeds",
        args.final_seeds,
    ]
    if args.skip_mvp:
        merge_cmd.append("--skip_mvp")
    subprocess.run(merge_cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
