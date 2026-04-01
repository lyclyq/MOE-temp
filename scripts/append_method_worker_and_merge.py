#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _csv_tokens(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Append one method worker to an existing parallel-method run, then re-merge plots.")
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--append_method", required=True)
    p.add_argument("--all_methods", required=True)
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

    append_method = str(args.append_method).strip()
    all_methods = _csv_tokens(args.all_methods)
    if append_method not in all_methods:
        raise SystemExit(f"--append_method {append_method!r} must be included in --all_methods")

    out_dir = Path(args.out_dir)
    worker_out = out_dir / "_method_workers" / append_method
    worker_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "pipeline_hpo_final_plot.py"),
        "--config",
        args.config,
        "--out_dir",
        str(worker_out),
        "--methods",
        append_method,
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
        cmd.append("--hpo_skip_train_eval")
    if args.disable_mem_probe:
        cmd.append("--disable_mem_probe")
    if args.skip_mvp:
        cmd.append("--skip_mvp")
    for item in args.set:
        cmd.extend(["--set", item])

    subprocess.run(cmd, cwd=str(ROOT), check=True)

    merge_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "merge_method_worker_outputs.py"),
        "--out_dir",
        str(out_dir),
        "--worker_root",
        str(out_dir / "_method_workers"),
        "--methods",
        ",".join(all_methods),
        "--final_seeds",
        args.final_seeds,
    ]
    if args.skip_mvp:
        merge_cmd.append("--skip_mvp")
    subprocess.run(merge_cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
