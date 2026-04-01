#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from path_utils import resolve_runs_path


ROOT = Path(__file__).resolve().parents[1]


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for x in str(s).split(","):
        t = x.strip()
        if t:
            out.append(int(t))
    return out


def _parse_csv_strs(s: str) -> List[str]:
    out: List[str] = []
    for x in str(s).split(","):
        t = x.strip()
        if t:
            out.append(t)
    return out


def _fmt_lr(x: float) -> str:
    return f"{float(x):.2e}".replace("+", "")


@dataclass
class Row:
    method: str
    lr: float
    seed: int
    best_val_acc: float
    final_val_acc: float
    score_05_05: float
    summary_json: str
    curve_csv: str


def _run_one(
    *,
    config: str,
    method: str,
    seed: int,
    lr: float,
    steps: int,
    warmup_ratio: float,
    lambda_align: float,
    ours_eps: float,
    ours_micro_batch_size: int,
    out_json: Path,
    out_curve: Path,
    retries: int,
    extra_set: List[str],
) -> Row:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run.py"),
        "--config",
        config,
        "--set",
        f"seed={int(seed)}",
        "--set",
        f"method.name={method}",
        "--set",
        f"train.lr={float(lr)}",
        "--set",
        f"train.steps={int(steps)}",
        "--set",
        f"train.warmup_ratio={float(warmup_ratio)}",
        "--set",
        f"method.ours.lambda_align={float(lambda_align)}",
        "--set",
        f"method.ours.eps={float(ours_eps)}",
        "--set",
        f"method.ours.micro_batch_size={int(ours_micro_batch_size)}",
        "--out",
        str(out_json),
        "--curve_out",
        str(out_curve),
    ]
    for kv in extra_set:
        cmd.extend(["--set", kv])

    print(f"[sweep] method={method} seed={seed} lr={_fmt_lr(lr)} steps={steps}")
    env = dict(os.environ)
    for attempt in range(retries + 1):
        try:
            subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
            break
        except subprocess.CalledProcessError:
            if attempt >= retries:
                raise
            print(f"[sweep] retry {attempt + 1}/{retries} for method={method} seed={seed} lr={_fmt_lr(lr)}")
            time.sleep(2.0)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    best = float(payload["best_val_acc"])
    final = float(payload["final_val_acc"])
    score = 0.5 * best + 0.5 * final
    return Row(
        method=method,
        lr=float(lr),
        seed=int(seed),
        best_val_acc=best,
        final_val_acc=final,
        score_05_05=score,
        summary_json=str(out_json),
        curve_csv=str(out_curve),
    )


def _write_per_run(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "lr",
        "seed",
        "best_val_acc",
        "final_val_acc",
        "score_05_05",
        "summary_json",
        "curve_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "method": r.method,
                    "lr": r.lr,
                    "seed": r.seed,
                    "best_val_acc": r.best_val_acc,
                    "final_val_acc": r.final_val_acc,
                    "score_05_05": r.score_05_05,
                    "summary_json": r.summary_json,
                    "curve_csv": r.curve_csv,
                }
            )


def _aggregate(rows: List[Row]) -> List[Dict[str, float | str]]:
    buckets: Dict[tuple[str, float], List[Row]] = {}
    for r in rows:
        buckets.setdefault((r.method, r.lr), []).append(r)

    out: List[Dict[str, float | str]] = []
    for (method, lr), rs in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        b = [x.best_val_acc for x in rs]
        f = [x.final_val_acc for x in rs]
        s = [x.score_05_05 for x in rs]
        out.append(
            {
                "method": method,
                "lr": float(lr),
                "n": int(len(rs)),
                "best_mean": float(statistics.fmean(b)),
                "best_std": float(statistics.pstdev(b) if len(b) > 1 else 0.0),
                "final_mean": float(statistics.fmean(f)),
                "final_std": float(statistics.pstdev(f) if len(f) > 1 else 0.0),
                "score_mean": float(statistics.fmean(s)),
                "score_std": float(statistics.pstdev(s) if len(s) > 1 else 0.0),
            }
        )
    return out


def _write_agg(path: Path, rows: List[Dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "lr",
        "n",
        "best_mean",
        "best_std",
        "final_mean",
        "final_std",
        "score_mean",
        "score_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _best_lr_by_method_mean(agg_rows: List[Dict[str, float | str]]) -> Dict[str, Dict[str, float]]:
    by_method: Dict[str, List[Dict[str, float | str]]] = {}
    for r in agg_rows:
        by_method.setdefault(str(r["method"]), []).append(r)

    out: Dict[str, Dict[str, float]] = {}
    for m, rs in by_method.items():
        best = max(rs, key=lambda x: float(x["score_mean"]))
        out[m] = {
            "lr": float(best["lr"]),
            "score_mean": float(best["score_mean"]),
            "best_mean": float(best["best_mean"]),
            "final_mean": float(best["final_mean"]),
        }
    return out


def _best_lr_by_method_seed(rows: List[Row]) -> Dict[str, Dict[str, Dict[str, float]]]:
    buckets: Dict[tuple[str, int], List[Row]] = {}
    for r in rows:
        buckets.setdefault((r.method, r.seed), []).append(r)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (method, seed), rs in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        best = max(rs, key=lambda x: float(x.score_05_05))
        out.setdefault(method, {})[str(seed)] = {
            "lr": float(best.lr),
            "score_05_05": float(best.score_05_05),
            "best_val_acc": float(best.best_val_acc),
            "final_val_acc": float(best.final_val_acc),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="LR sweep for baseline_avg / baseline_cagrad / ours at fixed step budget.")
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--out_dir", type=str, default="runs_lr_sweep_step100")
    p.add_argument("--methods", type=str, default="baseline_avg,baseline_cagrad,ours")
    p.add_argument("--seeds", type=str, default="2,3,5")
    p.add_argument("--lr_min", type=float, default=2.0e-6)
    p.add_argument("--lr_max", type=float, default=1.0e-3)
    p.add_argument("--num_lrs", type=int, default=9)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lambda_align", type=float, default=1.0e-3)
    p.add_argument("--ours_eps", type=float, default=1.0e-8)
    p.add_argument("--ours_micro_batch_size", type=int, default=8)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--set", action="append", default=[], help="extra key=value overrides")
    args = p.parse_args()

    methods = _parse_csv_strs(args.methods)
    seeds = _parse_csv_ints(args.seeds)
    if not methods:
        raise RuntimeError("empty --methods")
    if not seeds:
        raise RuntimeError("empty --seeds")
    if args.num_lrs <= 0:
        raise RuntimeError("--num_lrs must be > 0")
    if args.lr_min <= 0 or args.lr_max <= 0:
        raise RuntimeError("--lr_min/--lr_max must be > 0 for logspace")

    lrs = np.logspace(math.log10(float(args.lr_min)), math.log10(float(args.lr_max)), num=int(args.num_lrs)).astype(float).tolist()

    out_dir = resolve_runs_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Row] = []
    for method in methods:
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for lr in lrs:
            lr_dir = method_dir / f"lr_{_fmt_lr(lr).replace('.', 'p').replace('-', 'm')}"
            lr_dir.mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                out_json = lr_dir / f"{method}_s{seed}.json"
                out_curve = lr_dir / f"{method}_s{seed}_curve.csv"
                if args.skip_existing and out_json.exists() and out_curve.exists():
                    payload = json.loads(out_json.read_text(encoding="utf-8"))
                    best = float(payload["best_val_acc"])
                    final = float(payload["final_val_acc"])
                    rows.append(
                        Row(
                            method=method,
                            lr=float(lr),
                            seed=int(seed),
                            best_val_acc=best,
                            final_val_acc=final,
                            score_05_05=0.5 * best + 0.5 * final,
                            summary_json=str(out_json),
                            curve_csv=str(out_curve),
                        )
                    )
                    continue

                row = _run_one(
                    config=args.config,
                    method=method,
                    seed=int(seed),
                    lr=float(lr),
                    steps=int(args.steps),
                    warmup_ratio=float(args.warmup_ratio),
                    lambda_align=float(args.lambda_align),
                    ours_eps=float(args.ours_eps),
                    ours_micro_batch_size=int(args.ours_micro_batch_size),
                    out_json=out_json,
                    out_curve=out_curve,
                    retries=max(0, int(args.retries)),
                    extra_set=list(args.set),
                )
                rows.append(row)

    per_csv = out_dir / "lr_sweep_per_run.csv"
    agg_csv = out_dir / "lr_sweep_agg.csv"
    best_json = out_dir / "best_lr_summary.json"

    _write_per_run(per_csv, rows)
    agg_rows = _aggregate(rows)
    _write_agg(agg_csv, agg_rows)

    payload = {
        "config": str(args.config),
        "methods": methods,
        "seeds": seeds,
        "steps": int(args.steps),
        "warmup_ratio": float(args.warmup_ratio),
        "lambda_align": float(args.lambda_align),
        "ours_eps": float(args.ours_eps),
        "ours_micro_batch_size": int(args.ours_micro_batch_size),
        "lrs": [float(x) for x in lrs],
        "best_lr_by_method_mean": _best_lr_by_method_mean(agg_rows),
        "best_lr_by_method_seed": _best_lr_by_method_seed(rows),
    }
    best_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"saved -> {per_csv}")
    print(f"saved -> {agg_csv}")
    print(f"saved -> {best_json}")


if __name__ == "__main__":
    main()
