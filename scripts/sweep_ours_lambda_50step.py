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

import matplotlib.pyplot as plt
import numpy as np
from path_utils import resolve_runs_path


ROOT = Path(__file__).resolve().parents[1]


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    for x in s.split(","):
        t = x.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for x in s.split(","):
        t = x.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def _fmt_lambda(x: float) -> str:
    return f"{x:.1e}".replace("+0", "").replace("+", "")


@dataclass
class Row:
    lambda_align: float
    seed: int
    best_val_acc: float
    final_val_acc: float
    score_05_05: float
    summary_json: str
    curve_csv: str


def _to_float(v: object) -> float:
    if v is None:
        return float("nan")
    try:
        s = str(v).strip()
        if not s:
            return float("nan")
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _mean_std(vals: List[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.fmean(vals)), float(statistics.pstdev(vals))


def _collect_trend_rows(per_rows: List[Row]) -> List[Dict[str, float]]:
    # key: (lambda_align, step) -> metric lists across seeds
    buckets: Dict[tuple[float, float], Dict[str, List[float]]] = {}
    for row in per_rows:
        path = Path(row.curve_csv)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                step = _to_float(rec.get("step", ""))
                if not math.isfinite(step):
                    continue
                train_acc = _to_float(rec.get("train_acc", ""))
                val_acc = _to_float(rec.get("val_acc", ""))
                key = (float(row.lambda_align), float(step))
                if key not in buckets:
                    buckets[key] = {"train_acc": [], "val_acc": []}
                if math.isfinite(train_acc):
                    buckets[key]["train_acc"].append(float(train_acc))
                if math.isfinite(val_acc):
                    buckets[key]["val_acc"].append(float(val_acc))

    out: List[Dict[str, float]] = []
    for (lam, step), metrics in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        train_mean, train_std = _mean_std(metrics["train_acc"])
        val_mean, val_std = _mean_std(metrics["val_acc"])
        out.append(
            {
                "lambda_align": float(lam),
                "step": float(step),
                "n_train": float(len(metrics["train_acc"])),
                "train_acc_mean": train_mean,
                "train_acc_std": train_std,
                "n_val": float(len(metrics["val_acc"])),
                "val_acc_mean": val_mean,
                "val_acc_std": val_std,
            }
        )
    return out


def _write_trend_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "lambda_align",
        "step",
        "n_train",
        "train_acc_mean",
        "train_acc_std",
        "n_val",
        "val_acc_mean",
        "val_acc_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _plot_trend(
    *,
    out_png: Path,
    lambdas: List[float],
    trend_rows: List[Dict[str, float]],
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=150)
    metric_specs = [
        ("train_acc_mean", "train_acc_std", "Train Acc"),
        ("val_acc_mean", "val_acc_std", "Val Acc"),
    ]

    for ax, (mean_key, std_key, title) in zip(axes, metric_specs):
        for lam in lambdas:
            rows_l = [r for r in trend_rows if float(r["lambda_align"]) == float(lam)]
            rows_l.sort(key=lambda r: float(r["step"]))
            if not rows_l:
                continue
            xs = [float(r["step"]) for r in rows_l if math.isfinite(float(r[mean_key]))]
            ys = [float(r[mean_key]) for r in rows_l if math.isfinite(float(r[mean_key]))]
            sd = [float(r[std_key]) for r in rows_l if math.isfinite(float(r[mean_key]))]
            if not xs:
                continue
            label = _fmt_lambda(float(lam))
            ax.plot(xs, ys, marker="o", linewidth=1.25, label=label)
            ax.fill_between(
                xs,
                [y - s for y, s in zip(ys, sd)],
                [y + s for y, s in zip(ys, sd)],
                alpha=0.12,
            )

        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        fig.legend(
            uniq.values(),
            uniq.keys(),
            loc="upper center",
            ncol=min(6, len(uniq)),
            title="lambda_align",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_png)


def _run_one(
    *,
    config: str,
    seed: int,
    lam: float,
    lr: float,
    steps: int,
    eval_every_steps: int,
    out_json: Path,
    out_curve: Path,
    cuda_visible_devices: str,
    hf_home: str,
    retries: int,
    extra_set: List[str],
) -> Row:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run.py"),
        "--config",
        config,
        "--set",
        f"seed={seed}",
        "--set",
        "method.name=ours",
        "--set",
        f"method.ours.lambda_align={lam}",
        "--set",
        f"train.lr={lr}",
        "--set",
        f"train.steps={steps}",
        "--set",
        f"train.eval_every_steps={eval_every_steps}",
        "--out",
        str(out_json),
        "--curve_out",
        str(out_curve),
    ]
    for kv in extra_set:
        cmd.extend(["--set", kv])

    env = dict(os.environ)
    if cuda_visible_devices.strip():
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices.strip()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if hf_home.strip():
        env["HF_HOME"] = hf_home.strip()
        env["HF_DATASETS_CACHE"] = str(Path(hf_home.strip()) / "datasets")
        env["TRANSFORMERS_CACHE"] = str(Path(hf_home.strip()) / "transformers")

    print(
        f"[sweep] seed={seed} lambda={lam:.2e} lr={lr:.2e} "
        f"steps={steps} -> {out_json.name}"
    )
    for attempt in range(retries + 1):
        try:
            subprocess.run(cmd, env=env, check=True, cwd=str(ROOT))
            break
        except subprocess.CalledProcessError:
            if attempt >= retries:
                raise
            print(f"[sweep] run failed, retry {attempt + 1}/{retries}")
            time.sleep(2.0)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    best = float(payload["best_val_acc"])
    final = float(payload["final_val_acc"])
    score = 0.5 * best + 0.5 * final
    return Row(
        lambda_align=lam,
        seed=seed,
        best_val_acc=best,
        final_val_acc=final,
        score_05_05=score,
        summary_json=str(out_json),
        curve_csv=str(out_curve),
    )


def _write_rows(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "lambda_align",
                "seed",
                "best_val_acc",
                "final_val_acc",
                "score_05_05",
                "summary_json",
                "curve_csv",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.lambda_align,
                    r.seed,
                    r.best_val_acc,
                    r.final_val_acc,
                    r.score_05_05,
                    r.summary_json,
                    r.curve_csv,
                ]
            )


def _aggregate(rows: List[Row], lambdas: List[float], seeds: List[int]) -> List[Dict[str, float]]:
    by_key: Dict[tuple[float, int], Row] = {(r.lambda_align, r.seed): r for r in rows}
    out: List[Dict[str, float]] = []
    for lam in lambdas:
        vals_best: List[float] = []
        vals_final: List[float] = []
        vals_score: List[float] = []
        for sd in seeds:
            r = by_key.get((lam, sd))
            if r is None:
                continue
            vals_best.append(r.best_val_acc)
            vals_final.append(r.final_val_acc)
            vals_score.append(r.score_05_05)
        if not vals_score:
            continue
        out.append(
            {
                "lambda_align": lam,
                "n": float(len(vals_score)),
                "best_mean": float(statistics.fmean(vals_best)),
                "best_std": float(statistics.pstdev(vals_best) if len(vals_best) > 1 else 0.0),
                "final_mean": float(statistics.fmean(vals_final)),
                "final_std": float(statistics.pstdev(vals_final) if len(vals_final) > 1 else 0.0),
                "score_mean": float(statistics.fmean(vals_score)),
                "score_std": float(statistics.pstdev(vals_score) if len(vals_score) > 1 else 0.0),
            }
        )
    return out


def _write_agg(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "lambda_align",
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


def _plot(
    *,
    out_png: Path,
    lambdas: List[float],
    seeds: List[int],
    per_rows: List[Row],
    agg_rows: List[Dict[str, float]],
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    idx_map = {lam: i for i, lam in enumerate(lambdas)}
    x = np.arange(len(lambdas), dtype=np.float64)
    labels = [_fmt_lambda(l) for l in lambdas]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    specs = [
        ("best_val_acc", "best_mean", "best_std", "Val Max"),
        ("final_val_acc", "final_mean", "final_std", "Val Final"),
        ("score_05_05", "score_mean", "score_std", "Score 0.5/0.5"),
    ]

    by_lam = {float(r["lambda_align"]): r for r in agg_rows}
    for ax, (seed_key, mean_key, std_key, title) in zip(axes, specs):
        mu = np.array([float(by_lam.get(l, {}).get(mean_key, np.nan)) for l in lambdas], dtype=np.float64)
        sd = np.array([float(by_lam.get(l, {}).get(std_key, np.nan)) for l in lambdas], dtype=np.float64)
        ax.plot(x, mu, marker="o", label="mean")
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.2, label="mean±std")

        for sd_seed in seeds:
            ys = np.full((len(lambdas),), np.nan, dtype=np.float64)
            for r in per_rows:
                if r.seed == sd_seed:
                    ys[idx_map[r.lambda_align]] = float(getattr(r, seed_key))
            ax.plot(x, ys, alpha=0.35, linewidth=1.0, marker=".", label=f"seed{sd_seed}")

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("lambda_align")
        ax.grid(True, alpha=0.3)

    handles, labels_ = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels_, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=min(5, len(uniq)))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_png)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sweep ours.lambda_align at fixed lr/steps and report 0.5*max+0.5*final score."
    )
    p.add_argument("--config", type=str, default="configs/multitask_glue3_rte_mrpc_cola_real.yaml")
    p.add_argument("--out_dir", type=str, default="runs_lambda_sweep_50step")
    p.add_argument(
        "--lambdas",
        type=str,
        default="0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,1e0",
    )
    p.add_argument("--seeds", type=str, default="2,3,5")
    p.add_argument("--lr", type=float, default=2.0e-5)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--eval_every_steps", type=int, default=30)
    p.add_argument("--cuda_visible_devices", type=str, default="2")
    p.add_argument("--hf_home", type=str, default="", help="optional isolated HF cache root")
    p.add_argument("--retries", type=int, default=1, help="retry count for failed runs")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--set", action="append", default=[], help="extra key=value overrides")
    args = p.parse_args()

    lambdas = _parse_csv_floats(args.lambdas)
    seeds = _parse_csv_ints(args.seeds)
    if not lambdas:
        raise RuntimeError("empty --lambdas")
    if not seeds:
        raise RuntimeError("empty --seeds")

    out_dir = resolve_runs_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_home = str((out_dir / "_hf_cache").resolve()) if not str(args.hf_home).strip() else str(args.hf_home).strip()
    Path(hf_home).mkdir(parents=True, exist_ok=True)

    per_rows: List[Row] = []
    for lam in lambdas:
        lam_tag = _fmt_lambda(lam).replace("-", "m")
        lam_dir = out_dir / f"lam_{lam_tag}"
        lam_dir.mkdir(parents=True, exist_ok=True)
        for sd in seeds:
            out_json = lam_dir / f"ours_s{sd}.json"
            out_curve = lam_dir / f"ours_s{sd}_curve.csv"
            if args.skip_existing and out_json.exists() and out_curve.exists():
                payload = json.loads(out_json.read_text(encoding="utf-8"))
                best = float(payload["best_val_acc"])
                final = float(payload["final_val_acc"])
                per_rows.append(
                    Row(
                        lambda_align=lam,
                        seed=sd,
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
                seed=sd,
                lam=lam,
                lr=float(args.lr),
                steps=int(args.steps),
                eval_every_steps=int(args.eval_every_steps),
                out_json=out_json,
                out_curve=out_curve,
                cuda_visible_devices=args.cuda_visible_devices,
                hf_home=hf_home,
                retries=max(0, int(args.retries)),
                extra_set=list(args.set),
            )
            per_rows.append(row)

    per_csv = out_dir / "sweep_per_run.csv"
    agg_csv = out_dir / "sweep_agg.csv"
    fig_png = out_dir / "sweep_score_plot.png"
    best_json = out_dir / "best_lambda.json"
    trend_csv = out_dir / "sweep_trend_mean_std.csv"
    trend_png = out_dir / "sweep_trend_train_val.png"

    _write_rows(per_csv, per_rows)
    agg_rows = _aggregate(per_rows, lambdas=lambdas, seeds=seeds)
    _write_agg(agg_csv, agg_rows)
    _plot(out_png=fig_png, lambdas=lambdas, seeds=seeds, per_rows=per_rows, agg_rows=agg_rows)
    trend_rows = _collect_trend_rows(per_rows)
    _write_trend_csv(trend_csv, trend_rows)
    _plot_trend(out_png=trend_png, lambdas=lambdas, trend_rows=trend_rows)

    if not agg_rows:
        raise RuntimeError("no aggregate rows generated")
    best = max(agg_rows, key=lambda x: float(x["score_mean"]))
    best_json.write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")

    print(f"saved -> {per_csv}")
    print(f"saved -> {agg_csv}")
    print(f"saved -> {fig_png}")
    print(f"saved -> {trend_csv}")
    print(f"saved -> {trend_png}")
    print(f"saved -> {best_json}")


if __name__ == "__main__":
    main()
