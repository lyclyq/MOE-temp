#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _parse_csv(path: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            step = int(float(row["step"]))
            out[step] = {
                "train_step_loss": float(row["train_step_loss"]),
                "train_acc": float(row["train_acc"]),
                "train_loss": float(row["train_loss"]),
                "val_acc": float(row["val_acc"]),
                "val_loss": float(row["val_loss"]),
            }
    return out


def _band(arr: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    # arr: [S, T]
    mu = arr.mean(axis=0)
    if mode == "minmax":
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
    else:
        sd = arr.std(axis=0)
        lo = mu - sd
        hi = mu + sd
    return lo, hi


def main() -> None:
    p = argparse.ArgumentParser(description="Plot seed-mean curves with shaded band.")
    p.add_argument("--runs_dir", type=str, required=True, help="directory containing *_curve.csv files")
    p.add_argument("--methods", type=str, default="ours,baseline_avg,baseline_cagrad")
    p.add_argument("--seeds", type=str, default="2,3,5")
    p.add_argument("--band", type=str, default="std", choices=["std", "minmax"])
    p.add_argument("--out", type=str, default="seed_mean_band.png")
    p.add_argument("--val_out", type=str, default="", help="optional val-only figure path")
    p.add_argument("--summary_out", type=str, default="seed_mean_band_summary.json")
    p.add_argument("--val_table_out", type=str, default="", help="optional CSV table for final val comparison")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    metric_names = ["val_acc", "val_loss", "train_acc", "train_loss"]
    titles = {
        "val_acc": "Validation Accuracy",
        "val_loss": "Validation Loss",
        "train_acc": "Train Accuracy",
        "train_loss": "Train Loss",
    }

    data: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = {}
    for method in methods:
        data[method] = {}
        for sd in seeds:
            path = runs_dir / f"{method}_s{sd}_curve.csv"
            if not path.exists():
                raise RuntimeError(f"missing curve file: {path}")
            data[method][sd] = _parse_csv(path)

    summary: Dict[str, Dict[str, object]] = {}
    curves: Dict[str, Dict[str, object]] = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    ax_map = {
        "val_acc": axes[0, 0],
        "val_loss": axes[0, 1],
        "train_acc": axes[1, 0],
        "train_loss": axes[1, 1],
    }

    for method in methods:
        step_sets = [set(data[method][sd].keys()) for sd in seeds]
        common_steps = sorted(list(set.intersection(*step_sets))) if step_sets else []
        if not common_steps:
            raise RuntimeError(f"no common steps across seeds for method={method}")

        summary[method] = {
            "seeds": seeds,
            "num_common_steps": len(common_steps),
            "steps": common_steps,
        }
        curves[method] = {}

        for metric in metric_names:
            arr = np.array(
                [[data[method][sd][st][metric] for st in common_steps] for sd in seeds],
                dtype=np.float64,
            )
            mu = arr.mean(axis=0)
            lo, hi = _band(arr, args.band)

            ax = ax_map[metric]
            ax.plot(common_steps, mu, label=method)
            ax.fill_between(common_steps, lo, hi, alpha=0.2)
            curves[method][metric] = {
                "steps": common_steps,
                "mean": mu,
                "lo": lo,
                "hi": hi,
            }

            summary[method][f"{metric}_last_mean"] = float(mu[-1])
            summary[method][f"{metric}_last_std"] = float(arr.std(axis=0)[-1])

    for metric in metric_names:
        ax = ax_map[metric]
        ax.set_title(titles[metric])
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    ax_map["val_acc"].legend()
    fig.suptitle(f"Seed Mean Curves with {args.band} Band")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)

    val_out_path = Path(args.val_out) if args.val_out else out_path.with_name(f"{out_path.stem}_val_only{out_path.suffix}")
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    for method in methods:
        for j, metric in enumerate(["val_acc", "val_loss"]):
            c = curves[method][metric]
            steps_arr = c["steps"]
            mu = c["mean"]
            lo = c["lo"]
            hi = c["hi"]
            ax = axes2[j]
            ax.plot(steps_arr, mu, label=method)
            ax.fill_between(steps_arr, lo, hi, alpha=0.2)
    axes2[0].set_title("Validation Accuracy")
    axes2[1].set_title("Validation Loss")
    for ax in axes2:
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    axes2[0].legend()
    fig2.suptitle(f"Validation Curves with {args.band} Band")
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    val_out_path.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(val_out_path)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    val_table_path = Path(args.val_table_out) if args.val_table_out else out_path.with_name(f"{out_path.stem}_val_last.csv")
    val_table_path.parent.mkdir(parents=True, exist_ok=True)
    with val_table_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "val_acc_last_mean", "val_acc_last_std", "val_loss_last_mean", "val_loss_last_std"])
        for method in methods:
            s = summary[method]
            w.writerow(
                [
                    method,
                    s["val_acc_last_mean"],
                    s["val_acc_last_std"],
                    s["val_loss_last_mean"],
                    s["val_loss_last_std"],
                ]
            )

    print(f"saved plot -> {out_path}")
    print(f"saved val-only plot -> {val_out_path}")
    print(f"saved summary -> {summary_path}")
    print(f"saved val table -> {val_table_path}")


if __name__ == "__main__":
    main()
