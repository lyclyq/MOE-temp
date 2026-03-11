#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _fmt_lambda(x: float) -> str:
    return f"{x:.1e}".replace("+0", "").replace("+", "")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot val acc mean comparison from sweep_agg.csv.")
    p.add_argument("--agg_csv", type=str, required=True, help="path to sweep_agg.csv")
    p.add_argument("--out", type=str, required=True, help="output png path")
    args = p.parse_args()

    agg_csv = Path(args.agg_csv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with agg_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "lambda_align": float(row["lambda_align"]),
                    "best_mean": float(row["best_mean"]),
                    "best_std": float(row["best_std"]),
                    "final_mean": float(row["final_mean"]),
                    "final_std": float(row["final_std"]),
                }
            )

    if not rows:
        raise RuntimeError(f"no rows in {agg_csv}")

    rows.sort(key=lambda r: r["lambda_align"])
    x = list(range(len(rows)))
    x_labels = [_fmt_lambda(r["lambda_align"]) for r in rows]

    best_mean = [r["best_mean"] for r in rows]
    best_std = [r["best_std"] for r in rows]
    final_mean = [r["final_mean"] for r in rows]
    final_std = [r["final_std"] for r in rows]

    plt.figure(figsize=(10, 4), dpi=150)
    plt.plot(x, best_mean, marker="o", label="val_max_mean")
    plt.fill_between(
        x,
        [m - s for m, s in zip(best_mean, best_std)],
        [m + s for m, s in zip(best_mean, best_std)],
        alpha=0.2,
    )

    plt.plot(x, final_mean, marker="s", label="val_final_mean")
    plt.fill_between(
        x,
        [m - s for m, s in zip(final_mean, final_std)],
        [m + s for m, s in zip(final_mean, final_std)],
        alpha=0.2,
    )

    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.xlabel("lambda_align")
    plt.ylabel("val_acc")
    plt.title("Lambda Sweep: Val Acc Mean Compare")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
