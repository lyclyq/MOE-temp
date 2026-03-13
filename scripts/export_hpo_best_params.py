#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HPO best params to CSV and a simple score plot.")
    parser.add_argument("--hpo_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_png", type=str, default="")
    args = parser.parse_args()

    hpo_dir = Path(args.hpo_dir)
    best_path = hpo_dir / "best_configs.json"
    if not best_path.exists():
        raise RuntimeError(f"missing best_configs.json: {best_path}")
    best_cfg = _read_json(best_path)

    rows: List[Dict[str, Any]] = []
    methods: List[str] = []
    scores: List[float] = []
    for method, payload in sorted(best_cfg.items(), key=lambda kv: kv[0]):
        row: Dict[str, Any] = {
            "method": method,
            "method_alias": payload.get("method_alias", method),
            "method_name": payload.get("method_name", ""),
            "candidate_id": payload.get("candidate_id", ""),
            "score_mean": float(payload.get("score_mean", 0.0)),
            "score_std": float(payload.get("score_std", 0.0)),
        }
        for key, val in sorted(dict(payload.get("params", {})).items()):
            row[f"param::{key}"] = val
        for key, val in sorted(dict(payload.get("fixed_overrides", {})).items()):
            row[f"fixed::{key}"] = val
        if "reused_from" in payload:
            row["reused_from"] = payload["reused_from"]
        if "computed_from" in payload:
            row["computed_from"] = payload["computed_from"]
        rows.append(row)
        methods.append(method)
        scores.append(float(payload.get("score_mean", 0.0)))

    out_csv = Path(args.out_csv) if args.out_csv else hpo_dir / "hpo_best_params.csv"
    _write_csv(out_csv, rows)

    out_png = Path(args.out_png) if args.out_png else hpo_dir / "hpo_best_scores.png"
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=150)
    x = np.arange(len(methods))
    ax.bar(x, scores, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("HPO best score")
    ax.set_title("Best HPO Candidate by Method")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    print(f"saved csv -> {out_csv}")
    print(f"saved plot -> {out_png}")


if __name__ == "__main__":
    main()
