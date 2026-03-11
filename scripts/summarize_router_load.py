#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize router load columns from curve csv files.")
    p.add_argument("--runs_dir", type=str, required=True)
    p.add_argument("--methods", type=str, default="ours,baseline_avg,baseline_cagrad")
    p.add_argument("--seeds", type=str, default="2,3,5")
    p.add_argument("--out_csv", type=str, default="")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]

    out_rows: List[Dict[str, str]] = []
    all_load_cols: List[str] = []

    for method in methods:
        for seed in seeds:
            path = runs_dir / f"{method}_s{seed}_curve.csv"
            if not path.exists():
                continue
            rows = _read_rows(path)
            if not rows:
                continue
            last = rows[-1]
            load_cols = sorted([k for k in last.keys() if k.startswith("router_load_e")])
            for k in load_cols:
                if k not in all_load_cols:
                    all_load_cols.append(k)
            row = {
                "method": method,
                "seed": str(seed),
                "step": str(last.get("step", "")),
                "router_entropy": str(last.get("router_entropy", "")),
            }
            for k in load_cols:
                row[k] = str(last.get(k, ""))
            out_rows.append(row)

    all_load_cols = sorted(all_load_cols)
    fields = ["method", "seed", "step", "router_entropy"] + all_load_cols

    out_csv = Path(args.out_csv) if args.out_csv else runs_dir / "router_load_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: row.get(k, "") for k in fields})

    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
