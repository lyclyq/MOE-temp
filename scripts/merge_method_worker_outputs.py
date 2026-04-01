#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
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


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _run_hpo_exports(hpo_root: Path) -> None:
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_hpo_best_params.py"),
            "--hpo_dir",
            str(hpo_root),
            "--out_csv",
            str(hpo_root / "hpo_best_params.csv"),
            "--out_png",
            str(hpo_root / "hpo_best_scores.png"),
        ]
    )


def _run_plotters(*, final_dir: Path, methods: Sequence[str], final_seeds: Sequence[int], skip_mvp: bool) -> None:
    methods_csv = ",".join(methods)
    seeds_csv = ",".join(str(int(x)) for x in final_seeds)

    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_seed_mean_band.py"),
            "--runs_dir",
            str(final_dir),
            "--methods",
            methods_csv,
            "--seeds",
            seeds_csv,
            "--band",
            "std",
            "--out",
            str(final_dir / "seed_mean_band_std.png"),
            "--summary_out",
            str(final_dir / "seed_mean_band_std_summary.json"),
            "--val_table_out",
            str(final_dir / "seed_mean_band_val_last.csv"),
        ]
    )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_router_load.py"),
            "--runs_dir",
            str(final_dir),
            "--methods",
            methods_csv,
            "--seeds",
            seeds_csv,
            "--out_csv",
            str(final_dir / "router_load_summary.csv"),
        ]
    )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_paper_metrics.py"),
            "--final_dir",
            str(final_dir),
            "--methods",
            methods_csv,
            "--seeds",
            seeds_csv,
            "--band",
            "std",
            "--out_dir",
            str(final_dir),
        ]
    )
    if skip_mvp:
        return
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_mvp_12pack.py"),
            "--runs_dir",
            str(final_dir),
            "--methods",
            methods_csv,
            "--seeds",
            seeds_csv,
            "--band",
            "std",
            "--out_dir",
            str(final_dir),
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Merge per-method worker outputs back into a standard pipeline output directory.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--worker_root", type=str, required=True)
    p.add_argument("--methods", type=str, required=True)
    p.add_argument("--final_seeds", type=str, required=True)
    p.add_argument("--skip_mvp", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    worker_root = Path(args.worker_root)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    final_seeds = [int(x.strip()) for x in args.final_seeds.split(",") if x.strip()]

    hpo_root = out_dir / "hpo"
    final_root = out_dir / "final"
    if hpo_root.exists():
        shutil.rmtree(hpo_root)
    if final_root.exists():
        shutil.rmtree(final_root)
    hpo_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    merged_best_cfg: Dict[str, Any] = {}
    merged_hpo_rows: List[Dict[str, Any]] = []
    final_per_run: List[Dict[str, Any]] = []

    for method in methods:
        worker_dir = worker_root / method
        if not worker_dir.exists():
            raise RuntimeError(f"missing worker output for method={method}: {worker_dir}")

        worker_hpo_root = worker_dir / "hpo"
        worker_final_root = worker_dir / "final"
        shutil.copytree(worker_hpo_root / method, hpo_root / method, dirs_exist_ok=True)

        best_cfg = _read_json(worker_hpo_root / "best_configs.json")
        if method not in best_cfg:
            raise RuntimeError(f"worker best_configs.json missing method={method}: {worker_hpo_root / 'best_configs.json'}")
        merged_best_cfg[method] = best_cfg[method]
        merged_hpo_rows.extend(_read_csv(worker_hpo_root / "hpo_agg_all_methods.csv"))

        for path in worker_final_root.glob(f"{method}_s*"):
            if path.is_file():
                shutil.copy2(path, final_root / path.name)

        for row in _read_csv(worker_final_root / "final_per_run.csv"):
            if str(row.get("method", "")) != method:
                continue
            seed = str(row.get("seed", "")).strip()
            row["summary_json"] = str(final_root / f"{method}_s{seed}.json")
            row["curve_csv"] = str(final_root / f"{method}_s{seed}_curve.csv")
            final_per_run.append(dict(row))

    _write_csv(hpo_root / "hpo_agg_all_methods.csv", merged_hpo_rows)
    (hpo_root / "best_configs.json").write_text(json.dumps(merged_best_cfg, indent=2, sort_keys=True), encoding="utf-8")
    _run_hpo_exports(hpo_root)

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in final_per_run:
        by_method.setdefault(str(row["method"]), []).append(row)

    final_agg_rows: List[Dict[str, Any]] = []
    for method, rows in sorted(by_method.items(), key=lambda kv: kv[0]):
        bests = [float(r["best_val_acc"]) for r in rows]
        finals = [float(r["final_val_acc"]) for r in rows]
        scores = [float(r["score_05_05"]) for r in rows]
        final_agg_rows.append(
            {
                "method": method,
                "n_seeds": len(rows),
                "best_mean": float(statistics.fmean(bests)),
                "best_std": float(statistics.pstdev(bests) if len(bests) > 1 else 0.0),
                "final_mean": float(statistics.fmean(finals)),
                "final_std": float(statistics.pstdev(finals) if len(finals) > 1 else 0.0),
                "score_mean": float(statistics.fmean(scores)),
                "score_std": float(statistics.pstdev(scores) if len(scores) > 1 else 0.0),
            }
        )
    final_agg_rows.sort(key=lambda row: float(row["score_mean"]), reverse=True)
    _write_csv(final_root / "final_per_run.csv", final_per_run)
    _write_csv(final_root / "final_agg.csv", final_agg_rows)
    (final_root / "final_best_configs_snapshot.json").write_text(
        json.dumps(merged_best_cfg, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _run_plotters(final_dir=final_root, methods=methods, final_seeds=final_seeds, skip_mvp=bool(args.skip_mvp))

    manifest = {
        "out_dir": str(out_dir),
        "worker_root": str(worker_root),
        "methods": methods,
        "final_seeds": final_seeds,
    }
    (out_dir / "method_parallel_merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[merge] done -> {out_dir}")


if __name__ == "__main__":
    main()
