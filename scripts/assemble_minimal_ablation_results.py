#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from path_utils import resolve_runs_path


ROOT = Path(__file__).resolve().parents[1]


def _parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).replace(" ", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _link_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError(f"missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            raise RuntimeError(f"target exists as directory: {dst}")
        try:
            if dst.resolve() == src.resolve():
                return
        except Exception:
            pass
        dst.unlink()
    dst.symlink_to(src)


def _merge_best_configs(
    *,
    reuse_dir: Path,
    compute_dir: Path,
    out_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    reuse_best = _read_json(reuse_dir / "hpo" / "best_configs.json")
    compute_best = _read_json(compute_dir / "hpo" / "best_configs.json")
    merged: Dict[str, Dict[str, Any]] = {}
    for method in ["baseline", "ours"]:
        if method not in reuse_best:
            raise RuntimeError(f"missing method={method} in {reuse_dir / 'hpo' / 'best_configs.json'}")
        merged[method] = dict(reuse_best[method])
        merged[method]["reused_from"] = str(reuse_dir)
    if "ablation" not in compute_best:
        raise RuntimeError(f"missing method=ablation in {compute_dir / 'hpo' / 'best_configs.json'}")
    merged["ablation"] = dict(compute_best["ablation"])
    merged["ablation"]["computed_from"] = str(compute_dir)

    hpo_out = out_dir / "hpo"
    hpo_out.mkdir(parents=True, exist_ok=True)
    (hpo_out / "best_configs.json").write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_hpo_best_params.py"),
            "--hpo_dir",
            str(hpo_out),
            "--out_csv",
            str(hpo_out / "hpo_best_params.csv"),
            "--out_png",
            str(hpo_out / "hpo_best_scores.png"),
        ],
        cwd=str(ROOT),
        check=True,
    )
    return merged


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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _build_final_tables(
    *,
    final_dir: Path,
    methods: List[str],
    seeds: List[int],
    best_configs: Dict[str, Dict[str, Any]],
) -> None:
    per_run_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for method in methods:
        grouped[method] = []
        for seed in seeds:
            summary_path = final_dir / f"{method}_s{seed}.json"
            curve_path = final_dir / f"{method}_s{seed}_curve.csv"
            payload = _read_json(summary_path)
            row = {
                "method": method,
                "seed": int(seed),
                "best_val_acc": float(payload["best_val_acc"]),
                "final_val_acc": float(payload["final_val_acc"]),
                "score_05_05": 0.5 * float(payload["best_val_acc"]) + 0.5 * float(payload["final_val_acc"]),
                "reused": method in {"baseline", "ours"},
                "summary_json": str(summary_path),
                "curve_csv": str(curve_path),
            }
            grouped[method].append(row)
            per_run_rows.append(row)
    _write_csv(final_dir / "final_per_run.csv", per_run_rows)

    agg_rows: List[Dict[str, Any]] = []
    for method in methods:
        rows = grouped[method]
        bests = [float(r["best_val_acc"]) for r in rows]
        finals = [float(r["final_val_acc"]) for r in rows]
        scores = [float(r["score_05_05"]) for r in rows]
        agg_rows.append(
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
    agg_rows.sort(key=lambda x: float(x["score_mean"]), reverse=True)
    _write_csv(final_dir / "final_agg.csv", agg_rows)
    (final_dir / "final_best_configs_snapshot.json").write_text(
        json.dumps(best_configs, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _run_plotters(*, final_dir: Path, seeds: List[int], skip_mvp: bool) -> None:
    methods = ["baseline", "ablation", "ours"]
    methods_csv = ",".join(methods)
    seeds_csv = ",".join(str(int(x)) for x in seeds)
    cmds = [
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
        ],
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
        ],
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
        ],
    ]
    if not skip_mvp:
        cmds.append(
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
    for cmd in cmds:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Assemble minimal ablation results by reusing main baseline/ours runs.")
    p.add_argument("--compute_dir", type=str, required=True)
    p.add_argument("--reuse_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--final_seeds", type=str, default="2,3,5,7,11")
    p.add_argument("--skip_mvp", action="store_true")
    args = p.parse_args()

    compute_dir = resolve_runs_path(args.compute_dir)
    reuse_dir = resolve_runs_path(args.reuse_dir)
    out_dir = resolve_runs_path(args.out_dir)
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_csv_ints(args.final_seeds)
    if not seeds:
        raise RuntimeError("empty --final_seeds")

    for method in ["baseline", "ours"]:
        for seed in seeds:
            _link_file(reuse_dir / "final" / f"{method}_s{seed}.json", final_dir / f"{method}_s{seed}.json")
            _link_file(
                reuse_dir / "final" / f"{method}_s{seed}_curve.csv",
                final_dir / f"{method}_s{seed}_curve.csv",
            )

    for seed in seeds:
        _link_file(compute_dir / "final" / f"ablation_s{seed}.json", final_dir / f"ablation_s{seed}.json")
        _link_file(
            compute_dir / "final" / f"ablation_s{seed}_curve.csv",
            final_dir / f"ablation_s{seed}_curve.csv",
        )

    best_configs = _merge_best_configs(reuse_dir=reuse_dir, compute_dir=compute_dir, out_dir=out_dir)
    _build_final_tables(final_dir=final_dir, methods=["baseline", "ablation", "ours"], seeds=seeds, best_configs=best_configs)

    manifest = {
        "compute_dir": str(compute_dir),
        "reuse_dir": str(reuse_dir),
        "out_dir": str(out_dir),
        "reused_methods": ["baseline", "ours"],
        "computed_methods": ["ablation"],
        "final_seeds": seeds,
    }
    (out_dir / "reuse_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _run_plotters(final_dir=final_dir, seeds=seeds, skip_mvp=bool(args.skip_mvp))


if __name__ == "__main__":
    main()
