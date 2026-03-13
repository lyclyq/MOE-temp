#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_key(name: str) -> str:
    out: List[str] = []
    prev_us = False
    for ch in str(name).strip().lower():
        keep = ch.isalnum()
        if keep:
            out.append(ch)
            prev_us = False
            continue
        if not prev_us:
            out.append("_")
            prev_us = True
    return "".join(out).strip("_") or "task"


def _parse_curve(path: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "step" not in row:
                continue
            step = int(float(row["step"]))
            vals: Dict[str, float] = {}
            for key, val in row.items():
                if val is None or val == "":
                    continue
                try:
                    vals[key] = float(val)
                except ValueError:
                    continue
            out[step] = vals
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _band(arr: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    mu = arr.mean(axis=0)
    if mode == "minmax":
        return arr.min(axis=0), arr.max(axis=0)
    sd = arr.std(axis=0)
    return mu - sd, mu + sd


def _common_steps(
    curves: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    methods: Sequence[str],
    seeds: Sequence[int],
    required_keys: Sequence[str],
) -> List[int]:
    step_sets: List[set[int]] = []
    for method in methods:
        for seed in seeds:
            rows = curves.get(method, {}).get(seed, {})
            ok = {step for step, vals in rows.items() if all(k in vals for k in required_keys)}
            if not ok:
                return []
            step_sets.append(ok)
    if not step_sets:
        return []
    return sorted(set.intersection(*step_sets))


def _series_matrix(
    curves: Dict[int, Dict[int, Dict[str, float]]],
    seeds: Sequence[int],
    steps: Sequence[int],
    value_fn: Callable[[Dict[str, float]], float],
) -> np.ndarray:
    return np.array(
        [[value_fn(curves[seed][step]) for step in steps] for seed in seeds],
        dtype=np.float64,
    )


def _save_missing_plot(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=150)
    ax.axis("off")
    ax.text(0.01, 0.60, title, fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.01, 0.32, message, fontsize=10, transform=ax.transAxes)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


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


def _summary_stats(vals: Sequence[float]) -> tuple[float, float]:
    arr = np.array(list(vals), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std())


def _plot_curve_metric(
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    curves: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    methods: Sequence[str],
    seeds: Sequence[int],
    metric_key: str,
    band_mode: str,
) -> Dict[str, Dict[str, float]]:
    steps = _common_steps(curves, methods, seeds, [metric_key])
    if not steps:
        _save_missing_plot(out_path, title, f"Missing metric {metric_key!r} in one or more curve CSV files.")
        return {}
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
    summary: Dict[str, Dict[str, float]] = {}
    for method in methods:
        arr = _series_matrix(curves[method], seeds, steps, lambda row: float(row[metric_key]))
        mu = arr.mean(axis=0)
        lo, hi = _band(arr, band_mode)
        ax.plot(steps, mu, label=method)
        ax.fill_between(steps, lo, hi, alpha=0.2)
        summary[method] = {
            "last_mean": float(mu[-1]),
            "last_std": float(arr.std(axis=0)[-1]),
        }
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return summary


def _task_name_map(summaries: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for method_map in summaries.values():
        for payload in method_map.values():
            diag = payload.get("final_diagnostics", {})
            for task_name in diag.get("task_names", []) or []:
                out[_safe_key(task_name)] = str(task_name)
            repro = payload.get("data_reproducibility", {})
            for task_name in repro.get("task_order", []) or []:
                out[_safe_key(task_name)] = str(task_name)
    return out


def _matrix_seed_mean(
    summaries: Dict[str, Dict[int, Dict[str, Any]]],
    method: str,
    seeds: Sequence[int],
    key: str,
) -> np.ndarray | None:
    mats: List[np.ndarray] = []
    for seed in seeds:
        payload = summaries.get(method, {}).get(seed)
        if not payload:
            continue
        diag = payload.get("final_diagnostics", {})
        mat = diag.get(key)
        if not isinstance(mat, list) or not mat:
            continue
        mats.append(np.array(mat, dtype=np.float64))
    if not mats:
        return None
    return np.stack(mats, axis=0).mean(axis=0)


def _vector_seed_stats(
    summaries: Dict[str, Dict[int, Dict[str, Any]]],
    method: str,
    seeds: Sequence[int],
    key: str,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    vals: List[np.ndarray] = []
    for seed in seeds:
        payload = summaries.get(method, {}).get(seed)
        if not payload:
            continue
        diag = payload.get("final_diagnostics", {})
        vec = diag.get(key)
        if not isinstance(vec, list) or not vec:
            continue
        vals.append(np.array(vec, dtype=np.float64))
    if not vals:
        return None, None
    arr = np.stack(vals, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


def _plot_method_heatmaps(
    *,
    out_path: Path,
    title: str,
    methods: Sequence[str],
    seeds: Sequence[int],
    summaries: Dict[str, Dict[int, Dict[str, Any]]],
    matrix_key: str,
    labels: Sequence[str] | None,
    csv_path: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods) + 0.8, 4.6), dpi=150)
    if len(methods) == 1:
        axes = [axes]

    missing = False
    im = None
    for ax, method in zip(axes, methods):
        mat = _matrix_seed_mean(summaries, method, seeds, matrix_key)
        if mat is None:
            missing = True
            ax.axis("off")
            ax.text(0.05, 0.5, f"{method}\nmissing {matrix_key}", transform=ax.transAxes)
            continue
        im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        tick_labels = list(labels) if labels else [str(i) for i in range(int(mat.shape[0]))]
        ax.set_xticks(range(len(tick_labels)))
        ax.set_yticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=35, ha="right")
        ax.set_yticklabels(tick_labels)
        ax.set_title(method)
        for i, row_name in enumerate(tick_labels):
            for j, col_name in enumerate(tick_labels):
                rows.append(
                    {
                        "method": method,
                        "row": row_name,
                        "col": col_name,
                        "value": float(mat[i, j]),
                    }
                )

    if im is not None:
        fig.subplots_adjust(left=0.08, right=0.90, bottom=0.16, top=0.82, wspace=0.40)
        cax = fig.add_axes([0.92, 0.22, 0.012, 0.58])
        fig.colorbar(im, cax=cax)
    else:
        fig.subplots_adjust(left=0.08, right=0.90, bottom=0.16, top=0.82, wspace=0.40)
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    if missing and not rows:
        _save_missing_plot(out_path, title, f"Missing {matrix_key!r} in final summaries.")
    _write_csv(csv_path, rows)


def _plot_expert_purity(
    *,
    out_path: Path,
    csv_path: Path,
    methods: Sequence[str],
    seeds: Sequence[int],
    summaries: Dict[str, Dict[int, Dict[str, Any]]],
) -> None:
    rows: List[Dict[str, Any]] = []
    series: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in methods:
        mu, sd = _vector_seed_stats(summaries, method, seeds, "expert_purity")
        if mu is None or sd is None:
            continue
        series[method] = (mu, sd)
        for idx, val in enumerate(mu):
            rows.append(
                {
                    "method": method,
                    "expert": f"e{idx}",
                    "purity_mean": float(val),
                    "purity_std": float(sd[idx]),
                }
            )
    _write_csv(csv_path, rows)
    if not series:
        _save_missing_plot(out_path, "Expert Purity", "Missing expert_purity vectors in final summaries.")
        return

    first = next(iter(series.values()))[0]
    x = np.arange(len(first))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(1, 1, figsize=(9, 4), dpi=150)
    items = list(series.items())
    for i, (method, (mu, sd)) in enumerate(items):
        off = (i - (len(items) - 1) / 2.0) * width
        ax.bar(x + off, mu, width=width, yerr=sd, capsize=3, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([f"e{i}" for i in range(len(first))])
    ax.set_xlabel("expert")
    ax.set_ylabel("purity")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Expert Purity (Final, seed mean)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _build_task_metric_rows(
    summaries: Dict[str, Dict[int, Dict[str, Any]]],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    task_name_map = _task_name_map(summaries)
    all_task_keys: List[str] = []
    seen: set[str] = set()
    for method in methods:
        for seed in seeds:
            payload = summaries.get(method, {}).get(seed, {})
            final_step = payload.get("final_step_metrics", {})
            for key in final_step.keys():
                if not key.startswith("task_acc__"):
                    continue
                tok = key.split("__", 1)[1]
                if tok in seen:
                    continue
                seen.add(tok)
                all_task_keys.append(tok)

    rows: List[Dict[str, Any]] = []
    for method in methods:
        for task_key in all_task_keys:
            acc_vals: List[float] = []
            loss_vals: List[float] = []
            for seed in seeds:
                payload = summaries.get(method, {}).get(seed, {})
                final_step = payload.get("final_step_metrics", {})
                acc = final_step.get(f"task_acc__{task_key}")
                loss = final_step.get(f"task_loss__{task_key}")
                if isinstance(acc, (float, int)):
                    acc_vals.append(float(acc))
                if isinstance(loss, (float, int)):
                    loss_vals.append(float(loss))
            acc_mean, acc_std = _summary_stats(acc_vals)
            loss_mean, loss_std = _summary_stats(loss_vals)
            rows.append(
                {
                    "method": method,
                    "task_key": task_key,
                    "task_name": task_name_map.get(task_key, task_key),
                    "n_seeds": len(acc_vals),
                    "task_acc_mean": acc_mean,
                    "task_acc_std": acc_std,
                    "task_loss_mean": loss_mean,
                    "task_loss_std": loss_std,
                }
            )
    return rows


def _build_mechanism_rows(
    summaries: Dict[str, Dict[int, Dict[str, Any]]],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    metric_keys = [
        "val_acc",
        "overall_task_conflict_cos",
        "avg_intra_expert_coherence",
        "expert_purity_mean",
        "expert_diversity_mean",
        "load_cv",
        "load_max_min_ratio",
        "utilization_ratio",
        "router_entropy",
        "inter_expert_similarity",
        "task_grad_norm_cv",
    ]
    rows: List[Dict[str, Any]] = []
    for method in methods:
        row: Dict[str, Any] = {"method": method}
        for key in metric_keys:
            vals: List[float] = []
            for seed in seeds:
                payload = summaries.get(method, {}).get(seed, {})
                final_step = payload.get("final_step_metrics", {})
                val = final_step.get(key)
                if isinstance(val, (float, int)):
                    vals.append(float(val))
            mean, std = _summary_stats(vals)
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        rows.append(row)
    return rows


def _build_overhead_rows(
    summaries: Dict[str, Dict[int, Dict[str, Any]]],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    metric_keys = ["wall_time_sec", "mean_train_step_time_sec", "peak_cuda_memory_mb"]
    rows: List[Dict[str, Any]] = []
    for method in methods:
        row: Dict[str, Any] = {"method": method}
        for key in metric_keys:
            vals: List[float] = []
            for seed in seeds:
                payload = summaries.get(method, {}).get(seed, {})
                overhead = payload.get("overhead", {})
                val = overhead.get(key)
                if isinstance(val, (float, int)):
                    vals.append(float(val))
            mean, std = _summary_stats(vals)
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-style plots and CSV summaries from final runs.")
    parser.add_argument("--final_dir", type=str, required=True)
    parser.add_argument("--methods", type=str, default="baseline,cagrad,ours")
    parser.add_argument("--seeds", type=str, default="2,3,5")
    parser.add_argument("--band", type=str, default="std", choices=["std", "minmax"])
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    final_dir = Path(args.final_dir)
    out_dir = Path(args.out_dir) if args.out_dir else final_dir
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(tok.strip()) for tok in args.seeds.split(",") if tok.strip()]

    curves: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = {m: {} for m in methods}
    summaries: Dict[str, Dict[int, Dict[str, Any]]] = {m: {} for m in methods}
    for method in methods:
        for seed in seeds:
            curve_path = final_dir / f"{method}_s{seed}_curve.csv"
            summary_path = final_dir / f"{method}_s{seed}.json"
            if curve_path.exists():
                curves[method][seed] = _parse_curve(curve_path)
            if summary_path.exists():
                summaries[method][seed] = _read_json(summary_path)

    manifest: Dict[str, Any] = {"methods": methods, "seeds": seeds, "plots": [], "csvs": []}

    curve_specs = [
        ("val_acc", "Validation Score", "validation accuracy", "curve_validation_score.png"),
        (
            "avg_intra_expert_coherence",
            "Intra-expert Coherence",
            "mean cosine",
            "curve_intra_expert_coherence.png",
        ),
        ("load_cv", "Load Imbalance (CV)", "coefficient of variation", "curve_load_cv.png"),
        ("router_entropy", "Routing Entropy", "entropy", "curve_routing_entropy.png"),
    ]
    curve_manifest: Dict[str, Any] = {}
    for key, title, ylabel, fname in curve_specs:
        out_path = out_dir / fname
        summary = _plot_curve_metric(
            out_path=out_path,
            title=title,
            ylabel=ylabel,
            curves=curves,
            methods=methods,
            seeds=seeds,
            metric_key=key,
            band_mode=args.band,
        )
        curve_manifest[key] = summary
        manifest["plots"].append(str(out_path))

    task_labels: List[str] | None = None
    for method in methods:
        for seed in seeds:
            diag = summaries.get(method, {}).get(seed, {}).get("final_diagnostics", {})
            task_names = diag.get("task_names")
            if isinstance(task_names, list) and task_names:
                task_labels = [str(x) for x in task_names]
                break
        if task_labels:
            break

    task_conf_png = out_dir / "task_conflict_matrix_final.png"
    task_conf_csv = out_dir / "task_conflict_matrix_final.csv"
    _plot_method_heatmaps(
        out_path=task_conf_png,
        title="Task Conflict Matrix (Final, seed mean)",
        methods=methods,
        seeds=seeds,
        summaries=summaries,
        matrix_key="task_conflict_matrix",
        labels=task_labels,
        csv_path=task_conf_csv,
    )
    manifest["plots"].append(str(task_conf_png))
    manifest["csvs"].append(str(task_conf_csv))

    expert_labels: List[str] | None = None
    for method in methods:
        for seed in seeds:
            diag = summaries.get(method, {}).get(seed, {}).get("final_diagnostics", {})
            names = diag.get("expert_names")
            if isinstance(names, list) and names:
                expert_labels = [str(x) for x in names]
                break
        if expert_labels:
            break

    inter_png = out_dir / "inter_expert_similarity_final.png"
    inter_csv = out_dir / "inter_expert_similarity_final.csv"
    _plot_method_heatmaps(
        out_path=inter_png,
        title="Inter-expert Similarity (Final, seed mean)",
        methods=methods,
        seeds=seeds,
        summaries=summaries,
        matrix_key="inter_expert_similarity_matrix",
        labels=expert_labels,
        csv_path=inter_csv,
    )
    manifest["plots"].append(str(inter_png))
    manifest["csvs"].append(str(inter_csv))

    purity_png = out_dir / "expert_purity_final.png"
    purity_csv = out_dir / "expert_purity_final.csv"
    _plot_expert_purity(
        out_path=purity_png,
        csv_path=purity_csv,
        methods=methods,
        seeds=seeds,
        summaries=summaries,
    )
    manifest["plots"].append(str(purity_png))
    manifest["csvs"].append(str(purity_csv))

    task_metric_rows = _build_task_metric_rows(summaries, methods, seeds)
    task_metric_csv = out_dir / "task_metrics_final.csv"
    _write_csv(task_metric_csv, task_metric_rows)
    manifest["csvs"].append(str(task_metric_csv))

    mechanism_rows = _build_mechanism_rows(summaries, methods, seeds)
    mechanism_csv = out_dir / "mechanism_summary.csv"
    _write_csv(mechanism_csv, mechanism_rows)
    manifest["csvs"].append(str(mechanism_csv))

    overhead_rows = _build_overhead_rows(summaries, methods, seeds)
    overhead_csv = out_dir / "overhead_summary.csv"
    _write_csv(overhead_csv, overhead_rows)
    manifest["csvs"].append(str(overhead_csv))

    manifest["curve_summary"] = curve_manifest
    manifest_path = out_dir / "paper_metrics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
