#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_curve(path: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "step" not in row:
                continue
            step = int(float(row["step"]))
            vals: Dict[str, float] = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    vals[k] = float(v)
                except ValueError:
                    continue
            out[step] = vals
    return out


def _band(arr: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    mu = arr.mean(axis=0)
    if mode == "minmax":
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
    else:
        sd = arr.std(axis=0)
        lo = mu - sd
        hi = mu + sd
    return lo, hi


def _common_steps(
    curves: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    methods: List[str],
    seeds: List[int],
    required_keys: List[str],
) -> List[int]:
    step_sets: List[set[int]] = []
    for method in methods:
        for seed in seeds:
            rows = curves[method][seed]
            ok_steps = {st for st, vals in rows.items() if all(k in vals for k in required_keys)}
            step_sets.append(ok_steps)
    if not step_sets:
        return []
    return sorted(set.intersection(*step_sets))


def _series_matrix(
    curves: Dict[int, Dict[int, Dict[str, float]]],
    seeds: List[int],
    steps: List[int],
    value_fn: Callable[[Dict[str, float]], float],
) -> np.ndarray:
    return np.array(
        [[value_fn(curves[sd][st]) for st in steps] for sd in seeds],
        dtype=np.float64,
    )


def _save_missing_plot(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=150)
    ax.axis("off")
    ax.text(0.01, 0.55, title, fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.01, 0.30, message, fontsize=10, transform=ax.transAxes)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_metric(
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    curves: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    methods: List[str],
    seeds: List[int],
    steps: List[int],
    value_fn: Callable[[Dict[str, float]], float],
    band_mode: str,
) -> Dict[str, Dict[str, float]]:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
    summary: Dict[str, Dict[str, float]] = {}
    for method in methods:
        arr = _series_matrix(curves[method], seeds, steps, value_fn)
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


def _plot_lconflict_effect(
    *,
    out_path: Path,
    curves: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    seeds: List[int],
    band_mode: str,
) -> bool:
    required = ["l_conflict", "router_conflict_grad_l2"]
    steps = _common_steps(curves, ["ours"], seeds, required)
    if not steps:
        _save_missing_plot(
            out_path,
            "OURS L_conflict -> Router Effect",
            "Missing l_conflict/router_conflict_grad_l2 columns. Rerun training with updated trainer.",
        )
        return False

    arr_lc = _series_matrix(curves["ours"], seeds, steps, lambda r: -float(r["l_conflict"]))
    arr_gn = _series_matrix(curves["ours"], seeds, steps, lambda r: float(r["router_conflict_grad_l2"]))
    mu_lc = arr_lc.mean(axis=0)
    lo_lc, hi_lc = _band(arr_lc, band_mode)
    mu_gn = arr_gn.mean(axis=0)
    lo_gn, hi_gn = _band(arr_gn, band_mode)

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 4), dpi=150)
    ax2 = ax1.twinx()
    l1 = ax1.plot(steps, mu_lc, color="tab:blue", label="alignment_reward_mean")
    ax1.fill_between(steps, lo_lc, hi_lc, color="tab:blue", alpha=0.2)
    l2 = ax2.plot(steps, mu_gn, color="tab:red", label="router_conflict_grad_l2_mean")
    ax2.fill_between(steps, lo_gn, hi_gn, color="tab:red", alpha=0.2)
    ax1.set_title("OURS: L_conflict Signal and Router Gradient Effect")
    ax1.set_xlabel("step")
    ax1.set_ylabel("alignment reward (-L_conflict)")
    ax2.set_ylabel("router conflict grad L2")
    ax1.grid(True, alpha=0.3)
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_lconflict_per_expert(
    *,
    out_path: Path,
    curves: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    seeds: List[int],
) -> bool:
    first_curve = curves["ours"][seeds[0]]
    expert_cols = sorted(
        {
            k
            for row in first_curve.values()
            for k in row.keys()
            if k.startswith("l_conflict_e")
        },
        key=lambda x: int(x.replace("l_conflict_e", "")),
    )
    if not expert_cols:
        _save_missing_plot(
            out_path,
            "OURS Expert-wise L_conflict Trend",
            "Missing l_conflict_e* columns. Rerun training with updated trainer.",
        )
        return False

    steps = _common_steps(curves, ["ours"], seeds, expert_cols)
    if not steps:
        _save_missing_plot(
            out_path,
            "OURS Expert-wise L_conflict Trend",
            "No common steps found for l_conflict_e* across seeds.",
        )
        return False

    fig, ax = plt.subplots(1, 1, figsize=(9, 4), dpi=150)
    for col in expert_cols:
        arr = _series_matrix(curves["ours"], seeds, steps, lambda r, c=col: -float(r[c]))
        mu = arr.mean(axis=0)
        ax.plot(steps, mu, marker="o", linewidth=1.4, label=col.replace("l_conflict_", ""))
    ax.set_title("OURS: Expert-wise Alignment Reward Trend")
    ax.set_xlabel("step")
    ax.set_ylabel("alignment reward per expert (-L_conflict_e)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_load_pies_for_step(
    *,
    out_path: Path,
    step: int,
    methods: List[str],
    seeds: List[int],
    curves: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    load_cols: List[str],
) -> None:
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4.0), dpi=150)
    if len(methods) == 1:
        axes = [axes]
    labels = [c.replace("router_load_", "") for c in load_cols]
    for i, method in enumerate(methods):
        arr = np.array(
            [[curves[method][sd][step][c] for c in load_cols] for sd in seeds],
            dtype=np.float64,
        )
        mean_load = arr.mean(axis=0)
        ax = axes[i]
        ax.pie(mean_load, labels=labels, autopct="%1.1f%%", startangle=90, counterclock=False)
        ax.set_title(f"{method}\nstep={step} (seed mean)")
    fig.suptitle("Validation Router Load Pie Comparison")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_variance_score(
    *,
    out_path: Path,
    csv_out: Path,
    curves: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    methods: List[str],
    seeds: List[int],
    load_cols: List[str],
) -> None:
    variances: Dict[str, List[float]] = {}
    rows: List[Dict[str, object]] = []
    for method in methods:
        steps = _common_steps(curves, [method], seeds, load_cols)
        if not steps:
            variances[method] = [0.0 for _ in load_cols]
            continue
        vals: List[List[float]] = [[] for _ in load_cols]
        for sd in seeds:
            for j, col in enumerate(load_cols):
                seq = [float(curves[method][sd][st][col]) for st in steps]
                score = 0.5 * seq[-1] + 0.5 * max(seq)
                vals[j].append(score)
        variances[method] = []
        for j, col in enumerate(load_cols):
            arr = np.array(vals[j], dtype=np.float64)
            var = float(np.var(arr))
            variances[method].append(var)
            rows.append(
                {
                    "method": method,
                    "expert": col.replace("router_load_", ""),
                    "score_var": var,
                    "score_mean": float(arr.mean()),
                    "score_std": float(arr.std()),
                }
            )

    x = np.arange(len(load_cols))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(1, 1, figsize=(9, 4), dpi=150)
    for i, method in enumerate(methods):
        off = (i - (len(methods) - 1) / 2.0) * width
        ax.bar(x + off, variances[method], width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("router_load_", "") for c in load_cols])
    ax.set_xlabel("expert")
    ax.set_ylabel("variance across seeds")
    ax.set_title("Expert Val-Load Score Variance (0.5*final + 0.5*max)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "expert", "score_var", "score_mean", "score_std"])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate MVP 12-plot diagnostics from run curve CSV files.")
    p.add_argument("--runs_dir", type=str, required=True)
    p.add_argument("--methods", type=str, default="ours,baseline_avg,baseline_cagrad")
    p.add_argument("--seeds", type=str, default="2,3,5")
    p.add_argument("--band", type=str, default="std", choices=["std", "minmax"])
    p.add_argument("--pie_targets", type=str, default="120,240,360")
    p.add_argument("--out_dir", type=str, default="")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else runs_dir
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    pie_targets = [int(x.strip()) for x in args.pie_targets.split(",") if x.strip()]

    curves: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = {m: {} for m in methods}
    for method in methods:
        for seed in seeds:
            path = runs_dir / f"{method}_s{seed}_curve.csv"
            if not path.exists():
                raise RuntimeError(f"missing curve file: {path}")
            curves[method][seed] = _parse_curve(path)

    chart_paths: List[Path] = []
    summary: Dict[str, object] = {"methods": methods, "seeds": seeds}

    metric_specs = [
        ("val_acc", "Validation Accuracy", "accuracy", lambda r: float(r["val_acc"]), "metric_val_acc.png"),
        ("val_loss", "Validation Loss", "loss", lambda r: float(r["val_loss"]), "metric_val_loss.png"),
        ("train_acc", "Train Accuracy", "accuracy", lambda r: float(r["train_acc"]), "metric_train_acc.png"),
        ("train_loss", "Train Loss", "loss", lambda r: float(r["train_loss"]), "metric_train_loss.png"),
        (
            "train_minus_val_acc",
            "Train - Validation Accuracy",
            "accuracy gap",
            lambda r: float(r["train_acc"]) - float(r["val_acc"]),
            "metric_train_minus_val_acc.png",
        ),
    ]

    metric_summary: Dict[str, object] = {}
    for metric_name, title, ylabel, value_fn, fname in metric_specs:
        need = ["train_acc", "val_acc"] if metric_name == "train_minus_val_acc" else [metric_name]
        common = _common_steps(curves, methods, seeds, need)
        if not common:
            raise RuntimeError(f"no common steps found for metric={metric_name}")
        out_path = out_dir / fname
        sm = _plot_metric(
            out_path=out_path,
            title=title,
            ylabel=ylabel,
            curves=curves,
            methods=methods,
            seeds=seeds,
            steps=common,
            value_fn=value_fn,
            band_mode=args.band,
        )
        chart_paths.append(out_path)
        metric_summary[metric_name] = {"steps": common, "last": sm}

    lconf_effect_path = out_dir / "ours_lconflict_router_effect.png"
    _plot_lconflict_effect(out_path=lconf_effect_path, curves=curves, seeds=seeds, band_mode=args.band)
    chart_paths.append(lconf_effect_path)

    lconf_expert_path = out_dir / "ours_lconflict_per_expert.png"
    _plot_lconflict_per_expert(out_path=lconf_expert_path, curves=curves, seeds=seeds)
    chart_paths.append(lconf_expert_path)

    load_cols_all: List[set[str]] = []
    for method in methods:
        for seed in seeds:
            cols = {
                k
                for row in curves[method][seed].values()
                for k in row.keys()
                if k.startswith("router_load_e")
            }
            load_cols_all.append(cols)
    load_cols = sorted(
        list(set.intersection(*load_cols_all)) if load_cols_all else [],
        key=lambda x: int(x.replace("router_load_e", "")),
    )
    if not load_cols:
        for idx, name in enumerate(
            [
                "load_pie_compare_step_a.png",
                "load_pie_compare_step_b.png",
                "load_pie_compare_step_c.png",
                "load_pie_compare_final.png",
            ]
        ):
            pth = out_dir / name
            _save_missing_plot(
                pth,
                f"Load Pie Comparison {idx + 1}",
                "Missing router_load_e* columns. Cannot draw load pies.",
            )
            chart_paths.append(pth)
        var_out = out_dir / "expert_seed_variance_score_05final_05max.png"
        _save_missing_plot(
            var_out,
            "Expert Val-Load Score Variance",
            "Missing router_load_e* columns. Cannot compute seed variance chart.",
        )
        chart_paths.append(var_out)
    else:
        load_steps = _common_steps(curves, methods, seeds, load_cols)
        if not load_steps:
            raise RuntimeError("no common steps found for router_load_e*")
        final_step = int(max(load_steps))

        chosen_steps: List[int] = []
        for target in pie_targets[:3]:
            chosen = min(load_steps, key=lambda s: (abs(s - target), s))
            chosen_steps.append(int(chosen))

        for i, st in enumerate(chosen_steps):
            out_path = out_dir / f"load_pie_compare_step_{i + 1}_target{pie_targets[i]}_actual{st}.png"
            _plot_load_pies_for_step(
                out_path=out_path,
                step=st,
                methods=methods,
                seeds=seeds,
                curves=curves,
                load_cols=load_cols,
            )
            chart_paths.append(out_path)

        final_out = out_dir / f"load_pie_compare_final_step_{final_step}.png"
        _plot_load_pies_for_step(
            out_path=final_out,
            step=final_step,
            methods=methods,
            seeds=seeds,
            curves=curves,
            load_cols=load_cols,
        )
        chart_paths.append(final_out)

        var_out = out_dir / "expert_seed_variance_score_05final_05max.png"
        var_csv = out_dir / "expert_seed_variance_score_05final_05max.csv"
        _plot_variance_score(
            out_path=var_out,
            csv_out=var_csv,
            curves=curves,
            methods=methods,
            seeds=seeds,
            load_cols=load_cols,
        )
        chart_paths.append(var_out)
        summary["expert_score_variance_csv"] = str(var_csv)

    summary["metric_summary"] = metric_summary
    summary["charts"] = [str(p) for p in chart_paths]
    summary["chart_count"] = len(chart_paths)

    summary_path = out_dir / "mvp_plot_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"saved manifest -> {summary_path}")
    print(f"generated charts -> {len(chart_paths)}")


if __name__ == "__main__":
    main()
