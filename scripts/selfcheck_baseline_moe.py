#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Iterator, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moe_gc import MoEClassifier, build_multitask_data, load_config  # noqa: E402
from path_utils import resolve_runs_path  # noqa: E402


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _split_indices(n: int, micro_bs: int) -> List[Tuple[int, int]]:
    if n <= 0:
        return []
    if micro_bs <= 0 or micro_bs >= n:
        return [(0, n)]
    out: List[Tuple[int, int]] = []
    s = 0
    while s < n:
        e = min(n, s + micro_bs)
        out.append((s, e))
        s = e
    return out


def _resolve_micro_batch_size(cfg: dict) -> int:
    tr = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    if "micro_batch_size" in tr:
        return int(tr.get("micro_batch_size", 16))
    ours = ((cfg.get("method", {}) or {}).get("ours", {}) or {})
    return int(ours.get("micro_batch_size", 16))


def _iter_forever(loader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for input_ids, attention_mask, labels in loader:
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }


def _flatten_grads(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    vecs: List[torch.Tensor] = []
    for p in params:
        if p.grad is None:
            vecs.append(torch.zeros_like(p).flatten())
        else:
            vecs.append(p.grad.detach().flatten())
    if not vecs:
        return torch.zeros(0)
    return torch.cat(vecs, dim=0)


def _project_simplex(v: torch.Tensor) -> torch.Tensor:
    if v.numel() == 0:
        return v
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    idx = torch.arange(1, v.numel() + 1, dtype=v.dtype, device=v.device)
    cond = u - cssv / idx > 0
    if not torch.any(cond):
        return torch.full_like(v, 1.0 / float(v.numel()))
    rho = int(torch.nonzero(cond, as_tuple=False)[-1].item()) + 1
    theta = float(cssv[rho - 1].item()) / float(rho)
    w = torch.clamp(v - theta, min=0.0)
    z = float(w.sum().item())
    if z <= 0.0:
        return torch.full_like(v, 1.0 / float(v.numel()))
    return w / z


def _cagrad_direction(grads: torch.Tensor, *, c: float, eps: float, inner_steps: int, inner_lr: float) -> torch.Tensor:
    if grads.ndim != 2:
        raise RuntimeError(f"grads must be [T, D], got {tuple(grads.shape)}")
    T = int(grads.shape[0])
    g0 = grads.mean(dim=0)
    if T <= 1 or float(c) <= 0.0:
        return g0

    gram = grads @ grads.t()
    trace_g = float(torch.trace(gram).item())
    step_scale = float(inner_lr) / (trace_g + 1.0e-12)

    w = torch.full((T,), 1.0 / float(T), dtype=grads.dtype, device=grads.device)
    for _ in range(max(1, int(inner_steps))):
        grad_w = gram @ w
        w = _project_simplex(w - float(step_scale) * grad_w)

    s = (w.unsqueeze(1) * grads).sum(dim=0)
    g0_norm = torch.norm(g0)
    s_norm = torch.norm(s)
    scale = float(c) * g0_norm / (s_norm + float(eps))
    return g0 + scale * s


def _router_checks(model: MoEClassifier, batch: Dict[str, torch.Tensor]) -> Dict[str, float | bool]:
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        probs = out["router_probs"]

    row_sum = probs.sum(dim=1)
    row_sum_max_abs_err = float((row_sum - 1.0).abs().max().item())
    nonzero = (probs > 0).sum(dim=1)
    k_cfg = int(model.top_k)
    if str(model.routing_mode).lower() == "topk":
        nz_max = int(nonzero.max().item())
        topk_ok = bool(nz_max <= k_cfg)
    else:
        nz_min = int(nonzero.min().item())
        topk_ok = bool(nz_min == int(probs.shape[1]))

    return {
        "router_prob_sum_max_abs_err": float(row_sum_max_abs_err),
        "router_prob_valid": bool(row_sum_max_abs_err <= 1.0e-5),
        "routing_nonzero_pattern_ok": bool(topk_ok),
    }


def _collect_per_task_grads(
    model: MoEClassifier,
    batch_by_task: Dict[str, Dict[str, torch.Tensor]],
    micro_bs: int,
) -> Tuple[torch.Tensor, float, Dict[str, int]]:
    params = model.trainable_params()
    grad_vecs: List[torch.Tensor] = []
    loss_sum = 0.0
    micro_windows_by_task: Dict[str, int] = {}

    for task_name, batch in batch_by_task.items():
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        bs = int(input_ids.shape[0])
        windows = _split_indices(bs, micro_bs)
        if not windows:
            windows = [(0, bs)]
        micro_windows_by_task[task_name] = int(len(windows))

        task_grad_accum: Dict[torch.nn.Parameter, torch.Tensor] = {}
        task_loss_wsum = 0.0
        wsum = 0.0

        for s, e in windows:
            ids_m = input_ids[s:e]
            mask_m = attention_mask[s:e]
            labels_m = labels[s:e]
            w = float(e - s) / float(max(1, bs))

            model.zero_grad(set_to_none=True)
            out = model(
                input_ids=ids_m,
                attention_mask=mask_m,
                labels=labels_m,
            )
            loss = out["loss"]
            loss.backward()
            task_loss_wsum += float(loss.detach().item()) * w
            wsum += w

            with torch.no_grad():
                for p in params:
                    if p.grad is None:
                        continue
                    if p not in task_grad_accum:
                        task_grad_accum[p] = p.grad.detach().clone() * w
                    else:
                        task_grad_accum[p].add_(p.grad.detach(), alpha=w)

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, g in task_grad_accum.items():
                p.grad = g

        grad_vecs.append(_flatten_grads(params))
        loss_sum += float(task_loss_wsum / max(1.0e-12, wsum))

    return (
        torch.stack(grad_vecs, dim=0),
        float(loss_sum / max(1, len(grad_vecs))),
        micro_windows_by_task,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Self-check baseline MoE structure and update paths.")
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--set", action="append", default=[], help="override key=value")
    p.add_argument("--out", type=str, default="selfcheck/baseline_selfcheck_summary.json")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    _set_seed(int(cfg.get("seed", 0)))

    data = build_multitask_data(cfg)
    model = MoEClassifier(cfg["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_iters = {k: _iter_forever(v) for k, v in data.train_loaders.items()}
    batch_by_task: Dict[str, Dict[str, torch.Tensor]] = {}
    for task_name, it in train_iters.items():
        raw = next(it)
        batch_by_task[task_name] = {
            "input_ids": raw["input_ids"].to(device),
            "attention_mask": raw["attention_mask"].to(device),
            "labels": raw["labels"].to(device),
        }
    per_task_batch_size = {k: int(v["input_ids"].shape[0]) for k, v in batch_by_task.items()}
    unique_bs = sorted(set(per_task_batch_size.values()))
    micro_bs = _resolve_micro_batch_size(cfg)

    any_task = next(iter(batch_by_task.keys()))
    router_ck = _router_checks(model, batch_by_task[any_task])

    grads, mean_task_loss, micro_windows_by_task = _collect_per_task_grads(model, batch_by_task, micro_bs)
    g_avg = grads.mean(dim=0)

    ccfg = cfg.get("method", {}).get("baseline_cagrad", {})
    g_cagrad = _cagrad_direction(
        grads,
        c=float(ccfg.get("c", 0.4)),
        eps=float(ccfg.get("eps", 1.0e-8)),
        inner_steps=int(ccfg.get("inner_steps", 10)),
        inner_lr=float(ccfg.get("inner_lr", 0.1)),
    )

    summary = {
        "num_tasks_in_step": int(len(batch_by_task)),
        "task_names": sorted(list(batch_by_task.keys())),
        "per_task_batch_size": per_task_batch_size,
        "unique_batch_sizes": [int(x) for x in unique_bs],
        "micro_batch_size_cfg": int(micro_bs),
        "micro_windows_by_task": micro_windows_by_task,
        "grads_shape": [int(x) for x in grads.shape],
        "mean_task_loss": float(mean_task_loss),
        "model": {
            "backbone_backend": str(cfg.get("model", {}).get("backbone_backend", "")),
            "backbone": str(cfg.get("model", {}).get("backbone", "")),
            "expert_type": str(cfg.get("model", {}).get("expert_type", "")),
            "routing_mode": str(cfg.get("model", {}).get("routing_mode", "")),
            "num_experts": int(cfg.get("model", {}).get("num_experts", 0)),
            "top_k": int(cfg.get("model", {}).get("top_k", 0)),
        },
        "router_checks": router_ck,
        "baseline_avg": {
            "direction_norm": float(torch.norm(g_avg).item()),
            "is_finite": bool(torch.isfinite(g_avg).all().item()),
        },
        "baseline_cagrad": {
            "direction_norm": float(torch.norm(g_cagrad).item()),
            "is_finite": bool(torch.isfinite(g_cagrad).all().item()),
            "delta_vs_avg_l2": float(torch.norm(g_cagrad - g_avg).item()),
        },
        "passes": {
            "router_prob_valid": bool(router_ck["router_prob_valid"]),
            "routing_nonzero_pattern_ok": bool(router_ck["routing_nonzero_pattern_ok"]),
            "baseline_avg_finite": bool(torch.isfinite(g_avg).all().item()),
            "baseline_cagrad_finite": bool(torch.isfinite(g_cagrad).all().item()),
            "same_batch_size_across_tasks": bool(len(unique_bs) == 1),
        },
    }

    out_path = resolve_runs_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
