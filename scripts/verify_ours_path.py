#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Iterator, List, Tuple

import torch
import torch.nn.functional as F

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


def _clone_grad_map(params: Iterable[torch.nn.Parameter]) -> Dict[int, torch.Tensor | None]:
    out: Dict[int, torch.Tensor | None] = {}
    for p in params:
        out[id(p)] = None if p.grad is None else p.grad.detach().clone()
    return out


def _grad_delta_stats(
    params: Iterable[torch.nn.Parameter],
    before: Dict[int, torch.Tensor | None],
) -> Dict[str, float]:
    max_abs = 0.0
    l2_sq = 0.0
    nnz = 0
    for p in params:
        g_before = before.get(id(p))
        g_after = None if p.grad is None else p.grad.detach()

        if g_before is None and g_after is None:
            continue
        if g_before is None:
            d = g_after
        elif g_after is None:
            d = -g_before
        else:
            d = g_after - g_before

        cur_max = float(d.abs().max().item()) if d.numel() > 0 else 0.0
        max_abs = max(max_abs, cur_max)
        l2_sq += float((d * d).sum().item())
        if cur_max > 0.0:
            nnz += 1

    return {
        "max_abs": float(max_abs),
        "l2": float(l2_sq ** 0.5),
        "num_changed_params": float(nnz),
    }


def _route_with_mode(model: MoEClassifier, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = model.router(h)
    mode = str(model.routing_mode).strip().lower()
    if mode == "softmax":
        probs = F.softmax(logits, dim=-1)
    else:
        k = min(int(model.top_k), int(logits.shape[-1]))
        top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(dim=-1, index=top_idx, src=top_vals)
        probs = F.softmax(masked, dim=-1)
    return logits, probs


def _weighted_intra_cosine(grads: torch.Tensor, probs: torch.Tensor, eps: float = 1.0e-8) -> List[float]:
    # grads: [M, D], probs: [M, K]
    if grads.numel() == 0:
        return []
    g = F.normalize(grads, p=2, dim=1, eps=eps)
    cos = g @ g.t()

    m, k = int(probs.shape[0]), int(probs.shape[1])
    mask = torch.triu(torch.ones((m, m), device=grads.device, dtype=torch.bool), diagonal=1)
    out: List[float] = []

    for j in range(k):
        w = probs[:, j]
        w2 = torch.outer(w, w)
        den = w2[mask].sum()
        if float(den.item()) <= eps:
            out.append(0.0)
        else:
            num = (w2[mask] * cos[mask]).sum()
            out.append(float(num.item() / float(den.item())))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Verify OURS forward/backward path and diagnostics.")
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--set", action="append", default=[], help="override key=value")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--print_every", type=int, default=1)
    p.add_argument("--out_jsonl", type=str, default="selfcheck/ours_diagnostics.jsonl")
    p.add_argument("--out_summary", type=str, default="selfcheck/ours_diagnostics_summary.json")
    p.add_argument("--detach_tol", type=float, default=1.0e-10)
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    cfg.setdefault("method", {})
    cfg["method"]["name"] = "ours"

    _set_seed(int(cfg.get("seed", 0)))

    data = build_multitask_data(cfg)
    model = MoEClassifier(cfg["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    ours_cfg = cfg["method"].get("ours", {})
    micro_bs = _resolve_micro_batch_size(cfg)
    lam = float(ours_cfg.get("lambda_align", 0.0))
    eps = float(ours_cfg.get("eps", 1.0e-8))

    lr = float(cfg["train"].get("lr", 1.0e-3))
    wd = float(cfg["train"].get("weight_decay", 0.0))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    steps = int(args.steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_iters = {k: _iter_forever(v) for k, v in data.train_loaders.items()}

    task_params = model.task_params()
    router_params = model.router_params()
    all_params = model.trainable_params()

    out_jsonl = resolve_runs_path(args.out_jsonl)
    out_summary = resolve_runs_path(args.out_summary)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    if out_jsonl.exists():
        out_jsonl.unlink()

    pass_detach_all = True
    max_task_grad_drift = 0.0
    max_router_grad_delta = 0.0
    final_lconf = 0.0

    for step in range(1, steps + 1):
        batch_by_task: Dict[str, Dict[str, torch.Tensor]] = {}
        for task_name, it in train_iters.items():
            raw = next(it)
            batch_by_task[task_name] = {
                "input_ids": raw["input_ids"].to(device),
                "attention_mask": raw["attention_mask"].to(device),
                "labels": raw["labels"].to(device),
            }

        total_samples = int(sum(int(b["input_ids"].shape[0]) for b in batch_by_task.values()))
        if total_samples <= 0:
            raise RuntimeError("empty total_samples")

        optimizer.zero_grad(set_to_none=True)

        grad_accum: Dict[torch.nn.Parameter, torch.Tensor] = {}
        grad_micro: List[torch.Tensor] = []
        hidden_micro: List[torch.Tensor] = []
        task_loss_wsum = 0.0
        wsum = 0.0

        for batch in batch_by_task.values():
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            windows = _split_indices(int(input_ids.shape[0]), micro_bs)
            if not windows:
                windows = [(0, int(input_ids.shape[0]))]

            for s, e in windows:
                ids_m = input_ids[s:e]
                mask_m = attention_mask[s:e]
                labels_m = labels[s:e]
                w = float(e - s) / float(total_samples)

                optimizer.zero_grad(set_to_none=True)
                out = model(input_ids=ids_m, attention_mask=mask_m, labels=labels_m)
                loss = out["loss"]
                loss.backward()

                task_loss_wsum += float(loss.detach().item()) * w
                wsum += w
                grad_micro.append(_flatten_grads(task_params).detach())
                hidden_micro.append(out["hidden"].detach())

                with torch.no_grad():
                    for p in all_params:
                        if p.grad is None:
                            continue
                        if p not in grad_accum:
                            grad_accum[p] = p.grad.detach().clone() * w
                        else:
                            grad_accum[p].add_(p.grad.detach(), alpha=w)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, g in grad_accum.items():
                p.grad = g

        # Snapshot before conflict backward.
        task_before = _clone_grad_map(task_params)
        router_before = _clone_grad_map(router_params)

        grads = torch.stack(grad_micro, dim=0)

        probs_rows: List[torch.Tensor] = []
        logits_rows_detached: List[torch.Tensor] = []
        for h in hidden_micro:
            logits, probs_tok = _route_with_mode(model, h)
            probs_rows.append(probs_tok.mean(dim=0))
            logits_rows_detached.append(logits.mean(dim=0).detach())

        probs = torch.stack(probs_rows, dim=0)
        G = probs.t() @ grads
        load = probs.sum(dim=0) + eps
        lconf = -((G.pow(2).sum(dim=1) / load).sum())
        final_lconf = float(lconf.detach().item())

        # Make task grads leaf/no-graph then backprop conflict; task grads should remain unchanged.
        for p in task_params:
            if p.grad is not None:
                p.grad = p.grad.detach()

        (lam * lconf).backward()

        for p in task_params:
            if p.grad is not None:
                p.grad = p.grad.detach()

        task_delta = _grad_delta_stats(task_params, task_before)
        router_delta = _grad_delta_stats(router_params, router_before)

        max_task_grad_drift = max(max_task_grad_drift, float(task_delta["max_abs"]))
        max_router_grad_delta = max(max_router_grad_delta, float(router_delta["l2"]))

        detach_ok = bool(task_delta["max_abs"] <= float(args.detach_tol))
        pass_detach_all = bool(pass_detach_all and detach_ok)

        probs_mean = probs.mean(dim=0)
        top1 = torch.argmax(probs, dim=1)
        top1_counts = torch.bincount(top1, minlength=int(probs.shape[1])).detach().cpu().tolist()

        routing_entropy = float((-(probs * torch.log(probs + 1.0e-12)).sum(dim=1)).mean().item())
        intra_cos = _weighted_intra_cosine(grads, probs, eps=eps)

        record = {
            "step": int(step),
            "num_tasks_in_step": int(len(batch_by_task)),
            "task_names": sorted(list(batch_by_task.keys())),
            "total_samples_step": int(total_samples),
            "micro_batch_size_cfg": int(micro_bs),
            "num_micro": int(grads.shape[0]),
            "task_loss_weighted": float(task_loss_wsum / max(1.0e-12, wsum)),
            "l_conflict": float(lconf.detach().item()),
            "alignment_reward": float((-lconf).detach().item()),
            "lambda_align": float(lam),
            "routing_entropy": float(routing_entropy),
            "expert_load": [float(x) for x in probs_mean.detach().cpu().tolist()],
            "expert_top1_counts": [int(x) for x in top1_counts],
            "expert_logits_mean": [float(x) for x in torch.stack(logits_rows_detached, dim=0).mean(dim=0).cpu().tolist()],
            "expert_intra_cosine": [float(x) for x in intra_cos],
            "detach_check_pass": bool(detach_ok),
            "task_grad_drift_max_abs": float(task_delta["max_abs"]),
            "task_grad_drift_l2": float(task_delta["l2"]),
            "router_grad_delta_l2": float(router_delta["l2"]),
        }

        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

        if step % int(args.print_every) == 0 or step == 1 or step == steps:
            print(
                f"[verify step {step:03d}/{steps}] "
                f"L_task={record['task_loss_weighted']:.4f} "
                f"L_conf={record['l_conflict']:.4f} "
                f"num_micro={record['num_micro']} "
                f"entropy={record['routing_entropy']:.4f} "
                f"detach_ok={record['detach_check_pass']} "
                f"task_drift={record['task_grad_drift_max_abs']:.2e} "
                f"router_delta_l2={record['router_grad_delta_l2']:.2e}"
            )

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    summary = {
        "steps": int(steps),
        "pass_detach_all_steps": bool(pass_detach_all),
        "max_task_grad_drift": float(max_task_grad_drift),
        "max_router_grad_delta_l2": float(max_router_grad_delta),
        "final_l_conflict": float(final_lconf),
        "jsonl": str(out_jsonl),
    }
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
