from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch

from .data import MultiTaskData, build_train_iters
from .model import MoEClassifier


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


def _resolve_device_batch_size(
    cfg: dict,
    *,
    effective_batch_size: int,
    logical_micro_batch_size: int,
) -> int:
    if effective_batch_size <= 0:
        return 1
    tr = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    raw = tr.get("device_batch_size", 0)
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        target = int(effective_batch_size)
    elif raw in {None, "", 0, "0"}:
        target = int(effective_batch_size)
    else:
        target = int(raw)
    target = max(1, min(int(effective_batch_size), int(target)))
    micro = max(1, int(logical_micro_batch_size))
    if target <= micro or effective_batch_size <= micro:
        return min(int(effective_batch_size), micro)
    groups = max(1, target // micro)
    return min(int(effective_batch_size), groups * micro)


def _slice_to_device_batch(
    batch: Dict[str, torch.Tensor],
    s: int,
    e: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"][s:e].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"][s:e].to(device, non_blocking=True),
        "labels": batch["labels"][s:e].to(device, non_blocking=True),
    }


def _group_micro_windows(
    windows: List[Tuple[int, int]],
    *,
    device_batch_size: int,
    logical_micro_batch_size: int,
) -> List[List[Tuple[int, int]]]:
    if not windows:
        return []
    group_span = max(1, int(device_batch_size) // max(1, int(logical_micro_batch_size)))
    out: List[List[Tuple[int, int]]] = []
    for i in range(0, len(windows), group_span):
        out.append(windows[i:i + group_span])
    return out


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


def _flatten_grads_cpu(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    vecs: List[torch.Tensor] = []
    for p in params:
        if p.grad is None:
            vecs.append(torch.zeros((int(p.numel()),), dtype=torch.float32, device="cpu"))
        else:
            vecs.append(p.grad.detach().to(device="cpu", dtype=torch.float32).flatten())
    if not vecs:
        return torch.zeros((0,), dtype=torch.float32, device="cpu")
    return torch.cat(vecs, dim=0)


def _assign_grads(params: Iterable[torch.nn.Parameter], vec: torch.Tensor) -> None:
    off = 0
    for p in params:
        n = int(p.numel())
        g = vec[off: off + n]
        off += n
        g = g.to(device=p.device, dtype=p.dtype).view_as(p)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)


def _unwrap_model(model: torch.nn.Module) -> MoEClassifier:
    if isinstance(model, torch.nn.DataParallel):
        inner = model.module
        if not isinstance(inner, MoEClassifier):
            raise RuntimeError(f"unexpected wrapped model type: {type(inner)!r}")
        return inner
    if not isinstance(model, MoEClassifier):
        raise RuntimeError(f"unexpected model type: {type(model)!r}")
    return model


def _scalar_loss(loss: torch.Tensor) -> torch.Tensor:
    # DataParallel may return a vector of per-device scalar losses.
    if loss.ndim == 0:
        return loss
    return loss.mean()


def _safe_metric_key(name: str) -> str:
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
    text = "".join(out).strip("_")
    return text or "task"


def _cosine_scalar(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-12) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    an = float(torch.linalg.vector_norm(a).item())
    bn = float(torch.linalg.vector_norm(b).item())
    if an <= eps or bn <= eps:
        return 0.0
    return float(torch.dot(a, b).item() / max(eps, an * bn))


def _mean_upper_triangle(mat: List[List[float]]) -> float:
    n = len(mat)
    if n <= 1:
        return 1.0
    vals: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(float(mat[i][j]))
    if not vals:
        return 1.0
    return float(sum(vals) / float(len(vals)))


def _load_stats_from_tensor(loads: torch.Tensor) -> Dict[str, float]:
    if loads.numel() <= 0:
        return {}
    mean = float(loads.mean().item())
    sd = float(loads.std(unbiased=False).item())
    mn = float(loads.min().item())
    mx = float(loads.max().item())
    uniform = 1.0 / float(loads.numel())
    util_thresh = 0.5 * uniform
    utilized = int((loads >= util_thresh).sum().item())
    return {
        "load_cv": float(sd / max(mean, 1.0e-12)),
        "load_max_min_ratio": float(mx / max(mn, 1.0e-12)),
        "utilization_ratio": float(utilized / float(loads.numel())),
    }


def _evaluate_gradient_diagnostics(
    model: torch.nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    logical_micro_batch_size: int,
    device_batch_size: int,
    max_batches_per_task: int = 1,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    base_model = _unwrap_model(model)
    all_params = base_model.trainable_params()
    expert_param_groups = [
        [p for p in expert.parameters() if p.requires_grad]
        for expert in base_model.experts
    ]
    task_names = list(loaders.keys())
    if not task_names:
        return {}, {
            "task_names": [],
            "task_conflict_matrix": [],
            "expert_names": [],
            "expert_purity": [],
            "expert_diversity": [],
            "intra_expert_coherence": [],
            "inter_expert_similarity_matrix": [],
        }

    was_training = model.training
    model.eval()
    task_grad_all: List[torch.Tensor] = []
    task_grad_by_expert: List[List[torch.Tensor]] = []

    try:
        for task_name, loader in loaders.items():
            grad_all_acc: torch.Tensor | None = None
            grad_exp_acc: List[torch.Tensor | None] = [None for _ in expert_param_groups]
            seen = 0
            for input_ids, attention_mask, labels in loader:
                if max_batches_per_task > 0 and seen >= max_batches_per_task:
                    break
                batch_cpu = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                micro_windows = _split_indices(int(input_ids.shape[0]), logical_micro_batch_size)
                grouped = _group_micro_windows(
                    micro_windows,
                    device_batch_size=device_batch_size,
                    logical_micro_batch_size=logical_micro_batch_size,
                )
                for group in grouped:
                    gs = group[0][0]
                    ge = group[-1][1]
                    batch_gpu = _slice_to_device_batch(batch_cpu, gs, ge, device)
                    for s, e in group:
                        ls = s - gs
                        le = e - gs
                        model.zero_grad(set_to_none=True)
                        out = model(
                            input_ids=batch_gpu["input_ids"][ls:le],
                            attention_mask=batch_gpu["attention_mask"][ls:le],
                            labels=batch_gpu["labels"][ls:le],
                        )
                        loss = _scalar_loss(out["loss"])
                        loss.backward()

                        g_all = _flatten_grads_cpu(all_params)
                        if grad_all_acc is None:
                            grad_all_acc = g_all
                        else:
                            grad_all_acc.add_(g_all)
                        for ei, exp_group in enumerate(expert_param_groups):
                            g_exp = _flatten_grads_cpu(exp_group)
                            if grad_exp_acc[ei] is None:
                                grad_exp_acc[ei] = g_exp
                            else:
                                grad_exp_acc[ei].add_(g_exp)
                        seen += 1

            if seen <= 0 or grad_all_acc is None:
                raise RuntimeError(f"no diagnostic batches collected for task={task_name!r}")

            denom = float(seen)
            task_grad_all.append(grad_all_acc.div(denom))
            task_grad_by_expert.append(
                [
                    (g if g is not None else torch.zeros((0,), dtype=torch.float32, device="cpu")).div(denom)
                    for g in grad_exp_acc
                ]
            )
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

    n_tasks = len(task_names)
    n_experts = len(expert_param_groups)
    task_conflict_matrix = [[1.0 for _ in range(n_tasks)] for _ in range(n_tasks)]
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            cos = _cosine_scalar(task_grad_all[i], task_grad_all[j])
            task_conflict_matrix[i][j] = cos
            task_conflict_matrix[j][i] = cos
    overall_task_conflict = _mean_upper_triangle(task_conflict_matrix)

    per_expert_coherence: List[float] = []
    expert_purity: List[float] = []
    expert_diversity: List[float] = []
    expert_aggregate_grads: List[torch.Tensor] = []
    for ei in range(n_experts):
        grads = [task_grad_by_expert[ti][ei] for ti in range(n_tasks)]
        coh_mat = [[1.0 for _ in range(n_tasks)] for _ in range(n_tasks)]
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                cos = _cosine_scalar(grads[i], grads[j])
                coh_mat[i][j] = cos
                coh_mat[j][i] = cos
        coherence = _mean_upper_triangle(coh_mat)
        per_expert_coherence.append(coherence)

        norms = torch.tensor(
            [float(torch.linalg.vector_norm(g).item()) for g in grads],
            dtype=torch.float32,
            device="cpu",
        )
        norm_sum = float(norms.sum().item())
        if n_tasks <= 1:
            purity = 1.0
            diversity = 0.0
        elif norm_sum <= 1.0e-12:
            purity = 0.0
            diversity = 0.0
        else:
            probs = norms / norm_sum
            purity = float(probs.max().item())
            entropy = -(probs.clamp_min(1.0e-12) * probs.clamp_min(1.0e-12).log()).sum()
            diversity = float(entropy.item() / max(1.0e-12, float(torch.log(torch.tensor(float(n_tasks))).item())))
        expert_purity.append(purity)
        expert_diversity.append(diversity)
        expert_aggregate_grads.append(torch.stack(grads, dim=0).mean(dim=0))

    inter_expert_similarity_matrix = [[1.0 for _ in range(n_experts)] for _ in range(n_experts)]
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            cos = _cosine_scalar(expert_aggregate_grads[i], expert_aggregate_grads[j])
            inter_expert_similarity_matrix[i][j] = cos
            inter_expert_similarity_matrix[j][i] = cos
    inter_expert_similarity = _mean_upper_triangle(inter_expert_similarity_matrix)

    task_grad_norms = [float(torch.linalg.vector_norm(g).item()) for g in task_grad_all]
    if task_grad_norms:
        norm_t = torch.tensor(task_grad_norms, dtype=torch.float32, device="cpu")
        norm_mean = float(norm_t.mean().item())
        task_grad_norm_cv = float(norm_t.std(unbiased=False).item() / max(norm_mean, 1.0e-12))
    else:
        task_grad_norm_cv = 0.0

    scalars: Dict[str, float] = {
        "overall_task_conflict_cos": float(overall_task_conflict),
        "avg_intra_expert_coherence": float(
            sum(per_expert_coherence) / float(max(1, len(per_expert_coherence)))
        ),
        "inter_expert_similarity": float(inter_expert_similarity),
        "expert_purity_mean": float(sum(expert_purity) / float(max(1, len(expert_purity)))),
        "expert_diversity_mean": float(sum(expert_diversity) / float(max(1, len(expert_diversity)))),
        "task_grad_norm_cv": float(task_grad_norm_cv),
    }
    for ei, val in enumerate(per_expert_coherence):
        scalars[f"intra_expert_coherence_e{ei}"] = float(val)
    for ei, val in enumerate(expert_purity):
        scalars[f"expert_purity_e{ei}"] = float(val)
    for ei, val in enumerate(expert_diversity):
        scalars[f"expert_diversity_e{ei}"] = float(val)
    for i in range(n_tasks):
        for j in range(n_tasks):
            scalars[f"task_conflict_t{i}_t{j}"] = float(task_conflict_matrix[i][j])
    for i in range(n_experts):
        for j in range(n_experts):
            scalars[f"inter_expert_similarity_e{i}_e{j}"] = float(inter_expert_similarity_matrix[i][j])

    structured: Dict[str, Any] = {
        "task_names": list(task_names),
        "task_conflict_matrix": task_conflict_matrix,
        "overall_task_conflict_cos": float(overall_task_conflict),
        "expert_names": [f"e{i}" for i in range(n_experts)],
        "intra_expert_coherence": [float(v) for v in per_expert_coherence],
        "avg_intra_expert_coherence": float(
            sum(per_expert_coherence) / float(max(1, len(per_expert_coherence)))
        ),
        "expert_purity": [float(v) for v in expert_purity],
        "expert_diversity": [float(v) for v in expert_diversity],
        "inter_expert_similarity_matrix": inter_expert_similarity_matrix,
        "inter_expert_similarity": float(inter_expert_similarity),
        "task_grad_norms": [float(v) for v in task_grad_norms],
        "task_grad_norm_cv": float(task_grad_norm_cv),
    }
    return scalars, structured


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


def _evaluate_metrics(
    model: torch.nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    logical_micro_batch_size: int,
    device_batch_size: int,
    max_batches_per_task: int = 0,
) -> Dict[str, object]:
    model.eval()
    vals_acc: List[float] = []
    vals_loss: List[float] = []
    metrics: Dict[str, object] = {}
    router_sum: torch.Tensor | None = None
    router_cnt = 0
    router_ent_sum = 0.0
    with torch.no_grad():
        for task_name, loader in loaders.items():
            ok = 0
            tot = 0
            n_batches = 0
            loss_sum = 0.0
            for input_ids, attention_mask, labels in loader:
                if max_batches_per_task > 0 and n_batches >= max_batches_per_task:
                    break
                batch_cpu = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                micro_windows = _split_indices(int(input_ids.shape[0]), logical_micro_batch_size)
                grouped = _group_micro_windows(
                    micro_windows,
                    device_batch_size=device_batch_size,
                    logical_micro_batch_size=logical_micro_batch_size,
                )
                loss_batch_sum = 0.0
                batch_samples = 0
                for group in grouped:
                    gs = group[0][0]
                    ge = group[-1][1]
                    batch_gpu = _slice_to_device_batch(batch_cpu, gs, ge, device)
                    for s, e in group:
                        ls = s - gs
                        le = e - gs
                        model_out = model(
                            input_ids=batch_gpu["input_ids"][ls:le],
                            attention_mask=batch_gpu["attention_mask"][ls:le],
                            labels=batch_gpu["labels"][ls:le],
                        )
                        logits = model_out["logits"]
                        labels_m = batch_gpu["labels"][ls:le]
                        pred = torch.argmax(logits, dim=-1)
                        ok += int((pred == labels_m).sum().item())
                        mb_n = int(labels_m.numel())
                        tot += mb_n
                        batch_samples += mb_n
                        loss_batch_sum += float(_scalar_loss(model_out["loss"]).detach().item()) * float(mb_n)

                        probs = model_out.get("router_probs")
                        if probs is not None:
                            pp = probs.detach()
                            if pp.ndim == 1:
                                pp = pp.unsqueeze(0)
                            ppc = pp.to(dtype=torch.float32, device="cpu")
                            if router_sum is None:
                                router_sum = ppc.sum(dim=0)
                            else:
                                router_sum.add_(ppc.sum(dim=0))
                            router_cnt += int(ppc.shape[0])
                            ent = -(ppc.clamp_min(1.0e-12) * ppc.clamp_min(1.0e-12).log()).sum(dim=1)
                            router_ent_sum += float(ent.sum().item())
                loss_sum += float(loss_batch_sum / max(1, batch_samples))
                n_batches += 1
            task_acc = float(ok / max(1, tot))
            task_loss = float(loss_sum / max(1, n_batches))
            vals_acc.append(task_acc)
            vals_loss.append(task_loss)
            task_key = _safe_metric_key(task_name)
            metrics[f"task_acc__{task_key}"] = task_acc
            metrics[f"task_loss__{task_key}"] = task_loss
    model.train()
    metrics.update(
        {
        "acc": float(sum(vals_acc) / max(1, len(vals_acc))),
        "loss": float(sum(vals_loss) / max(1, len(vals_loss))),
        }
    )
    if router_sum is not None and router_cnt > 0:
        loads = router_sum / float(router_cnt)
        metrics["router_entropy"] = float(router_ent_sum / float(router_cnt))
        for i in range(int(loads.numel())):
            metrics[f"router_load_e{i}"] = float(loads[i].item())
        metrics.update({k: float(v) for k, v in _load_stats_from_tensor(loads).items()})
    return metrics


@dataclass
class TrainSummary:
    best_val_acc: float
    final_val_acc: float
    step_metrics: List[Dict[str, float]]
    final_step_metrics: Dict[str, float]
    final_diagnostics: Dict[str, Any]
    overhead: Dict[str, float]
    batch_semantics: Dict[str, Any]


def _train_step_baseline(
    *,
    cfg: dict,
    model: torch.nn.Module,
    batch_by_task: Dict[str, Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    method_name: str,
    device: torch.device,
) -> float:
    logical_micro_bs = _resolve_micro_batch_size(cfg)
    base_model = _unwrap_model(model)
    params = base_model.trainable_params()
    need_cagrad = method_name == "baseline_cagrad"
    grad_vecs: List[torch.Tensor] = []
    task_grad_maps: List[Dict[torch.nn.Parameter, torch.Tensor]] = []
    losses: List[float] = []

    for batch in batch_by_task.values():
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        bs = int(input_ids.shape[0])
        if bs <= 0:
            raise RuntimeError("empty task batch in baseline step")

        windows = _split_indices(bs, logical_micro_bs)
        if not windows:
            windows = [(0, bs)]
        device_bs = _resolve_device_batch_size(
            cfg,
            effective_batch_size=bs,
            logical_micro_batch_size=logical_micro_bs,
        )
        grouped = _group_micro_windows(
            windows,
            device_batch_size=device_bs,
            logical_micro_batch_size=logical_micro_bs,
        )

        task_grad_accum: Dict[torch.nn.Parameter, torch.Tensor] = {}
        task_loss_wsum = 0.0
        wsum = 0.0

        for group in grouped:
            gs = group[0][0]
            ge = group[-1][1]
            batch_gpu = _slice_to_device_batch(batch, gs, ge, device)
            for s, e in group:
                ls = s - gs
                le = e - gs
                w = float(e - s) / float(bs)
                optimizer.zero_grad(set_to_none=True)
                out = model(
                    input_ids=batch_gpu["input_ids"][ls:le],
                    attention_mask=batch_gpu["attention_mask"][ls:le],
                    labels=batch_gpu["labels"][ls:le],
                )
                loss = _scalar_loss(out["loss"])
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

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, g in task_grad_accum.items():
                p.grad = g

        task_grad_maps.append(task_grad_accum)
        if need_cagrad:
            # Keep cagrad vectors on CPU to avoid a multi-GB GPU allocation spike.
            grad_vecs.append(_flatten_grads_cpu(params))
        losses.append(float(task_loss_wsum / max(1.0e-12, wsum)))

    if not task_grad_maps:
        raise RuntimeError("no task gradients collected in baseline step")

    # Single-task baseline does not need full-vector combine/assign path.
    # Keeping per-parameter grads avoids a large temporary allocation peak.
    if len(task_grad_maps) == 1:
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, g in task_grad_maps[0].items():
                p.grad = g
        return float(sum(losses) / max(1, len(losses)))

    if method_name == "baseline_avg":
        tcount = float(len(task_grad_maps))
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p in params:
                acc: torch.Tensor | None = None
                for gm in task_grad_maps:
                    g = gm.get(p)
                    if g is None:
                        continue
                    if acc is None:
                        acc = g.detach().clone()
                    else:
                        acc.add_(g)
                if acc is not None:
                    p.grad = acc.mul_(1.0 / tcount)
        return float(sum(losses) / max(1, len(losses)))

    if not grad_vecs:
        raise RuntimeError("baseline_cagrad requires task gradient vectors")
    grads = torch.stack(grad_vecs, dim=0)
    if method_name == "baseline_cagrad":
        ccfg = cfg["method"].get("baseline_cagrad", {})
        direction = _cagrad_direction(
            grads,
            c=float(ccfg.get("c", 0.4)),
            eps=float(ccfg.get("eps", 1.0e-8)),
            inner_steps=int(ccfg.get("inner_steps", 10)),
            inner_lr=float(ccfg.get("inner_lr", 0.1)),
        )
    else:
        direction = grads.mean(dim=0)

    optimizer.zero_grad(set_to_none=True)
    _assign_grads(params, direction)
    return float(sum(losses) / max(1, len(losses)))


def _train_step_ours(
    *,
    cfg: dict,
    model: torch.nn.Module,
    batch_by_task: Dict[str, Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    ours_cfg = cfg["method"]["ours"]
    logical_micro_bs = _resolve_micro_batch_size(cfg)
    lam = float(ours_cfg.get("lambda_align", 0.0))
    eps = float(ours_cfg.get("eps", 1.0e-8))
    use_load_norm = bool(ours_cfg.get("use_load_norm", True))

    base_model = _unwrap_model(model)
    all_params = base_model.trainable_params()
    task_params = base_model.task_params()
    conflict_params = base_model.conflict_params()

    total_samples = int(sum(int(b["input_ids"].shape[0]) for b in batch_by_task.values()))
    if total_samples <= 0:
        raise RuntimeError("empty total samples")

    optimizer.zero_grad(set_to_none=True)
    grad_accum: Dict[torch.nn.Parameter, torch.Tensor] = {}
    grad_micro: List[torch.Tensor] = []
    hidden_micro: List[torch.Tensor] = []
    losses: List[float] = []
    weights: List[float] = []

    for batch in batch_by_task.values():
        input_ids = batch["input_ids"]
        bs = int(input_ids.shape[0])
        micro_windows = _split_indices(bs, logical_micro_bs)
        device_bs = _resolve_device_batch_size(
            cfg,
            effective_batch_size=bs,
            logical_micro_batch_size=logical_micro_bs,
        )
        grouped = _group_micro_windows(
            micro_windows,
            device_batch_size=device_bs,
            logical_micro_batch_size=logical_micro_bs,
        )
        for group in grouped:
            gs = group[0][0]
            ge = group[-1][1]
            batch_gpu = _slice_to_device_batch(batch, gs, ge, device)
            for s, e in group:
                ls = s - gs
                le = e - gs
                w = float(e - s) / float(total_samples)

                optimizer.zero_grad(set_to_none=True)
                out = model(
                    input_ids=batch_gpu["input_ids"][ls:le],
                    attention_mask=batch_gpu["attention_mask"][ls:le],
                    labels=batch_gpu["labels"][ls:le],
                )
                loss = _scalar_loss(out["loss"])
                loss.backward()

                losses.append(float(loss.detach().item()) * w)
                weights.append(w)
                grad_micro.append(_flatten_grads_cpu(conflict_params).detach())
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

    diag: Dict[str, float] = {}
    if lam > 0.0 and grad_micro:
        grads = torch.stack(grad_micro, dim=0)
        probs_rows: List[torch.Tensor] = []
        for h in hidden_micro:
            pm = base_model.route_probs_from_hidden(h).mean(dim=0)
            probs_rows.append(pm)
        probs = torch.stack(probs_rows, dim=0)
        grads = grads.to(device=probs.device, dtype=probs.dtype)

        G = probs.t() @ grads
        load = probs.sum(dim=0) + eps
        if use_load_norm:
            per_expert_conflict = -(G.pow(2).sum(dim=1) / load)
        else:
            # Ablation: remove load normalization and keep only raw squared-norm reward.
            per_expert_conflict = -(G.pow(2).sum(dim=1))
        lnorm = per_expert_conflict.sum()

        for p in task_params:
            if p.grad is not None:
                p.grad = p.grad.detach()

        router_before: Dict[torch.nn.Parameter, torch.Tensor] = {}
        with torch.no_grad():
            for p in conflict_params:
                if p.grad is not None:
                    router_before[p] = p.grad.detach().clone()
        (lam * lnorm).backward()

        delta_sq = 0.0
        with torch.no_grad():
            for p in conflict_params:
                g_before = router_before.get(p)
                g_after = p.grad.detach() if p.grad is not None else None
                if g_before is None and g_after is None:
                    continue
                if g_before is None:
                    diff = g_after  # type: ignore[assignment]
                elif g_after is None:
                    diff = -g_before
                else:
                    diff = g_after - g_before
                delta_sq += float(diff.to(dtype=torch.float32).pow(2).sum().item())
        router_delta_l2 = delta_sq ** 0.5

        for p in task_params:
            if p.grad is not None:
                p.grad = p.grad.detach()

        diag["l_conflict"] = float(lnorm.detach().item())
        diag["alignment_reward"] = float((-lnorm).detach().item())
        diag["router_conflict_grad_l2"] = float(router_delta_l2)
        for i, v in enumerate(per_expert_conflict.detach().cpu().tolist()):
            diag[f"l_conflict_e{i}"] = float(v)
    else:
        diag["l_conflict"] = 0.0
        diag["alignment_reward"] = 0.0
        diag["router_conflict_grad_l2"] = 0.0

    return float(sum(losses) / max(1.0e-12, sum(weights))), diag


def train(cfg: dict, data: MultiTaskData, model: MoEClassifier) -> TrainSummary:
    _set_seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_dp = bool((cfg.get("train", {}) or {}).get("data_parallel", False))
    if device.type == "cuda" and use_dp:
        ndev = int(torch.cuda.device_count())
        if ndev > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(ndev)))
            print(f"[train] data_parallel enabled on {ndev} visible GPUs")
    model.train()

    lr = float(cfg["train"].get("lr", 1.0e-3))
    wd = float(cfg["train"].get("weight_decay", 0.0))
    steps = int(cfg["train"]["steps"])
    warmup_ratio = float(cfg["train"].get("warmup_ratio", cfg["train"].get("warmup", 0.0)))
    warmup_ratio = max(0.0, min(1.0, warmup_ratio))
    warmup_steps = int(round(float(steps) * warmup_ratio))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    log_every = int(cfg["train"].get("log_every", 10))
    eval_every_steps = int(cfg["train"].get("eval_every_steps", 30))
    eval_max_batches = int(cfg["train"].get("eval_max_batches", 0))
    diag_max_batches = int(cfg["train"].get("diag_max_batches", 1))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_iters = build_train_iters(data.train_loaders)
    logical_micro_batch_size = _resolve_micro_batch_size(cfg)
    num_tasks = int(len(data.train_loaders))
    effective_batch_per_task = int((cfg.get("train", {}) or {}).get("batch_size", 0))
    device_streaming_chunk_size = _resolve_device_batch_size(
        cfg,
        effective_batch_size=max(1, effective_batch_per_task),
        logical_micro_batch_size=logical_micro_batch_size,
    )

    method_name = str(cfg["method"]["name"]).strip()
    if method_name not in {"ours", "baseline_avg", "baseline_cagrad"}:
        raise RuntimeError(f"unsupported method.name={method_name!r}")

    best_val = -1.0
    final_val = 0.0
    step_metrics: List[Dict[str, float]] = []
    final_step_metrics: Dict[str, float] = {}
    final_diagnostics: Dict[str, Any] = {}
    wall_start = time.perf_counter()
    train_step_time_sum = 0.0
    train_step_count = 0
    peak_cuda_mem_bytes = 0
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    for step in range(1, steps + 1):
        step_start = time.perf_counter()
        if warmup_steps > 0 and step <= warmup_steps:
            lr_scale = float(step) / float(max(1, warmup_steps))
        else:
            lr_scale = 1.0
        for pg in optimizer.param_groups:
            pg["lr"] = float(lr) * float(lr_scale)

        batch_by_task: Dict[str, Dict[str, torch.Tensor]] = {}
        for task_name, it in train_iters.items():
            raw = next(it)
            batch_by_task[task_name] = {
                "input_ids": raw["input_ids"],
                "attention_mask": raw["attention_mask"],
                "labels": raw["labels"],
            }

        step_diag: Dict[str, float] = {}
        if method_name == "ours":
            step_loss, step_diag = _train_step_ours(
                cfg=cfg,
                model=model,
                batch_by_task=batch_by_task,
                optimizer=optimizer,
                device=device,
            )
        else:
            step_loss = _train_step_baseline(
                cfg=cfg,
                model=model,
                batch_by_task=batch_by_task,
                optimizer=optimizer,
                method_name=method_name,
                device=device,
            )

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_step_time_sum += float(time.perf_counter() - step_start)
        train_step_count += 1
        if device.type == "cuda":
            try:
                peak_cuda_mem_bytes = max(peak_cuda_mem_bytes, int(torch.cuda.max_memory_allocated(device)))
            except Exception:
                pass

        should_eval = False
        if eval_every_steps > 0 and step % eval_every_steps == 0:
            should_eval = True
        if step == steps:
            should_eval = True

        if should_eval:
            val_m = _evaluate_metrics(
                model,
                data.val_loaders,
                device,
                logical_micro_batch_size=logical_micro_batch_size,
                device_batch_size=device_streaming_chunk_size,
                max_batches_per_task=eval_max_batches,
            )
            tr_m = _evaluate_metrics(
                model,
                data.train_loaders,
                device,
                logical_micro_batch_size=logical_micro_batch_size,
                device_batch_size=device_streaming_chunk_size,
                max_batches_per_task=eval_max_batches,
            )
            diag_scalar: Dict[str, float] = {}
            diag_struct: Dict[str, Any] = {}
            if diag_max_batches != 0:
                diag_scalar, diag_struct = _evaluate_gradient_diagnostics(
                    model,
                    data.val_loaders,
                    device,
                    logical_micro_batch_size=logical_micro_batch_size,
                    device_batch_size=device_streaming_chunk_size,
                    max_batches_per_task=max(1, diag_max_batches),
                )

            val_acc = float(val_m["acc"])
            val_loss = float(val_m["loss"])
            tr_acc = float(tr_m["acc"])
            tr_loss = float(tr_m["loss"])

            final_val = val_acc
            best_val = max(best_val, val_acc)
            step_row = {
                "step": float(step),
                "train_step_loss": float(step_loss),
                "train_acc": tr_acc,
                "train_loss": tr_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
                **step_diag,
                **{
                    k: float(v)
                    for k, v in val_m.items()
                    if k not in {"acc", "loss"}
                },
                **diag_scalar,
            }
            step_metrics.append(step_row)
            final_step_metrics = dict(step_row)
            final_diagnostics = dict(diag_struct)

        if step % log_every == 0 or step == 1 or step == steps:
            if step_metrics:
                latest = step_metrics[-1]
                val_acc_show = float(latest["val_acc"])
                val_loss_show = float(latest["val_loss"])
                tr_acc_show = float(latest["train_acc"])
                tr_loss_show = float(latest["train_loss"])
                msg = (
                    f"[step {step:04d}/{steps}] method={method_name} "
                    f"loss={step_loss:.4f} "
                    f"train_acc={tr_acc_show:.4f} train_loss={tr_loss_show:.4f} "
                    f"val_acc={val_acc_show:.4f} val_loss={val_loss_show:.4f} "
                )
                if "router_entropy" in latest:
                    msg += f"router_H={float(latest['router_entropy']):.4f} "
                msg += f"best={best_val:.4f}"
            else:
                val_acc_show = float("nan")
                val_loss_show = float("nan")
                tr_acc_show = float("nan")
                tr_loss_show = float("nan")
                msg = (
                    f"[step {step:04d}/{steps}] method={method_name} "
                    f"loss={step_loss:.4f} "
                    f"train_acc={tr_acc_show:.4f} train_loss={tr_loss_show:.4f} "
                    f"val_acc={val_acc_show:.4f} val_loss={val_loss_show:.4f} "
                    f"best={best_val:.4f}"
                )
            print(msg)
            if step_metrics and any(k.startswith("router_load_") for k in latest.keys()):
                loads = [
                    float(latest[k])
                    for k in sorted(latest.keys())
                    if k.startswith("router_load_e")
                ]
                print(f"           router_load={loads}")

    return TrainSummary(
        best_val_acc=float(best_val),
        final_val_acc=float(final_val),
        step_metrics=step_metrics,
        final_step_metrics=final_step_metrics,
        final_diagnostics=final_diagnostics,
        overhead={
            "wall_time_sec": float(time.perf_counter() - wall_start),
            "mean_train_step_time_sec": float(train_step_time_sum / max(1, train_step_count)),
            "peak_cuda_memory_mb": float(peak_cuda_mem_bytes / (1024.0 * 1024.0)),
        },
        batch_semantics={
            "configured_effective_batch_per_task": int(effective_batch_per_task),
            "configured_effective_update_samples": int(effective_batch_per_task * max(1, num_tasks)),
            "num_tasks_per_update": int(num_tasks),
            "logical_micro_batch_size": int(logical_micro_batch_size),
            "gpu_streaming_chunk_size": int(device_streaming_chunk_size),
            "batch_hpo_semantics": "train.batch_size is the per-task effective batch size; total samples per update equals num_tasks * train.batch_size.",
            "device_transfer_semantics": "GPU staging chunks are only outer transport groups of whole logical micro-batches; they do not redefine vote partitioning or the effective update batch.",
        },
    )
