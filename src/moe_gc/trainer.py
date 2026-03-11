from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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
    max_batches_per_task: int = 0,
) -> Dict[str, object]:
    model.eval()
    vals_acc: List[float] = []
    vals_loss: List[float] = []
    router_sum: torch.Tensor | None = None
    router_cnt = 0
    router_ent_sum = 0.0
    with torch.no_grad():
        for loader in loaders.values():
            ok = 0
            tot = 0
            n_batches = 0
            loss_sum = 0.0
            for input_ids, attention_mask, labels in loader:
                if max_batches_per_task > 0 and n_batches >= max_batches_per_task:
                    break
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = out["logits"]
                pred = torch.argmax(logits, dim=-1)
                ok += int((pred == labels).sum().item())
                tot += int(labels.numel())
                loss_sum += float(_scalar_loss(out["loss"]).detach().item())
                n_batches += 1

                probs = out.get("router_probs")
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
            vals_acc.append(float(ok / max(1, tot)))
            vals_loss.append(float(loss_sum / max(1, n_batches)))
    model.train()
    out: Dict[str, object] = {
        "acc": float(sum(vals_acc) / max(1, len(vals_acc))),
        "loss": float(sum(vals_loss) / max(1, len(vals_loss))),
    }
    if router_sum is not None and router_cnt > 0:
        loads = router_sum / float(router_cnt)
        out["router_entropy"] = float(router_ent_sum / float(router_cnt))
        for i in range(int(loads.numel())):
            out[f"router_load_e{i}"] = float(loads[i].item())
    return out


@dataclass
class TrainSummary:
    best_val_acc: float
    final_val_acc: float
    step_metrics: List[Dict[str, float]]


def _train_step_baseline(
    *,
    cfg: dict,
    model: torch.nn.Module,
    batch_by_task: Dict[str, Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    method_name: str,
) -> float:
    micro_bs = _resolve_micro_batch_size(cfg)
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

        windows = _split_indices(bs, micro_bs)
        if not windows:
            windows = [(0, bs)]

        task_grad_accum: Dict[torch.nn.Parameter, torch.Tensor] = {}
        task_loss_wsum = 0.0
        wsum = 0.0

        for s, e in windows:
            ids_m = input_ids[s:e]
            mask_m = attention_mask[s:e]
            labels_m = labels[s:e]
            w = float(e - s) / float(bs)
            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=ids_m,
                attention_mask=mask_m,
                labels=labels_m,
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
) -> Tuple[float, Dict[str, float]]:
    ours_cfg = cfg["method"]["ours"]
    micro_bs = _resolve_micro_batch_size(cfg)
    lam = float(ours_cfg.get("lambda_align", 0.0))
    eps = float(ours_cfg.get("eps", 1.0e-8))

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
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        for s, e in _split_indices(int(input_ids.shape[0]), micro_bs):
            ids_m = input_ids[s:e]
            mask_m = attention_mask[s:e]
            labels_m = labels[s:e]
            w = float(e - s) / float(total_samples)

            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=ids_m,
                attention_mask=mask_m,
                labels=labels_m,
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
        per_expert_conflict = -(G.pow(2).sum(dim=1) / load)
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_iters = build_train_iters(data.train_loaders)

    method_name = str(cfg["method"]["name"]).strip()
    if method_name not in {"ours", "baseline_avg", "baseline_cagrad"}:
        raise RuntimeError(f"unsupported method.name={method_name!r}")

    best_val = -1.0
    final_val = 0.0
    step_metrics: List[Dict[str, float]] = []

    for step in range(1, steps + 1):
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
                "input_ids": raw["input_ids"].to(device),
                "attention_mask": raw["attention_mask"].to(device),
                "labels": raw["labels"].to(device),
            }

        step_diag: Dict[str, float] = {}
        if method_name == "ours":
            step_loss, step_diag = _train_step_ours(
                cfg=cfg,
                model=model,
                batch_by_task=batch_by_task,
                optimizer=optimizer,
            )
        else:
            step_loss = _train_step_baseline(
                cfg=cfg,
                model=model,
                batch_by_task=batch_by_task,
                optimizer=optimizer,
                method_name=method_name,
            )

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        should_eval = False
        if eval_every_steps > 0 and step % eval_every_steps == 0:
            should_eval = True
        if step == steps:
            should_eval = True

        if should_eval:
            val_m = _evaluate_metrics(model, data.val_loaders, device, max_batches_per_task=eval_max_batches)
            tr_m = _evaluate_metrics(model, data.train_loaders, device, max_batches_per_task=eval_max_batches)

            val_acc = float(val_m["acc"])
            val_loss = float(val_m["loss"])
            tr_acc = float(tr_m["acc"])
            tr_loss = float(tr_m["loss"])

            final_val = val_acc
            best_val = max(best_val, val_acc)
            step_metrics.append(
                {
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
                        if k.startswith("router_load_") or k == "router_entropy"
                    },
                }
            )

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
    )
