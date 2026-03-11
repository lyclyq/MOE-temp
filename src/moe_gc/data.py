from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class MultiTaskData:
    train_loaders: Dict[str, DataLoader]
    val_loaders: Dict[str, DataLoader]
    reproducibility: Dict[str, Any] = field(default_factory=dict)


def _task_seed(name: str) -> int:
    return int(sum(ord(c) for c in name) % (2**31 - 1))


def _loader_seed(cfg: dict, *, task: str, split: str) -> int:
    mod = (2**31 - 1)
    if split not in {"train", "val"}:
        raise RuntimeError(f"split must be train/val, got {split!r}")
    dcfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    global_seed = int(cfg.get("seed", 0))
    base = int(dcfg.get("loader_seed_base", global_seed * 1009 + 17))
    split_off = 0 if split == "train" else 1
    return int((base + 8191 * _task_seed(task) + split_off) % mod)


def _make_task_dataset(
    *,
    name: str,
    n: int,
    seq_len: int,
    vocab_size: int,
    num_classes: int,
    base_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    task_off = _task_seed(name)
    # Keep per-task label rule fixed across train/val; only sample draws/noise differ by base_seed.
    g_rule = torch.Generator().manual_seed(int(777_777 + task_off))
    g_data = torch.Generator().manual_seed(int(base_seed + task_off))

    input_ids = torch.randint(low=0, high=vocab_size, size=(n, seq_len), generator=g_data, dtype=torch.long)
    attention_mask = torch.ones((n, seq_len), dtype=torch.long)

    # Token-table rule: produces task-specific label patterns from token ids.
    cls_table = torch.randn((vocab_size, num_classes), generator=g_rule, dtype=torch.float32)
    logits = cls_table[input_ids].mean(dim=1)
    logits = logits + 0.15 * torch.randn((n, num_classes), generator=g_data)
    labels = torch.argmax(logits, dim=1).to(torch.long)
    return input_ids, attention_mask, labels


def _iter_forever(loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for input_ids, attention_mask, labels in loader:
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }


def build_train_iters(loaders: Dict[str, DataLoader]) -> Dict[str, Iterator[Dict[str, torch.Tensor]]]:
    return {k: _iter_forever(v) for k, v in loaders.items()}


def build_synthetic_multitask(cfg: dict) -> MultiTaskData:
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["train"]

    datasets: List[str] = list(dcfg["datasets"])
    if not datasets:
        raise RuntimeError("data.datasets cannot be empty")

    batch_size = int(tcfg["batch_size"])
    train_size = int(dcfg["train_size"])
    val_size = int(dcfg["val_size"])
    seq_len = int(mcfg.get("seq_len", 64))
    vocab_size = int(mcfg.get("vocab_size", 30522))
    num_classes = int(mcfg["num_classes"])
    seed = int(cfg.get("seed", 0))

    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}
    repro: Dict[str, Any] = {
        "data_source": "synthetic",
        "global_seed": seed,
        "loader_seed_base": int(dcfg.get("loader_seed_base", seed * 1009 + 17)),
        "task_order": list(datasets),
        "tasks": {},
    }

    for task in datasets:
        task_off = _task_seed(task)
        tr_ids, tr_mask, tr_y = _make_task_dataset(
            name=task,
            n=train_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_classes=num_classes,
            base_seed=seed,
        )
        va_ids, va_mask, va_y = _make_task_dataset(
            name=task,
            n=val_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_classes=num_classes,
            base_seed=seed + 991,
        )
        tr = TensorDataset(tr_ids, tr_mask, tr_y)
        va = TensorDataset(va_ids, va_mask, va_y)

        tr_loader_seed = _loader_seed(cfg, task=task, split="train")
        va_loader_seed = _loader_seed(cfg, task=task, split="val")
        tr_gen = torch.Generator().manual_seed(int(tr_loader_seed))
        va_gen = torch.Generator().manual_seed(int(va_loader_seed))
        train_loaders[task] = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=False, generator=tr_gen)
        val_loaders[task] = DataLoader(va, batch_size=batch_size, shuffle=False, drop_last=False, generator=va_gen)
        repro["tasks"][task] = {
            "task_seed_offset": int(task_off),
            "synthetic_rule_seed": int(777_777 + task_off),
            "synthetic_train_data_seed": int(seed + task_off),
            "synthetic_val_data_seed": int(seed + 991 + task_off),
            "train_loader_shuffle_seed": int(tr_loader_seed),
            "val_loader_shuffle_seed": int(va_loader_seed),
        }

    return MultiTaskData(train_loaders=train_loaders, val_loaders=val_loaders, reproducibility=repro)


def _normalize_glue_task(name: str) -> str:
    s = str(name).strip()
    if "/" in s:
        pref, task = s.split("/", 1)
        if pref.lower() != "glue":
            raise RuntimeError(f"only glue/<task> is supported, got: {name!r}")
        s = task
    return s.lower()


def _sentence_keys(task: str) -> Tuple[str, Optional[str]]:
    mp: Dict[str, Tuple[str, Optional[str]]] = {
        "cola": ("sentence", None),
        "sst2": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "stsb": ("sentence1", "sentence2"),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    if task not in mp:
        raise RuntimeError(f"unsupported GLUE task: {task!r}")
    return mp[task]


def _hash_token(tok: str, vocab_size: int) -> int:
    if vocab_size <= 2:
        return 1
    hv = hashlib.md5(tok.encode("utf-8")).digest()
    x = int.from_bytes(hv[:8], byteorder="little", signed=False)
    return 2 + (x % (vocab_size - 2))


def _simple_split(text: str) -> List[str]:
    return [t for t in str(text).strip().lower().split() if t]


def _encode_pair_hashed(
    text_a: str,
    text_b: Optional[str],
    seq_len: int,
    vocab_size: int,
) -> Tuple[List[int], List[int]]:
    toks = _simple_split(text_a)
    ids = [_hash_token(t, vocab_size) for t in toks]
    if text_b is not None:
        ids.append(1)
        ids.extend(_hash_token(t, vocab_size) for t in _simple_split(text_b))

    ids = ids[:seq_len]
    mask = [1] * len(ids)
    if len(ids) < seq_len:
        pad_n = seq_len - len(ids)
        ids.extend([0] * pad_n)
        mask.extend([0] * pad_n)
    return ids, mask


def _hf_tokenizer_name(model_cfg: dict) -> str:
    provided = str(model_cfg.get("hf_pretrained_name", "") or "").strip()
    if provided:
        return provided
    bb = str(model_cfg.get("backbone", "deberta")).strip().lower()
    mp = {
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-base",
        "distilbert": "distilbert-base-uncased",
    }
    if bb not in mp:
        raise RuntimeError(f"unsupported backbone for hf tokenizer: {bb!r}")
    return mp[bb]


def _build_glue_task_loader(
    *,
    task_name: str,
    cfg: dict,
) -> Tuple[DataLoader, DataLoader, int, Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "datasets is required for data.source=glue. Install with: pip install datasets"
        ) from e

    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    batch_size = int(tcfg["batch_size"])
    seq_len = int(mcfg.get("seq_len", mcfg.get("max_seq_len", 64)))
    backend = str(mcfg.get("backbone_backend", "tiny")).strip().lower()
    vocab_size = int(mcfg.get("vocab_size", 30522))

    task = _normalize_glue_task(task_name)
    s1_key, s2_key = _sentence_keys(task)
    ds = load_dataset("glue", task)

    if "train" not in ds:
        raise RuntimeError(f"GLUE {task} missing train split")
    if "validation" in ds:
        va_name = "validation"
    elif "validation_matched" in ds:
        va_name = "validation_matched"
    else:
        raise RuntimeError(f"GLUE {task} missing validation split")

    tr = ds["train"]
    va = ds[va_name]

    train_cap = int(dcfg.get("train_size", 0))
    val_cap = int(dcfg.get("val_size", 0))
    if train_cap > 0 and train_cap < len(tr):
        tr = tr.select(range(train_cap))
    if val_cap > 0 and val_cap < len(va):
        va = va.select(range(val_cap))

    num_labels = int(ds["train"].features["label"].num_classes)

    if backend == "hf":
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "transformers is required for model.backbone_backend=hf"
            ) from e
        tok_name = _hf_tokenizer_name(mcfg)
        tok = AutoTokenizer.from_pretrained(
            tok_name,
            local_files_only=bool(mcfg.get("hf_local_files_only", False)),
            use_fast=True,
        )

        def encode_row(ex: dict) -> Tuple[List[int], List[int]]:
            a = str(ex[s1_key])
            b = None if s2_key is None else str(ex[s2_key])
            out = tok(
                a,
                b,
                truncation=True,
                max_length=seq_len,
                padding="max_length",
            )
            return list(out["input_ids"]), list(out["attention_mask"])

    else:
        def encode_row(ex: dict) -> Tuple[List[int], List[int]]:
            a = str(ex[s1_key])
            b = None if s2_key is None else str(ex[s2_key])
            return _encode_pair_hashed(a, b, seq_len, vocab_size)

    def to_tensor_dataset(split) -> TensorDataset:
        ids_all: List[List[int]] = []
        mask_all: List[List[int]] = []
        y_all: List[int] = []
        for ex in split:
            y = int(ex["label"])
            if y < 0:
                continue
            ids, mask = encode_row(ex)
            ids_all.append(ids)
            mask_all.append(mask)
            y_all.append(y)

        input_ids = torch.tensor(ids_all, dtype=torch.long)
        attention_mask = torch.tensor(mask_all, dtype=torch.long)
        labels = torch.tensor(y_all, dtype=torch.long)
        return TensorDataset(input_ids, attention_mask, labels)

    tr_ds = to_tensor_dataset(tr)
    va_ds = to_tensor_dataset(va)

    tr_loader_seed = _loader_seed(cfg, task=task_name, split="train")
    va_loader_seed = _loader_seed(cfg, task=task_name, split="val")
    tr_gen = torch.Generator().manual_seed(int(tr_loader_seed))
    va_gen = torch.Generator().manual_seed(int(va_loader_seed))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=tr_gen)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False, generator=va_gen)
    repro = {
        "task": str(task_name),
        "num_labels": int(num_labels),
        "train_size": int(len(tr_ds)),
        "val_size": int(len(va_ds)),
        "train_loader_shuffle_seed": int(tr_loader_seed),
        "val_loader_shuffle_seed": int(va_loader_seed),
    }
    return tr_loader, va_loader, num_labels, repro


def build_glue_multitask(cfg: dict) -> MultiTaskData:
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tasks: List[str] = list(dcfg.get("datasets", []))
    if not tasks:
        raise RuntimeError("data.datasets cannot be empty for data.source=glue")

    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}
    num_labels_ref: Optional[int] = None
    repro: Dict[str, Any] = {
        "data_source": "glue",
        "global_seed": int(cfg.get("seed", 0)),
        "loader_seed_base": int(dcfg.get("loader_seed_base", int(cfg.get("seed", 0)) * 1009 + 17)),
        "task_order": list(tasks),
        "tasks": {},
    }
    for name in tasks:
        tr_loader, va_loader, nlab, task_repro = _build_glue_task_loader(task_name=name, cfg=cfg)
        train_loaders[name] = tr_loader
        val_loaders[name] = va_loader
        repro["tasks"][name] = task_repro
        if num_labels_ref is None:
            num_labels_ref = int(nlab)
        elif int(nlab) != int(num_labels_ref):
            raise RuntimeError(
                f"mixed label count across tasks: got {nlab} for {name}, expected {num_labels_ref}"
            )

    if num_labels_ref is None:
        raise RuntimeError("no glue task loaded")
    cfg_num_classes = int(mcfg.get("num_classes", num_labels_ref))
    if cfg_num_classes != int(num_labels_ref):
        raise RuntimeError(
            f"model.num_classes={cfg_num_classes} mismatch GLUE labels={num_labels_ref}. "
            "Please set model.num_classes accordingly."
        )
    return MultiTaskData(train_loaders=train_loaders, val_loaders=val_loaders, reproducibility=repro)


def build_multitask_data(cfg: dict) -> MultiTaskData:
    dcfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    source = str(dcfg.get("source", "synthetic")).strip().lower()
    if source == "synthetic":
        return build_synthetic_multitask(cfg)
    if source == "glue":
        return build_glue_multitask(cfg)
    raise RuntimeError(f"unsupported data.source={source!r}, expected synthetic|glue")
