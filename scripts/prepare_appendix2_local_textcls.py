#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _save_subset(ds, out_dir: Path, train_n: int, val_n: int, val_split: str) -> None:
    from datasets import DatasetDict

    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    subset = DatasetDict(
        train=ds["train"].select(range(min(train_n, len(ds["train"])))),
        **{val_split: ds[val_split].select(range(min(val_n, len(ds[val_split]))))},
    )
    subset.save_to_disk(str(out_dir))


def _normalize_dataset_token(name: str) -> str:
    token = str(name).strip().lower()
    aliases = {
        "sst2": "glue_sst2",
        "glue/sst2": "glue_sst2",
        "glue_sst2": "glue_sst2",
        "imdb": "imdb",
        "yelp": "yelp_polarity",
        "yelp_polarity": "yelp_polarity",
        "amazon": "amazon_polarity",
        "amazon_polarity": "amazon_polarity",
    }
    if token not in aliases:
        raise RuntimeError(f"unsupported textcls dataset token: {name!r}")
    return aliases[token]


def _dataset_specs() -> dict[str, tuple[str, str | None, str]]:
    return {
        "glue_sst2": ("glue", "sst2", "validation"),
        "imdb": ("imdb", None, "test"),
        "yelp_polarity": ("yelp_polarity", None, "test"),
        "amazon_polarity": ("amazon_polarity", None, "test"),
    }


def _parse_datasets(csv_text: str) -> list[str]:
    datasets: list[str] = []
    for part in str(csv_text).split(","):
        token = part.strip()
        if not token:
            continue
        datasets.append(_normalize_dataset_token(token))
    if not datasets:
        raise RuntimeError("empty --datasets")
    return list(dict.fromkeys(datasets))


def _datasets_cache_roots() -> list[Path]:
    candidates = [
        str(os.environ.get("HF_DATASETS_CACHE", "")).strip(),
        str(Path(os.environ["HF_HOME"]) / "datasets") if os.environ.get("HF_HOME") else "",
        str(Path.home() / ".cache" / "huggingface" / "datasets"),
        str(Path.home() / "hf_cache" / "datasets"),
    ]
    roots: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        if not item:
            continue
        path = Path(item).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        roots.append(path)
    return roots


def _purge_dataset_cache(dataset_name: str) -> None:
    aliases = {
        str(dataset_name).strip(),
        str(dataset_name).strip().replace("/", "_"),
    }
    for root in _datasets_cache_roots():
        if not root.exists():
            continue
        for alias in aliases:
            target = root / alias
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)


def _load_dataset_with_repair(load_dataset, dataset_name: str, subset: str | None):
    display_name = dataset_name if subset is None else f"{dataset_name}/{subset}"
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            kwargs = {}
            if attempt > 0:
                kwargs["download_mode"] = "force_redownload"
            return load_dataset(dataset_name, subset, **kwargs) if subset is not None else load_dataset(dataset_name, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt > 0:
                break
            print(f"[prepare-repair] clearing dataset cache for {display_name}: {type(e).__name__}: {e}")
            _purge_dataset_cache(dataset_name)
    assert last_error is not None
    raise last_error


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="local_datasets/textcls")
    ap.add_argument("--train_size", type=int, default=20000)
    ap.add_argument("--val_size", type=int, default=4000)
    ap.add_argument("--datasets", default="glue_sst2,yelp_polarity,amazon_polarity")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("datasets is required to prepare local text classification subsets") from e

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    mirror_enabled = bool(str(os.environ.get("HF_ENDPOINT", "")).strip())
    specs = _dataset_specs()
    dataset_names = _parse_datasets(args.datasets)

    for name in dataset_names:
        dataset_name, subset, val_split = specs[name]
        display_name = dataset_name if subset is None else f"{dataset_name}/{subset}"
        source_desc = "mirror" if mirror_enabled else "default hub"
        print(f"[prepare] downloading {display_name} via {source_desc}")
        ds = _load_dataset_with_repair(load_dataset, dataset_name, subset)
        _save_subset(ds, out_root / name, args.train_size, args.val_size, val_split)
        print((out_root / name).resolve())


if __name__ == "__main__":
    main()
