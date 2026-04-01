#!/usr/bin/env python3
from __future__ import annotations

import os

from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer


def main() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    for model_name in ("roberta-base", "microsoft/deberta-v3-base"):
        AutoConfig.from_pretrained(model_name)
        AutoTokenizer.from_pretrained(model_name)
        AutoModel.from_pretrained(model_name)
        print(f"[model-ok] {model_name}")

    for dataset_name, subset in (
        ("glue", "sst2"),
        ("glue", "cola"),
        ("glue", "rte"),
        ("glue", "mrpc"),
    ):
        ds = load_dataset(dataset_name, subset)
        print(f"[dataset-ok] {dataset_name}/{subset} splits={list(ds.keys())}")


if __name__ == "__main__":
    main()
