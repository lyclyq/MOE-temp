#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _csv_tokens(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="roberta-base,microsoft/deberta-v3-base,gpt2-medium")
    ap.add_argument("--textcls_datasets", default="glue_sst2,yelp_polarity,amazon_polarity,imdb")
    ap.add_argument("--out_root", default="local_datasets/textcls")
    ap.add_argument("--train_size", type=int, default=20000)
    ap.add_argument("--val_size", type=int, default=4000)
    ap.add_argument("--use_mirror", action="store_true")
    args = ap.parse_args()

    if args.use_mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required to predownload offline model assets") from e

    for model_name in _csv_tokens(args.models):
        AutoConfig.from_pretrained(model_name)
        AutoTokenizer.from_pretrained(model_name)
        AutoModel.from_pretrained(model_name)
        print(f"[model-ok] {model_name}")

    prep_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_appendix2_local_textcls.py"),
        "--out_root",
        str(args.out_root),
        "--train_size",
        str(int(args.train_size)),
        "--val_size",
        str(int(args.val_size)),
        "--datasets",
        ",".join(_csv_tokens(args.textcls_datasets)),
    ]
    subprocess.run(prep_cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
