#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import shlex
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moe_gc import MoEClassifier, build_multitask_data, load_config, train
from path_utils import resolve_runs_path


def _write_curve_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = ["step", "train_step_loss", "train_acc", "train_loss", "val_acc", "val_loss"]
    extra_fields = sorted({k for r in rows for k in r.keys() if k not in set(base_fields)})
    fields = base_fields + extra_fields
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _config_sha256(cfg: Dict[str, Any]) -> str:
    txt = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def _runtime_info() -> Dict[str, Any]:
    import torch

    out: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        out["cuda_device_count"] = int(torch.cuda.device_count())
        out["cuda_device_names"] = [str(torch.cuda.get_device_name(i)) for i in range(int(torch.cuda.device_count()))]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--set", action="append", default=[], help="override key=value")
    p.add_argument("--out", type=str, default="manual_run/run_summary.json")
    p.add_argument("--curve_out", type=str, default="", help="optional csv path for step metrics")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    data = build_multitask_data(cfg)

    model = MoEClassifier(cfg["model"])

    summary = train(cfg, data, model)
    out_path = resolve_runs_path(args.out)
    curve_path = resolve_runs_path(args.curve_out) if args.curve_out else out_path.with_name(f"{out_path.stem}_curve.csv")
    payload = {
        "method": cfg["method"]["name"],
        "best_val_acc": summary.best_val_acc,
        "final_val_acc": summary.final_val_acc,
        "num_eval_points": len(summary.step_metrics),
        "seed": int(cfg.get("seed", 0)),
        "config_path": str(Path(args.config)),
        "config_path_resolved": str(Path(args.config).resolve()),
        "cli_overrides": list(args.set),
        "resolved_config": cfg,
        "resolved_config_sha256": _config_sha256(cfg),
        "data_reproducibility": dict(data.reproducibility),
        "reproducibility": {
            "command": " ".join(shlex.quote(x) for x in sys.argv),
            "cwd": str(Path.cwd()),
            "summary_path": str(out_path.resolve()),
            "curve_path": str(curve_path.resolve()),
            "runtime": _runtime_info(),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_curve_csv(curve_path, summary.step_metrics)
    print(f"saved summary -> {out_path}")
    print(f"saved curve -> {curve_path}")


if __name__ == "__main__":
    main()
