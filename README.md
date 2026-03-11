# MOE Grad Conflict Routing (Scaffold)

This repo contains a runnable scaffold for the paper method in `optimization_metagraph (8).pdf` with three methods:

- `ours`: MoE routing with detached micro-batch gradient alignment penalty.
- `baseline_avg`: per-step multi-dataset average gradient direction.
- `baseline_cagrad`: per-step multi-dataset CAGrad direction.

Model controls:

- Backbone backend: `tiny` / `hf` (`hf` needs `transformers`)
- `hf` options: `model.hf_load_pretrained`, `model.hf_pretrained_name`, `model.hf_local_files_only`
- Backbone family: `roberta` / `deberta` / `distilbert`
- Expert type: `lora` / `ffn`
- Routing mode: `softmax` / `topk`

## Key training semantics

- Multi-task step: each dataset contributes **one batch per step**.
- `ours`: each dataset batch is further split into micro-batches.
- `train.micro_batch_size` controls micro granularity for both `ours` and baselines (fair comparison).
- `train.eval_every_steps` controls periodic evaluation/logging (default 30 steps).
- `ours` conflict penalty (router only):
  - `g_tilde_m = stopgrad(g_m)` from expert/shared trainable gradient vectors.
  - `G_k = sum_m p_mk * g_tilde_m`
  - `L_norm = -sum_k ||G_k||^2 / (load_k + eps)`
  - expert/shared params update from task loss only; router updates from task + `lambda * L_norm`.
- `baseline_avg`: compute one gradient vector per dataset batch, then use mean direction.
- `baseline_cagrad`: compute one gradient vector per dataset batch, then combine by CAGrad simplex solver.
  Both baselines now also run micro-batch forwards/backwards inside each task batch before forming task-level gradients.

## Quick start

```bash
cd /home/yuli0398/Optimization/MOE-grad-conflict-routing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run.py --config configs/base.yaml
```

## Run specific methods

```bash
python scripts/run.py --config configs/base.yaml --set method.name=ours
python scripts/run.py --config configs/base.yaml --set method.name=baseline_avg
python scripts/run.py --config configs/base.yaml --set method.name=baseline_cagrad
```

Each `--out` JSON now includes full reproducibility metadata:

- full resolved config (`resolved_config`) + hash (`resolved_config_sha256`)
- CLI overrides (`cli_overrides`)
- per-task train/val dataloader shuffle seeds (`data_reproducibility.tasks.*`)
- runtime environment and exact command (`reproducibility`)

## Example combinations

```bash
# deberta + LoRA experts + top-k routing
python scripts/run.py --config configs/base.yaml \
  --set model.backbone_backend=tiny \
  --set model.backbone=deberta \
  --set model.expert_type=lora \
  --set model.routing_mode=topk \
  --set model.top_k=2

# distilbert + FFN experts + softmax routing
python scripts/run.py --config configs/base.yaml \
  --set model.backbone_backend=tiny \
  --set model.backbone=distilbert \
  --set model.expert_type=ffn \
  --set model.routing_mode=softmax

# HuggingFace architecture backend (random init)
python scripts/run.py --config configs/base.yaml \
  --set model.backbone_backend=hf \
  --set model.backbone=roberta

# HuggingFace pretrained weights (if model files available)
python scripts/run.py --config configs/base.yaml \
  --set model.backbone_backend=hf \
  --set model.hf_load_pretrained=true \
  --set model.hf_pretrained_name=roberta-base
```

## Ours Path Verification

Use this to verify OURS forward/backward semantics (`L_task` + detached `L_conflict`) and inspect expert clustering dynamics.

```bash
# recommended fast smoke backbone
python scripts/verify_ours_path.py \
  --config configs/base.yaml \
  --steps 20 \
  --set model.backbone_backend=tiny \
  --set model.backbone=distilbert \
  --set model.expert_type=ffn \
  --set model.routing_mode=softmax \
  --set train.micro_batch_size=8 \
  --set method.ours.lambda_align=1e-3
```

Switch to micro-batch size 4:

```bash
python scripts/verify_ours_path.py \
  --config configs/base.yaml \
  --steps 20 \
  --set train.micro_batch_size=4
```

Generated artifacts:

- `ours_diagnostics.jsonl`: per-step metrics (`L_task`, `L_conflict`, expert logits/load/entropy, detach checks)
- `ours_diagnostics_summary.json`: pass/fail summary for detach channel and gradient drift bounds

## Glue4 Smoke (One Command)

Preset config:

- [multitask_glue4_smoke.yaml](/home/yuli0398/Optimization/MOE-grad-conflict-routing/configs/multitask_glue4_smoke.yaml)
  Uses `glue/rte, glue/mrpc, glue/sst2, glue/cola`, `batch_size=32`, `micro_batch_size=8`.

Run both checks in one shot:

```bash
./scripts/verify_multitask_glue4_pipeline.sh
```

Outputs:

- `runs/verify_glue4/baseline_selfcheck_glue4.json`
- `runs/verify_glue4/ours_diagnostics_glue4.jsonl`
- `runs/verify_glue4/ours_diagnostics_glue4_summary.json`

## 200-Step S235 + Plot

Runs methods `ours`, `baseline_avg`, `baseline_cagrad` on seeds `2,3,5` with:

- `train.steps=200`
- `train.micro_batch_size=8`
- `train.eval_every_steps=30`

and writes per-run curve CSV (`train/val acc/loss`) plus seed-mean plot with shaded band.

```bash
./scripts/run_multitask_s235_200_plot.sh
```

Outputs under `runs/runs_200_s235/`:

- `<method>_s<seed>.json`
- `<method>_s<seed>_curve.csv`
- `seed_mean_band_std.png`
- `seed_mean_band_std_summary.json`
- `router_load_summary.csv`
- `mvp_plot_manifest.json`

MVP diagnostics (auto-generated, 12 charts):

- `metric_val_acc.png`
- `metric_val_loss.png`
- `metric_train_acc.png`
- `metric_train_loss.png`
- `metric_train_minus_val_acc.png`
- `ours_lconflict_router_effect.png`
- `ours_lconflict_per_expert.png`
- `load_pie_compare_step_1_target120_actual*.png`
- `load_pie_compare_step_2_target240_actual*.png`
- `load_pie_compare_step_3_target360_actual*.png`
- `load_pie_compare_final_step_*.png`
- `expert_seed_variance_score_05final_05max.png`
