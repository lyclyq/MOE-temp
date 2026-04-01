# RTE FFN Ours Rerun Memo

This memo is a hold-place plan for the `single_task / roberta / rte / ffn / e4 / k2` rerun.

## Rerun list

- Add `single_task / roberta / rte / ffn / e4 / k2 / ours` to the rerun queue.
- Do not delete the `RTE + FFN` table entry yet.
- Replace only the `ours` branch after rerun is complete.
- Keep the existing `baseline` and `cagrad` data from:
  - `runs_server/paper_suite/single_gpu/single_task/roberta_rte_ffn_e4_k2`

## Reason

- `RTE + FFN + ours` is unstable in the current paper-suite result.
- The existing canonical comparison should keep `baseline` and `cagrad` fixed.
- The rerun should only refresh `ours`, then regenerate the mixed plots/tables.

## Fixed target setting

- task: `rte`
- backbone: `roberta`
- expert_type: `ffn`
- num_experts: `4`
- top_k: `2`
- routing_mode: `topk`
- methods to rerun: `ours` only

## HPO and final budget

- HPO seeds: `(2, 3)`
- HPO steps: `200`
- HPO trials: `80`
- local top-k knobs for multi-dim recheck: `4`
- local grid points per knob: `3`
- local grid size for `ours`: `3^4 = 81`
- final seeds: `(2, 3, 5, 7, 11)`
- final steps: `1000`
- eval_every: `50`

## LR policy for this rerun

- Use the current `baseline` best LR as the center for the `ours` LR sweep.
- Baseline LR center for `roberta_rte_ffn_e4_k2`:
  - `2.5551784142791192e-05`
  - source: `runs_server/paper_suite/single_gpu/single_task/roberta_rte_ffn_e4_k2/hpo/baseline/best_config.json`
- Expand from that center exponentially to the same global boundaries that the current pipeline uses.
- Current pipeline LR boundaries:
  - low: `2.0e-6`
  - high: `5.0e-3`
  - source: `scripts/pipeline_hpo_final_plot.py`

## Important note about the current pipeline

- The current `scripts/pipeline_hpo_final_plot.py` supports:
  - `--methods ours`
  - `--hpo_trials 80`
  - `--hpo_steps 200`
  - `--final_seeds 2,3,5,7,11`
  - `--final_steps 1000`
  - `--local_topk 4`
  - `--local_grid_points 3`
- But it does not yet support a first-class "baseline-centered coarse LR sweep".
- If run today without code changes, setting `train.lr=<baseline_lr>` only changes the anchor/base value, not the full coarse LR sampling rule.
- Therefore the bash file added next to this memo is a staging script, not a final run command yet.

## Canonical output plan

- Scratch rerun directory:
  - `runs/paper_suite_rerun/single_gpu/single_task/roberta_rte_ffn_e4_k2_ours_lr_centered`
- Canonical paper-suite directory to refresh:
  - `runs_server/paper_suite/single_gpu/single_task/roberta_rte_ffn_e4_k2`

## What should be replaced in the canonical directory

- Replace `hpo/ours/*`
- Replace `final/ours_s2.json`
- Replace `final/ours_s2_curve.csv`
- Replace `final/ours_s3.json`
- Replace `final/ours_s3_curve.csv`
- Replace `final/ours_s5.json`
- Replace `final/ours_s5_curve.csv`
- Replace `final/ours_s7.json`
- Replace `final/ours_s7_curve.csv`
- Replace `final/ours_s11.json`
- Replace `final/ours_s11_curve.csv`

## What should be regenerated after replacement

- `final/final_per_run.csv`
- `final/final_agg.csv`
- `final/final_best_configs_snapshot.json`
- `final/seed_mean_band_std.png`
- `final/seed_mean_band_std_summary.json`
- `final/seed_mean_band_val_last.csv`
- `final/seed_mean_band_std_val_only.png`
- `final/router_load_summary.csv`
- `final/paper_metrics_manifest.json`
- `final/metric_*.png`
- `final/curve_*.png`
- `final/task_conflict_matrix_final.*`
- `final/inter_expert_similarity_final.*`
- `final/expert_purity_final.*`
- `final/overhead_summary.csv`
- `final/mechanism_summary.csv`
- `final/task_metrics_final.csv`
- `final/mvp_plot_manifest.json`
- `final/load_pie_compare_*.png`
- `final/ours_lconflict_*.png`

## Optional HPO-side refresh

- If the canonical `hpo/` directory is refreshed, also update:
  - `hpo/best_configs.json`
  - `hpo/hpo_agg_all_methods.csv`
  - optionally regenerate:
    - `hpo/hpo_best_params.csv`
    - `hpo/hpo_best_scores.png`

## Execution order when this is actually run

1. Add the baseline-centered LR sweep hook or custom candidate generation for `ours`.
2. Run the `ours`-only rerun into the scratch directory.
3. Backup the current canonical `ours` artifacts.
4. Copy scratch `hpo/ours` and `final/ours_s*.{json,csv}` into the canonical directory.
5. Merge the scratch `ours` row into canonical `final_per_run.csv`.
6. Merge the scratch `ours` row into canonical `final_agg.csv`.
7. Merge the scratch `ours` entry into canonical `final_best_configs_snapshot.json`.
8. Re-run plotters on the canonical `final/` directory with methods `baseline,cagrad,ours`.
9. Sanity check that `baseline` and `cagrad` files did not change.

## Quick sanity checklist

- `baseline` and `cagrad` data stay untouched.
- Only `ours` is rerun.
- `ours` HPO uses `(2, 3)` and `200` steps.
- `ours` final uses `(2, 3, 5, 7, 11)` and `1000` steps.
- LR sweep center is the baseline best LR, not the config default LR.
- Final plots are regenerated from the mixed canonical directory.
