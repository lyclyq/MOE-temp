Merged on 2026-04-06.

This canonical directory keeps the original `baseline` and `cagrad` branches from:

- `paperoutput/runs_server/paper_suite/single_gpu/single_task/roberta_rte_ffn_e4_k2`

Its `ours` branch was refreshed from:

- `paperoutput/runs_server/paper_suite/single_gpu/rerun/roberta_rte_ffn_e4_k2_ours_lr_centered`

Replaced content:

- `hpo/ours/*`
- `hpo/best_configs.json` for `ours`
- `hpo/hpo_agg_all_methods.csv` rows for `ours`
- `final/ours_s{2,3,5,7,11}.json`
- `final/ours_s{2,3,5,7,11}_curve.csv`

Regenerated content:

- `final/final_per_run.csv`
- `final/final_agg.csv`
- `final/final_best_configs_snapshot.json`
- `final/router_load_summary.csv`
- `final/seed_mean_band_*`
- `final/paper_metrics_manifest.json`
- `final/mvp_plot_manifest.json`
- all derived plots and summary CSVs under `final/`

Key metric change for `ours`:

- old `final_mean`: `0.6837545126353791`
- new `final_mean`: `0.7494584837545126`
