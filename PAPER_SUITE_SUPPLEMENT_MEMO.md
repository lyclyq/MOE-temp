# Paper Suite Supplement Memo

This memo groups the current supplement plan into two output roots:

- single-task add and rerun:
  - `runs/paper_suite_supplement/single_task_add_rerun`
- multi-task add:
  - `runs/paper_suite_supplement/multi_task_add`

## Overall budget policy

- HPO steps: `200`
- HPO trials: `80`
- HPO seeds: `(2, 3)`
- final steps: `1000`
- final seeds: `(2, 3, 5, 7, 11)`
- eval_every: `50`

## Single-task supplement

### Additions

- add `single / roberta / cola / lora / e4 / k2`
- add `single / roberta / cola / ffn / e4 / k2`

### Rerun

- rerun `single / roberta / rte / ffn / e4 / k2 / ours`
- keep `baseline` and `cagrad` fixed
- refresh only the `ours` branch into the canonical paper-suite directory after rerun

### Notes

- `single_task` mainline currently has configs for `rte`, `mrpc`, `sst2`; this supplement adds `cola`
- a ready config is now available at:
  - `configs/singletask_cola_real.yaml`
- the `RTE + FFN + ours` rerun still depends on the baseline-centered LR sweep hook described in:
  - `RTE_FFN_OURS_RERUN_MEMO.md`

### Single-task supplement output layout

- add:
  - `runs/paper_suite_supplement/single_task_add_rerun/single_task/roberta_cola_lora_e4_k2`
  - `runs/paper_suite_supplement/single_task_add_rerun/single_task/roberta_cola_ffn_e4_k2`
- rerun scratch:
  - `runs/paper_suite_supplement/single_task_add_rerun/rerun/roberta_rte_ffn_e4_k2_ours_lr_centered`

## Multi-task supplement

### Requested mix

- requested by plan: `cola + mnli + qnli`

### Current blocker

- the current GLUE multi-task loader requires the tasks in one mix to share the same label count
- code path:
  - `src/moe_gc/data.py:397`
  - `src/moe_gc/data.py:399`
- task label counts in the current loader map:
  - `CoLA`: binary
  - `QNLI`: binary
  - `MNLI`: 3-class
- task support map:
  - `src/moe_gc/data.py:165`
  - `src/moe_gc/data.py:170`
  - `src/moe_gc/data.py:171`

### Executable substitute for now

- use `cola + qnli + qqp` as the current executable supplement mix
- reason:
  - same binary label count
  - all three are already supported by the current GLUE loader
  - it still adds a new `CoLA`-containing multi-task mix that has not been part of the current paper-suite mainline
- ready config:
  - `configs/multitask_glue3_cola_qnli_qqp_real.yaml`

### Future exact requested mix

- if we later want the exact `cola + mnli + qnli` mix, we first need:
  - mixed-label multi-task support
  - or separate task heads / per-task classifier logic
  - or a dedicated 3-class compatible redesign for the mix

### Multi-task supplement output layout

- add:
  - `runs/paper_suite_supplement/multi_task_add/multi_task/glue3_cola_qnli_qqp_roberta_lora_e4_k2`
  - `runs/paper_suite_supplement/multi_task_add/multi_task/glue3_cola_qnli_qqp_roberta_ffn_e4_k2`

## Suggested execution order

1. run the single-task `CoLA` additions
2. keep the `RTE + FFN + ours` rerun staged until the centered-LR hook is ready
3. run the multi-task binary supplement mix `cola + qnli + qqp`
4. after all supplement runs are complete, decide whether any of them should be promoted into the main paper table set

## Shell entry points

- single-task supplement:
  - `run_paper_suite_supplement_single_placeholder.sh`
- multi-task supplement:
  - `run_paper_suite_supplement_multi_placeholder.sh`
- linear master controller:
  - `run_paper_suite_supplement_master_placeholder.sh`

## Linear master order

- the linear master shell should run in this order:
  1. `RTE + FFN + ours` rerun staging step
  2. single-task `CoLA` additions
  3. multi-task supplement additions
- intended usage:
  - first source of truth is the master shell
  - single-task and multi-task placeholder shells remain available as separate entry points

## Assumptions used here

- backbone default: `roberta`
- expert types for supplement runs: `lora,ffn`
- methods: `baseline,cagrad,ours`
- these are supplement runs, not replacements for the current canonical paper-suite directories

## Remote deployment and sync

### Remote runtime setup

- remote project root:
  - `/root/Optimization/MOE-grad-conflict-routing`
- remote runtime bootstrap script:
  - `scripts/setup_remote_cn_runtime.sh`
- intended mirror policy:
  - Ubuntu apt mirror: `tsinghua` or `aliyun`
  - pip mirror: `tsinghua` or `aliyun`
- default runtime layout on the server:
  - project venv:
    - `/root/Optimization/MOE-grad-conflict-routing/.venv`
  - supplement outputs:
    - `/root/Optimization/MOE-grad-conflict-routing/runs/paper_suite_supplement`

### Auto-pull policy

- pull direction:
  - remote server -> local workstation
- local pull root:
  - `runs/paper_suite_supplement_pull/1p95p193p128`
- no-overwrite rule:
  - remote supplement runs must stay under `runs/paper_suite_supplement/...`
  - local sync uses `rsync --ignore-existing`
  - local sync target is a dedicated pull-only folder, not the canonical local `runs_server/...` tree
- pull script:
  - `scripts/pull_remote_paper_suite_supplement.sh`
- timer installer:
  - `scripts/install_paper_suite_supplement_pull_timer.sh`
- systemd units:
  - `systemd/paper-suite-supplement-pull.service`
  - `systemd/paper-suite-supplement-pull.timer`
- schedule:
  - every day at `05:00`
  - every day at `17:00`

### Operational note

- the pull timer assumes SSH key auth from local -> remote
- recommended key path:
  - `~/.ssh/moe_paper_suite_sync_ed25519`
- once the public key is appended to the remote root user's `authorized_keys`, the timer can run unattended
