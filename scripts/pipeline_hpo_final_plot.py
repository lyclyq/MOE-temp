#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from path_utils import resolve_runs_path


ROOT = Path(__file__).resolve().parents[1]


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for tok in str(s).replace(" ", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def _parse_csv_strs(s: str) -> List[str]:
    out: List[str] = []
    for tok in str(s).replace(" ", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(t)
    return out


def _fmt_f(v: float) -> str:
    return f"{float(v):.10g}"


def _canonical_method_token(tok: str) -> str:
    t = str(tok).strip().lower()
    alias = {
        "baseline": "baseline",
        "baseline_avg": "baseline",
        "avg": "baseline",
        "cagrad": "cagrad",
        "baseline_cagrad": "cagrad",
        "ours": "ours",
        "ablation": "ablation",
        "ablation_nonorm": "ablation",
        "ablation_non_normalized": "ablation",
        "ours_nonorm": "ablation",
        "ours_non_normalized": "ablation",
        "non_normalized": "ablation",
    }
    if t not in alias:
        raise RuntimeError(f"unsupported method token: {tok!r}")
    return alias[t]


@dataclass(frozen=True)
class KnobSpec:
    key: str
    scale: str  # "log" | "linear"
    lo: float
    hi: float
    local_factor: float = 2.0
    local_span: float = 0.15

    def clip(self, x: float) -> float:
        return float(min(self.hi, max(self.lo, x)))

    def sample(self, rng: random.Random) -> float:
        if self.scale == "log":
            lo = math.log10(self.lo)
            hi = math.log10(self.hi)
            return float(10.0 ** rng.uniform(lo, hi))
        if self.scale == "linear":
            return float(rng.uniform(self.lo, self.hi))
        raise RuntimeError(f"unknown scale={self.scale!r}")

    def local_points(self, center: float, points: int) -> List[float]:
        if points <= 1:
            return [self.clip(center)]
        half = points // 2
        idxs = list(range(-half, half + 1))
        if len(idxs) > points:
            idxs = idxs[:points]
        vals: List[float] = []
        for i in idxs:
            if self.scale == "log":
                v = center * (self.local_factor ** i)
            else:
                delta = max((self.hi - self.lo) * self.local_span, 1.0e-12)
                v = center + float(i) * delta
            vals.append(self.clip(float(v)))
        uniq = sorted({round(v, 12) for v in vals})
        return [float(v) for v in uniq]


@dataclass(frozen=True)
class MethodSpec:
    alias: str
    method_name: str
    fixed_overrides: Dict[str, str]
    knobs: Tuple[KnobSpec, KnobSpec, KnobSpec]


@dataclass
class Candidate:
    cid: str
    stage: str
    params: Dict[str, float]


@dataclass
class RunJob:
    order: int
    method_alias: str
    candidate_id: str
    seed: int
    steps: int
    eval_every: int
    out_json: Path
    out_curve: Path
    set_kv: List[str]


@dataclass
class RunResult:
    method_alias: str
    candidate_id: str
    seed: int
    best_val_acc: float
    final_val_acc: float
    score_05_05: float
    out_json: str
    out_curve: str
    reused: bool


def _method_specs() -> Dict[str, MethodSpec]:
    lr_knob = KnobSpec("train.lr", "log", 2.0e-6, 5.0e-3, local_factor=2.0)

    return {
        "baseline": MethodSpec(
            alias="baseline",
            method_name="baseline_avg",
            fixed_overrides={},
            knobs=(
                lr_knob,
                KnobSpec("train.weight_decay", "linear", 0.0, 0.05, local_span=0.20),
                KnobSpec("train.grad_clip", "linear", 0.5, 2.0, local_span=0.20),
            ),
        ),
        "cagrad": MethodSpec(
            alias="cagrad",
            method_name="baseline_cagrad",
            fixed_overrides={},
            knobs=(
                lr_knob,
                KnobSpec("method.baseline_cagrad.c", "linear", 0.1, 1.0, local_span=0.20),
                KnobSpec("method.baseline_cagrad.inner_lr", "log", 1.0e-2, 5.0e-1, local_factor=2.0),
            ),
        ),
        "ours": MethodSpec(
            alias="ours",
            method_name="ours",
            fixed_overrides={"method.ours.use_load_norm": "true"},
            knobs=(
                lr_knob,
                KnobSpec("method.ours.lambda_align", "log", 1.0e-5, 3.0e-2, local_factor=2.0),
                KnobSpec("method.ours.eps", "log", 1.0e-9, 1.0e-6, local_factor=3.0),
            ),
        ),
        "ablation": MethodSpec(
            alias="ablation",
            method_name="ours",
            fixed_overrides={"method.ours.use_load_norm": "false"},
            knobs=(
                lr_knob,
                KnobSpec("method.ours.lambda_align", "log", 1.0e-5, 3.0e-2, local_factor=2.0),
                KnobSpec("method.ours.eps", "log", 1.0e-9, 1.0e-6, local_factor=3.0),
            ),
        ),
    }


def _load_base_config(config_path: str, cli_set: Sequence[str]) -> Dict[str, Any]:
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from moe_gc import load_config  # type: ignore

    return load_config(config_path, list(cli_set))


def _get_by_path(d: Dict[str, Any], key: str) -> Any:
    cur: Any = d
    for tok in key.split("."):
        if not isinstance(cur, dict) or tok not in cur:
            return None
        cur = cur[tok]
    return cur


def _candidate_fingerprint(params: Dict[str, float]) -> str:
    items = sorted((str(k), float(v)) for k, v in params.items())
    return "|".join(f"{k}={v:.12g}" for k, v in items)


def _build_coarse_candidates(
    *,
    spec: MethodSpec,
    base_cfg: Dict[str, Any],
    trials: int,
    seed: int,
) -> List[Candidate]:
    if trials <= 0:
        raise RuntimeError("hpo trials must be > 0")
    rng = random.Random(int(seed))
    out: List[Candidate] = []
    seen: set[str] = set()

    default_params: Dict[str, float] = {}
    for kb in spec.knobs:
        cur = _get_by_path(base_cfg, kb.key)
        if isinstance(cur, (int, float)):
            v = kb.clip(float(cur))
        else:
            if kb.scale == "log":
                v = kb.clip(math.sqrt(kb.lo * kb.hi))
            else:
                v = kb.clip(0.5 * (kb.lo + kb.hi))
        default_params[kb.key] = float(v)
    fp = _candidate_fingerprint(default_params)
    seen.add(fp)
    out.append(Candidate(cid="coarse_default", stage="coarse", params=default_params))

    idx = 0
    while len(out) < trials:
        cand = {kb.key: float(kb.sample(rng)) for kb in spec.knobs}
        fp = _candidate_fingerprint(cand)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(Candidate(cid=f"coarse_{idx:03d}", stage="coarse", params=cand))
        idx += 1
    return out


def _build_local_candidates(
    *,
    spec: MethodSpec,
    top_candidates: Sequence[Candidate],
    already_seen: set[str],
    grid_points: int,
) -> List[Candidate]:
    out: List[Candidate] = []
    local_idx = 0
    for rank, cand in enumerate(top_candidates):
        grids: List[List[float]] = []
        for kb in spec.knobs:
            center = float(cand.params[kb.key])
            grids.append(kb.local_points(center=center, points=grid_points))
        for vals in itertools.product(*grids):
            cc = {kb.key: float(vals[i]) for i, kb in enumerate(spec.knobs)}
            fp = _candidate_fingerprint(cc)
            if fp in already_seen:
                continue
            already_seen.add(fp)
            out.append(Candidate(cid=f"local_t{rank+1}_{local_idx:03d}", stage="local", params=cc))
            local_idx += 1
    return out


def _to_set_args(
    *,
    method: MethodSpec,
    candidate: Candidate,
    cli_set: Sequence[str],
    seed: int,
    steps: int,
    eval_every: int,
) -> List[str]:
    kvs: List[str] = []
    kvs.extend(list(cli_set))
    kvs.append(f"seed={int(seed)}")
    kvs.append(f"method.name={method.method_name}")
    kvs.append(f"train.steps={int(steps)}")
    kvs.append(f"train.eval_every_steps={int(eval_every)}")
    for k, v in method.fixed_overrides.items():
        kvs.append(f"{k}={v}")
    for k, v in sorted(candidate.params.items()):
        kvs.append(f"{k}={_fmt_f(v)}")
    return kvs


def _read_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_is_valid(path: Path, curve_path: Path) -> bool:
    if (not path.exists()) or (not curve_path.exists()):
        return False
    try:
        payload = _read_summary(path)
    except Exception:
        return False
    if "best_val_acc" not in payload or "final_val_acc" not in payload:
        return False
    return True


def _run_job_once(job: RunJob, config: str, gpu: int) -> None:
    cmd: List[str] = [
        sys.executable,
        str(ROOT / "scripts" / "run.py"),
        "--config",
        str(config),
    ]
    for kv in job.set_kv:
        cmd.extend(["--set", kv])
    cmd.extend(["--out", str(job.out_json), "--curve_out", str(job.out_curve)])

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def _run_job_with_retry(job: RunJob, config: str, gpu: int, retries: int) -> RunResult:
    reused = False
    if _summary_is_valid(job.out_json, job.out_curve):
        reused = True
    else:
        job.out_json.parent.mkdir(parents=True, exist_ok=True)
        job.out_curve.parent.mkdir(parents=True, exist_ok=True)
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                _run_job_once(job, config=config, gpu=gpu)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= retries:
                    break
                print(
                    f"[retry] method={job.method_alias} cand={job.candidate_id} seed={job.seed} "
                    f"gpu={gpu} attempt={attempt+1}/{retries}"
                )
                time.sleep(2.0)
        if last_err is not None:
            raise last_err
        if not _summary_is_valid(job.out_json, job.out_curve):
            raise RuntimeError(f"run output invalid: {job.out_json}")

    payload = _read_summary(job.out_json)
    best = float(payload["best_val_acc"])
    final = float(payload["final_val_acc"])
    score = 0.5 * best + 0.5 * final
    return RunResult(
        method_alias=job.method_alias,
        candidate_id=job.candidate_id,
        seed=int(job.seed),
        best_val_acc=best,
        final_val_acc=final,
        score_05_05=score,
        out_json=str(job.out_json),
        out_curve=str(job.out_curve),
        reused=bool(reused),
    )


def _run_jobs_parallel(
    *,
    jobs: Sequence[RunJob],
    config: str,
    gpus: Sequence[int],
    retries: int,
    phase_name: str,
) -> List[RunResult]:
    if not jobs:
        return []
    if not gpus:
        raise RuntimeError("empty gpus list")

    print(f"[{phase_name}] launch jobs={len(jobs)} gpus={list(gpus)}")
    out: List[RunResult | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        fut_to_idx = {}
        for i, job in enumerate(jobs):
            gpu = int(gpus[i % len(gpus)])
            fut = ex.submit(_run_job_with_retry, job, config, gpu, retries)
            fut_to_idx[fut] = i

        done = 0
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            try:
                out[i] = fut.result()
            except Exception:  # noqa: BLE001
                for ff in fut_to_idx:
                    ff.cancel()
                raise
            done += 1
            if done % 8 == 0 or done == len(jobs):
                print(f"[{phase_name}] done {done}/{len(jobs)}")

    return [x for x in out if x is not None]


def _agg_candidate_scores(
    results: Sequence[RunResult],
    candidates: Sequence[Candidate],
    method_alias: str,
) -> List[Dict[str, Any]]:
    c_map = {c.cid: c for c in candidates}
    bucket: Dict[str, List[RunResult]] = {}
    for r in results:
        bucket.setdefault(r.candidate_id, []).append(r)

    rows: List[Dict[str, Any]] = []
    for cid, rs in bucket.items():
        bests = [r.best_val_acc for r in rs]
        finals = [r.final_val_acc for r in rs]
        scores = [r.score_05_05 for r in rs]
        row = {
            "method": method_alias,
            "candidate_id": cid,
            "stage": c_map[cid].stage if cid in c_map else "",
            "n_seeds": len(rs),
            "score_mean": float(statistics.fmean(scores)),
            "score_std": float(statistics.pstdev(scores) if len(scores) > 1 else 0.0),
            "best_mean": float(statistics.fmean(bests)),
            "final_mean": float(statistics.fmean(finals)),
        }
        if cid in c_map:
            for k, v in sorted(c_map[cid].params.items()):
                row[k] = float(v)
        rows.append(row)
    rows.sort(key=lambda x: float(x["score_mean"]), reverse=True)
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k in seen:
                continue
            seen.add(k)
            fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _ensure_manifest(path: Path, expected: Dict[str, Any]) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")
        return
    cur = json.loads(path.read_text(encoding="utf-8"))
    if cur != expected:
        raise RuntimeError(
            "pipeline manifest mismatch, refusing resume.\n"
            + json.dumps({"expected": expected, "found": cur}, indent=2, sort_keys=True)
        )


def _run_plotters(
    *,
    final_dir: Path,
    methods: Sequence[str],
    final_seeds: Sequence[int],
    skip_mvp: bool,
) -> None:
    methods_csv = ",".join(methods)
    seeds_csv = ",".join(str(int(x)) for x in final_seeds)

    cmd_plot = [
        sys.executable,
        str(ROOT / "scripts" / "plot_seed_mean_band.py"),
        "--runs_dir",
        str(final_dir),
        "--methods",
        methods_csv,
        "--seeds",
        seeds_csv,
        "--band",
        "std",
        "--out",
        str(final_dir / "seed_mean_band_std.png"),
        "--summary_out",
        str(final_dir / "seed_mean_band_std_summary.json"),
        "--val_table_out",
        str(final_dir / "seed_mean_band_val_last.csv"),
    ]
    subprocess.run(cmd_plot, cwd=str(ROOT), check=True)

    cmd_load = [
        sys.executable,
        str(ROOT / "scripts" / "summarize_router_load.py"),
        "--runs_dir",
        str(final_dir),
        "--methods",
        methods_csv,
        "--seeds",
        seeds_csv,
        "--out_csv",
        str(final_dir / "router_load_summary.csv"),
    ]
    subprocess.run(cmd_load, cwd=str(ROOT), check=True)

    if skip_mvp:
        return
    cmd_mvp = [
        sys.executable,
        str(ROOT / "scripts" / "plot_mvp_12pack.py"),
        "--runs_dir",
        str(final_dir),
        "--methods",
        methods_csv,
        "--seeds",
        seeds_csv,
        "--band",
        "std",
        "--out_dir",
        str(final_dir),
    ]
    subprocess.run(cmd_mvp, cwd=str(ROOT), check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Generic pipeline: HPO (coarse+local) -> final -> plot.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True, help="relative path will be placed under runs/")
    p.add_argument("--methods", type=str, default="baseline,cagrad,ours")
    p.add_argument("--gpus", type=str, default="0,1")
    p.add_argument("--hpo_seeds", type=str, default="2,3")
    p.add_argument("--final_seeds", type=str, default="2,3,5,7,11")
    p.add_argument("--hpo_trials", type=int, default=96, help="coarse random trials per method")
    p.add_argument("--hpo_steps", type=int, default=50)
    p.add_argument("--final_steps", type=int, default=600)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--local_topk", type=int, default=3)
    p.add_argument("--local_grid_points", type=int, default=3)
    p.add_argument("--hpo_seed_base", type=int, default=20260311)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--skip_mvp", action="store_true")
    p.add_argument("--set", action="append", default=[], help="extra key=value overrides passed to run.py")
    args = p.parse_args()

    method_tokens = [_canonical_method_token(x) for x in _parse_csv_strs(args.methods)]
    method_tokens = list(dict.fromkeys(method_tokens))
    specs_all = _method_specs()
    specs = [specs_all[m] for m in method_tokens]

    gpus = _parse_csv_ints(args.gpus)
    if not gpus:
        raise RuntimeError("empty --gpus")
    hpo_seeds = _parse_csv_ints(args.hpo_seeds)
    final_seeds = _parse_csv_ints(args.final_seeds)
    if not hpo_seeds:
        raise RuntimeError("empty --hpo_seeds")
    if not final_seeds:
        raise RuntimeError("empty --final_seeds")

    out_dir = resolve_runs_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": str(args.config),
        "methods": [s.alias for s in specs],
        "hpo_seeds": [int(x) for x in hpo_seeds],
        "final_seeds": [int(x) for x in final_seeds],
        "hpo_trials": int(args.hpo_trials),
        "hpo_steps": int(args.hpo_steps),
        "final_steps": int(args.final_steps),
        "eval_every": int(args.eval_every),
        "local_topk": int(args.local_topk),
        "local_grid_points": int(args.local_grid_points),
        "hpo_seed_base": int(args.hpo_seed_base),
        "extra_set": list(args.set),
    }
    _ensure_manifest(out_dir / "pipeline_manifest.json", manifest)

    base_cfg = _load_base_config(args.config, args.set)
    hpo_root = out_dir / "hpo"
    final_root = out_dir / "final"
    hpo_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    all_best_cfg: Dict[str, Dict[str, Any]] = {}
    global_hpo_rows: List[Dict[str, Any]] = []

    for mi, spec in enumerate(specs):
        print(f"\n[method] {spec.alias} -> {spec.method_name}")
        method_dir = hpo_root / spec.alias
        method_dir.mkdir(parents=True, exist_ok=True)

        coarse = _build_coarse_candidates(
            spec=spec,
            base_cfg=base_cfg,
            trials=int(args.hpo_trials),
            seed=int(args.hpo_seed_base) + mi * 10007,
        )

        jobs: List[RunJob] = []
        order = 0
        for cand in coarse:
            for sd in hpo_seeds:
                run_dir = method_dir / "trial_runs" / cand.cid
                out_json = run_dir / f"{spec.alias}_s{sd}.json"
                out_curve = run_dir / f"{spec.alias}_s{sd}_curve.csv"
                jobs.append(
                    RunJob(
                        order=order,
                        method_alias=spec.alias,
                        candidate_id=cand.cid,
                        seed=int(sd),
                        steps=int(args.hpo_steps),
                        eval_every=int(args.eval_every),
                        out_json=out_json,
                        out_curve=out_curve,
                        set_kv=_to_set_args(
                            method=spec,
                            candidate=cand,
                            cli_set=args.set,
                            seed=int(sd),
                            steps=int(args.hpo_steps),
                            eval_every=int(args.eval_every),
                        ),
                    )
                )
                order += 1

        coarse_results = _run_jobs_parallel(
            jobs=jobs,
            config=args.config,
            gpus=gpus,
            retries=max(0, int(args.retries)),
            phase_name=f"{spec.alias}:coarse",
        )
        coarse_agg = _agg_candidate_scores(coarse_results, coarse, spec.alias)
        if not coarse_agg:
            raise RuntimeError(f"no coarse hpo rows for method={spec.alias}")

        top_n = max(1, int(args.local_topk))
        top_ids = [str(r["candidate_id"]) for r in coarse_agg[:top_n]]
        top_cands = [c for c in coarse if c.cid in set(top_ids)]

        seen = {_candidate_fingerprint(c.params) for c in coarse}
        local = _build_local_candidates(
            spec=spec,
            top_candidates=top_cands,
            already_seen=seen,
            grid_points=max(1, int(args.local_grid_points)),
        )

        local_results: List[RunResult] = []
        local_agg: List[Dict[str, Any]] = []
        if local:
            jobs_local: List[RunJob] = []
            for cand in local:
                for sd in hpo_seeds:
                    run_dir = method_dir / "trial_runs" / cand.cid
                    out_json = run_dir / f"{spec.alias}_s{sd}.json"
                    out_curve = run_dir / f"{spec.alias}_s{sd}_curve.csv"
                    jobs_local.append(
                        RunJob(
                            order=order,
                            method_alias=spec.alias,
                            candidate_id=cand.cid,
                            seed=int(sd),
                            steps=int(args.hpo_steps),
                            eval_every=int(args.eval_every),
                            out_json=out_json,
                            out_curve=out_curve,
                            set_kv=_to_set_args(
                                method=spec,
                                candidate=cand,
                                cli_set=args.set,
                                seed=int(sd),
                                steps=int(args.hpo_steps),
                                eval_every=int(args.eval_every),
                            ),
                        )
                    )
                    order += 1

            local_results = _run_jobs_parallel(
                jobs=jobs_local,
                config=args.config,
                gpus=gpus,
                retries=max(0, int(args.retries)),
                phase_name=f"{spec.alias}:local",
            )
            local_agg = _agg_candidate_scores(local_results, local, spec.alias)

        all_candidates = list(coarse) + list(local)
        all_results = list(coarse_results) + list(local_results)
        all_agg = _agg_candidate_scores(all_results, all_candidates, spec.alias)
        if not all_agg:
            raise RuntimeError(f"no merged hpo rows for method={spec.alias}")
        best_row = all_agg[0]
        best_cid = str(best_row["candidate_id"])
        best_cand = None
        for c in all_candidates:
            if c.cid == best_cid:
                best_cand = c
                break
        if best_cand is None:
            raise RuntimeError(f"best candidate not found: {best_cid}")

        all_best_cfg[spec.alias] = {
            "method_alias": spec.alias,
            "method_name": spec.method_name,
            "candidate_id": best_cid,
            "score_mean": float(best_row["score_mean"]),
            "score_std": float(best_row["score_std"]),
            "params": {k: float(v) for k, v in sorted(best_cand.params.items())},
            "fixed_overrides": dict(spec.fixed_overrides),
            "top_candidates_from_coarse": top_ids,
        }

        (method_dir / "best_config.json").write_text(
            json.dumps(all_best_cfg[spec.alias], indent=2, sort_keys=True),
            encoding="utf-8",
        )

        rows_per_run: List[Dict[str, Any]] = []
        for rr in all_results:
            rows_per_run.append(
                {
                    "method": rr.method_alias,
                    "candidate_id": rr.candidate_id,
                    "seed": rr.seed,
                    "best_val_acc": rr.best_val_acc,
                    "final_val_acc": rr.final_val_acc,
                    "score_05_05": rr.score_05_05,
                    "reused": rr.reused,
                    "summary_json": rr.out_json,
                    "curve_csv": rr.out_curve,
                }
            )
        _write_csv(method_dir / "hpo_per_run.csv", rows_per_run)
        _write_csv(method_dir / "hpo_agg.csv", all_agg)
        (method_dir / "candidate_manifest.json").write_text(
            json.dumps(
                {
                    "coarse_candidates": [c.__dict__ for c in coarse],
                    "local_candidates": [c.__dict__ for c in local],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        global_hpo_rows.extend(all_agg)

    _write_csv(hpo_root / "hpo_agg_all_methods.csv", global_hpo_rows)
    (hpo_root / "best_configs.json").write_text(json.dumps(all_best_cfg, indent=2, sort_keys=True), encoding="utf-8")

    final_jobs: List[RunJob] = []
    order = 0
    for spec in specs:
        best = all_best_cfg[spec.alias]
        best_cand = Candidate(
            cid=str(best["candidate_id"]),
            stage="best",
            params={str(k): float(v) for k, v in dict(best["params"]).items()},
        )
        for sd in final_seeds:
            out_json = final_root / f"{spec.alias}_s{sd}.json"
            out_curve = final_root / f"{spec.alias}_s{sd}_curve.csv"
            final_jobs.append(
                RunJob(
                    order=order,
                    method_alias=spec.alias,
                    candidate_id=best_cand.cid,
                    seed=int(sd),
                    steps=int(args.final_steps),
                    eval_every=int(args.eval_every),
                    out_json=out_json,
                    out_curve=out_curve,
                    set_kv=_to_set_args(
                        method=spec,
                        candidate=best_cand,
                        cli_set=args.set,
                        seed=int(sd),
                        steps=int(args.final_steps),
                        eval_every=int(args.eval_every),
                    ),
                )
            )
            order += 1

    final_results = _run_jobs_parallel(
        jobs=final_jobs,
        config=args.config,
        gpus=gpus,
        retries=max(0, int(args.retries)),
        phase_name="final",
    )

    final_per_run: List[Dict[str, Any]] = []
    for rr in final_results:
        final_per_run.append(
            {
                "method": rr.method_alias,
                "seed": rr.seed,
                "best_val_acc": rr.best_val_acc,
                "final_val_acc": rr.final_val_acc,
                "score_05_05": rr.score_05_05,
                "reused": rr.reused,
                "summary_json": rr.out_json,
                "curve_csv": rr.out_curve,
            }
        )
    _write_csv(final_root / "final_per_run.csv", final_per_run)

    final_agg_rows: List[Dict[str, Any]] = []
    by_method: Dict[str, List[RunResult]] = {}
    for rr in final_results:
        by_method.setdefault(rr.method_alias, []).append(rr)
    for m, rs in sorted(by_method.items(), key=lambda x: x[0]):
        bests = [x.best_val_acc for x in rs]
        finals = [x.final_val_acc for x in rs]
        scores = [x.score_05_05 for x in rs]
        final_agg_rows.append(
            {
                "method": m,
                "n_seeds": len(rs),
                "best_mean": float(statistics.fmean(bests)),
                "best_std": float(statistics.pstdev(bests) if len(bests) > 1 else 0.0),
                "final_mean": float(statistics.fmean(finals)),
                "final_std": float(statistics.pstdev(finals) if len(finals) > 1 else 0.0),
                "score_mean": float(statistics.fmean(scores)),
                "score_std": float(statistics.pstdev(scores) if len(scores) > 1 else 0.0),
            }
        )
    final_agg_rows.sort(key=lambda x: float(x["score_mean"]), reverse=True)
    _write_csv(final_root / "final_agg.csv", final_agg_rows)
    (final_root / "final_best_configs_snapshot.json").write_text(
        json.dumps(all_best_cfg, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _run_plotters(
        final_dir=final_root,
        methods=[s.alias for s in specs],
        final_seeds=final_seeds,
        skip_mvp=bool(args.skip_mvp),
    )

    print(f"\n[pipeline done] out_dir={out_dir}")
    print(f"[hpo best] {(hpo_root / 'best_configs.json')}")
    print(f"[final agg] {(final_root / 'final_agg.csv')}")


if __name__ == "__main__":
    main()
