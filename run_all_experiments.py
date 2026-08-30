"""run_all_experiments.py

Orchestrates the complete Section-4 experiment suite for all four models:

    1) 5-seed "thinking-layer" ablation  (top/bottom 20% layer removal)
    2) 5-seed multi-dataset benchmark    (BF16/INT8/INT4/LLM-QRP + 6 baselines)
    3) Spearman rho                      (criticality vs. downstream drop)
    4) Deployment efficiency             (peak VRAM / latency / tokens-per-sec)

Each step is timed and each result is logged both to stdout and to
``experiments_run_log.txt`` so that total wall time is known.  Individual
stages can be skipped via command-line flags, and the full suite runs
sequentially on one GPU.

Usage:
    ai_env\\Scripts\\python.exe run_all_experiments.py [--skip-ablation]
        [--skip-bench] [--skip-spearman] [--skip-deploy] [--seeds 0 1 2 3 4]
        [--models ...]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / "ai_env" / "Scripts" / "python.exe"
OUT = ROOT / "results"
LOG = ROOT / "experiments_run_log.txt"

DEFAULT_MODELS = [
    "HuggingFaceTB/SmolLM2-135M",
    "ibm-granite/granite-4.0-350m-base",
    "Qwen/Qwen2.5-0.5B",
    "LiquidAI/LFM2-350M",
]

BENCH_FLAGS = [
    "--with-gptq", "--with-awq", "--with-spqr",
    "--with-slim", "--with-smoothquant", "--with-atom",
]


def log(msg: str, total: float | None = None) -> None:
    line = msg
    if total is not None:
        line = f"{msg}  (total so far: {total / 60.0:.1f} min)"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def run(cmd: list[str], tag: str, total: float) -> float:
    log(f"[{tag}] running: {' '.join(cmd)}", total)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        log(f"[{tag}] FAILED (exit {proc.returncode}) after {dt:.1f}s", total)
        raise SystemExit(1)
    log(f"[{tag}] done in {dt / 60.0:.2f} min", total + dt)
    return dt


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the full Section-4 experiment suite")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-spearman", action="store_true")
    ap.add_argument("--skip-deploy", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    args = ap.parse_args()

    if not PY.exists():
        print(f"[ERROR] venv python not found: {PY}")
        return 1

    total: float = 0.0
    with LOG.open("w") as f:
        f.write(f"experiment run started {time.ctime()}\n")

    for model in args.models:
        log("\n" + "=" * 70)
        log(f"MODEL: {model}", total)

        if not args.skip_ablation:
            for seed in args.seeds:
                dt = run(
                    [str(PY), "-m", "qrp.ablate.run_ablation",
                     "--model-name", model, "--output-folder", str(OUT),
                     "--samples", "100", "--seed", str(seed),
                     "--output-suffix", f"seed{seed}"],
                    f"ablation {model} seed{seed}", total)
                total += dt

        if not args.skip_bench:
            for seed in args.seeds:
                dt = run(
                    [str(PY), "-m", "qrp.quantize.run_multi_dataset_benchmark",
                     "--model-name", model, "--output-folder", str(OUT),
                     "--samples", "100", "--datasets", "gsm8k,tfqa,mmlu",
                     *BENCH_FLAGS, "--seed", str(seed),
                     "--output-suffix", f"seed{seed}"],
                    f"bench {model} seed{seed}", total)
                total += dt

        if not args.skip_spearman:
            dt = run(
                [str(PY), "-m", "qrp.analysis.spearman_rho",
                 "--model-name", model, "--output-folder", str(OUT),
                 "--samples", "50", "--seed", "0"],
                f"spearman {model}", total)
            total += dt

        if not args.skip_deploy:
            dt = run(
                [str(PY), "-m", "qrp.deploy.measure_deployment",
                 "--model-name", model, "--output-folder", str(OUT),
                 "--n-prompts", "5", "--output-tokens", "64"],
                f"deploy {model}", total)
            total += dt

    log("\n" + "=" * 70)
    log("Regenerating cross-model figures and combined tex tables", total)
    dt = run([str(PY), "qrp/visualization/cross_model.py", "--results-root", str(OUT)],
             "cross-model figures", total)
    total += dt
    dt = run([str(PY), "-m", "qrp.quantize.aggregate_benchmarks", "--results-dir", str(OUT)],
             "aggregate tex tables", total)
    total += dt

    log("=" * 70)
    log(f"ALL EXPERIMENTS COMPLETE. Total wall time: {total / 60.0:.1f} min", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
