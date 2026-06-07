#!/usr/bin/env python3
"""Run `scripts/run_batch.py` for multiple prover models sequentially.

Minimal orchestration: calls the existing `scripts/run_batch.py` via subprocess
so each model's results are isolated under `--output`.

Example:
  python scripts/multi_model_runner.py --input data/algos.csv --config config/default.yaml \
    --models gpt-4 openai/gpt-oss-120b
"""
import argparse
import subprocess
from pathlib import Path
import sys


def sanitize_label(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(description="Run multiple models via run_batch.py")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=False, default=None)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--provider", default="nvidia")
    parser.add_argument("--judge-provider", default=None)
    parser.add_argument("--judges", nargs="+", default=None)
    parser.add_argument("--output-root", default="batch_results")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--multi-judge", action="store_true", default=False)
    args = parser.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        label = sanitize_label(model)
        out_dir = out_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "scripts/run_batch.py", "--input", args.input, "--provider", args.provider, "--model", model, "--output", str(out_dir)]
        if args.judge_provider:
            cmd += ["--judge-provider", args.judge_provider]
        if args.config:
            cmd += ["--config", args.config]
        if args.judges:
            cmd += ["--judges"] + args.judges
        if args.multi_judge:
            cmd.append("--multi-judge")
        if args.api_key:
            cmd += ["--api-key", args.api_key]
        if args.max_iter:
            cmd += ["--max-iter", str(args.max_iter)]

        print(f"Running model {model} -> {out_dir}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
