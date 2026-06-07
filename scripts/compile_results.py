#!/usr/bin/env python3
"""Compile and render ablation results using the existing run_ablation.py analysis.

This wrapper invokes `scripts/run_ablation.py` in `--analyze-only` mode so users
can regenerate plots/reports from previously produced run CSVs.

Example:
  python scripts/compile_results.py --input data/algos.csv --config config/default.yaml --output-dir ablation_out
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Regenerate ablation plots/reports")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="ablation_out")
    parser.add_argument("--grid-step", type=float, default=0.2)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "scripts/run_ablation.py",
           "--input", args.input,
           "--config", args.config,
           "--output-dir", str(out),
           "--grid-step", str(args.grid_step),
           "--analyze-only"]

    print("Regenerating ablation analysis (analyze-only) ...")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
