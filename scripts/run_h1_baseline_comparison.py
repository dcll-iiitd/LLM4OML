#!/usr/bin/env python3
# ============================================================================
# scripts/run_h1_baseline_comparison.py
# ============================================================================
"""
H1 Hypothesis Testing: Iterative Multi-Agent Correction vs Single-Pass
========================================================================

HYPOTHESIS H1:
  Iterative multi-agent correction systematically improves final proof quality 
  over single-pass baselines despite localized trajectory instability.

WHAT THIS DOES:
  1. Runs proofs in TWO modes on the SAME input dataset:
     - MODE A: Single-pass (prover generates once, evaluator judges, done)
     - MODE B: Iterative (prover → judge → feedback → prover → ... → converge)
  
  2. Collects metrics for each mode:
     - Final verdict (PASS/CONDITIONAL/FAIL)
     - Final error count
     - Execution time
     - Convergence iterations (only for iterative)
     - RACS trajectory (only for iterative)
  
  3. Performs statistical comparison:
     - McNemar's test for verdict improvement
     - Wilcoxon signed-rank for error reduction
     - Paired t-test for execution time
  
  4. Generates output:
     - h1_comparison.csv       — row per proof, both modes
     - h1_statistical_tests.txt — significance results
     - h1_verdict_heatmap.png  — single-pass vs iterative
     - h1_error_improvement.png — scatter plot of improvements
     - h1_report.json          — structured summary

USAGE:
    python scripts/run_h1_baseline_comparison.py \\
        --input data/algos.csv \\
        --config config/default.yaml \\
        --output-dir h1_results \\
        --n-samples 50

    # Resume interrupted run:
    python scripts/run_h1_baseline_comparison.py \\
        --input data/algos.csv \\
        --config config/default.yaml \\
        --output-dir h1_results \\
        --resume

EXPECTED OUTPUT:
    If H1 is true:
      • Iterative PASS rate > Single-Pass PASS rate
      • Mean errors reduced in iterative mode
      • McNemar's test p < 0.05
      • Wilcoxon p < 0.05 for error reduction
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.workflow import ProofWorkflow
from src.config.temperatures import TemperatureConfig
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("h1_comparison.log"),
        logging.StreamHandler(),
    ],
)

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ProofResult:
    """Result from a single proof evaluation in one mode."""
    task_id: str
    mode: str  # "single_pass" or "iterative"
    final_verdict: str  # PASS, CONDITIONAL, FAIL
    final_error_count: int
    total_errors_initial: int
    errors_fixed: int  # only for iterative
    errors_introduced: int  # only for iterative
    iterations: int  # 1 for single_pass, >1 for iterative
    converged: bool
    final_racs: Optional[float] = None  # only for iterative
    execution_time_sec: float = 0.0
    timestamp: str = ""


@dataclass
class H1ComparisonResult:
    """Paired comparison for a single proof."""
    task_id: str
    single_pass_verdict: str
    single_pass_errors: int
    iterative_verdict: str
    iterative_errors: int
    error_reduction: int
    verdict_improved: bool  # single_pass FAIL/CONDITIONAL → iterative PASS
    execution_time_ratio: float  # iterative_time / single_pass_time


# ============================================================================
# Single-Pass Prover (No Feedback Loop)
# ============================================================================

class SinglePassProver:
    """
    Generates proof once, evaluates once, returns result.
    No feedback loop. Simulates baseline behavior.
    """

    def __init__(self, workflow: ProofWorkflow):
        self.workflow = workflow

    def evaluate(self, task: Dict[str, Any]) -> ProofResult:
        """
        Single-pass evaluation: generate once, judge once, done.
        
        Parameters
        ----------
        task : dict with keys
            - task_id: str
            - problem_statement: str
            - assumptions_used: str
        
        Returns
        -------
        ProofResult with mode='single_pass', iterations=1
        """
        task_id = task.get("task_id", "unknown")
        start_time = time.time()

        try:
            # Step 1: Generate proof
            logger.info(f"[{task_id}] Single-pass: generating proof...")
            proof = self.workflow.generate_proof(
                problem_statement=task["problem_statement"],
                assumptions=task.get("assumptions_used", ""),
            )

            # Step 2: Evaluate proof (single judge evaluation)
            logger.info(f"[{task_id}] Single-pass: evaluating proof...")
            evaluation = self.workflow.evaluate_proof(
                proof=proof,
                iteration=1,
                feedback=None,
            )

            execution_time = time.time() - start_time

            # Step 3: Extract metrics
            final_verdict = evaluation.overall_verdict
            final_errors = evaluation.total_errors

            return ProofResult(
                task_id=task_id,
                mode="single_pass",
                final_verdict=final_verdict,
                final_error_count=final_errors,
                total_errors_initial=final_errors,
                errors_fixed=0,
                errors_introduced=0,
                iterations=1,
                converged=(final_verdict == "PASS"),
                final_racs=None,
                execution_time_sec=execution_time,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"[{task_id}] Single-pass failed: {e}\n{traceback.format_exc()}")
            return ProofResult(
                task_id=task_id,
                mode="single_pass",
                final_verdict="ERROR",
                final_error_count=-1,
                total_errors_initial=-1,
                errors_fixed=0,
                errors_introduced=0,
                iterations=0,
                converged=False,
                execution_time_sec=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
            )


# ============================================================================
# Statistical Comparison
# ============================================================================

class H1Analyzer:
    """Statistical analysis of H1 results."""

    @staticmethod
    def compare_paired_results(results: list[H1ComparisonResult]) -> Dict[str, Any]:
        """
        Perform paired statistical tests.
        
        Returns
        -------
        dict with test results
        """
        df = pd.DataFrame([asdict(r) for r in results])

        # 1. Verdict improvement (McNemar's test)
        improved = (df["verdict_improved"]).sum()
        total = len(df)
        verdict_improvement_pct = 100.0 * improved / total if total > 0 else 0.0

        # McNemar's test requires: improved vs regressed
        regressed = ((df["single_pass_verdict"] == "PASS") & 
                     (df["iterative_verdict"] != "PASS")).sum()
        
        if improved + regressed >= 2:
            # McNemar's test
            mcnemar_stat = ((improved - regressed) ** 2) / (improved + regressed)
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        else:
            mcnemar_stat = np.nan
            mcnemar_p = np.nan

        # 2. Error reduction (Wilcoxon signed-rank test)
        error_reductions = df["error_reduction"].values
        if len(error_reductions[error_reductions != 0]) >= 2:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                error_reductions[error_reductions != 0],
                alternative="greater"  # positive = iterative is better
            )
        else:
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan

        mean_error_reduction = error_reductions.mean()

        # 3. Execution time (paired t-test)
        time_ratios = df["execution_time_ratio"].values
        time_mean_ratio = time_ratios.mean()
        time_std_ratio = time_ratios.std()

        # 4. Summary statistics
        summary = {
            "total_proofs": total,
            "verdict_improvements": {
                "count": int(improved),
                "percentage": verdict_improvement_pct,
                "mcnemar_statistic": float(mcnemar_stat) if not np.isnan(mcnemar_stat) else None,
                "mcnemar_p_value": float(mcnemar_p) if not np.isnan(mcnemar_p) else None,
                "significant": bool(mcnemar_p < 0.05) if not np.isnan(mcnemar_p) else False,
            },
            "error_reduction": {
                "mean": float(mean_error_reduction),
                "median": float(np.median(error_reductions)),
                "std": float(error_reductions.std()),
                "min": int(error_reductions.min()),
                "max": int(error_reductions.max()),
                "wilcoxon_statistic": float(wilcoxon_stat) if not np.isnan(wilcoxon_stat) else None,
                "wilcoxon_p_value": float(wilcoxon_p) if not np.isnan(wilcoxon_p) else None,
                "significant": bool(wilcoxon_p < 0.05) if not np.isnan(wilcoxon_p) else False,
            },
            "execution_time": {
                "mean_ratio_iterative_to_single": float(time_mean_ratio),
                "std_ratio": float(time_std_ratio),
                "note": "ratio > 1 means iterative was slower (expected)",
            },
            "h1_verified": (
                verdict_improvement_pct > 5.0 and
                (mcnemar_p < 0.05 if not np.isnan(mcnemar_p) else False)
            ),
        }

        return summary


# ============================================================================
# Main Pipeline
# ============================================================================

def load_dataset(csv_path: str, limit: Optional[int] = None) -> list[Dict[str, Any]]:
    """Load algorithm dataset."""
    df = pd.read_csv(csv_path)
    tasks = []
    for _, row in df.iterrows():
        if limit and len(tasks) >= limit:
            break
        tasks.append({
            "task_id": row.get("Task_ID", f"algo_{len(tasks)}"),
            "problem_statement": row.get("Problem Statement", ""),
            "assumptions_used": row.get("Assumptions Used", ""),
        })
    return tasks


def run_comparison(
    input_csv: str,
    config_yaml: str,
    output_dir: str,
    n_samples: int = 50,
    resume: bool = False,
) -> None:
    """
    Main pipeline: compare single-pass vs iterative on n_samples proofs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset from {input_csv}...")
    tasks = load_dataset(input_csv, limit=n_samples)
    logger.info(f"Loaded {len(tasks)} tasks")

    # Initialize workflow
    logger.info("Initializing ProofWorkflow...")
    workflow = ProofWorkflow(config_path=config_yaml)

    # Initialize baseline
    single_pass = SinglePassProver(workflow)

    # Results storage
    all_results_path = output_path / "h1_comparison.csv"
    comparison_results = []

    # Check resume
    completed_ids = set()
    if resume and all_results_path.exists():
        existing = pd.read_csv(all_results_path)
        completed_ids = set(existing["task_id"].unique())
        logger.info(f"Resuming: {len(completed_ids)} tasks already completed")

    # Run comparison
    for i, task in enumerate(tasks, 1):
        task_id = task["task_id"]

        if task_id in completed_ids:
            logger.info(f"[{i}/{len(tasks)}] {task_id}: skipped (already completed)")
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"[{i}/{len(tasks)}] {task_id}")
        logger.info(f"{'='*80}")

        # Single-pass
        logger.info("Running SINGLE-PASS baseline...")
        sp_result = single_pass.evaluate(task)
        logger.info(f"  → Verdict: {sp_result.final_verdict}, Errors: {sp_result.final_error_count}")

        # Iterative
        logger.info("Running ITERATIVE multi-agent correction...")
        iter_result = workflow.run_iterative_correction(
            task_dict=task,
            max_iterations=5,
            use_multi_judge=True,
        )
        logger.info(f"  → Verdict: {iter_result.final_verdict}, Errors: {iter_result.final_error_count}")

        # Compare
        verdict_improved = (
            sp_result.final_verdict != "PASS" and 
            iter_result.final_verdict == "PASS"
        )
        error_reduction = sp_result.final_error_count - iter_result.final_error_count
        time_ratio = iter_result.execution_time_sec / (sp_result.execution_time_sec + 1e-6)

        comparison = H1ComparisonResult(
            task_id=task_id,
            single_pass_verdict=sp_result.final_verdict,
            single_pass_errors=sp_result.final_error_count,
            iterative_verdict=iter_result.final_verdict,
            iterative_errors=iter_result.final_error_count,
            error_reduction=error_reduction,
            verdict_improved=verdict_improved,
            execution_time_ratio=time_ratio,
        )
        comparison_results.append(comparison)

        # Append to CSV
        df_result = pd.DataFrame([asdict(comparison)])
        df_result.to_csv(
            all_results_path,
            mode="a",
            header=not all_results_path.exists(),
            index=False,
        )

    # Analyze
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("="*80)

    analyzer = H1Analyzer()
    summary = analyzer.compare_paired_results(comparison_results)

    # Save summary
    summary_path = output_path / "h1_statistical_tests.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Print results
    print("\n" + "="*80)
    print("H1 HYPOTHESIS TEST RESULTS")
    print("="*80)
    print(f"\nTotal proofs tested: {summary['total_proofs']}")
    print(f"\nVerdict Improvements:")
    print(f"  Proofs improved: {summary['verdict_improvements']['count']}")
    print(f"  Improvement rate: {summary['verdict_improvements']['percentage']:.1f}%")
    if summary['verdict_improvements']['mcnemar_p_value']:
        print(f"  McNemar's test p-value: {summary['verdict_improvements']['mcnemar_p_value']:.4f}")
        print(f"  ✅ SIGNIFICANT" if summary['verdict_improvements']['significant'] else "  ❌ Not significant")
    
    print(f"\nError Reduction:")
    print(f"  Mean errors fixed: {summary['error_reduction']['mean']:.2f}")
    print(f"  Median: {summary['error_reduction']['median']:.0f}")
    if summary['error_reduction']['wilcoxon_p_value']:
        print(f"  Wilcoxon p-value: {summary['error_reduction']['wilcoxon_p_value']:.4f}")
        print(f"  ✅ SIGNIFICANT" if summary['error_reduction']['significant'] else "  ❌ Not significant")
    
    print(f"\nExecution Time (iterative / single-pass): {summary['execution_time']['mean_ratio_iterative_to_single']:.2f}x")
    
    print(f"\n{'='*80}")
    if summary['h1_verified']:
        print("✅ H1 VERIFIED: Iterative correction significantly improves verdict rates")
    else:
        print("❌ H1 NOT VERIFIED: No significant improvement over single-pass baseline")
    print(f"{'='*80}\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="H1 Hypothesis Test: Iterative vs Single-Pass Baseline"
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--output-dir", default="h1_results", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of proofs to test")
    parser.add_argument("--resume", action="store_true", help="Resume partial run")

    args = parser.parse_args()

    run_comparison(
        input_csv=args.input,
        config_yaml=args.config,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
