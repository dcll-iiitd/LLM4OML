# ============================================================================
# File: scripts/run_single.py
# ============================================================================
"""Run a single convergence proof generation and evaluation."""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.workflow import ProofWorkflow
from src.config.temperatures import TemperatureConfig


def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate a convergence proof")
    parser.add_argument("--algorithm", required=True, help="Algorithm description")
    parser.add_argument("--assumptions", required=True, help="Algorithm assumptions")

    # Provider
    parser.add_argument(
        "--provider", default="nvidia",
        choices=["nvidia", "openrouter", "openai", "anthropic"],
    )
    parser.add_argument(
        "--judge-provider", default=None,
        choices=["nvidia", "openrouter", "openai", "anthropic"],
    )
    parser.add_argument("--api-key", default=None, help="API key (overrides env var)")

    # Models
    parser.add_argument("--model", default=None, help="Prover model")
    parser.add_argument("--judges", nargs="+", default=None, help="Judge model(s)")
    parser.add_argument("--multi-judge", action="store_true")

    # Generation parameters
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output", default="output")

    # Per-phase temperature overrides (optional)
    parser.add_argument("--temp-generation", type=float, default=0.7)
    parser.add_argument("--temp-correction", type=float, default=0.4)
    parser.add_argument("--temp-evaluation", type=float, default=0.05)
    parser.add_argument("--temp-verification", type=float, default=0.05)

    args = parser.parse_args()
    load_dotenv()

    if args.model is None:
        defaults = {
            "nvidia": "openai/gpt-oss-120b",
            "openrouter": "anthropic/claude-3.5-sonnet",
            "openai": "gpt-4",
            "anthropic": "claude-3-5-sonnet-20241022",
        }
        args.model = defaults[args.provider]
        print(f"Using default model for {args.provider}: {args.model}")

    evaluator_models = args.judges or [args.model]

    temp_config = TemperatureConfig(
        generation=args.temp_generation,
        correction=args.temp_correction,
        evaluation=args.temp_evaluation,
        verification=args.temp_verification,
    )

    workflow = ProofWorkflow(
        provider_name=args.provider,
        judge_provider_name=args.judge_provider,
        prover_model=args.model,
        evaluator_models=evaluator_models,
        api_key=args.api_key,
        use_multi_judge=args.multi_judge,
        max_iterations=args.max_iter,
        timeout=args.timeout,
        max_retries=args.max_retries,
        output_dir=args.output,
        temp_config=temp_config,
        per_judge_timeout=600,
    )

    print(f"\nProvider       : {args.provider}")
    print(f"Judge provider : {args.judge_provider or args.provider}")
    print(f"Prover model   : {args.model}")
    print(f"Judge model(s) : {evaluator_models}")
    print(f"Temperatures   : gen={temp_config.generation}, corr={temp_config.correction}, "
          f"eval={temp_config.evaluation}")
    print(f"Max iterations : {args.max_iter}")

    final_state, tracker = workflow.run(args.algorithm, args.assumptions)

    log = tracker.get_log()
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Final Verdict    : {log['final_results']['verdict']}")
    print(f"Total Iterations : {log['final_results']['total_iterations']}")
    print(f"Convergence      : {log['final_results']['convergence_achieved']}")

    stats = log.get("statistics", {})
    crs_prog = stats.get("crs_progression", [])
    if crs_prog:
        print(f"CRS Progression  : {' → '.join(f'{x:.3f}' for x in crs_prog)}")
    print("=" * 60)


if __name__ == "__main__":
    main()