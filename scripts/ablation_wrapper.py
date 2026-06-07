#!/usr/bin/env python3
"""Minimal ablation wrapper that re-uses scripts/run_batch.BatchProcessor
but injects CRS weights into ProofWorkflow for each weight configuration.

This keeps changes minimal: it subclasses BatchProcessor and overrides the
per-algorithm runner to pass `crs_weights` through to ProofWorkflow.

Example:
  python scripts/ablation_wrapper.py --input data/algos.csv --config config/default.yaml
"""
import argparse
import sys
from pathlib import Path
import time
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import run_ablation as ra
from scripts.run_batch import BatchProcessor, load_yaml_config
from src.graph.workflow import ProofWorkflow
from src.config.temperatures import TemperatureConfig


class AblationBatchProcessor(BatchProcessor):
    def __init__(self, *args, crs_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._crs_weights = crs_weights or {}

    def _process_single_algorithm(self, index, row, results_df):
        # Adapted from BatchProcessor._process_single_algorithm but injects crs_weights
        algo_text = str(row["Problem Statement"])
        assump_text = str(row.get("Assumptions Used", ""))

        algo_dir = self.output_dir / f"algorithm_{index:03d}"
        start = time.time()

        workflow = ProofWorkflow(
            provider_name=self.provider_name,
            prover_model=self.prover_model,
            evaluator_models=self.evaluator_models,
            api_key=self.api_key,
            use_multi_judge=self.use_multi_judge,
            max_iterations=self.max_iterations,
            timeout=self.timeout,
            max_retries=self.max_retries,
            output_dir=str(algo_dir),
            temp_config=self.temp_config,
            crs_weights=self._crs_weights,
            per_judge_timeout=self.per_judge_timeout,
        )

        final_state, tracker = workflow.run(algo_text, assump_text)
        elapsed = time.time() - start
        exec_log = tracker.get_log()

        self._update_results_row(index, results_df, exec_log, final_state, elapsed)

        self.batch_log["algorithms"].append({
            "row_index": index,
            "algorithm": algo_text,
            "status": "completed",
            "processing_time": elapsed,
            "final_verdict": final_state.get("verdict") if final_state else "UNKNOWN",
            "iterations": exec_log.get("final_results", {}).get("total_iterations", 0),
            "output_directory": str(algo_dir),
            "crs_final": (exec_log.get("statistics", {}).get("crs_progression") or [None])[-1],
        })


def sanitize_label(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(description="Minimal ablation wrapper")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="ablation_out")
    parser.add_argument("--grid-step", type=float, default=0.2)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Prover model(s) to run; if omitted uses YAML/defaults")
    parser.add_argument("--provider", default="nvidia")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--multi-judge", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    load_dotenv()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    yaml_cfg = load_yaml_config(args.config)

    # Determine list of models to run
    models = args.models
    if models is None:
        yaml_model = None
        try:
            yaml_model = yaml_cfg.get("models", {}).get("prover", {}).get("name")
        except Exception:
            yaml_model = None
        models = [yaml_model] if yaml_model else [None]

    weight_configs = ra.build_weight_grid(step=args.grid_step)

    if args.dry_run:
        for cfg in weight_configs:
            print(ra.config_label(cfg))
        return

    for model in models:
        model_label = sanitize_label(str(model)) if model else "default_model"
        for cfg in weight_configs:
            label = ra.config_label(cfg)
            run_out = out_root / model_label / "runs" / label
            run_out.mkdir(parents=True, exist_ok=True)

            print(f"\nRunning model={model or '(default)'}  config={label}  → {run_out}")

            # Instantiate processor for this config
            temp_config = TemperatureConfig()
            processor = AblationBatchProcessor(
                input_csv=args.input,
                output_dir=str(run_out),
                provider_name=args.provider,
                prover_model=model or "",
                evaluator_models=[model] if model else [],
                api_key=args.api_key,
                use_multi_judge=args.multi_judge,
                max_iterations=args.max_iter or int(yaml_cfg.get("workflow", {}).get("max_iterations", 3)),
                timeout=args.timeout,
                max_retries=args.max_retries,
                temp_config=temp_config,
                yaml_config=yaml_cfg,
                per_judge_timeout=int(yaml_cfg.get("models", {}).get("evaluators", {}).get("per_judge_timeout", 600)),
                crs_weights={"w_err": cfg["w_err"], "w_rp_pen": cfg["w_rp_pen"]},
            )

            processor.process()


if __name__ == "__main__":
    main()
