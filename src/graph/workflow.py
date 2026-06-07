# ============================================================================
# File: src/graph/workflow.py  (patched for ablation study — crs_weights param added)
# ============================================================================
"""LangGraph workflow with provider abstraction, per-phase temperatures,
and optional CRS weight injection for ablation studies.

ABLATION PATCH (v3.1):
  ProofWorkflow now accepts `crs_weights: dict` and forwards it to
  WorkflowNodes, which passes it to CorrectionMetricsCalculator.
  No other behaviour changes.
"""

import logging
from typing import Optional, Tuple, Dict

from langgraph.graph import StateGraph, END

from ..models.types import AgentState
from ..utils.state import StateManager
from ..utils.execution_tracker import ExecutionTracker
from .nodes import WorkflowNodes
from ..config.temperatures import TemperatureConfig, DEFAULT_TEMPERATURES

logger = logging.getLogger(__name__)


class ProofWorkflow:
    """
    Convergence proof generation and evaluation workflow.

    Temperature design:
        Temperatures are now phase-aware (see src/config/temperatures.py):
          - Prover generation : 0.70  (creative)
          - Prover correction : 0.40  (targeted)
          - Judge evaluation  : 0.05  (deterministic JSON)
          - Judge verification: 0.05  (deterministic JSON)

        The `temperature` constructor parameter sets the prover's instance-
        default but is overridden per-call by the phase logic. Judges always
        use near-zero temperature regardless of this setting.

    CRS weight injection (ablation):
        Pass `crs_weights={"w_err": 0.6, "w_rp_pen": 0.3}` to override the
        default CRS formula weights for this run.  w_tfp is derived as
        1 - w_err automatically.
    """

    def __init__(
        self,
        provider_name: str = "nvidia",
        prover_model: str = "openai/gpt-oss-120b",
        evaluator_models: Optional[list] = None,
        api_key: Optional[str] = None,
        use_multi_judge: bool = True,
        max_iterations: int = 3,
        temperature: float = 0.5,
        timeout: int = 600,
        max_retries: int = 3,
        output_dir: str = "output",
        temp_config: TemperatureConfig = DEFAULT_TEMPERATURES,
        # ── ABLATION PATCH ────────────────────────────────────────────────────
        crs_weights: Optional[Dict[str, float]] = None,
        per_judge_timeout: int = 600,
        judge_provider_name: Optional[str] = None,
        # ─────────────────────────────────────────────────────────────────────
    ):
        if evaluator_models is None:
            evaluator_models = [prover_model]

        self.max_iterations = max_iterations
        self.state_manager = StateManager()
        self.temp_config = temp_config
        self.crs_weights = crs_weights or {}

        self.nodes = WorkflowNodes(
            provider_name=provider_name,
            prover_model=prover_model,
            evaluator_models=evaluator_models,
            api_key=api_key,
            use_multi_judge=use_multi_judge,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            temp_config=temp_config,
            crs_weights=crs_weights,        # ← ABLATION PATCH
            per_judge_timeout=per_judge_timeout,
            judge_provider_name=judge_provider_name,
        )

        self.config = {
            "provider": provider_name,
            "judge_provider": judge_provider_name or provider_name,
            "prover_model": prover_model,
            "evaluator_models": evaluator_models,
            "use_multi_judge": use_multi_judge,
            "max_iterations": max_iterations,
            "temperature_config": {
                "generation": temp_config.generation,
                "correction": temp_config.correction,
                "evaluation": temp_config.evaluation,
                "verification": temp_config.verification,
            },
            "timeout": timeout,
            "max_retries": max_retries,
            # ── ABLATION PATCH: record injected weights in config ──────────
            "crs_weights": self.crs_weights,
            # ─────────────────────────────────────────────────────────────────
        }

        self.output_dir = output_dir
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("prover", self._tracked_prover_node)
        workflow.add_node("evaluator", self._tracked_evaluator_node)
        workflow.set_entry_point("prover")
        workflow.add_edge("prover", "evaluator")
        workflow.add_conditional_edges(
            "evaluator",
            lambda state: self.state_manager.should_continue(state, self.max_iterations),
            {"correct": "prover", "end": END},
        )
        return workflow.compile()

    # ------------------------------------------------------------------
    # Tracked node wrappers
    # ------------------------------------------------------------------

    def _tracked_prover_node(self, state: AgentState) -> dict:
        print(f"\n{'='*60}")
        print(f"ITERATION {state['iteration'] + 1}: PROOF GENERATION")
        print(f"  Phase      : {'generation' if state['iteration'] == 0 else 'correction'}")
        print(f"  Temperature: {self.temp_config.generation if state['iteration'] == 0 else self.temp_config.correction:.2f}")
        print(f"{'='*60}")

        result = self.nodes.prover_node(state)

        self.tracker.track_iteration(
            iteration=result["iteration"],
            node_name="prover",
            state={**state, **result},
        )
        print(f"  ✓ Proof generated ({len(result['current_proof'])} chars)")
        return result

    def _tracked_evaluator_node(self, state: AgentState) -> dict:
        print(f"\n{'='*60}")
        print(f"ITERATION {state['iteration']}: EVALUATION")
        print(f"  Temperature: {self.temp_config.evaluation:.2f} (deterministic)")
        # ── show active weights ───────────────────────────────────────────────
        if self.crs_weights:
            w_err = self.crs_weights.get("w_err", 0.5)
            print(f"  CRS Weights: w_err={w_err:.2f}  w_tfp={1-w_err:.2f}  "
                  f"w_rp={self.crs_weights.get('w_rp_pen', 0.4):.2f}")
        print(f"{'='*60}")

        result = self.nodes.evaluator_node(state)

        self.tracker.track_iteration(
            iteration=result["iteration"],
            node_name="evaluator",
            state={**state, **result},
        )

        print(f"  ✓ Verdict: {result['verdict']}")
        cm = result.get("correction_metrics")
        if cm:
            crs = cm.get("correction_reasoning_score", 0) if isinstance(cm, dict) else 0
            err = cm.get("error_resolution_rate", 0) if isinstance(cm, dict) else 0
            rp  = cm.get("regression_penalty", 0) if isinstance(cm, dict) else 0
            tfp = cm.get("targeted_fix_precision", 0) if isinstance(cm, dict) else 0
            print(f"  ✓ CRS: {crs:.3f}  (ERR={err:.3f}, RP={rp:.3f}, TFP={tfp:.3f})")

        return result

    # ------------------------------------------------------------------
    # Public run method
    # ------------------------------------------------------------------

    def run(
        self, algorithm: str, assumptions: str
    ) -> Tuple[dict, ExecutionTracker]:
        """
        Run the complete workflow.

        Returns
        -------
        (final_state, tracker)
        """
        self.tracker = ExecutionTracker(output_dir=self.output_dir)
        self.tracker.set_metadata(algorithm, assumptions, self.config)

        initial_state = self.state_manager.initialize_state(algorithm, assumptions)

        print(f"\n{'#'*60}")
        print("# CONVERGENCE PROOF GENERATION STARTED")
        print(f"#   Algorithm   : {algorithm[:50]}...")
        print(f"#   Max Iter    : {self.max_iterations}")
        print(f"#   Multi-Judge : {self.config['use_multi_judge']}")
        print(f"#   Temperatures: gen={self.temp_config.generation}, "
              f"corr={self.temp_config.correction}, "
              f"eval={self.temp_config.evaluation}")
        if self.crs_weights:
            w_err = self.crs_weights.get("w_err", 0.5)
            print(f"#   CRS Weights : w_err={w_err:.2f}  w_tfp={1-w_err:.2f}  "
                  f"w_rp={self.crs_weights.get('w_rp_pen', 0.4):.2f}")
        print(f"{'#'*60}\n")

        final_state = None
        for output in self.app.stream(initial_state):
            for _, node_data in output.items():
                final_state = node_data

        if final_state:
            self.tracker.finalize(
                final_verdict=final_state.get("verdict", "UNKNOWN"),
                final_proof=final_state.get("current_proof", ""),
            )
            log_path = self.tracker.save()
            self.tracker.save_proofs_separately()

            print(f"\n{'#'*60}")
            print("# EXECUTION COMPLETE")
            print(f"#   Final Verdict : {final_state.get('verdict')}")
            print(f"#   Total Iter    : {final_state.get('iteration')}")
            print(f"#   Log saved to  : {log_path}")
            print(f"{'#'*60}\n")

        return final_state, self.tracker