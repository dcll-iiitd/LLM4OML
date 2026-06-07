# ============================================================================
# File: src/graph/nodes.py  (patched for ablation study — crs_weights param added)
# ============================================================================
"""
LangGraph node implementations.

Key fixes over v2:
  1. CRS is now computed from iteration 1 onward (first iteration uses
     compute_first_iteration_crs; subsequent use the delta-based formula).
  2. Multi-judge path propagates fully resolved per-type error sets so that
     ERR, RP, and TFP are computed correctly.
  3. flagged_steps_current is always populated and passed through state so
     TFP has non-trivial data to work with.

ABLATION PATCH (v3.1):
  4. WorkflowNodes now accepts an optional `crs_weights` dict that is forwarded
     to CorrectionMetricsCalculator.  This lets the ablation orchestrator swap
     weights without touching any other part of the pipeline.
     Default is None → uses the existing defaults (0.50 / 0.50 / 0.40).
"""

import logging
from typing import Dict, Any, Optional, Set

from ..models.types import AgentState, ErrorSet
from ..models.schemas import EvaluationMetrics, ConsensusEvaluation
from ..agents.prover_agent import ProverAgent
from ..evaluators.multi_judge import MultiJudgeEvaluator
from ..evaluators.single_judge import SingleJudge
from ..metrics.correction_metrics import CorrectionMetricsCalculator
from ..metrics.error_tracker import ErrorTracker
from ..utils.state import StateManager
from ..config.temperatures import TemperatureConfig, DEFAULT_TEMPERATURES

logger = logging.getLogger(__name__)


class WorkflowNodes:
    """All node logic for the convergence proof workflow."""

    def __init__(
        self,
        provider_name: str,
        prover_model: str,
        evaluator_models: list,
        api_key: Optional[str] = None,
        use_multi_judge: bool = True,
        temperature: float = 0.5,          # instance default (overridden per phase)
        timeout: int = 600,
        max_retries: int = 3,
        temp_config: TemperatureConfig = DEFAULT_TEMPERATURES,
        # ── ABLATION PATCH ────────────────────────────────────────────────────
        crs_weights: Optional[Dict[str, float]] = None,
        per_judge_timeout: int = 600,
        judge_provider_name: Optional[str] = None,
        # ─────────────────────────────────────────────────────────────────────
    ):
        self.temp_config = temp_config

        self.prover = ProverAgent(
            provider_name=provider_name,
            model_name=prover_model,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            temp_config=temp_config,
        )

        actual_judge_provider = judge_provider_name or provider_name

        if use_multi_judge:
            self.evaluator = MultiJudgeEvaluator(
                provider_name=actual_judge_provider,
                judge_models=evaluator_models,
                api_key=api_key,
                temperature=temp_config.evaluation,
                timeout=timeout,
                max_retries=max_retries,
                temp_config=temp_config,
                per_judge_timeout=per_judge_timeout,
            )
        else:
            self.evaluator = SingleJudge(
                provider_name=actual_judge_provider,
                model_name=evaluator_models[0],
                judge_id="single_judge",
                api_key=api_key,
                temperature=temp_config.evaluation,
                timeout=timeout,
                max_retries=max_retries,
                temp_config=temp_config,
            )

        self.use_multi_judge = use_multi_judge

        # ── ABLATION PATCH: inject weights into calculator ────────────────────
        _w = crs_weights or {}
        w_err    = float(_w.get("w_err",    0.60))
        w_tfp    = float(_w.get("w_tfp",    1.0 - w_err))   # honour explicit or derive
        w_rp_pen = float(_w.get("w_rp_pen", 0.20))

        # Normalise so w_err + w_tfp == 1.0 (required by CorrectionMetricsCalculator)
        total = w_err + w_tfp
        if abs(total - 1.0) > 1e-4:
            logger.warning(
                "crs_weights w_err+w_tfp=%.4f ≠ 1.0; normalising.", total
            )
            w_err = w_err / total
            w_tfp = w_tfp / total

        self.correction_calc = CorrectionMetricsCalculator(
            w_err=w_err,
            w_tfp=w_tfp,
            w_rp_pen=w_rp_pen,
        )
        # ─────────────────────────────────────────────────────────────────────

    # ------------------------------------------------------------------
    # Prover node
    # ------------------------------------------------------------------

    def prover_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate or correct the convergence proof."""
        result = self.prover.execute(
            algorithm=state["algorithm_description"],
            assumptions=state["assumptions"],
            iteration=state["iteration"],
            previous_proof=state.get("current_proof", ""),
            feedback=state.get("feedback", ""),
        )
        return result

    # ------------------------------------------------------------------
    # Evaluator node
    # ------------------------------------------------------------------

    def evaluator_node(self, state: AgentState) -> Dict[str, Any]:
        """Evaluate the current proof (single or multi-judge)."""
        logger.info("[EvaluatorNode] Evaluating (iteration %d)...", state["iteration"])

        proof = state["current_proof"]
        iteration = state["iteration"]
        feedback = state.get("feedback", "")

        if self.use_multi_judge:
            result = self._multi_judge_evaluation(proof, feedback, iteration, state)
        else:
            result = self._single_judge_evaluation(proof, feedback, iteration, state)

        return result

    # ------------------------------------------------------------------
    # Single-judge evaluation
    # ------------------------------------------------------------------

    def _single_judge_evaluation(
        self,
        proof: str,
        feedback: str,
        iteration: int,
        state: AgentState,
    ) -> Dict[str, Any]:
        metrics: EvaluationMetrics = self.evaluator.evaluate(proof, feedback, iteration)
        error_set = ErrorTracker.error_set_from_metrics(metrics)
        flagged = set(metrics.flagged_steps)

        correction_metrics = self._compute_crs(
            iteration=iteration,
            state=state,
            curr_error_set=error_set,
            flagged_steps=flagged,
            metrics=metrics,
        )

        return {
            "feedback": metrics.detailed_feedback,
            "metrics": {
                "HA": metrics.hallucination_error,
                "MS": metrics.missing_step,
                "OP": metrics.operator_error,
                "completeness_score": metrics.completeness_score,
                "assumption_use_score": metrics.assumption_use_score,
            },
            "verdict": metrics.overall_verdict,
            "error_set_previous": state.get("error_set_current", StateManager.empty_error_set()),
            "error_set_current": error_set,
            "flagged_steps_current": flagged,
            "correction_metrics": correction_metrics.dict() if correction_metrics else None,
            "iteration": iteration,
        }

    # ------------------------------------------------------------------
    # Multi-judge evaluation
    # ------------------------------------------------------------------

    def _multi_judge_evaluation(
        self,
        proof: str,
        feedback: str,
        iteration: int,
        state: AgentState,
    ) -> Dict[str, Any]:
        consensus: ConsensusEvaluation = self.evaluator.evaluate_parallel(
            proof, feedback, iteration
        )

        # Build fully resolved error set from merged consensus data
        error_set: ErrorSet = {
            "hallucinations": set(consensus.merged_hallucination_steps),
            "missing_steps": set(consensus.merged_missing_step_indices),
            "operator_errors": set(consensus.merged_operator_error_steps),
            "assumption_violations": set(consensus.merged_assumption_violation_steps),
        }
        flagged: Set[int] = set(consensus.merged_flagged_steps)

        # Build a synthetic EvaluationMetrics for first-iteration CRS
        synthetic_metrics = EvaluationMetrics(
            hallucination_error=any(e.metrics.hallucination_error for e in consensus.evaluations),
            missing_step=any(e.metrics.missing_step for e in consensus.evaluations),
            operator_error=any(e.metrics.operator_error for e in consensus.evaluations),
            completeness_score=round(consensus.mean_completeness),
            assumption_use_score=round(consensus.mean_assumption_score),
            correctness_verdict="CORRECT", # Synthetic placeholder; not aggregated natively here
            critical_errors=[],
            overall_verdict=consensus.consensus_verdict,
            detailed_feedback="",
            hallucination_steps=set(consensus.merged_hallucination_steps),
            missing_step_indices=set(consensus.merged_missing_step_indices),
            operator_error_steps=set(consensus.merged_operator_error_steps),
            assumption_violation_steps=set(consensus.merged_assumption_violation_steps),
            flagged_steps=flagged,
        )

        correction_metrics = self._compute_crs(
            iteration=iteration,
            state=state,
            curr_error_set=error_set,
            flagged_steps=flagged,
            metrics=synthetic_metrics,
        )

        # Aggregate feedback from all judges
        feedback_parts = [
            f"=== CONSENSUS: {consensus.consensus_verdict} ===",
            f"Mean Completeness: {consensus.mean_completeness:.2f}",
            f"Mean Assumption Score: {consensus.mean_assumption_score:.2f}",
            f"Outlier Judges: {consensus.outlier_judges or 'none'}",
            "",
        ]
        for ev in consensus.evaluations:
            jrs_val = consensus.judge_reliability_scores[
                next(
                    (i for i, e in enumerate(consensus.evaluations) if e.judge_id == ev.judge_id),
                    0,
                )
            ]
            feedback_parts.append(
                f"--- {ev.judge_id} ({ev.model_name}, JRS={jrs_val:.3f}) ---"
            )
            feedback_parts.append(ev.metrics.detailed_feedback)
            feedback_parts.append("")
        full_feedback = "\n".join(feedback_parts)

        return {
            "feedback": full_feedback,
            "metrics": {
                "HA": synthetic_metrics.hallucination_error,
                "MS": synthetic_metrics.missing_step,
                "OP": synthetic_metrics.operator_error,
                "completeness_score": synthetic_metrics.completeness_score,
                "assumption_use_score": synthetic_metrics.assumption_use_score,
                "weighted_cfrs": consensus.weighted_cfrs,
            },
            "verdict": consensus.consensus_verdict,
            "verdict_variance": consensus.verdict_variance,
            "is_split_verdict": consensus.is_split_verdict,
            "error_set_previous": state.get("error_set_current", StateManager.empty_error_set()),
            "error_set_current": error_set,
            "flagged_steps_current": flagged,
            "correction_metrics": correction_metrics.dict() if correction_metrics else None,
            "judge_evaluations": [e.dict() for e in consensus.evaluations],
            "judge_reliability": consensus.judge_reliability_scores,
            "iteration": iteration,
        }

    # ------------------------------------------------------------------
    # CRS computation (always computed, even on first iteration)
    # ------------------------------------------------------------------

    def _compute_crs(
        self,
        iteration: int,
        state: AgentState,
        curr_error_set: ErrorSet,
        flagged_steps: Set[int],
        metrics: EvaluationMetrics,
    ):
        """
        Compute CRS for ANY iteration:
          - iteration == 1 → use compute_first_iteration_crs (static quality proxy)
          - iteration >= 2 → use delta-based CRS with previous error set
        """
        if iteration <= 1:
            # First evaluation — no previous error set exists
            return self.correction_calc.compute_first_iteration_crs(metrics)

        prev_error_set = state.get("error_set_current", StateManager.empty_error_set())
        total_steps = max(len(state.get("current_proof", "").split("\n")), 1)

        return self.correction_calc.compute_crs(
            prev_error_set=prev_error_set,
            curr_error_set=curr_error_set,
            flagged_steps=flagged_steps,
            total_steps_current=total_steps,
        )