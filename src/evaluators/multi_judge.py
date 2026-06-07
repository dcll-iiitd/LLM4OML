# ============================================================================
# File: src/evaluators/multi_judge.py
# ============================================================================
"""
Multi-judge ensemble evaluator with reliability-aware consensus.

Key fix over v2:
  - Merged error sets (union across judges) are now propagated back to the
    caller via ConsensusEvaluation fields so that CRS can be computed
    correctly. In v2 the multi-judge path collapsed all errors into
    assumption_violations only, making ERR/TFP unreliable.
  - Judge temperature is always near-zero (set in SingleJudge).
"""

import time
import logging
from typing import List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .single_judge import SingleJudge
from ..models.schemas import EvaluationMetrics, JudgeEvaluation, ConsensusEvaluation
from ..models.types import ErrorSet
from ..metrics.judge_metrics import JudgeReliabilityCalculator
from ..metrics.error_tracker import ErrorTracker
from ..config.temperatures import TemperatureConfig, DEFAULT_TEMPERATURES

logger = logging.getLogger(__name__)


class MultiJudgeEvaluator:
    """Ensemble of multiple LLM judges with reliability-aware consensus."""

    def __init__(
        self,
        provider_name: str,
        judge_models: List[str],
        api_key: Optional[str] = None,
        temperature: float = 0.05,   # always near-zero for evaluation
        max_tokens: int = 8192,
        timeout: int = 600,
        max_retries: int = 3,
        outlier_threshold: float = 2.0,
        temp_config: TemperatureConfig = DEFAULT_TEMPERATURES,
        per_judge_timeout: int = 600,
    ):
        self.per_judge_timeout = 600
        self.judges = [
            SingleJudge(
                provider_name=provider_name,
                model_name=model,
                judge_id=f"judge_{i}",
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
                temp_config=temp_config,
            )
            for i, model in enumerate(judge_models)
        ]
        self.reliability_calc = JudgeReliabilityCalculator()
        self.outlier_threshold = outlier_threshold

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate_parallel(self, proof, feedback=None, iteration=1):
        evaluations = []
        PER_JUDGE_TIMEOUT = 600

        with ThreadPoolExecutor(max_workers=len(self.judges)) as executor:
            futures = {
                executor.submit(
                    self._evaluate_single_judge, judge, proof, feedback, iteration
                ): judge
                for judge in self.judges
            }
            # Collect with individual deadlines
            for future, judge in futures.items():
                try:
                    result = future.result(timeout=PER_JUDGE_TIMEOUT)
                    evaluations.append(result)
                except TimeoutError:
                    logger.warning(
                        "[MultiJudge] %s timed out after %ds — skipping",
                        judge.judge_id, PER_JUDGE_TIMEOUT
                    )
                    future.cancel()
                except Exception as exc:
                    logger.error("[MultiJudge] %s failed: %s", judge.judge_id, exc)

        if not evaluations:
            raise RuntimeError("All judges failed or timed out. Cannot form consensus.")
        return self._compute_consensus(evaluations)

    # ------------------------------------------------------------------
    # Judge execution
    # ------------------------------------------------------------------

    def _evaluate_single_judge(
        self,
        judge: SingleJudge,
        proof: str,
        feedback: Optional[str],
        iteration: int,
    ) -> JudgeEvaluation:
        start = time.time()
        metrics = judge.evaluate(proof, feedback, iteration)
        return JudgeEvaluation(
            judge_id=judge.judge_id,
            model_name=judge.model_name,
            metrics=metrics,
            response_time=time.time() - start,
        )

    # ------------------------------------------------------------------
    # Consensus logic
    # ------------------------------------------------------------------

    def _compute_consensus(
        self, evaluations: List[JudgeEvaluation]
    ) -> ConsensusEvaluation:

        error_sets: List[ErrorSet] = [
            ErrorTracker.error_set_from_metrics(e.metrics) for e in evaluations
        ]
        completeness = [e.metrics.completeness_score for e in evaluations]
        assumption_use = [e.metrics.assumption_use_score for e in evaluations]

        # JRS
        combined_scores = [0.5 * c + 0.5 * a for c, a in zip(completeness, assumption_use)]
        judge_metrics = self.reliability_calc.compute_jrs(
            error_sets=error_sets, scores=combined_scores
        )
        jrs = np.array([jm["jrs"] for jm in judge_metrics])
        z_scores = [jm["z_score"] for jm in judge_metrics]

        # Outlier detection
        outliers: Set[int] = set(
            self.reliability_calc.detect_outliers(z_scores, self.outlier_threshold)
        )
        valid_indices = [i for i in range(len(evaluations)) if i not in outliers] or list(
            range(len(evaluations))
        )

        # Stable weights from JRS (floored at 1e-6)
        weights = np.array([max(float(jrs[i]), 1e-6) for i in valid_indices])
        weights /= weights.sum()

        # Ordinal-aware verdict (weighted median)
        verdict_to_score = {"PASS": 5, "PASS_MINOR": 4, "CONDITIONAL": 3, "FAIL": 2, "REJECT": 1}
        
        verdict_scores_list = [
            verdict_to_score.get(evaluations[i].metrics.overall_verdict, 2) 
            for i in valid_indices
        ]
        verdict_variance = float(np.var(verdict_scores_list)) if verdict_scores_list else 0.0
        judge_split = len(set(
            evaluations[i].metrics.overall_verdict for i in valid_indices
        )) > 1
        
        scored = sorted(
            (verdict_scores_list[j], weights[j])
            for j, i in enumerate(valid_indices)
        )
        cumulative = 0.0
        consensus_score = 2
        for score, w in scored:
            cumulative += w
            if cumulative >= 0.5:
                consensus_score = score
                break
        consensus_verdict = self._score_to_verdict(consensus_score)

        # Weighted quality means (valid judges only)
        mean_completeness = float(
            sum(weights[j] * completeness[i] for j, i in enumerate(valid_indices))
        )
        mean_assumption = float(
            sum(weights[j] * assumption_use[i] for j, i in enumerate(valid_indices))
        )
        weighted_cfrs = (mean_completeness + mean_assumption) / 10.0

        # ----------------------------------------------------------------
        # Merged error sets — CRITICAL FIX vs v2
        # Use majority-vote merging across all valid judges
        # ----------------------------------------------------------------
        valid_error_sets = [error_sets[i] for i in valid_indices]
        merged = ErrorTracker.merge_error_sets(valid_error_sets, mode="majority")

        # Also union-merge flagged steps (more inclusive is safer for correction)
        merged_flagged: Set[int] = set()
        for ev in [evaluations[i] for i in valid_indices]:
            merged_flagged |= set(ev.metrics.flagged_steps)

        # Consensus error set (for backward compat)
        consensus_error_set = {
            "consensus_errors": ErrorTracker.get_all_errors(merged)
        }

        return ConsensusEvaluation(
            num_judges=len(evaluations),
            evaluations=evaluations,
            mean_completeness=mean_completeness,
            mean_assumption_score=mean_assumption,
            consensus_verdict=consensus_verdict,
            verdict_variance=verdict_variance,
            is_split_verdict=judge_split,
            judge_reliability_scores=jrs.tolist(),
            outlier_judges=[evaluations[i].judge_id for i in outliers],
            consensus_error_set=consensus_error_set,
            weighted_cfrs=weighted_cfrs,
            # Fully resolved per-type merged sets for CRS downstream
            merged_hallucination_steps=set(merged.get("hallucinations", set())),
            merged_missing_step_indices=set(merged.get("missing_steps", set())),
            merged_operator_error_steps=set(merged.get("operator_errors", set())),
            merged_assumption_violation_steps=set(merged.get("assumption_violations", set())),
            merged_flagged_steps=merged_flagged,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_verdict(score: int) -> str:
        return {5: "PASS", 4: "PASS_MINOR", 3: "CONDITIONAL", 2: "FAIL", 1: "REJECT"}.get(
            score, "FAIL"
        )