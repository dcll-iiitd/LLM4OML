# ============================================================================
# File: src/evaluators/base_evaluator.py
# ============================================================================
"""Base evaluator interface."""

from abc import ABC, abstractmethod
from typing import Optional
from ..models.schemas import EvaluationMetrics


class BaseEvaluator(ABC):
    """Abstract base class for proof evaluators."""

    def __init__(self, model_name: str, judge_id: str):
        self.model_name = model_name
        self.judge_id = judge_id

    @abstractmethod
    def evaluate(
        self,
        proof: str,
        feedback: Optional[str] = None,
        iteration: int = 1,
    ) -> EvaluationMetrics:
        """
        Evaluate a convergence proof.

        Parameters
        ----------
        proof     : the proof text to evaluate
        feedback  : previous feedback (used for verification on iter > 1)
        iteration : current iteration number (1 = first eval, 2+ = re-eval)

        Returns
        -------
        EvaluationMetrics with populated step-level error sets
        """
        pass