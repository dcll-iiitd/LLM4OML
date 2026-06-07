# ============================================================================
# File: src/models/types.py
# ============================================================================
"""Type definitions for the convergence proof agent v3."""

from typing import TypedDict, List, Dict, Set, Optional, Literal
from typing_extensions import NotRequired

ErrorType = Literal["hallucination", "missing_step", "operator_error", "assumption_violation"]
VerdictType = Literal["PASS", "PASS_MINOR", "CONDITIONAL", "FAIL", "REJECT"]

VERDICT_SCORE: Dict[str, int] = {
    "PASS": 5,
    "PASS_MINOR": 4,
    "CONDITIONAL": 3,
    "FAIL": 2,
    "REJECT": 1,
}


class ErrorSet(TypedDict):
    """Set of error step indices by type."""
    hallucinations: Set[int]        # Hallucination error steps
    missing_steps: Set[int]         # Missing step indices
    operator_errors: Set[int]       # Operator/computation error steps
    assumption_violations: Set[int] # Assumption misuse steps


class IterationMetrics(TypedDict):
    """Metrics for a single evaluation iteration."""
    hallucination_error: bool
    missing_step: bool
    operator_error: bool
    completeness_score: int         # 0-5
    assumption_use_score: int       # 0-5
    error_set: ErrorSet
    flagged_steps: Set[int]         # Steps explicitly flagged by evaluator for next correction


class CorrectionMetrics(TypedDict):
    """Correction Reasoning Score components."""
    err: float   # Error Resolution Rate  [0,1]
    rp: float    # Regression Penalty     [0, +inf, clipped to 1]
    tfp: float   # Targeted Fix Precision [0,1]
    crs: float   # Overall CRS            [0,1] after clipping


class JudgeMetrics(TypedDict):
    """Per-judge reliability metrics."""
    esa: float      # Error Set Agreement (Jaccard w/ peers)
    sc: float       # Score Consistency (group variance, lower = better)
    z_score: float  # Standardized score deviation
    jrs: float      # Judge Reliability Score


class AgentState(TypedDict):
    """Complete state for the LangGraph workflow."""
    algorithm_description: str
    assumptions: str
    current_proof: str
    previous_proof: str
    feedback: str
    iteration: int
    metrics: Dict
    verdict: VerdictType

    # Error tracking across iterations
    error_set_current: NotRequired[ErrorSet]
    error_set_previous: NotRequired[ErrorSet]
    flagged_steps_current: NotRequired[Set[int]]

    # Correction metrics (populated from iteration 2 onward)
    correction_metrics: NotRequired[CorrectionMetrics]

    # Multi-judge fields
    judge_feedbacks: NotRequired[List[str]]
    judge_metrics: NotRequired[List[IterationMetrics]]
    judge_reliability: NotRequired[List[JudgeMetrics]]
    consensus_verdict: NotRequired[VerdictType]