# ============================================================================
# File: src/models/schemas.py
# ============================================================================
"""Pydantic models for structured outputs and validation."""

from pydantic import BaseModel, Field, validator
from typing import Set, List, Optional


class EvaluationMetrics(BaseModel):
    """Structured output for a single evaluation pass."""

    hallucination_error: bool = Field(description="HA: any non-trivial claim without justification")
    missing_step: bool = Field(description="MS: proof has a logical gap or skipped derivation")
    operator_error: bool = Field(description="OP: incorrect algebraic/analytic manipulation")
    completeness_score: int = Field(ge=0, le=5, description="0=empty 5=fully rigorous")
    assumption_use_score: int = Field(ge=0, le=5, description="0=assumptions ignored 5=all used correctly")
    correctness_verdict: str = Field(
        default="UNKNOWN",
        description="CORRECT|PROBABLY_CORRECT|INCOMPLETE|INCORRECT — mathematical validity only"
    )
    critical_errors: List[str] = Field(
        default_factory=list,
        description="List of mathematical invalidity descriptions (wrong rate, false inequality, etc.)"
    )
    overall_verdict: str = Field(description="PASS | PASS_MINOR | CONDITIONAL | FAIL | REJECT")
    detailed_feedback: str = Field(description="Full analysis with flagged steps annotated")

    # Step-level error sets — REQUIRED for CRS to work correctly
    hallucination_steps: Set[int] = Field(
        default_factory=set,
        description="Proof line/step numbers containing hallucinations"
    )
    missing_step_indices: Set[int] = Field(
        default_factory=set,
        description="Proof line/step numbers where derivation steps are missing"
    )
    operator_error_steps: Set[int] = Field(
        default_factory=set,
        description="Proof line/step numbers with operator/computation errors"
    )
    assumption_violation_steps: Set[int] = Field(
        default_factory=set,
        description="Proof line/step numbers where assumptions are misapplied"
    )
    flagged_steps: Set[int] = Field(
        default_factory=set,
        description="Steps explicitly requested for correction in next iteration"
    )

    @validator("overall_verdict")
    def validate_verdict(cls, v: str) -> str:
        allowed = {"PASS", "PASS_MINOR", "CONDITIONAL", "FAIL", "REJECT"}
        if v not in allowed:
            # Fuzzy fix for common LLM response variations
            upper = v.strip().upper()
            if upper in allowed:
                return upper
            raise ValueError(f"Verdict must be one of {allowed}, got: {v!r}")
        return v

    @property
    def all_error_steps(self) -> Set[int]:
        return (
            self.hallucination_steps
            | self.missing_step_indices
            | self.operator_error_steps
            | self.assumption_violation_steps
        )

    @property
    def total_errors(self) -> int:
        return len(self.all_error_steps)


class CorrectionMetricsOutput(BaseModel):
    """Output from CRS computation — fully resolved with no ceiling artifacts."""

    # Core components
    error_resolution_rate: float = Field(ge=0.0, le=1.0, description="ERR")
    regression_penalty: float = Field(ge=0.0, description="RP (unbounded before clip)")
    targeted_fix_precision: float = Field(ge=0.0, le=1.0, description="TFP")

    # Final score
    correction_reasoning_score: float = Field(ge=0.0, le=1.0, description="CRS (clipped to [0,1])")
    crs_raw: float = Field(description="CRS before clipping (can be > 1 or < 0)")

    # Transition diagnostics
    errors_fixed: int
    errors_introduced: int
    flagged_fixed: int
    total_flagged: int
    is_mixed_transition: bool = Field(default=False)

    # Previous / current error totals for audit
    prev_error_count: int = Field(default=0)
    curr_error_count: int = Field(default=0)


class JudgeEvaluation(BaseModel):
    """Single judge's evaluation result."""
    judge_id: str
    model_name: str
    metrics: EvaluationMetrics
    response_time: float = Field(description="Wall-clock seconds")


class ConsensusEvaluation(BaseModel):
    """Consensus built from multiple judge evaluations."""
    num_judges: int
    evaluations: List[JudgeEvaluation]

    mean_completeness: float
    mean_assumption_score: float
    consensus_verdict: str
    verdict_variance: float = Field(default=0.0, description="Variance in judge verdicts — high = uncertain case")
    is_split_verdict: bool = Field(default=False, description="True if judges disagreed on verdict category")

    judge_reliability_scores: List[float]
    outlier_judges: List[str] = Field(default_factory=list)

    # Fully resolved consensus error set
    consensus_error_set: dict   # {"consensus_errors": Set[int]}

    weighted_cfrs: float = Field(description="Weighted quality proxy (NOT CRS)")

    # Merged error sets for CRS computation downstream
    merged_hallucination_steps: Set[int] = Field(default_factory=set)
    merged_missing_step_indices: Set[int] = Field(default_factory=set)
    merged_operator_error_steps: Set[int] = Field(default_factory=set)
    merged_assumption_violation_steps: Set[int] = Field(default_factory=set)
    merged_flagged_steps: Set[int] = Field(default_factory=set)

    class Config:
        arbitrary_types_allowed = True