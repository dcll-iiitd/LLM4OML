# ============================================================================
# File: src/utils/state.py
# ============================================================================
"""State management utilities."""

from typing import Dict, Any
from ..models.types import AgentState, ErrorSet


_EMPTY_ERROR_SET: ErrorSet = {
    "hallucinations": set(),
    "missing_steps": set(),
    "operator_errors": set(),
    "assumption_violations": set(),
}


class StateManager:
    """Manages agent state transitions and updates."""

    @staticmethod
    def initialize_state(algorithm: str, assumptions: str) -> AgentState:
        """Create the initial agent state."""
        return {
            "algorithm_description": algorithm,
            "assumptions": assumptions,
            "current_proof": "",
            "previous_proof": "",
            "feedback": "",
            "iteration": 0,
            "metrics": {},
            "verdict": "FAIL",
            "error_set_current": dict(_EMPTY_ERROR_SET),
            "error_set_previous": dict(_EMPTY_ERROR_SET),
            "flagged_steps_current": set(),
        }

    @staticmethod
    def should_continue(state: AgentState, max_iterations: int = 3) -> str:
        """Routing function for LangGraph conditional edges."""
        verdict = state.get("verdict", "FAIL")
        iteration = state.get("iteration", 0)

        if iteration >= max_iterations:
            return "end"
        if verdict in ("PASS", "PASS_MINOR"):
            return "end"
        if "SYSTEM ERROR" in state.get("feedback", ""):
            return "end"

        return "correct"

    @staticmethod
    def empty_error_set() -> ErrorSet:
        return {k: set() for k in _EMPTY_ERROR_SET}