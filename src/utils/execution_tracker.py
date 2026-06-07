# ============================================================================
# File: src/utils/execution_tracker.py
# ============================================================================
"""Complete execution tracker — saves all iteration states to JSON."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from ..models.types import AgentState

logger = logging.getLogger(__name__)


class ExecutionTracker:
    """
    Tracks complete execution history for a single algorithm.
    All proof content, feedback, metrics, and error sets are preserved
    so that CRS can be audited and reproduced offline.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.execution_log: Dict[str, Any] = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "status": "running",
            },
            "configuration": {},
            "algorithm_description": "",
            "assumptions": "",
            "iterations": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_metadata(
        self,
        algorithm: str,
        assumptions: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.execution_log["algorithm_description"] = algorithm
        self.execution_log["assumptions"] = assumptions
        if config:
            self.execution_log["configuration"] = config

    def track_iteration(
        self,
        iteration: int,
        node_name: str,
        state: AgentState,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """Record the state produced by a single node in a single iteration."""
        # Ensure slots exist
        while len(self.execution_log["iterations"]) < iteration:
            self.execution_log["iterations"].append(
                {
                    "iteration_number": len(self.execution_log["iterations"]) + 1,
                    "nodes": {},
                }
            )

        iter_log = self.execution_log["iterations"][iteration - 1]
        entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "node_name": node_name,
        }

        if node_name == "prover":
            entry.update(
                {
                    "proof_content": state.get("current_proof", ""),
                    "proof_length": len(state.get("current_proof", "")),
                    "previous_proof_length": len(state.get("previous_proof", "")),
                    "iteration": state.get("iteration", 0),
                }
            )
        elif node_name == "evaluator":
            entry.update(
                {
                    "feedback": state.get("feedback", ""),
                    "verdict": state.get("verdict", ""),
                    "metrics": state.get("metrics", {}),
                    "error_set": self._serialise_error_set(
                        state.get("error_set_current", {})
                    ),
                    "flagged_steps": list(
                        state.get("flagged_steps_current", set())
                    ),
                    "correction_metrics": state.get("correction_metrics"),
                    "judge_evaluations": state.get("judge_evaluations", []),
                    "judge_reliability": state.get("judge_reliability", []),
                }
            )

        if additional_data:
            entry.update(additional_data)

        iter_log["nodes"][node_name] = entry
        self._update_summary(iteration)

    def finalize(self, final_verdict: str, final_proof: str):
        self.execution_log["metadata"]["end_time"] = datetime.now().isoformat()
        self.execution_log["metadata"]["status"] = "completed"
        self.execution_log["final_results"] = {
            "verdict": final_verdict,
            "total_iterations": len(self.execution_log["iterations"]),
            "final_proof_length": len(final_proof),
            "convergence_achieved": final_verdict in ("PASS", "PASS_MINOR"),
        }
        self._calculate_statistics()

    def save(self, filename: Optional[str] = None) -> Path:
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"execution_log_{ts}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.execution_log, f, indent=2, default=str)
        logger.info("Execution log saved to %s", filepath)
        return filepath

    def save_proofs_separately(self):
        proofs_dir = self.output_dir / "proofs"
        proofs_dir.mkdir(exist_ok=True)
        for iter_log in self.execution_log["iterations"]:
            if "prover" in iter_log["nodes"]:
                n = iter_log["iteration_number"]
                content = iter_log["nodes"]["prover"]["proof_content"]
                path = proofs_dir / f"proof_iteration_{n}.tex"
                path.write_text(content, encoding="utf-8")

    def get_log(self) -> Dict[str, Any]:
        return self.execution_log

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise_error_set(error_set: Dict) -> Dict:
        return {
            k: list(v) if isinstance(v, (set, frozenset)) else (v or [])
            for k, v in error_set.items()
        }

    def _update_summary(self, iteration: int):
        iter_log = self.execution_log["iterations"][iteration - 1]
        if "evaluator" in iter_log["nodes"]:
            ev = iter_log["nodes"]["evaluator"]
            m = ev.get("metrics", {})
            summary = {
                "verdict": ev.get("verdict"),
                "completeness_score": m.get("completeness_score"),
                "has_errors": any(
                    [m.get("HA", False), m.get("MS", False), m.get("OP", False)]
                ),
            }
            cm = ev.get("correction_metrics")
            if cm:
                summary["crs"] = (
                    cm.get("correction_reasoning_score")
                    if isinstance(cm, dict)
                    else getattr(cm, "correction_reasoning_score", None)
                )
            iter_log["summary"] = summary

    def _calculate_statistics(self):
        iterations = self.execution_log["iterations"]
        if not iterations:
            return

        stats: Dict[str, Any] = {
            "total_errors_identified": 0,
            "errors_by_type": {
                "hallucinations": 0,
                "missing_steps": 0,
                "operator_errors": 0,
                "assumption_violations": 0,
            },
            "verdict_progression": [],
            "crs_progression": [],
            "proof_length_progression": [],
        }

        for il in iterations:
            if "summary" in il:
                stats["verdict_progression"].append(il["summary"].get("verdict"))
                crs = il["summary"].get("crs")
                if crs is not None:
                    stats["crs_progression"].append(crs)

            if "evaluator" in il["nodes"]:
                es = il["nodes"]["evaluator"].get("error_set", {})
                for key in stats["errors_by_type"]:
                    steps = es.get(key, [])
                    n = len(steps) if steps else 0
                    stats["errors_by_type"][key] += n
                    stats["total_errors_identified"] += n

            if "prover" in il["nodes"]:
                stats["proof_length_progression"].append(
                    il["nodes"]["prover"].get("proof_length", 0)
                )

        self.execution_log["statistics"] = stats