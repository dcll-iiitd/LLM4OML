# ============================================================================
# File: src/agents/prover_agent.py
# ============================================================================
"""
Proof generation agent.

Temperature strategy:
  - iteration 0 (initial generation): HIGH (0.7) — encourages creative,
    diverse proof strategies.
  - iteration 1+ (correction): MODERATE (0.4) — targeted edits; high temp
    would cause the model to rewrite correct sections needlessly.
"""

import logging
from typing import Optional
from .base_agent import BaseAgent
from ..utils.parsers import ResponseParser
from ..config.temperatures import get_prover_temperature, TemperatureConfig, DEFAULT_TEMPERATURES

logger = logging.getLogger(__name__)


class ProverAgent(BaseAgent):
    """Agent for generating and correcting convergence proofs."""

    def __init__(self, *args, temp_config: TemperatureConfig = DEFAULT_TEMPERATURES, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp_config = temp_config

    def execute(
        self,
        algorithm: str,
        assumptions: str,
        iteration: int = 0,
        previous_proof: str = "",
        feedback: str = "",
    ) -> dict:
        """Generate or correct a convergence proof with phase-appropriate temperature."""
        phase = "generation" if iteration == 0 else "correction"
        temperature = get_prover_temperature(iteration, self.temp_config)

        logger.info(
            "[ProverAgent] Iteration %d (%s), temperature=%.2f", iteration + 1, phase, temperature
        )
        print(
            f"[ProverAgent] Iteration {iteration + 1} ({phase}), "
            f"model={self.model_name}, temperature={temperature:.2f}"
        )

        try:
            if iteration == 0:
                prompt = self.prompt_manager.get_generation_prompt(algorithm, assumptions)
            else:
                prompt = self.prompt_manager.get_correction_prompt(previous_proof, feedback)

            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages, temperature=temperature)
            proof = ResponseParser.clean_latex(response)

            return {
                "current_proof": proof,
                "previous_proof": previous_proof,
                "iteration": iteration + 1,
            }

        except Exception as exc:
            logger.error("[ProverAgent] Failed after all retries: %s", exc)
            error_note = f"\n% ERROR: {exc}\n% Proof generation failed after retries."
            return {
                "current_proof": (previous_proof + error_note) if previous_proof else error_note,
                "previous_proof": previous_proof,
                "iteration": iteration + 1,
            }