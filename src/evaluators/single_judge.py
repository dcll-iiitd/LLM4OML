# ============================================================================
# File: src/evaluators/single_judge.py
# ============================================================================
"""
Single LLM judge evaluator.

Temperature strategy:
  - ALWAYS near-zero (0.05) for evaluation/verification regardless of
    global config. Deterministic, structured JSON output is critical here.
    High temperature produces inconsistent verdicts and malformed JSON.
"""

import json
import re
import time
import logging
from typing import Optional

from .base_evaluator import BaseEvaluator
from ..models.schemas import EvaluationMetrics
from ..utils.prompts import PromptManager
from ..llm.provider_factory import LLMProviderFactory
from ..config.temperatures import get_judge_temperature, TemperatureConfig, DEFAULT_TEMPERATURES

logger = logging.getLogger(__name__)


class SingleJudge(BaseEvaluator):
    """Single LLM evaluator with deterministic temperature and robust JSON parsing."""

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        judge_id: str,
        api_key: Optional[str] = None,
        temperature: float = 0.05,   # default override — evaluation must be near-0
        max_tokens: int = 8192,      # bumped to 8192 to prevent reasoning models from getting cut off
        timeout: int = 600,
        max_retries: int = 3,
        temp_config: TemperatureConfig = DEFAULT_TEMPERATURES,
    ):
        super().__init__(model_name, judge_id)
        self.provider_name = provider_name
        self.temp_config = temp_config

        self.llm = LLMProviderFactory.create(
            provider_name=provider_name,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.prompt_manager = PromptManager()

    def evaluate(
        self,
        proof: str,
        feedback: Optional[str] = None,
        iteration: int = 1,
    ) -> EvaluationMetrics:
        """Evaluate a proof with deterministic temperature."""
        temperature = get_judge_temperature(iteration, self.temp_config)
        phase = "evaluation" if iteration <= 1 else "verification"

        logger.info(
            "[%s] %s (iter %d), temperature=%.2f",
            self.judge_id, phase, iteration, temperature,
        )
        print(
            f"[{self.judge_id}] {phase} (iter {iteration}), "
            f"model={self.model_name}, temperature={temperature:.2f}"
        )

        if iteration == 1:
            prompt = self.prompt_manager.get_initial_evaluation_prompt(proof)
        else:
            prompt = self.prompt_manager.get_verification_prompt(feedback or "", proof)

        result = self._parse_with_retries(prompt, temperature)
        if result is None:
            result = self._fallback_result()
        return result

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_with_retries(
        self, prompt: str, temperature: float, max_attempts: int = 3
    ) -> Optional[EvaluationMetrics]:
        """
        Try to get a valid JSON EvaluationMetrics response.
        Attempts in order:
          1. Plain prompt → parse
          2. Add explicit JSON instruction → parse
          3. Add format schema → parse
        """
        format_instructions = (
            "\n\nIMPORTANT: You MUST respond with ONLY valid JSON. No text before or after. "
            "DO NOT output any reasoning, <think> tags, or conversational text. Output the JSON object immediately. "
            "The JSON must have these exact keys: hallucination_error (bool), missing_step (bool), "
            "operator_error (bool), completeness_score (int 0-5), assumption_use_score (int 0-5), "
            "overall_verdict (str: PASS|PASS_MINOR|CONDITIONAL|FAIL|REJECT), "
            "detailed_feedback (str), hallucination_steps (list of ints), "
            "missing_step_indices (list of ints), operator_error_steps (list of ints), "
            "assumption_violation_steps (list of ints), flagged_steps (list of ints)."
        )

        attempts = [prompt, prompt + format_instructions, prompt + format_instructions]

        for attempt_idx, full_prompt in enumerate(attempts):
            try:
                messages = [{"role": "user", "content": full_prompt}]
                response = self.llm.invoke(messages, temperature=temperature)
                result = self._parse_response(response)
                if result is not None:
                    return result
                logger.warning(
                    "[%s] JSON parse attempt %d failed. Output start: %s ...", 
                    self.judge_id, attempt_idx + 1, response[:200].replace('\n', ' ')
                )
            except Exception as exc:
                logger.warning(
                    "[%s] API call attempt %d failed: %s", self.judge_id, attempt_idx + 1, exc
                )

        return None

    def _parse_response(self, text: str) -> Optional[EvaluationMetrics]:
        """Extract and validate JSON from LLM response text."""
        # Strip out <think> blocks if reasoning models still emitted them
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Also strip unclosed <think> blocks in case max_tokens cut it off
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        
        # 1. Try ```json ... ```
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                pass
            else:
                return self._coerce_data(data)

        # 2. Try finding the json block by locating the opening brace containing required schema keys
        key_idx = text.find('"hallucination_error"')
        if key_idx != -1:
            start_idx = text.rfind('{', 0, key_idx)
        else:
            start_idx = text.find('{')
            
        if start_idx == -1:
            return None
            
        # Walk backwards from the end of the text to find the matching '}' 
        # that parses correctly. Handles cases where output ends with random { or }.
        end_idx = len(text)
        data = None
        while True:
            end_idx = text.rfind('}', start_idx, end_idx)
            if end_idx == -1:
                break
                
            json_str = text[start_idx:end_idx+1]
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
            try:
                data = json.loads(json_str)
                break
            except json.JSONDecodeError:
                pass

        if data is None:
            logger.debug("[%s] Failed to parse JSON entirely.", self.judge_id)
            return None
            
        return self._coerce_data(data)

    def _coerce_data(self, data: dict) -> Optional[EvaluationMetrics]:
        # Coerce types that LLMs commonly get wrong
        for list_field in [
            "hallucination_steps",
            "missing_step_indices",
            "operator_error_steps",
            "assumption_violation_steps",
            "flagged_steps",
        ]:
            if list_field not in data or data[list_field] is None:
                data[list_field] = []
            elif isinstance(data[list_field], (int, float)):
                data[list_field] = [int(data[list_field])]
            elif isinstance(data[list_field], str):
                val = data[list_field].strip().lower()
                if val in ("none", "null", "[]", "set()", ""):
                    data[list_field] = []
                else:
                    # attempt to extract integers from strings like "1, 2" or "[1, 2]"
                    nums = re.findall(r'\d+', data[list_field])
                    data[list_field] = [int(n) for n in nums]
            elif not isinstance(data[list_field], list):
                # Fallback for unexpected types
                data[list_field] = []

        # Coerce booleans
        for bool_field in ["hallucination_error", "missing_step", "operator_error"]:
            if isinstance(data.get(bool_field), str):
                data[bool_field] = data[bool_field].lower() in ("true", "yes", "1")

        # Coerce int scores
        for int_field in ["completeness_score", "assumption_use_score"]:
            if isinstance(data.get(int_field), str):
                try:
                    data[int_field] = int(data[int_field])
                except ValueError:
                    data[int_field] = 0

        try:
            return EvaluationMetrics(**data)
        except Exception as exc:
            logger.debug("[%s] Pydantic validation error: %s", self.judge_id, exc)
            return None

    def _fallback_result(self) -> EvaluationMetrics:
        """Emergency fallback when all parse attempts fail."""
        return EvaluationMetrics(
            hallucination_error=False,
            missing_step=False,
            operator_error=False,
            completeness_score=0,
            assumption_use_score=0,
            correctness_verdict="INCORRECT",
            critical_errors=["SYSTEM ERROR: Parsing failed"],
            overall_verdict="FAIL",
            detailed_feedback=(
                f"SYSTEM ERROR: Evaluation parsing failed after all retries. "
                f"Judge: {self.judge_id}, Model: {self.model_name}"
            ),
        )