# ============================================================================
# File: src/llm/openai_provider.py
# ============================================================================
"""OpenAI (or compatible) provider."""

from typing import Optional, List, Dict
import requests
from .base_provider import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible provider (works with any /v1/chat/completions endpoint)."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1/chat/completions",
        **kwargs,
    ):
        super().__init__(model_name, api_key, **kwargs)
        self.base_url = base_url

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]