# ============================================================================
# File: src/llm/anthropic_provider.py
# ============================================================================
"""Anthropic Claude provider."""

from typing import Optional, List, Dict
import requests
from .base_provider import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Messages API provider."""

    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

        system_msg = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        payload: Dict = {
            "model": self.model_name,
            "messages": user_messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if system_msg:
            payload["system"] = system_msg

        response = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["content"][0]["text"]