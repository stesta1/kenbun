from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.logging import get_logger

logger = get_logger(__name__)


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        raise NotImplementedError


class OpenAICompatibleLLMClient(LLMClient):
    def __init__(self, base_url: str | None = None, model_name: str | None = None, temperature: float | None = None) -> None:
        self.base_url = base_url or settings.llm_server_base_url
        self.model_name = model_name or settings.llm_model_name
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self._client = httpx.Client(base_url=self.base_url, timeout=60.0)

    def generate(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        body = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        response = self._client.post("/chat/completions", json=body)
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    def close(self) -> None:
        self._client.close()
