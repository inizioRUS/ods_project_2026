from __future__ import annotations

import os

import httpx

from core.config import LLMConfig
from services.generation.base import BaseGenerator


class MistralGenerator(BaseGenerator):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = os.getenv("MISTRAL_API_KEY", "")
        self.base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1/chat/completions")

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        print(system_prompt)
        print(user_prompt)
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set")

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"].strip()
