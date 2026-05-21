from __future__ import annotations

from services.generation.base import BaseGenerator


class NoopGenerator(BaseGenerator):
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "LLM provider is disabled. Switch llm.provider to 'mistral' in config.yaml."
