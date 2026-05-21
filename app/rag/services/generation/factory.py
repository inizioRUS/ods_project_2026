from __future__ import annotations

from core.config import LLMConfig
from services.generation.base import BaseGenerator
from services.generation.mistral import MistralGenerator
from services.generation.noop import NoopGenerator


def build_generator(config: LLMConfig) -> BaseGenerator:
    if config.provider == "noop":
        return NoopGenerator()
    if config.provider == "mistral":
        return MistralGenerator(config)
    raise ValueError(f"Unknown llm provider: {config.provider}")
