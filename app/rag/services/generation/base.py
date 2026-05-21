from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError
