"""Runtime-neutral boundary for future Codex CLI or API adapters."""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, request: str) -> str:
        """Accept serialized text and return raw text; adapters own transport errors.

        Future agents will serialize prompts and validate returned model data.
        This interface makes no assumptions about authentication or runtime.
        """
        raise NotImplementedError
