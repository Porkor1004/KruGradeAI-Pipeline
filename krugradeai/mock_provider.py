"""Deterministic test provider with no network, subprocess or model invocation."""
from krugradeai.providers import LLMProvider


class MockLLMProvider(LLMProvider):
    """Return the configured raw response unchanged on every call."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[str] = []

    def generate(self, request: str) -> str:
        self.requests.append(request)
        return self.response
