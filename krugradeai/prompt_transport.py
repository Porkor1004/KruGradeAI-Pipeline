"""Shared instruction-prefix adapter; no role or runtime assumptions."""
from krugradeai.providers import LLMProvider


class PromptProvider(LLMProvider):
    """Replace only the instruction prefix; LLMGrader still owns serialization/validation."""
    def __init__(self, provider, instructions):
        self.provider, self.instructions, self.calls = provider, instructions, 0

    def generate(self, request):
        payload = request.split("\nINPUT_JSON\n", 1)[1]
        self.calls += 1
        return self.provider.generate(self.instructions + "\nINPUT_JSON\n" + payload)
