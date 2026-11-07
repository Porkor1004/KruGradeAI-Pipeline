"""Concrete provider-backed grader; no runtime-specific integration."""
import json
from decimal import Decimal
from typing import Literal

from krugradeai.agents import Grader
from krugradeai.models import Contract, GraderInput, GraderOutput, GradingRecord, Text
from krugradeai.prompts import build_grading_prompt
from krugradeai.providers import LLMProvider


class HumanReviewRequired(Exception):
    """No score is issued when evidence cannot support reliable grading."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class _ReviewRequest(Contract):
    human_review_required: Literal[True]
    review_reason: Text


class _ScoredResponse(GraderOutput):
    # Legacy records remain valid, but new model responses must provide feedback.
    overall_feedback: Text


class AIAdjudicatedGrade(GraderOutput):
    """Internally issued provisional grade, never accepted from the direct Grader."""


class CLIGraderResponse(Contract):
    """Single object root required by CLI structured output, with either valid outcome."""
    result: _ScoredResponse | _ReviewRequest


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


class LLMGrader(Grader):
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def grade(self, request: GraderInput) -> GraderOutput:
        """Return a validated grade, or raise for invalid output/human review.

        Provider exceptions propagate unchanged. No retries, repairs or external
        data access occur; a failed response never becomes a default score.
        """
        request = GraderInput.model_validate(request)
        if not request.response.text.strip():
            raise HumanReviewRequired("Student answer is empty; confirm submission or OCR content before scoring.")
        raw = self.provider.generate(build_grading_prompt(request))
        data = json.loads(raw, parse_float=Decimal, parse_constant=_reject_constant,
                          object_pairs_hook=_unique_object)
        if isinstance(data, dict) and "human_review_required" in data:
            review = _ReviewRequest.model_validate(data)
            raise HumanReviewRequired(review.review_reason)
        output = _ScoredResponse.model_validate(data)
        return GradingRecord(grader_input=request, grader_output=output).grader_output
