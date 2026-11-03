"""Provider-backed Refiner producing candidate AdaptationRules only."""
import json
from decimal import Decimal
from typing import Self

from pydantic import model_validator

from krugradeai.agents import Refiner
from krugradeai.models import AdaptationRule, AdaptationRules, Contract, RefinerInput, Text
from krugradeai.providers import LLMProvider
from krugradeai.refiner_prompt import build_refiner_prompt
from krugradeai.rule_validation import validate_rule_text


class _CandidateResponse(AdaptationRules):
    rules: tuple[Text | AdaptationRule, ...]  # Required in model output.


class _ValidatedCandidate(Contract):
    request: RefinerInput
    candidate: AdaptationRules

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        request, candidate = self.request, self.candidate
        if request.rubric is None:
            raise ValueError("Refiner requires the original rubric")
        reflections = (request.reflection, *request.additional_reflections)
        supported = any(f.status == "supported" and f.category == "grader_error"
                        for reflection in reflections for f in reflection.findings)
        if (not supported or any(r.human_review_required for r in reflections)) and candidate != request.current_rules:
            raise ValueError("Weak or unresolved evidence requires an unchanged candidate")
        known = {c.id for c in request.rubric.criteria}
        quotes = tuple(e.quote for r in reflections for f in r.findings for e in f.evidence
                       if e.source in ("student_answer", "reference_solution"))
        for rule in candidate.rules:
            if isinstance(rule, str):
                if rule not in request.current_rules.rules:
                    raise ValueError("New or revised rules require structured IDs and rationale")
                validate_rule_text(rule, known, quotes)
            else:
                if not set(rule.criterion_ids) <= known:
                    raise ValueError("Candidate rule references an unknown rubric criterion")
                validate_rule_text(rule.instruction, known, quotes)
                validate_rule_text(rule.rationale, known, quotes)
        return self


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def validate_candidate_rules(request: RefinerInput, candidate: AdaptationRules) -> AdaptationRules:
    """Reuse the Refiner's contextual safety boundary for injected agents."""
    return _ValidatedCandidate(request=request, candidate=candidate).candidate


class LLMRefiner(Refiner):
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def refine(self, request: RefinerInput) -> AdaptationRules:
        """Propose a validated candidate; never apply it or modify protected data."""
        original = request
        request = RefinerInput.model_validate(request)
        if request.rubric is None:
            raise ValueError("Refiner requires the original rubric")
        before = original.model_dump_json()
        validated_before = request.model_dump_json()
        try:
            raw = self.provider.generate(build_refiner_prompt(request))
        finally:
            # Frozen contracts prevent normal mutation; also detect deliberate
            # provider-side bypasses against the caller or validated copy.
            if original.model_dump_json() != before or request.model_dump_json() != validated_before:
                raise ValueError("Refiner input integrity violation: protected input was modified")
        data = json.loads(raw, parse_float=Decimal, parse_constant=_reject_constant,
                          object_pairs_hook=_unique_object)
        response = _CandidateResponse.model_validate(data)
        candidate = AdaptationRules.model_validate(response.model_dump())
        return validate_candidate_rules(request, candidate)
