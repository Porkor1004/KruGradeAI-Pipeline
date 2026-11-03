"""Provider-backed discrepancy analysis with contextual validation."""
import json
from decimal import Decimal
from typing import Literal, Self

from pydantic import StrictBool, ValidationError, model_validator

from krugradeai.agents import Reflector
from krugradeai.models import (
    Contract, CriterionDifference, ReflectionFinding, ReflectorInput,
    ReflectorOutput, ScoreDifference, Text,
)
from krugradeai.providers import LLMProvider
from krugradeai.reflector_prompt import build_reflector_prompt
from krugradeai.validation_diagnostics import annotate


class _ReflectionResponse(ReflectorOutput):
    # Keep legacy observation-only records usable, but require full new responses.
    overall_score_difference: ScoreDifference
    direction: Literal["agreement", "overgrading", "undergrading"]
    criterion_differences: tuple[CriterionDifference, ...]
    findings: tuple[ReflectionFinding, ...]
    human_review_required: StrictBool
    review_reasons: tuple[Text, ...]


class _ValidatedReflection(Contract):
    request: ReflectorInput
    output: _ReflectionResponse

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        label = self.request.ta_output
        if label is None:
            raise ValueError("Reference labels are required for reflection")
        record = self.request.record
        grade_input = record.grader_input
        grade = record.grader_output
        rubric = {c.id: c for c in grade_input.rubric.criteria}
        ai = {c.criterion_id: c for c in grade.criterion_scores}
        ta = {c.criterion_id: c.score for c in label.criterion_scores}
        output = self.output
        if output.overall_score_difference != grade.total_score - label.total_score:
            raise ValueError("Overall difference must equal AI total minus reference total")
        reported = {c.criterion_id: c.difference for c in output.criterion_differences}
        expected = {key: ai[key].score - score for key, score in ta.items()}
        if reported != expected:
            raise ValueError("Criterion differences must match exactly the available TA labels")
        if output.overall_score_difference != 0 and not output.findings:
            raise ValueError("A discrepancy requires evidence-backed findings or uncertainty analysis")
        if not grade_input.response.text.strip() and not output.human_review_required:
            raise ValueError("Blank student work requires human review")
        for finding in output.findings:
            key = finding.criterion_id
            if key is not None and key not in rubric:
                raise ValueError("Unknown finding criterion ID")
            if finding.category == "no_discrepancy":
                difference = output.overall_score_difference if key is None else expected.get(key)
                if difference is None or difference != 0:
                    raise ValueError("No-discrepancy claims require score agreement at the cited scope")
            if finding.category == "grader_error" and finding.status == "supported":
                if key not in ta:
                    raise ValueError("Supported criterion error requires the corresponding TA criterion label")
                if not any(e.source == "student_answer" for e in finding.evidence):
                    raise ValueError("Supported grading errors require student-answer evidence")
                if not any(e.source == "rubric" and e.criterion_id == key for e in finding.evidence):
                    raise ValueError("Supported grading errors require evidence from the cited rubric criterion")
            if finding.category == "ta_rubric_inconsistency" and finding.status == "supported":
                if key not in ta or not any(e.source == "ta_label" and e.criterion_id == key for e in finding.evidence):
                    raise ValueError("Supported source inconsistency requires the cited criterion label evidence")
                if not any(e.source == "rubric" and e.criterion_id == key for e in finding.evidence):
                    raise ValueError("Supported source inconsistency requires the cited rubric evidence")
            for evidence in finding.evidence:
                criterion_id = evidence.criterion_id
                if criterion_id is not None and criterion_id not in rubric:
                    raise ValueError("Unknown evidence criterion ID")
                if key is not None and criterion_id is not None and criterion_id != key:
                    raise ValueError("Evidence criterion contradicts finding criterion")
                if evidence.source == "rubric":
                    if criterion_id is None:
                        raise ValueError("Rubric evidence requires a criterion ID")
                    source = rubric[criterion_id].description
                elif evidence.source == "grader_rationale":
                    if criterion_id is None:
                        raise ValueError("Grader rationale evidence requires a criterion ID")
                    source = ai[criterion_id].rationale
                elif evidence.source == "student_answer":
                    source = grade_input.response.text
                elif evidence.source == "reference_solution":
                    source = grade_input.question.reference_solution or ""
                else:
                    if criterion_id is not None and criterion_id not in ta:
                        raise ValueError("Cannot cite an unavailable TA criterion label")
                    source = str(label.total_score if criterion_id is None else ta[criterion_id])
                    if evidence.quote != source:
                        raise ValueError("TA evidence must quote the exact label value")
                if evidence.quote not in source:
                    raise ValueError("Evidence quote does not occur in the cited source")
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


class LLMReflector(Reflector):
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def reflect(self, request: ReflectorInput) -> ReflectorOutput:
        """Compare immutable inputs; propagate provider and validation errors."""
        request = ReflectorInput.model_validate(request)
        if request.ta_output is None:
            raise ValueError("Reference labels are required for reflection")
        raw = self.provider.generate(build_reflector_prompt(request))
        data = json.loads(raw, parse_float=Decimal, parse_constant=_reject_constant,
                          object_pairs_hook=_unique_object)
        try:
            output = _ReflectionResponse.model_validate(data)
        except ValidationError as error:
            raise annotate(error, _ReflectionResponse, "reflector_output_contract") from None
        try:
            return _ValidatedReflection(request=request, output=output).output
        except ValidationError as error:
            raise annotate(error, _ValidatedReflection, "reflector_context") from None
