"""Immutable, dataset-independent contracts for KruGradeAI Phase 1."""
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Score = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
PositiveScore = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
ScoreDifference = Annotated[Decimal, Field(allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class Question(Contract):
    id: Text
    text: Text
    reference_solution: Text | None = None


class RubricCriterion(Contract):
    id: Text
    description: Text
    max_score: PositiveScore


class Rubric(Contract):
    id: Text
    criteria: Annotated[tuple[RubricCriterion, ...], Field(min_length=1)]
    total_score: PositiveScore

    @model_validator(mode="after")
    def validate_criteria(self) -> Self:
        ids = [criterion.id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("Rubric criterion IDs must be unique")
        if sum((c.max_score for c in self.criteria), Decimal(0)) != self.total_score:
            raise ValueError("Rubric total must equal the sum of criterion maxima")
        return self


class StudentResponse(Contract):
    id: Text
    question_id: Text
    text: str  # Blank submissions are legitimate student responses.


class AdaptationRule(Contract):
    """Identified interpretation instruction; never a scoring allocation."""

    id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")]
    instruction: Text
    criterion_ids: tuple[Text, ...] = ()
    rationale: Text

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        if len(self.criterion_ids) != len(set(self.criterion_ids)):
            raise ValueError("Rule criterion references must be unique")
        return self


class AdaptationRules(Contract):
    # Legacy strings remain usable; newly generated instructions have IDs.
    rules: tuple[Text | AdaptationRule, ...] = ()

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        ids = [r.id for r in self.rules if isinstance(r, AdaptationRule)]
        if len(ids) != len(set(ids)):
            raise ValueError("Adaptation rule IDs must be unique")
        return self


class GraderInput(Contract):
    question: Question
    rubric: Rubric
    response: StudentResponse
    adaptation_rules: AdaptationRules = Field(default_factory=AdaptationRules)

    @model_validator(mode="after")
    def validate_question(self) -> Self:
        if self.response.question_id != self.question.id:
            raise ValueError("Response must belong to the supplied question")
        return self


class CriterionScore(Contract):
    criterion_id: Text
    score: Score
    rationale: Text


class GraderOutput(Contract):
    criterion_scores: Annotated[tuple[CriterionScore, ...], Field(min_length=1)]
    total_score: Score
    overall_feedback: Text | None = None

    @model_validator(mode="after")
    def validate_scores(self) -> Self:
        ids = [score.criterion_id for score in self.criterion_scores]
        if len(ids) != len(set(ids)):
            raise ValueError("Criterion scores must be unique")
        if sum((c.score for c in self.criterion_scores), Decimal(0)) != self.total_score:
            raise ValueError("Total must equal the sum of criterion scores")
        return self


def validate_against_rubric(output: GraderOutput, rubric: Rubric) -> None:
    """Validate contextual boundaries unavailable to a standalone output."""
    maxima = {criterion.id: criterion.max_score for criterion in rubric.criteria}
    if {score.criterion_id for score in output.criterion_scores} != set(maxima):
        raise ValueError("Scores must cover exactly the rubric criteria")
    if any(score.score > maxima[score.criterion_id] for score in output.criterion_scores):
        raise ValueError("Criterion score exceeds rubric maximum")
    if output.total_score > rubric.total_score:
        raise ValueError("Total exceeds rubric maximum")


class GradingRecord(Contract):
    grader_input: GraderInput
    grader_output: GraderOutput

    @model_validator(mode="after")
    def validate_grading(self) -> Self:
        validate_against_rubric(self.grader_output, self.grader_input.rubric)
        return self


class CriterionLabelProvenance(Contract):
    """Source evidence, not certification of independent grading correctness."""

    source: Text
    label_origin: Literal["ai_derived", "human_derived"]
    operational_role: Literal["pseudo_ground_truth", "human_reference"]
    original_criterion_id: Text
    source_response_id: Text
    source_question_id: Text
    source_locator: Text
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    extraction_evidence: Annotated[tuple[Text, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_role(self) -> Self:
        expected = "pseudo_ground_truth" if self.label_origin == "ai_derived" else "human_reference"
        if self.operational_role != expected:
            raise ValueError("Criterion label role must preserve its human/AI origin")
        return self


class TACriterionScore(Contract):
    criterion_id: Text
    score: Score
    provenance: CriterionLabelProvenance | None = None


class TAReference(Contract):
    """Reference scores; an empty criterion tuple means total-only labels.

    The legacy name does not certify label trust: real imports declare human-
    or AI-derived supervision in RecordProvenance.label_origin. Source-provided
    criterion labels additionally carry their origin and operational role.
    """

    total_score: Score
    criterion_scores: tuple[TACriterionScore, ...] = ()

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        ids = [c.criterion_id for c in self.criterion_scores]
        if len(ids) != len(set(ids)):
            raise ValueError("TA criterion labels must be unique")
        if sum((c.score for c in self.criterion_scores), Decimal(0)) > self.total_score:
            raise ValueError("Known TA criterion scores exceed the TA total")
        return self


class ReflectorInput(Contract):
    reference_total_score_origin: Literal["original_ta", "ai_generated", "missing", "unknown"] = "unknown"
    record: GradingRecord
    ta_output: GraderOutput | TAReference | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if isinstance(self.ta_output, GraderOutput):
            validate_against_rubric(self.ta_output, self.record.grader_input.rubric)
        elif self.ta_output is not None:
            rubric = self.record.grader_input.rubric
            maxima = {c.id: c.max_score for c in rubric.criteria}
            labels = {c.criterion_id: c.score for c in self.ta_output.criterion_scores}
            if not labels.keys() <= maxima.keys():
                raise ValueError("Unknown TA criterion ID")
            if any(score > maxima[key] for key, score in labels.items()):
                raise ValueError("TA criterion score exceeds rubric maximum")
            remaining = sum((maximum for key, maximum in maxima.items() if key not in labels), Decimal(0))
            known = sum(labels.values(), Decimal(0))
            if not known <= self.ta_output.total_score <= known + remaining:
                raise ValueError("TA total is inconsistent with known labels and rubric bounds")
        return self


class CriterionDifference(Contract):
    criterion_id: Text
    difference: ScoreDifference


class ReflectionEvidence(Contract):
    source: Literal["student_answer", "rubric", "grader_rationale", "reference_solution", "ta_label"]
    quote: Text
    criterion_id: Text | None = None


class ReflectionFinding(Contract):
    criterion_id: Text | None = None
    status: Literal["supported", "hypothesis"]
    category: Literal["grader_error", "ta_rubric_inconsistency", "insufficient_evidence", "no_discrepancy"]
    claim: Text
    evidence: Annotated[tuple[ReflectionEvidence, ...], Field(min_length=1)]
    adaptation_suggestion: Text | None = None

    @model_validator(mode="after")
    def validate_finding(self) -> Self:
        if self.category == "insufficient_evidence" and self.status != "hypothesis":
            raise ValueError("Insufficient evidence cannot establish a definitive cause")
        if self.category == "grader_error" and self.status == "supported" and self.adaptation_suggestion is None:
            raise ValueError("Supported grading errors require an adaptation improvement suggestion")
        return self


class ReflectorOutput(Contract):
    observations: tuple[Text, ...]
    # Optional defaults preserve the Step 1 observation-only contract.
    overall_score_difference: ScoreDifference | None = None
    direction: Literal["agreement", "overgrading", "undergrading"] | None = None
    criterion_differences: tuple[CriterionDifference, ...] = ()
    findings: tuple[ReflectionFinding, ...] = ()
    human_review_required: Annotated[bool, Field(strict=True)] = False
    review_reasons: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def validate_reflection(self) -> Self:
        if (self.overall_score_difference is None) != (self.direction is None):
            raise ValueError("Difference and direction must be supplied together")
        if self.overall_score_difference is not None:
            expected = ("overgrading" if self.overall_score_difference > 0 else
                        "undergrading" if self.overall_score_difference < 0 else "agreement")
            if self.direction != expected:
                raise ValueError("Direction contradicts the AI-minus-TA difference")
        ids = [c.criterion_id for c in self.criterion_differences]
        if len(ids) != len(set(ids)):
            raise ValueError("Criterion differences must be unique")
        if self.human_review_required != bool(self.review_reasons):
            raise ValueError("Human review requires reasons; reasons require human review")
        if any(f.status == "hypothesis" or f.category in ("ta_rubric_inconsistency", "insufficient_evidence") for f in self.findings):
            if not self.human_review_required:
                raise ValueError("Uncertain findings and suspected label inconsistencies require human review")
        return self


class RefinerInput(Contract):
    current_rules: AdaptationRules
    reflection: ReflectorOutput
    rubric: Rubric | None = None
    additional_reflections: tuple[ReflectorOutput, ...] = ()

    @model_validator(mode="after")
    def validate_criterion_context(self) -> Self:
        if self.rubric is None:
            return self
        known = {c.id for c in self.rubric.criteria}
        for rule in self.current_rules.rules:
            if isinstance(rule, AdaptationRule) and not set(rule.criterion_ids) <= known:
                raise ValueError("Current rule references an unknown rubric criterion")
        for reflection in (self.reflection, *self.additional_reflections):
            references = [c.criterion_id for c in reflection.criterion_differences]
            for finding in reflection.findings:
                references.append(finding.criterion_id)
                references.extend(e.criterion_id for e in finding.evidence)
            if any(key is not None and key not in known for key in references):
                raise ValueError("Reflection references an unknown rubric criterion")
        return self


# The Refiner has no output wrapper or rubric field to modify.
RefinerOutput = AdaptationRules
