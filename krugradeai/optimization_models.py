"""Frozen split-aware orchestration inputs, evaluation results and history."""
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from krugradeai.metrics import EvaluationMetrics, validate_categories

from krugradeai.models import (
    AdaptationRules, Contract, GraderInput, GraderOutput, GradingRecord,
    Question, ReflectorOutput, Rubric, StudentResponse, TAReference, Text, Score,
    validate_against_rubric,
)

NonnegativeInt = Annotated[int, Field(ge=0, strict=True)]
PositiveInt = Annotated[int, Field(ge=1, strict=True)]
StopReason = Literal["max_iterations", "no_change", "no_discrepancies", "agent_failure", "human_review", "patience"]
Stage = Literal["grader", "reflector", "refiner", "integrity"]


class RecordProvenance(Contract):
    source: Text
    item_id: Text
    # Optional for legacy/synthetic records; explicit on real-data imports.
    label_origin: Literal["human_derived", "ai_derived"] | None = None


class DevelopmentRecord(Contract):
    id: Text
    question: Question
    rubric: Rubric
    response: StudentResponse
    ta_output: GraderOutput | TAReference
    split: Literal["development"] = "development"
    provenance: RecordProvenance | None = None

    @model_validator(mode="after")
    def validate_development(self) -> Self:
        GraderInput(question=self.question, rubric=self.rubric, response=self.response)
        if isinstance(self.ta_output, GraderOutput):
            validate_against_rubric(self.ta_output, self.rubric)
        else:
            maxima = {c.id: c.max_score for c in self.rubric.criteria}
            labels = {c.criterion_id: c.score for c in self.ta_output.criterion_scores}
            if not labels.keys() <= maxima.keys():
                raise ValueError("Unknown TA criterion ID")
            if any(score > maxima[key] for key, score in labels.items()):
                raise ValueError("TA criterion label exceeds rubric maximum")
            known = sum(labels.values(), Decimal(0))
            remaining = sum((score for key, score in maxima.items() if key not in labels), Decimal(0))
            if not known <= self.ta_output.total_score <= known + remaining:
                raise ValueError("TA total is inconsistent with available labels and rubric")
        return self


class ValidationRecord(DevelopmentRecord):
    split: Literal["validation"] = "validation"


class ValidationConfig(Contract):
    primary_metric: Literal["mae", "normalized_mae"] = "mae"
    patience: PositiveInt = 2
    min_delta: Score = Decimal(0)
    qwk_categories: tuple[Score, ...] | None = None
    comparison_record_ids: Annotated[tuple[Text, ...], Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        validate_categories(self.qwk_categories)
        if self.comparison_record_ids is not None and len(set(self.comparison_record_ids)) != len(self.comparison_record_ids):
            raise ValueError("Comparison population IDs must be unique")
        return self


class RuleVersion(Contract):
    version: NonnegativeInt
    parent_version: NonnegativeInt | None
    created_iteration: NonnegativeInt
    rules: AdaptationRules


class ProcessingFailure(Contract):
    stage: Stage
    record_id: Text | None = None
    error_type: Text
    message: Text


class HumanReviewFlag(Contract):
    stage: Stage
    record_id: Text | None = None
    reasons: Annotated[tuple[Text, ...], Field(min_length=1)]


class RecordHistory(Contract):
    record_id: Text
    grading: GradingRecord | None = None
    discrepancy: bool | None = None
    reflection: ReflectorOutput | None = None
    human_review_required: bool = False


class ValidationEvaluation(Contract):
    version: NonnegativeInt
    metrics: EvaluationMetrics
    comparison_metrics: EvaluationMetrics
    comparison_record_ids: tuple[Text, ...]
    eligible: bool
    invalid_reason: Text | None = None
    records: tuple[RecordHistory, ...]
    failures: tuple[ProcessingFailure, ...] = ()
    human_reviews: tuple[HumanReviewFlag, ...] = ()

    @model_validator(mode="after")
    def validate_population(self) -> Self:
        ids = [r.record_id for r in self.records]
        if len(ids) != len(set(ids)) or len(set(self.comparison_record_ids)) != len(self.comparison_record_ids):
            raise ValueError("Evaluation population IDs must be unique")
        if not self.comparison_record_ids or not set(self.comparison_record_ids) <= set(ids):
            raise ValueError("Comparison population must be a nonempty subset of validation records")
        for metric, population in ((self.metrics, ids), (self.comparison_metrics, self.comparison_record_ids)):
            successes = sum(r.grading is not None for r in self.records if r.record_id in population)
            if metric.expected_count != len(population) or metric.sample_count != successes:
                raise ValueError("Metrics coverage contradicts the evaluation records")
        eligible = self.comparison_metrics.sample_count == self.comparison_metrics.expected_count
        if self.eligible != eligible or (self.invalid_reason is None) != eligible:
            raise ValueError("Selection eligibility requires complete fixed-population coverage")
        return self


class ValidationSummary(Contract):
    config: ValidationConfig
    primary_metric: Literal["mae", "normalized_mae"]
    baseline_metrics: EvaluationMetrics
    evaluations: Annotated[tuple[ValidationEvaluation, ...], Field(min_length=1)]
    best_version: NonnegativeInt | None
    best_rules: AdaptationRules | None
    stop_reason: StopReason | Literal["no_valid_evaluation"]
    non_improving_iterations: NonnegativeInt
    failure_count: NonnegativeInt
    human_review_count: NonnegativeInt

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.config.primary_metric != self.primary_metric:
            raise ValueError("Selection metric contradicts configuration")
        if self.evaluations[0].version != 0 or self.baseline_metrics != self.evaluations[0].metrics:
            raise ValueError("Baseline metrics must describe evaluated version zero")
        if [e.version for e in self.evaluations] != list(range(len(self.evaluations))):
            raise ValueError("Every version must be evaluated exactly once, baseline first")
        first = self.evaluations[0]
        if any(e.comparison_record_ids != first.comparison_record_ids or
               tuple(r.record_id for r in e.records) != tuple(r.record_id for r in first.records)
               for e in self.evaluations):
            raise ValueError("Candidate evaluations must use the same fixed population")
        valid = [e for e in self.evaluations if e.eligible]
        chosen = min(valid, key=lambda e: (getattr(e.comparison_metrics, self.primary_metric), e.version)) if valid else None
        if self.best_version != (chosen.version if chosen is not None else None):
            raise ValueError("Best version must be the earliest best eligible evaluated candidate")
        if (self.best_rules is None) != (chosen is None):
            raise ValueError("Best rules require a valid evaluation")
        if (self.stop_reason == "no_valid_evaluation") != (chosen is None):
            raise ValueError("No-valid-evaluation status must match selection availability")
        if self.failure_count != sum(len(e.failures) for e in self.evaluations) or self.human_review_count != sum(len(e.human_reviews) for e in self.evaluations):
            raise ValueError("Validation failure/review summary counts are inconsistent")
        return self


class IterationHistory(Contract):
    iteration: PositiveInt
    rules_version: NonnegativeInt
    records: tuple[RecordHistory, ...]
    proposed_candidate: AdaptationRules | None = None
    candidate_version: NonnegativeInt | None = None
    failures: tuple[ProcessingFailure, ...] = ()
    human_reviews: tuple[HumanReviewFlag, ...] = ()
    stop_reason: StopReason | None = None


class OptimizationPath(Contract):
    rubric_id: Text
    rule_versions: Annotated[tuple[RuleVersion, ...], Field(min_length=1)]
    history: tuple[IterationHistory, ...]
    final_version: NonnegativeInt
    stop_reason: StopReason
    validation: ValidationSummary | None = None

    @model_validator(mode="after")
    def validate_versions(self) -> Self:
        if [v.version for v in self.rule_versions] != list(range(len(self.rule_versions))):
            raise ValueError("Rule versions must form a contiguous sequence from zero")
        for version in self.rule_versions:
            expected_parent = version.version - 1 if version.version else None
            if version.parent_version != expected_parent:
                raise ValueError("Rule version parent is inconsistent")
        if self.final_version != self.rule_versions[-1].version:
            raise ValueError("Final version must identify the latest candidate, not a best selection")
        if [h.iteration for h in self.history] != list(range(1, len(self.history) + 1)):
            raise ValueError("Iteration numbers must be contiguous from one")
        if self.validation is not None:
            if [e.version for e in self.validation.evaluations] != [v.version for v in self.rule_versions]:
                raise ValueError("Every generated rule version requires evaluation")
            best = self.validation.best_version
            if best is not None and self.validation.best_rules != self.rule_versions[best].rules:
                raise ValueError("Selected rules must exactly match the selected version")
        return self


class OptimizationResult(Contract):
    split: Literal["development"] = "development"
    max_iterations: NonnegativeInt
    paths: tuple[OptimizationPath, ...]
