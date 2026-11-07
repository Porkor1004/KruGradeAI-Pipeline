"""Validation grading/selection only; validation evidence never reaches optimizers."""
from collections.abc import Iterable
from decimal import Context, localcontext

from krugradeai.agents import Grader
from krugradeai.execution import _failure, _invoke
from krugradeai.grader import AIAdjudicatedGrade, HumanReviewRequired
from krugradeai.metrics import ScorePair, calculate_metrics
from krugradeai.model_routing import ModelRoutingError
from krugradeai.models import AdaptationRules, GraderInput, GraderOutput, GradingRecord
from krugradeai.optimization_models import (
    DevelopmentRecord, HumanReviewFlag, RecordHistory, ValidationConfig,
    ValidationEvaluation, ValidationRecord, ValidationSummary,
)


def _identities(records):
    return (
        {r.id for r in records},
        {(r.question.id, r.response.id) for r in records},
        {(r.provenance.source, r.provenance.item_id) for r in records if r.provenance is not None},
    )


def _validate_unique_identities(records):
    """Reject known duplicates; source/item IDs do not imply unknown attempt links."""
    identities = _identities(records)
    if len(identities[0]) != len(records) or len(identities[1]) != len(records):
        raise ValueError("Records must have unique record and response identities")
    if len(identities[2]) != sum(r.provenance is not None for r in records):
        raise ValueError("Records duplicate source provenance")


def prepare_validation(development: tuple[DevelopmentRecord, ...],
                       validation: Iterable[ValidationRecord], config: ValidationConfig) -> dict[str, tuple[ValidationRecord, ...]]:
    _validate_unique_identities(development)
    records = tuple(ValidationRecord.model_validate(r) for r in validation)
    for r in records:
        p = r.provenance
        if p and p.source_dataset is not None and not (p.source_dataset == 'untitled' and p.score_origin == 'original_ta' and p.ocr_review_origin == 'human' and p.answer_origin == 'original_student' and p.rubric_origin == 'teacher_supplied' and p.benchmark_eligible is True):
            raise ValueError('Explicit Validation provenance requires original TA labels and human-reviewed Untitled OCR')
    if not records:
        raise ValueError("Validation collection cannot be empty")
    identities = _identities(records)
    if len(identities[0]) != len(records) or len(identities[1]) != len(records):
        raise ValueError("Validation records must have unique record and response identities")
    sourced = [r for r in records if r.provenance is not None]
    if len(identities[2]) != len(sourced):
        raise ValueError("Validation records duplicate source provenance")
    if any(left & right for left, right in zip(_identities(development), identities)):
        raise ValueError("Development and validation identities/provenance overlap")
    dev_groups = {r.provenance.split_group for r in development if r.provenance and r.provenance.split_group}
    val_groups = {r.provenance.split_group for r in records if r.provenance and r.provenance.split_group}
    if dev_groups & val_groups:
        raise ValueError("Development and validation split groups overlap")
    rubrics = {r.rubric.id: r.rubric for r in development}
    groups = {key: [] for key in rubrics}
    for record in records:
        if record.rubric.id not in rubrics or record.rubric != rubrics[record.rubric.id]:
            raise ValueError("Validation rubrics must match their development paths exactly")
        groups[record.rubric.id].append(record)
    if config.comparison_record_ids is not None:
        if not set(config.comparison_record_ids) <= identities[0]:
            raise ValueError("Configured comparison population contains unknown validation IDs")
    for group in groups.values():
        if not group:
            raise ValueError("Every development rubric path needs validation records")
        if config.comparison_record_ids is not None and not any(r.id in config.comparison_record_ids for r in group):
            raise ValueError("Every rubric path needs a nonempty fixed comparison population")
    return {key: tuple(group) for key, group in groups.items()}


def evaluate_rules(grader: Grader, records: tuple[ValidationRecord, ...], rules: AdaptationRules,
                   version: int, config: ValidationConfig, *, fail_fast=False) -> ValidationEvaluation:
    records = tuple(ValidationRecord.model_validate(r) for r in records)
    return evaluate_labeled_records(grader, records, rules, version, config, fail_fast=fail_fast)


def evaluate_labeled_records(grader: Grader, records: tuple[DevelopmentRecord, ...], rules: AdaptationRules,
                             version: int, config: ValidationConfig, *, fail_fast=False) -> ValidationEvaluation:
    """Shared score-only evaluation kernel; partition routing belongs to callers.

    Preserve each record's split type. The validation wrapper and experiment
    runner enforce validation and held-out boundaries respectively.
    """
    if any(not isinstance(r, DevelopmentRecord) for r in records):
        raise ValueError("Evaluation requires typed labeled records")
    records = tuple(type(r).model_validate(r) for r in records)
    config = ValidationConfig.model_validate(config)
    rules = AdaptationRules.model_validate(rules)
    if not records or len({r.id for r in records}) != len(records):
        raise ValueError("Evaluation needs a nonempty collection of unique validation records")
    selected = tuple(r.id for r in records if config.comparison_record_ids is None or r.id in config.comparison_record_ids)
    if not selected:
        raise ValueError("A nonempty comparison population is required")
    outcomes, failures, reviews = [], [], []
    pairs = {}
    for record in records:
        if fail_fast and (failures or reviews):
            outcomes.append(RecordHistory(record_id=record.id))
            continue
        request = GraderInput(question=record.question, rubric=record.rubric,
                              response=record.response, adaptation_rules=rules)
        grading, review, assisted = None, False, False
        try:
            prediction = _invoke(grader.grade, request)
            assisted = isinstance(prediction, AIAdjudicatedGrade)
            output = GraderOutput.model_validate(prediction)
            grading = GradingRecord(grader_input=request, grader_output=output)
            pairs[record.id] = ScorePair(predicted=output.total_score, reference=record.ta_output.total_score,
                                         maximum=record.rubric.total_score)
        except HumanReviewRequired as error:
            review = True
            reviews.append(HumanReviewFlag(stage="grader", record_id=record.id, reasons=(error.reason,)))
        except ModelRoutingError:
            raise
        except Exception as error:
            failures.append(_failure("grader", record.id, error))
        outcomes.append(RecordHistory(record_id=record.id, grading=grading, ai_adjudicated=assisted, human_review_required=review))
    metrics = calculate_metrics(pairs.values(), expected_count=len(records), qwk_categories=config.qwk_categories)
    comparison = calculate_metrics((pairs[key] for key in selected if key in pairs),
                                   expected_count=len(selected), qwk_categories=config.qwk_categories)
    eligible = comparison.sample_count == comparison.expected_count
    return ValidationEvaluation(version=version, metrics=metrics, comparison_metrics=comparison,
                                comparison_record_ids=selected, eligible=eligible,
                                invalid_reason=None if eligible else "Incomplete coverage of the fixed comparison population",
                                records=tuple(outcomes), failures=tuple(failures), human_reviews=tuple(reviews))


class ValidationTracker:
    """Deterministic bookkeeping, not an LLM agent or an optimization loop."""

    def __init__(self, grader: Grader, records: tuple[ValidationRecord, ...], config: ValidationConfig, *, fail_fast=False, observer=None):
        self.fail_fast, self.observer = fail_fast, observer
        self.grader, self.records, self.config = grader, records, config
        self.evaluations = []
        self.best_version = None
        self.best_rules = None
        self.best_value = None
        self.non_improving = 0

    def observe(self, version: int, rules: AdaptationRules) -> bool:
        evaluation = evaluate_rules(self.grader, self.records, rules, version, self.config, fail_fast=self.fail_fast)
        baseline = not self.evaluations
        self.evaluations.append(evaluation)
        previous = self.best_value
        value = getattr(evaluation.comparison_metrics, self.config.primary_metric)
        if evaluation.eligible and value is not None and (previous is None or value < previous):
            self.best_value, self.best_version = value, version
            self.best_rules = AdaptationRules.model_validate(rules)
        if not baseline and evaluation.eligible and value is not None:
            with localcontext(Context(prec=50)):
                significant = evaluation.eligible and value is not None and (previous is None or previous - value > self.config.min_delta)
            self.non_improving = 0 if significant else self.non_improving + 1
        if self.observer:
            self.observer(self, evaluation)
        if self.fail_fast and (evaluation.failures or evaluation.human_reviews or not evaluation.eligible):
            raise RuntimeError("Validation stopped at the first failure or unresolved review; inspect validation-history.jsonl")
        return self.non_improving >= self.config.patience

    def summary(self, operational_reason: str) -> ValidationSummary:
        return ValidationSummary(config=self.config, primary_metric=self.config.primary_metric,
                                 baseline_metrics=self.evaluations[0].metrics,
                                 evaluations=tuple(self.evaluations), best_version=self.best_version,
                                 best_rules=self.best_rules,
                                 stop_reason=operational_reason if self.best_version is not None else "no_valid_evaluation",
                                 non_improving_iterations=self.non_improving,
                                 failure_count=sum(len(e.failures) for e in self.evaluations),
                                 human_review_count=sum(len(e.human_reviews) for e in self.evaluations))
