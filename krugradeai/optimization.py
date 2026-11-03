"""Development optimization with optional, isolated validation selection."""
from collections.abc import Iterable
from krugradeai.execution import _invoke, _failure

from krugradeai.agents import Grader, Reflector, Refiner
from krugradeai.grader import HumanReviewRequired
from krugradeai.models import (
    AdaptationRule, AdaptationRules, GraderInput, GraderOutput,
    GradingRecord, ReflectorInput, ReflectorOutput, RefinerInput,
)
from krugradeai.optimization_models import (
    DevelopmentRecord, HumanReviewFlag, IterationHistory, OptimizationPath,
    OptimizationResult, RecordHistory, RuleVersion,
    ValidationConfig, ValidationRecord,
)
from krugradeai.refiner import validate_candidate_rules
from krugradeai.validation import ValidationTracker, prepare_validation, _validate_unique_identities


def _has_discrepancy(record: GradingRecord, example: DevelopmentRecord) -> bool:
    output = record.grader_output
    if output.total_score != example.ta_output.total_score:
        return True
    ai = {c.criterion_id: c.score for c in output.criterion_scores}
    return any(ai[c.criterion_id] != c.score for c in example.ta_output.criterion_scores)


def _validate_reflection(output: ReflectorOutput, request: ReflectorInput) -> None:
    """Check optional numerical assertions from injected implementations.

    Concrete LLMReflector also validates quotes/evidence against source text.
    Legacy observation-only outputs remain usable, but cannot authorize changes.
    """
    if output.overall_score_difference is not None:
        if output.overall_score_difference != request.record.grader_output.total_score - request.ta_output.total_score:
            raise ValueError("Reflection contradicts verified total labels")
    if output.overall_score_difference is not None or output.criterion_differences:
        ai = {c.criterion_id: c.score for c in request.record.grader_output.criterion_scores}
        expected = {c.criterion_id: ai[c.criterion_id] - c.score for c in request.ta_output.criterion_scores}
        if {c.criterion_id: c.difference for c in output.criterion_differences} != expected:
            raise ValueError("Reflection criterion differences contradict available TA labels")
    RefinerInput(current_rules=request.record.grader_input.adaptation_rules,
                 reflection=output, rubric=request.record.grader_input.rubric)


class OptimizationLoop:
    """Inject the existing three agent interfaces; no provider/runtime knowledge."""

    def __init__(self, *, grader: Grader, reflector: Reflector, refiner: Refiner,
                 max_iterations: int = 3, validation_config: ValidationConfig | None = None,
                 review_policy: str = "stop_path") -> None:
        if type(max_iterations) is not int or max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        self.grader, self.reflector, self.refiner = grader, reflector, refiner
        self.max_iterations = max_iterations
        if review_policy not in ("stop_path", "exclude_reviewed_records"):
            raise ValueError("Unknown review policy")
        if review_policy == "exclude_reviewed_records" and max_iterations != 1:
            raise ValueError("Record exclusion is supported only for a single iteration")
        self.review_policy = review_policy
        self.validation_config = ValidationConfig.model_validate(validation_config if validation_config is not None else ValidationConfig())

    def run(self, development_records: Iterable[DevelopmentRecord],
            initial_rules: AdaptationRules | None = None, *,
            validation_records: Iterable[ValidationRecord] | None = None) -> OptimizationResult:
        records = tuple(DevelopmentRecord.model_validate(r) for r in development_records)
        if not records:
            raise ValueError("At least one development record is required")
        if len({r.id for r in records}) != len(records):
            raise ValueError("Development record IDs must be unique")
        _validate_unique_identities(records)
        rules = AdaptationRules.model_validate(initial_rules if initial_rules is not None else AdaptationRules())
        groups: dict[str, list[DevelopmentRecord]] = {}
        for record in records:
            group = groups.setdefault(record.rubric.id, [])
            if group and group[0].rubric != record.rubric:
                raise ValueError("A rubric ID cannot identify different rubric definitions")
            group.append(record)
            known = {c.id for c in record.rubric.criteria}
            if any(isinstance(r, AdaptationRule) and not set(r.criterion_ids) <= known for r in rules.rules):
                raise ValueError("Initial rules must be compatible with every rubric path")
        validation_groups = prepare_validation(records, validation_records, self.validation_config) if validation_records is not None else {}
        return OptimizationResult(max_iterations=self.max_iterations,
                                  paths=tuple(self._run_path(group, rules, validation_groups.get(key)) for key, group in groups.items()))

    def _run_path(self, examples: list[DevelopmentRecord], initial: AdaptationRules,
                  validation_records: tuple[ValidationRecord, ...] | None = None) -> OptimizationPath:
        versions = [RuleVersion(version=0, parent_version=None, created_iteration=0, rules=initial)]
        validation = ValidationTracker(self.grader, validation_records, self.validation_config) if validation_records is not None else None
        if validation is not None:
            validation.observe(0, initial)
        history = []
        stop_reason = "max_iterations"
        for iteration in range(1, self.max_iterations + 1):
            active = versions[-1]
            outcomes, failures, reviews, reflections = [], [], [], []
            for example in examples:
                grading, reflection, discrepancy = None, None, None
                record_review = False
                request = GraderInput(question=example.question, rubric=example.rubric,
                                      response=example.response, adaptation_rules=active.rules)
                try:
                    output = GraderOutput.model_validate(_invoke(self.grader.grade, request))
                    grading = GradingRecord(grader_input=request, grader_output=output)
                    discrepancy = _has_discrepancy(grading, example)
                except HumanReviewRequired as error:
                    record_review = True
                    reviews.append(HumanReviewFlag(stage="grader", record_id=example.id, reasons=(error.reason,)))
                except Exception as error:
                    failures.append(_failure("grader", example.id, error))
                if grading is not None and discrepancy:
                    reflection_request = ReflectorInput(record=grading, ta_output=example.ta_output)
                    try:
                        output = ReflectorOutput.model_validate(_invoke(self.reflector.reflect, reflection_request))
                        _validate_reflection(output, reflection_request)
                        reflection = output
                        if not output.human_review_required or self.review_policy == "stop_path":
                            reflections.append(output)
                        if output.human_review_required:
                            record_review = True
                            reviews.append(HumanReviewFlag(stage="reflector", record_id=example.id, reasons=output.review_reasons))
                    except HumanReviewRequired as error:
                        record_review = True
                        reviews.append(HumanReviewFlag(stage="reflector", record_id=example.id, reasons=(error.reason,)))
                    except Exception as error:
                        failures.append(_failure("reflector", example.id, error))
                outcomes.append(RecordHistory(record_id=example.id, grading=grading, discrepancy=discrepancy,
                                              reflection=reflection, human_review_required=record_review))
            proposed, candidate_version, terminal = None, None, None
            if failures:
                terminal = "agent_failure"
            elif reviews and (self.review_policy == "stop_path" or not any(
                    f.category == "grader_error" and f.status == "supported"
                    for r in reflections for f in r.findings)):
                terminal = "human_review"
            elif not reflections:
                terminal = "no_discrepancies"
            else:
                refinement = RefinerInput(current_rules=active.rules, reflection=reflections[0],
                                          additional_reflections=tuple(reflections[1:]), rubric=examples[0].rubric)
                try:
                    candidate = AdaptationRules.model_validate(_invoke(self.refiner.refine, refinement))
                    proposed = validate_candidate_rules(refinement, candidate)
                    if proposed == active.rules:
                        terminal = "no_change"
                    else:
                        candidate_version = active.version + 1
                        versions.append(RuleVersion(version=candidate_version, parent_version=active.version,
                                                    created_iteration=iteration, rules=proposed))
                except HumanReviewRequired as error:
                    reviews.append(HumanReviewFlag(stage="refiner", reasons=(error.reason,)))
                    terminal = "human_review"
                except Exception as error:
                    failures.append(_failure("refiner", None, error))
                    terminal = "agent_failure"
            if validation is not None and candidate_version is not None:
                if validation.observe(candidate_version, versions[-1].rules):
                    terminal = "patience"
            if terminal is None and iteration == self.max_iterations:
                terminal = "max_iterations"
            history.append(IterationHistory(iteration=iteration, rules_version=active.version,
                                            records=tuple(outcomes), proposed_candidate=proposed,
                                            candidate_version=candidate_version, failures=tuple(failures),
                                            human_reviews=tuple(reviews), stop_reason=terminal))
            if terminal is not None:
                stop_reason = terminal
                break
        return OptimizationPath(rubric_id=examples[0].rubric.id, rule_versions=tuple(versions),
                                history=tuple(history), final_version=versions[-1].version, stop_reason=stop_reason,
                                validation=validation.summary(stop_reason) if validation is not None else None)
