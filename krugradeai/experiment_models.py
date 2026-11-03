"""Phase 1 experiment and export contracts; held-out data has its own split."""
from typing import Annotated, Literal, Self
from pydantic import Field, model_validator
from krugradeai.metrics import EvaluationMetrics
from krugradeai.models import AdaptationRules, Contract, Text
from krugradeai.optimization_models import (
    DevelopmentRecord, ValidationRecord, OptimizationResult, ValidationConfig,
    ValidationEvaluation, NonnegativeInt,
)


class HeldoutRecord(DevelopmentRecord):
    split: Literal["test"] = "test"


class DatasetProvenance(Contract):
    kind: Literal["synthetic", "real"]
    description: Text
    sources: Annotated[tuple[Text, ...], Field(min_length=1)]


class ExperimentDataset(Contract):
    provenance: DatasetProvenance
    development: Annotated[tuple[DevelopmentRecord, ...], Field(min_length=1)]
    validation: Annotated[tuple[ValidationRecord, ...], Field(min_length=1)]
    heldout: Annotated[tuple[HeldoutRecord, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_partition_integrity(self) -> Self:
        seen_ids, seen_responses, seen_provenance = set(), set(), set()
        rubrics = {}
        partitions = (self.development, self.validation, self.heldout)
        for partition in partitions:
            for record in partition:
                if record.provenance is None:
                    raise ValueError("Experiments require source/item provenance for every record")
                identity = record.question.id, record.response.id
                provenance = record.provenance.source, record.provenance.item_id
                if record.id in seen_ids or identity in seen_responses or provenance in seen_provenance:
                    raise ValueError("Record identities or provenance overlap within/across partitions")
                if record.provenance.source not in self.provenance.sources:
                    raise ValueError("Record source is missing from dataset provenance")
                seen_ids.add(record.id)
                seen_responses.add(identity)
                seen_provenance.add(provenance)
                if record.rubric.id in rubrics and rubrics[record.rubric.id] != record.rubric:
                    raise ValueError("Rubric identity refers to conflicting immutable definitions")
                rubrics[record.rubric.id] = record.rubric
        scopes = [{r.rubric.id for r in partition} for partition in partitions]
        if not scopes[0] == scopes[1] == scopes[2]:
            raise ValueError("All three partitions must cover the same rubric scopes")
        return self


class ExperimentConfig(Contract):
    max_iterations: NonnegativeInt
    validation: ValidationConfig
    initial_rules: AdaptationRules
    runtime: Literal["mock", "codex_cli", "custom"]


class RubricScope(Contract):
    rubric_id: Text
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class HeldoutPathResult(Contract):
    rubric_id: Text
    status: Literal["completed", "incomplete", "no_valid_best"]
    selected_version: NonnegativeInt | None
    baseline: ValidationEvaluation | None = None
    selected: ValidationEvaluation | None = None

    @model_validator(mode="after")
    def validate_final_phase(self) -> Self:
        if self.status == "no_valid_best":
            if any(v is not None for v in (self.selected_version, self.baseline, self.selected)):
                raise ValueError("No-best paths must not fabricate held-out evaluations")
        else:
            if self.selected_version is None or self.baseline is None or self.selected is None:
                raise ValueError("Held-out comparisons require the baseline and selected version")
            if self.baseline.version != 0 or self.selected.version != self.selected_version:
                raise ValueError("Held-out version identity mismatch")
            if self.baseline.comparison_record_ids != self.selected.comparison_record_ids:
                raise ValueError("Both final passes must use the same test population")
            complete = self.baseline.eligible and self.selected.eligible
            if (self.status == "completed") != complete:
                raise ValueError("Final status must disclose incomplete coverage")
        return self


class ExperimentResult(Contract):
    experiment_id: Text
    provenance: DatasetProvenance
    config: ExperimentConfig
    rubric_scopes: tuple[RubricScope, ...]
    optimization: OptimizationResult
    heldout: tuple[HeldoutPathResult, ...]
    status: Literal["completed", "incomplete"]

    @model_validator(mode="after")
    def validate_locked_selection(self) -> Self:
        paths = {p.rubric_id: p for p in self.optimization.paths}
        if not paths:
            raise ValueError("A completed experiment requires optimization paths")
        if len(self.heldout) != len(paths) or {h.rubric_id for h in self.heldout} != set(paths):
            raise ValueError("Final evaluations must cover every optimization path exactly once")
        if len(self.rubric_scopes) != len(paths) or {s.rubric_id for s in self.rubric_scopes} != set(paths):
            raise ValueError("Rubric scopes must match optimization paths")
        for heldout in self.heldout:
            path = paths[heldout.rubric_id]
            validation = path.validation
            expected = validation.best_version if validation is not None else None
            if heldout.selected_version != expected:
                raise ValueError("Held-out results cannot change validation selection")
            if heldout.baseline is not None:
                for evaluation, rules in ((heldout.baseline, path.rule_versions[0].rules),
                                          (heldout.selected, validation.best_rules)):
                    if any(record.grading is not None and
                           (record.grading.grader_input.adaptation_rules != rules or
                            record.grading.grader_input.rubric.id != heldout.rubric_id)
                           for record in evaluation.records):
                        raise ValueError("Final grades must use the exact locked rules and rubric scope")
        if (self.status == "completed") != all(h.status == "completed" for h in self.heldout):
            raise ValueError("Experiment status must disclose incomplete final evaluation")
        return self


class GuidelinesArtifact(Contract):
    artifact_schema_version: Literal["1"] = "1"
    experiment_id: Text
    guideline_version: NonnegativeInt
    rubric_scope: RubricScope
    rules: AdaptationRules
    validation_metrics: EvaluationMetrics
    validation_comparison_metrics: EvaluationMetrics
    heldout_baseline_metrics: EvaluationMetrics
    heldout_selected_metrics: EvaluationMetrics
    heldout_status: Literal["completed", "incomplete"]
    configuration: ExperimentConfig
    provenance: DatasetProvenance
