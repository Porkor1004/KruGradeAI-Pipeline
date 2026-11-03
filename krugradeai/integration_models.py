"""Concise live integration telemetry; no prompts or answers in reports."""
from typing import Literal
from pydantic import Field, model_validator
from krugradeai.metrics import EvaluationMetrics
from krugradeai.models import Contract, Text
from krugradeai.optimization_models import ProcessingFailure, HumanReviewFlag


class AgentInvocation(Contract):
    index: int
    agent: Literal["grader", "reflector", "refiner"]
    phase: Text
    status: Literal["success", "failure", "human_review"]
    record_id: Text | None = None
    error_type: Text | None = None
    error_stage: Text | None = None
    human_review_required: bool = False
    execution_kind: Literal["actual_optimization", "diagnostic_probe"] = "actual_optimization"


class StageExecution(Contract):
    stage: Text
    execution_kind: Literal["actual_optimization", "diagnostic_probe", "skipped"]
    status: Literal["success", "failure", "human_review", "skipped"]


class CandidateProgress(Contract):
    proposal_attempted: bool = False
    proposed: bool = False
    accepted_as_valid: bool = False
    evaluated_on_validation: bool = False
    validation_eligible: bool | None = None
    selected_as_best: bool = False
    no_change: bool = False

    @model_validator(mode="after")
    def validate_progress(self):
        if self.accepted_as_valid and not self.proposed:
            raise ValueError("Accepted candidates require a proposal")
        if self.evaluated_on_validation and not self.accepted_as_valid:
            raise ValueError("Candidate evaluation requires accepted rules")
        if self.selected_as_best and (not self.evaluated_on_validation or not self.validation_eligible):
            raise ValueError("Selection requires an eligible candidate evaluation")
        if self.no_change and (self.proposed or self.accepted_as_valid):
            raise ValueError("No-change output is not a new candidate")
        return self


class CandidateMetrics(Contract):
    version: int
    metrics: EvaluationMetrics
    eligible: bool


class IntegrationReport(Contract):
    report_schema_version: Literal["1"] = "1"
    experiment_id: Text
    data_kind: Literal["synthetic"] = "synthetic"
    max_refinement_iterations: Literal[1] = 1
    status: Literal["passed", "incomplete", "failed"]
    invocations: tuple[AgentInvocation, ...]
    missing_stages: tuple[Text, ...]
    baseline_validation_metrics: EvaluationMetrics | None = None
    candidate_validation_metrics: tuple[CandidateMetrics, ...] = ()
    selected_version: int | None = None
    optimization_stop_reason: Text | None = None
    heldout_baseline_metrics: EvaluationMetrics | None = None
    heldout_selected_metrics: EvaluationMetrics | None = None
    heldout_status: Text | None = None
    failures: tuple[ProcessingFailure, ...] = ()
    human_reviews: tuple[HumanReviewFlag, ...] = ()
    setup_error_type: Text | None = None
    setup_error_stage: Text | None = None
    notes: tuple[Text, ...]
    mode: Literal["contract_coverage", "optimization_path"] = "contract_coverage"
    stage_execution: tuple[StageExecution, ...] = ()
    candidate_progress: CandidateProgress = Field(default_factory=CandidateProgress)
