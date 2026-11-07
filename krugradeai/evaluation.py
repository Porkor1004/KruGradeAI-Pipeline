"""Independent score-only evaluation; never imports optimization agents or loop."""
import hashlib
import json
import platform
from pathlib import Path
from uuid import uuid4
from typing import Literal

from pydantic import Field, model_validator
from krugradeai.artifact_paths import output_directory, claim_live_directory
from krugradeai.prediction_reporting import prediction_report
from krugradeai.codex_provider import CodexCLIProvider
from krugradeai.grader import LLMGrader
from krugradeai.metrics import ScorePair, calculate_metrics
from krugradeai.models import AdaptationRule, AdaptationRules, Contract, Rubric, Text
from krugradeai.mock_provider import MockLLMProvider
from krugradeai.optimization_models import DevelopmentRecord, ValidationConfig
from krugradeai.providers import LLMProvider
from krugradeai.prompt_transport import PromptProvider
from krugradeai.validation import evaluate_labeled_records, _validate_unique_identities


class EvaluationRecord(DevelopmentRecord):
    split: Literal["evaluation"] = "evaluation"


from krugradeai.model_routing import ModelRoutes


class EvaluationConfig(Contract):
    dataset: Text
    rubric_file: Text
    prompt_file: Text
    output_directory: Text
    runtime: Literal["mock", "codex_cli"] = "mock"
    mock_responses: Text | None = None
    models: ModelRoutes = Field(default_factory=ModelRoutes)
    role: Literal["evaluation_grader", "heldout_grader"] = "evaluation_grader"
    guideline_version: int | None = Field(default=None, ge=0, strict=True)
    model: Text | None = None
    executable: Text | None = None
    timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    adaptation_rules: AdaptationRules = Field(default_factory=AdaptationRules)

    @model_validator(mode="after")
    def runtime_settings(self):
        if self.runtime == "mock" and self.mock_responses is None:
            raise ValueError("Mock execution requires explicit scripted responses")
        if self.model and "models" in self.model_fields_set and self.model != getattr(self.models, self.role):
            raise ValueError("Conflicting evaluation model settings")
        return self


def load_config(path):
    path = Path(path).resolve()
    data = EvaluationConfig.model_validate_json(path.read_bytes()).model_dump()
    if data["model"]:
        data["models"][data["role"]] = data["model"]
    for key in ("dataset", "rubric_file", "prompt_file", "output_directory", "mock_responses"):
        if data[key] is not None:
            data[key] = str((path.parent / data[key]).resolve())
    if data["executable"] and any(c in data["executable"] for c in ("/", "\\")):
        data["executable"] = str((path.parent / data["executable"]).resolve())
    return EvaluationConfig.model_validate(data)


class ScriptedProvider(LLMProvider):
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, request):
        return MockLLMProvider(json.dumps(next(self.responses))).generate(request)


def run(config_path, *, live=False, dry_run=False):
    config = load_config(config_path)
    if live and dry_run:
        raise ValueError("Live and dry-run are mutually exclusive")
    if config.runtime == "codex_cli" and not (live or dry_run):
        raise ValueError("Codex execution requires --live")
    paths = [Path(config_path).resolve(), Path(config.dataset), Path(config.rubric_file), Path(config.prompt_file)]
    if config.mock_responses:
        paths.append(Path(config.mock_responses))
    snapshots = {str(p): p.read_bytes() for p in paths}
    rubric = Rubric.model_validate_json(snapshots[config.rubric_file])
    known = {criterion.id for criterion in rubric.criteria}
    if any(isinstance(rule, AdaptationRule) and not set(rule.criterion_ids) <= known
           for rule in config.adaptation_rules.rules):
        raise ValueError("Adaptation rules reference unknown rubric criteria")
    instructions = snapshots[config.prompt_file].decode("utf-8")
    if not instructions.strip() or "\nINPUT_JSON\n" in instructions:
        raise ValueError("Supply instruction text without an embedded input block")
    records = tuple(EvaluationRecord.model_validate(dict(json.loads(line), rubric=rubric))
                    for line in snapshots[config.dataset].decode("utf-8").splitlines() if line.strip())
    if not records:
        raise ValueError("Empty dataset")
    _validate_unique_identities(records)
    definitions = {}
    for record in records:
        if record.question.id in definitions and definitions[record.question.id] != record.question:
            raise ValueError("Question IDs must have consistent definitions")
        definitions[record.question.id] = record.question
        if record.provenance is None or record.provenance.label_origin is None:
            raise ValueError("Every evaluation label needs explicit human/AI provenance")
    responses = None
    if config.runtime == "mock":
        responses = json.loads(snapshots[config.mock_responses])
        if not isinstance(responses, list) or len(responses) != len(records):
            raise ValueError("One scripted response per record is required")
    plan = dict(records=len(records), maximum_calls=len(records), runtime=config.runtime)
    if dry_run:
        return plan
    destination = Path(config.output_directory)
    if any(destination == p or destination in p.parents for p in paths):
        raise ValueError("Output must not contain inputs")
    if destination.exists():
        raise FileExistsError("Use a fresh output directory")
    (claim_live_directory if live else output_directory)(destination)
    experiment_id = uuid4().hex
    (destination / "status.json").write_text(json.dumps(dict(status="running",experiment_id=experiment_id)), encoding="utf-8")
    try:
        backend = ScriptedProvider(responses) if responses is not None else CodexCLIProvider(
            model=getattr(config.models, config.role), role=config.role, disable_cli_retries=True, invocation_log=destination/"provider-invocations.jsonl", executable=config.executable, timeout_seconds=config.timeout_seconds)
        provider = PromptProvider(backend, instructions)
        result = evaluate_labeled_records(LLMGrader(provider), records, config.adaptation_rules, 0, ValidationConfig(), fail_fast=live)
        outcomes = {r.record_id: r for r in result.records}
        groups = {}
        for origin in ("human_derived", "ai_derived", "unverified"):
            group = [r for r in records if r.provenance.label_origin == origin]
            pairs = [ScorePair(predicted=outcomes[r.id].grading.grader_output.total_score,
                               reference=r.ta_output.total_score, maximum=r.rubric.total_score)
                     for r in group if outcomes[r.id].grading is not None]
            groups[origin] = calculate_metrics(pairs, expected_count=len(group)).model_dump(mode="json")
        score_ocr_groups = {}
        for score, ocr in sorted({(r.provenance.score_origin or 'unspecified', r.provenance.ocr_review_origin or 'unknown') for r in records}):
            cohort = [r for r in records if (r.provenance.score_origin or 'unspecified', r.provenance.ocr_review_origin or 'unknown') == (score, ocr)]
            pairs = [ScorePair(predicted=outcomes[r.id].grading.grader_output.total_score,
                               reference=r.ta_output.total_score, maximum=r.rubric.total_score)
                     for r in cohort if outcomes[r.id].grading is not None]
            score_ocr_groups[score+'/'+ocr] = calculate_metrics(pairs, expected_count=len(cohort)).model_dump(mode='json')
        if any(Path(p).read_bytes() != content for p, content in snapshots.items()):
            raise ValueError("Inputs changed during evaluation")
        code = Path(__file__).parent
        report = dict(schema_version=1, experiment_id=experiment_id, status="completed" if result.eligible else "incomplete",
            configuration=config.model_dump(mode="json"), input_sha256={p: hashlib.sha256(b).hexdigest() for p,b in snapshots.items()},
            code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code.glob("*.py"))},
            python_version=platform.python_version(), provider_calls=provider.calls,
            model_requested=getattr(config.models, config.role), effective_model=backend.effective_model if responses is None else "no_model_mock",
            metrics=result.metrics.model_dump(mode="json"), metrics_by_label_origin=groups,
            metrics_by_score_and_ocr=score_ocr_groups,
            prediction_breakdown=dict(prediction_report(result, records), scope=rubric.id,
                version=config.guideline_version, selection_eligible=False, selection_mae=None, selected_version=None,
                patience=None, generation_history=[], selection_reason="Independent evaluation is score-only"),
            evaluation=result.model_dump(mode="json"),
            interpretation="Original TA labels are reported separately by OCR-review provenance. AI-generated labels measure pseudo-label agreement; mixed metrics are descriptive only.")
        (destination / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        (destination / "status.json").write_text(json.dumps(dict(status=report["status"],experiment_id=experiment_id)), encoding="utf-8")
        return report
    except (Exception, KeyboardInterrupt) as error:
        (destination / "status.json").write_text(json.dumps(dict(status="failed", experiment_id=experiment_id, error_type=type(error).__name__)), encoding="utf-8")
        raise
