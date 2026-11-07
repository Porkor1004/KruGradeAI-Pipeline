"""Portable configuration entry point for the existing three-agent loop only."""
import argparse
from krugradeai.model_routing import ModelRoutingError
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from uuid import uuid4
from krugradeai.artifact_paths import output_directory, claim_live_directory
from krugradeai.prediction_reporting import prediction_report
from krugradeai.codex_provider import CodexCLIProvider, CodexOutputError
from krugradeai.grader import LLMGrader
from krugradeai.models import AdaptationRules, Contract, ReflectorOutput, Text
from krugradeai.optimization import OptimizationLoop
from krugradeai.optimization_models import DevelopmentRecord, ValidationRecord, ValidationConfig
from krugradeai.providers import LLMProvider
from krugradeai.reflector import LLMReflector
from krugradeai.refiner import LLMRefiner
from krugradeai.synthetic import build_mock_agents
from krugradeai.validation import _validate_unique_identities, prepare_validation
from krugradeai.prompts import GRADING_PROMPT
from krugradeai.prompt_transport import PromptProvider
from krugradeai.models import GraderInput, StudentResponse


from krugradeai.model_routing import ModelRoutes


class PreflightBudgetError(ValueError):
    """Safe, numeric preflight error for the CLI."""


class OptimizationConfig(Contract):
    development: Text
    output_directory: Text
    runtime: Literal["synthetic_mock", "schema_mock", "codex_cli"] = "synthetic_mock"
    prompt_file: Text | None = None
    validation: Text | None = None
    validation_config: ValidationConfig = Field(default_factory=ValidationConfig)
    models: ModelRoutes = Field(default_factory=ModelRoutes)
    model: Text | None = None  # Legacy Grader-only override
    executable: Text | None = None
    timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    max_iterations: int = Field(default=1, ge=1, le=10, strict=True)
    max_calls: int = Field(default=32, ge=1, strict=True)
    review_policy: Literal["strict", "ai_assisted_pilot"] = "strict"
    cli_request_retries: Literal[0] = 0
    cli_stream_retries: Literal[0] = 0
    retry_malformed_output: bool = False
    initial_rules: AdaptationRules = Field(default_factory=AdaptationRules)

    @model_validator(mode="after")
    def explicit_model(self):
        if self.model and "models" in self.model_fields_set and self.models.grader != self.model:
            raise ValueError("Conflicting Grader model settings")
        return self


class _Budget:
    def __init__(self, maximum):
        self.maximum, self.calls = maximum, 0


class _BoundedProvider(LLMProvider):
    def __init__(self, provider, budget, retry_malformed_output=False):
        self.provider, self.budget = provider, budget
        self.retry_malformed_output = retry_malformed_output

    def generate(self, request):
        for attempt in range(1 + int(self.retry_malformed_output)):
            if self.budget.calls >= self.budget.maximum:
                raise RuntimeError("Shared optimization call budget exhausted")
            self.budget.calls += 1
            try:
                return self.provider.generate(request)
            except CodexOutputError:
                if attempt or not self.retry_malformed_output:
                    raise




def load_config(path):
    path = Path(path).resolve()
    config = OptimizationConfig.model_validate_json(path.read_bytes())
    # Input/output paths resolve relative to the config, never a developer's CWD.
    data = config.model_dump()
    if config.model:
        data["models"]["grader"] = config.model
    for field in ("development", "output_directory", "validation", "prompt_file"):
        if data[field] is None:
            continue
        data[field] = str((path.parent/data[field]).resolve())
    if config.executable and ("/" in config.executable or "\\" in config.executable):
        data["executable"] = str((path.parent/config.executable).resolve())
    return OptimizationConfig.model_validate(data)


def prepare_run(config):
    content = Path(config.development).read_bytes()
    records = tuple(DevelopmentRecord.model_validate_json(line)
                    for line in content.decode("utf-8").splitlines() if line.strip())
    if not records or len({r.id for r in records}) != len(records):
        raise ValueError("Use nonempty, uniquely identified Development records")
    if len({(r.question.id,r.response.id) for r in records}) != len(records):
        raise ValueError("Duplicate response identities")
    _validate_unique_identities(records)
    definitions = {}
    for record in records:
        key = record.rubric.id
        if key in definitions and definitions[key] != (record.question,record.rubric):
            raise ValueError("A rubric path must identify the same question and rubric")
        definitions[key] = record.question,record.rubric
        known = {criterion.id for criterion in record.rubric.criteria}
        if any(not isinstance(rule,str) and not set(rule.criterion_ids) <= known for rule in config.initial_rules.rules):
            raise ValueError("Initial rules must match every rubric path")
    if config.runtime == "synthetic_mock":
        # This provider is a scripted fixture, never a generic offline grader.
        from krugradeai.synthetic import make_synthetic_dataset
        allowed = {r.model_dump_json() for r in make_synthetic_dataset(multiple_rubrics=True).development}
        if any(r.model_dump_json() not in allowed for r in records):
            raise ValueError("synthetic_mock accepts only the shipped synthetic Development fixtures")
    required = config.max_iterations*(2*len(records)+len(definitions))
    if config.max_calls < required:
        raise ValueError("Budget must reserve the conservative Grader/Reflector/Refiner bound")
    return records, required, content


def run(config_path, *, live=False, dry_run=False):
    config = load_config(config_path)
    if live and dry_run:
        raise ValueError("Live and dry-run are mutually exclusive")
    if config.runtime == "codex_cli" and not (live or dry_run):
        raise ValueError("Codex execution requires explicit --live")
    if config.review_policy == "ai_assisted_pilot" and config.runtime != "codex_cli":
        raise ValueError("AI-assisted pilot requires the Codex CLI runtime")
    records, required, content = prepare_run(config)
    validation_records, validation_content = None, None
    if config.validation:
        validation_content = Path(config.validation).read_bytes()
        validation_records = tuple(ValidationRecord.model_validate_json(line)
            for line in validation_content.decode("utf-8").splitlines() if line.strip())
        prepare_validation(records, validation_records, config.validation_config)
        if config.runtime == "codex_cli":
            from krugradeai.synthetic import make_synthetic_dataset
            fixtures = set(r.model_dump_json() for r in make_synthetic_dataset(multiple_rubrics=True).validation)
            for row in validation_records:
                if row.model_dump_json() in fixtures:
                    continue
                provenance = row.provenance
                if not provenance or not (provenance.source_dataset == "untitled" and
                        provenance.score_origin == "original_ta" and provenance.benchmark_eligible is True):
                    raise ValueError("Real Validation requires explicit benchmark-eligible original TA provenance")
        required += (config.max_iterations + 1) * len(validation_records)
        if config.max_calls < required:
            raise ValueError("Budget must also reserve baseline and candidate Validation calls")
    instructions = Path(config.prompt_file).read_text(encoding="utf-8") if config.prompt_file else GRADING_PROMPT
    if not instructions.strip() or "\nINPUT_JSON\n" in instructions:
        raise ValueError("Supply instruction text without an embedded input block")
    plan = dict(records=len(records), rubric_paths=len({r.rubric.id for r in records}),
                reserved_maximum_calls=required, configured_max_calls=config.max_calls,
                max_iterations=config.max_iterations, runtime=config.runtime,
                validation_loaded=validation_records is not None, heldout_loaded=False)
    if config.review_policy == "ai_assisted_pilot":
        plan["review_policy"] = config.review_policy
        plan["potential_review_calls"] = 2*config.max_iterations*len(records) + (config.max_iterations+1)*len(validation_records or ())
    # Ordinary calls can retry malformed output once if explicitly enabled.
    # Adjudication and diagnostic passes have no retries.
    projected = required * (2 if config.retry_malformed_output else 1) + plan.get("potential_review_calls", 0)
    plan.update(projected_maximum_calls=projected, reserved_maximum_calls=projected,
                cli_request_retries=0, cli_stream_retries=0)
    if projected > config.max_calls:
        raise PreflightBudgetError(f"Projected maximum {projected} exceeds hard cap {config.max_calls}")
    if config.runtime == "codex_cli" and config.validation_config.comparison_record_ids is not None:
        raise ValueError("Live experiments require full fixed Validation coverage")
    if dry_run:
        return plan
    destination = Path(config.output_directory)
    # Never write into source inputs, datasets or prior outputs.
    if destination == Path(config.development) or destination in Path(config.development).parents:
        raise ValueError("Output must not contain the input dataset")
    if config.prompt_file and (destination == Path(config.prompt_file) or destination in Path(config.prompt_file).parents):
        raise ValueError("Output must not contain the prompt file")
    if config.validation and (destination == Path(config.validation) or destination in Path(config.validation).parents):
        raise ValueError("Output must not contain Validation inputs")
    if destination.exists():
        raise FileExistsError("Use a fresh output directory; no overwrite/resume")
    (claim_live_directory if live else output_directory)(destination)
    plan["experiment_id"] = uuid4().hex
    budget = _Budget(config.max_calls)
    stage = "configuration"

    def record_status(status, error_type=None):
        payload = dict(experiment_id=plan["experiment_id"], status=status, stage=stage, provider_calls=budget.calls)
        if error_type is not None:
            payload["error_type"] = error_type
        (destination/"status.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")

    try:
        record_status("running")
        (destination/"configuration.json").write_text(config.model_dump_json(indent=2),encoding="utf-8")
        stage = "providers"
        validation_grader = None
        if config.runtime == "synthetic_mock":
            agents = build_mock_agents()
        elif config.runtime == "schema_mock":
            from krugradeai.schema_mock import SchemaMockProvider
            agents = (LLMGrader(SchemaMockProvider('grader')),
                      LLMReflector(SchemaMockProvider('reflector')),
                      LLMRefiner(SchemaMockProvider('refiner')))
        else:
            def provider(role, output_model=None):
                return CodexCLIProvider(model=getattr(config.models, role), role=role,
                    invocation_log=destination/"provider-invocations.jsonl", executable=config.executable,
                    timeout_seconds=config.timeout_seconds, output_model=output_model, disable_cli_retries=True)
            agents = (LLMGrader(provider("grader")), LLMReflector(provider("reflector", ReflectorOutput)),
                      LLMRefiner(provider("refiner", AdaptationRules)))
            if validation_records:
                backend = _BoundedProvider(provider("validation_grader"), budget, config.retry_malformed_output)
                validation_grader = LLMGrader(PromptProvider(backend, instructions) if config.prompt_file else backend)
        for agent in agents:
            agent.provider = _BoundedProvider(agent.provider, budget, config.retry_malformed_output)
        if config.prompt_file:
            agents[0].provider = PromptProvider(agents[0].provider, instructions)
        if config.review_policy == "ai_assisted_pilot":
            from krugradeai.assisted_review import AssistedGrader, AssistedReflector, ReviewDecision
            def reviewer(role, schema):
                provider = CodexCLIProvider(model="gpt-6-astra", role=role, output_model=schema, disable_cli_retries=True,
                    invocation_log=destination/"provider-invocations.jsonl", executable=config.executable,
                    timeout_seconds=config.timeout_seconds)
                return _BoundedProvider(provider, budget)
            audit_path = destination/"assisted-reviews.jsonl"
            assessor = reviewer("ai_adjudicator", ReviewDecision)
            agents = (AssistedGrader(agents[0], assessor, audit_path, "development"),
                      AssistedReflector(agents[1], reviewer("reflector_diagnostic", ReflectorOutput), audit_path),
                      agents[2])
            if validation_grader is not None:
                validation_grader = AssistedGrader(validation_grader, assessor, audit_path, "validation")
        validation_history = {}
        def observe(tracker, evaluation):
            scope = tracker.records[0].rubric.id
            entry = prediction_report(evaluation, tracker.records)
            entry.update(scope=scope, best_version=tracker.best_version,
                         patience=tracker.non_improving, selection_metric=tracker.config.primary_metric,
                         status="ineligible" if not evaluation.eligible else "baseline" if evaluation.version==0
                         else "new_best" if tracker.best_version==evaluation.version else "retained_earlier_best")
            validation_history.setdefault(scope, []).append(entry)
            with (destination/"validation-history.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(entry)+"\n")
        protected = tuple(r.model_dump_json() for r in records)
        loop = OptimizationLoop(fail_fast=live, validation_observer=observe, validation_grader=validation_grader, grader=agents[0],reflector=agents[1],refiner=agents[2],
                                max_iterations=config.max_iterations, validation_config=config.validation_config)
        stage = "optimization"
        result = loop.run(records,config.initial_rules, validation_records=validation_records)
        stage = "integrity"
        if protected != tuple(r.model_dump_json() for r in records) or Path(config.development).read_bytes() != content:
            raise ValueError("Development inputs changed")
        if config.validation and Path(config.validation).read_bytes() != validation_content:
            raise ValueError("Validation inputs changed")
        stage = "artifacts"
        # Export only selected rules when Validation exists; otherwise mark proposals.
        exports = []
        for index, path in enumerate(result.paths, 1):
            selected = path.validation
            version = selected.best_version if selected else path.final_version
            if version is None:
                continue
            rules = selected.best_rules if selected else path.rule_versions[version].rules
            record = next(r for r in records if r.rubric.id == path.rubric_id)
            definition = dict(question=record.question.model_dump(mode="json"),
                rubric=record.rubric.model_dump(mode="json"), adaptation_rules=rules.model_dump(mode="json"))
            (destination / f"prompt-{index}.json").write_text(json.dumps(definition, indent=2)+"\n", encoding="utf-8")
            response = StudentResponse(id="REPLACE_WITH_RESPONSE_ID", question_id=record.question.id,
                                       text="REPLACE_WITH_ORIGINAL_STUDENT_ANSWER")
            template = instructions + "\nINPUT_JSON\n" + GraderInput(
                question=record.question, rubric=record.rubric, response=response,
                adaptation_rules=rules).model_dump_json()
            (destination / f"prompt-{index}.grader.txt").write_text(template, encoding="utf-8")
            exports.append(dict(rubric_id=path.rubric_id, version=version,
                                validated=selected is not None, definition=f"prompt-{index}.json"))
        (destination / "grader-instructions.txt").write_text(instructions, encoding="utf-8")
        with (destination/"optimization.json").open("x",encoding="utf-8") as stream:
            stream.write(result.model_dump_json(indent=2)+"\n")
        status = "incomplete" if any(p.stop_reason in ("agent_failure", "human_review") or
            (p.validation is not None and p.validation.best_version is None) for p in result.paths) else "completed"
        summary = dict(**plan, provider_calls=budget.calls, status=status,
                       paths=[dict(rubric_id=p.rubric_id,stop_reason=p.stop_reason,
                                   latest_version=p.final_version) for p in result.paths],
                       prompt_exports=exports,
                       selection="Validation-selected rules" if validation_records is not None else "No Validation provided: latest proposal is not a best-guidelines claim")
        summary["scope_reports"] = [dict(scope=p.rubric_id, selected_version=p.validation.best_version if p.validation else None,
            formal_test_version=(p.validation.best_version if p.validation and p.validation.best_version is not None else 0),
            selection_reason=("earliest eligible minimum of the configured Validation metric" if p.validation and p.validation.best_version is not None
                              else "no eligible Validation version" if p.validation else "Development-only proposal; use G0 for formal Test"),
            generation_history=[dict(iteration=i.iteration, parent_version=i.rules_version,
                                     candidate_version=i.candidate_version, stop_reason=i.stop_reason) for i in p.history],
            validation_history=validation_history.get(p.rubric_id, []), stop_reason=p.stop_reason) for p in result.paths]
        with (destination/"summary.json").open("x",encoding="utf-8") as stream:
            json.dump(summary,stream,indent=2)
        stage = "finished"
        record_status(status)
        return summary
    except (Exception, KeyboardInterrupt) as error:
        # Keep partial artifacts and the original exception; never persist its
        # message, which may contain prompts, answers or provider diagnostics.
        try:
            record_status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          type(error).__name__)
        except OSError:
            pass  # A storage failure may prevent even the status from being saved.
        raise



def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--live",action="store_true")
    mode.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    try:
        result=run(args.config,live=args.live,dry_run=args.dry_run)
    except PreflightBudgetError as error:
        parser.exit(1,f"Optimization refused: {error}\n")
    except ModelRoutingError as error:
        parser.exit(1, f"Model routing failed: {error}\n")
    except Exception as error:
        parser.exit(1,f"Optimization refused/incomplete ({type(error).__name__}); no private input printed, no automatic retry.\n")
    print(json.dumps(result,indent=2))
    return 1 if result.get("status") == "incomplete" else 0


if __name__=="__main__":
    raise SystemExit(main())
