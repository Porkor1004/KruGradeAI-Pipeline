"""Portable configuration entry point for the existing three-agent loop only."""
import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from krugradeai.artifact_paths import output_directory
from krugradeai.codex_provider import CodexCLIProvider
from krugradeai.grader import LLMGrader
from krugradeai.models import AdaptationRules, Contract, ReflectorOutput, Text
from krugradeai.optimization import OptimizationLoop
from krugradeai.optimization_models import DevelopmentRecord
from krugradeai.providers import LLMProvider
from krugradeai.reflector import LLMReflector
from krugradeai.refiner import LLMRefiner
from krugradeai.synthetic import build_mock_agents
from krugradeai.validation import _validate_unique_identities


class OptimizationConfig(Contract):
    development: Text
    output_directory: Text
    runtime: Literal["synthetic_mock", "codex_cli"] = "synthetic_mock"
    model: Text | None = None
    executable: Text | None = None
    timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    max_iterations: int = Field(default=1, ge=1, le=10, strict=True)
    max_calls: int = Field(default=32, ge=1, strict=True)
    initial_rules: AdaptationRules = Field(default_factory=AdaptationRules)

    @model_validator(mode="after")
    def explicit_model(self):
        if self.runtime == "codex_cli" and self.model is None:
            raise ValueError("Live optimization requires an explicit model")
        return self


class _Budget:
    def __init__(self, maximum):
        self.maximum, self.calls = maximum, 0


class _BoundedProvider(LLMProvider):
    def __init__(self, provider, budget):
        self.provider, self.budget = provider, budget

    def generate(self, request):
        if self.budget.calls >= self.budget.maximum:
            raise RuntimeError("Shared optimization call budget exhausted")
        self.budget.calls += 1
        return self.provider.generate(request)


def load_config(path):
    path = Path(path).resolve()
    config = OptimizationConfig.model_validate_json(path.read_bytes())
    # Input/output paths resolve relative to the config, never a developer's CWD.
    data = config.model_dump()
    for field in ("development", "output_directory"):
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
    records, required, content = prepare_run(config)
    plan = dict(records=len(records), rubric_paths=len({r.rubric.id for r in records}),
                reserved_maximum_calls=required, configured_max_calls=config.max_calls,
                max_iterations=config.max_iterations, runtime=config.runtime,
                validation_loaded=False, heldout_loaded=False)
    if dry_run:
        return plan
    destination = Path(config.output_directory)
    # Never write into source inputs, datasets or prior outputs.
    if destination == Path(config.development) or destination in Path(config.development).parents:
        raise ValueError("Output must not contain the input dataset")
    if destination.exists():
        raise FileExistsError("Use a fresh output directory; no overwrite/resume")
    output_directory(destination)
    budget = _Budget(config.max_calls)
    stage = "configuration"

    def record_status(status, error_type=None):
        payload = dict(status=status, stage=stage, provider_calls=budget.calls)
        if error_type is not None:
            payload["error_type"] = error_type
        (destination/"status.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")

    try:
        record_status("running")
        (destination/"configuration.json").write_text(config.model_dump_json(indent=2),encoding="utf-8")
        stage = "providers"
        if config.runtime == "synthetic_mock":
            agents = build_mock_agents()
        else:
            settings = dict(model=config.model, executable=config.executable, timeout_seconds=config.timeout_seconds)
            agents = (LLMGrader(CodexCLIProvider(**settings)),
                      LLMReflector(CodexCLIProvider(**settings, output_model=ReflectorOutput)),
                      LLMRefiner(CodexCLIProvider(**settings, output_model=AdaptationRules)))
        for agent in agents:
            agent.provider = _BoundedProvider(agent.provider, budget)
        protected = tuple(r.model_dump_json() for r in records)
        loop = OptimizationLoop(grader=agents[0],reflector=agents[1],refiner=agents[2],
                                max_iterations=config.max_iterations)
        stage = "optimization"
        result = loop.run(records,config.initial_rules)
        stage = "integrity"
        if protected != tuple(r.model_dump_json() for r in records) or Path(config.development).read_bytes() != content:
            raise ValueError("Development inputs changed")
        stage = "artifacts"
        with (destination/"optimization.json").open("x",encoding="utf-8") as stream:
            stream.write(result.model_dump_json(indent=2)+"\n")
        status = "incomplete" if any(p.stop_reason in ("agent_failure", "human_review") for p in result.paths) else "completed"
        summary = dict(**plan, provider_calls=budget.calls, status=status,
                       paths=[dict(rubric_id=p.rubric_id,stop_reason=p.stop_reason,
                                   latest_version=p.final_version) for p in result.paths],
                       selection="No Validation provided: latest proposal is not a best-guidelines claim")
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
    except Exception as error:
        parser.exit(1,f"Optimization refused/incomplete ({type(error).__name__}); no private input printed, no automatic retry.\n")
    print(json.dumps(result,indent=2))
    return 1 if result.get("status") == "incomplete" else 0


if __name__=="__main__":
    raise SystemExit(main())
