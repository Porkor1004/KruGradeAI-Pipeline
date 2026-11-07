"""One synthetic CLI request per configured role. Explicit opt-in, no retries."""
import argparse
import json
from pathlib import Path

from krugradeai.codex_provider import CodexCLIProvider
from krugradeai.model_routing import ModelRoutes, ModelRoutingError
from krugradeai.models import (AdaptationRules, GraderInput, GradingRecord,
                               ReflectorInput, ReflectorOutput, RefinerInput)
from krugradeai.grader import LLMGrader
from krugradeai.reflector import LLMReflector
from krugradeai.refiner import LLMRefiner
from krugradeai.synthetic import make_synthetic_dataset, build_mock_agents


def run(destination):
    destination.mkdir(parents=True, exist_ok=False)
    fixture = make_synthetic_dataset().development[0]
    grading = GraderInput(question=fixture.question, rubric=fixture.rubric, response=fixture.response)
    grader, reflector, _ = build_mock_agents()
    reflection = ReflectorInput(record=GradingRecord(grader_input=grading, grader_output=grader.grade(grading)),
                                ta_output=fixture.ta_output)
    refinement = RefinerInput(current_rules=AdaptationRules(), reflection=reflector.reflect(reflection), rubric=fixture.rubric)
    report = {}
    for role, model in ModelRoutes().model_dump().items():
        provider = CodexCLIProvider(role=role, model=model, invocation_log=destination/'invocations.jsonl',
                                    output_model={'reflector': ReflectorOutput, 'refiner': AdaptationRules}.get(role))
        try:
            if role == 'reflector':
                LLMReflector(provider).reflect(reflection)
            elif role == 'refiner':
                LLMRefiner(provider).refine(refinement)
            else:
                LLMGrader(provider).grade(grading)
            report[role] = 'accepted'
        except ModelRoutingError:
            report[role] = 'route_or_turn_rejected'
            break
        except Exception as error:
            report[role] = type(error).__name__
        finally:
            (destination/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return int(len(report) != 6 or any(v != 'accepted' for v in report.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    opts = parser.parse_args()
    raise SystemExit(run(opts.output_dir))
