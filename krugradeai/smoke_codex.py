"""Explicit opt-in live transport check with a synthetic grading example.

Never imported or executed as part of unit-test discovery.
"""
import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from krugradeai.codex_provider import CodexCLIError, CodexCLIProvider
from krugradeai.grader import HumanReviewRequired, LLMGrader
from krugradeai.models import GraderInput, GraderOutput, Question, Rubric, RubricCriterion, StudentResponse


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Opt-in live Codex test using one harmless synthetic grading request.")
    parser.add_argument("--live", action="store_true", help="Explicitly authorize one live Codex inference request (uses account quota).")
    parser.add_argument("--executable", help="Installed Codex executable or supported Windows npm shim.")
    parser.add_argument("--model", help="Codex model identifier; omit to use CLI defaults.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout in seconds.")
    parser.add_argument("--working-directory", help="Existing dedicated working directory; default is an isolated temporary directory.")
    args = parser.parse_args(argv)
    if not args.live:
        print("No live call made. Pass --live explicitly to run one synthetic grading request.", file=sys.stderr)
        return 2
    request = GraderInput(
        question=Question(id="smoke-addition", text="Compute 7 + 5.", reference_solution="7 + 5 = 12."),
        rubric=Rubric(id="smoke-rubric", criteria=(RubricCriterion(
            id="calculation", description="Award one point for the correct sum, otherwise zero.", max_score="1"),), total_score="1"),
        response=StudentResponse(id="synthetic-response", question_id="smoke-addition", text="7 + 5 = 12."),
    )
    try:
        provider = CodexCLIProvider(executable=args.executable, model=args.model,
                                    timeout_seconds=args.timeout, working_directory=args.working_directory,
                                    output_model=GraderOutput)
        result = LLMGrader(provider).grade(request)
    except CodexCLIError as error:
        print(f"Smoke test failed [{error.stage}]: {error}", file=sys.stderr)
        return 1
    except HumanReviewRequired:
        print("Smoke test requested human review instead of a grade.", file=sys.stderr)
        return 1
    except ValidationError:
        print("Smoke test returned an invalid grade; no result accepted.", file=sys.stderr)
        return 1
    except ValueError:
        print("Smoke test configuration is invalid; check model, timeout and paths.", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
