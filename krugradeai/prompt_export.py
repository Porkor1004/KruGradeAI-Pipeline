"""Export frozen grading instructions, never student answers or reference scores."""
import argparse
import json
from pathlib import Path

from krugradeai.artifact_paths import output_directory
from krugradeai.codex_provider import RUNTIME_POLICY
from krugradeai.final_evaluation import DEFAULT_FROZEN, FrozenGuidelines, freeze_selection, sha
from krugradeai.models import AdaptationRules, Contract, GraderInput, Question, Rubric, StudentResponse
from krugradeai.prompts import GRADING_PROMPT, build_grading_prompt
from krugradeai.real_data import load_prepared_partition


class PromptDefinition(Contract):
    question: Question
    rubric: Rubric
    adaptation_rules: AdaptationRules


def render_prompt(definition: PromptDefinition, response: StudentResponse) -> str:
    """Standalone assembly identical to LLMGrader; requires no optimizer/artifacts."""
    definition = PromptDefinition.model_validate(definition)
    return build_grading_prompt(GraderInput(**definition.model_dump(), response=response))


def codex_stdin(request: str) -> str:
    """Mirror the unchanged CLI provider's textual transport (not a system role)."""
    return RUNTIME_POLICY + "\nTASK_REQUEST_JSON\n" + json.dumps({"task_request": request}, ensure_ascii=False)


def export_prompts(frozen_path=DEFAULT_FROZEN, destination="exports/final-grader-prompts"):
    frozen_path = Path(frozen_path).resolve()
    frozen_bytes = frozen_path.read_bytes()
    snapshot = FrozenGuidelines.model_validate_json(frozen_bytes)
    if freeze_selection(snapshot.source_report, snapshot.data_directory,
                        model_verification=snapshot.model_verification) != snapshot:
        raise ValueError("Frozen selection or grading implementation changed")
    # Development is the only record partition read. Never export its answers/labels.
    development = load_prepared_partition(snapshot.data_directory, "development")
    definitions = []
    for selected in snapshot.paths:
        matching = [r for r in development if r.rubric.id == selected.rubric.id]
        if not matching or any(r.rubric != selected.rubric or r.question != matching[0].question for r in matching):
            raise ValueError("Missing or inconsistent question/rubric scope")
        definitions.append((selected, PromptDefinition(question=matching[0].question,
            rubric=selected.rubric, adaptation_rules=selected.selected_rules)))
    if len({d.question.id for _, d in definitions}) != len(definitions):
        raise ValueError("Question IDs must identify distinct export scopes")
    destination = Path(destination).resolve()
    for protected in (Path(snapshot.data_directory).resolve(),):
        if destination == protected or protected in destination.parents or destination in protected.parents:
            raise ValueError("Export must be separate from original experiment inputs")
    for protected in (frozen_path, Path(snapshot.source_report).resolve()):
        if destination == protected or destination in protected.parents:
            raise ValueError("Export must not contain original experiment inputs")
    if destination.exists():
        raise FileExistsError("Export exists; no files overwritten")
    output_directory(destination)
    entries = []
    for index, (selected, definition) in enumerate(sorted(definitions, key=lambda pair: pair[1].question.id), 1):
        name = f"Q{index}"
        # Explicit template, not invented student work or an executable grading request.
        response = StudentResponse(id="REPLACE_WITH_RESPONSE_ID", question_id=definition.question.id,
                                   text="REPLACE_WITH_ORIGINAL_STUDENT_ANSWER")
        agent_text = render_prompt(definition, response)
        payloads = {f"{name}.json": definition.model_dump_json(indent=2)+"\n",
                    f"{name}.grader.txt": agent_text,
                    f"{name}.codex-stdin.txt": codex_stdin(agent_text)}
        for filename, text in payloads.items():
            with (destination/filename).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        entries.append(dict(name=name, question_id=definition.question.id, rubric_id=selected.rubric.id,
            selected_version=selected.selected_version,
            rubric_sha256=sha(selected.rubric.model_dump_json().encode()),
            rules_sha256=sha(selected.selected_rules.model_dump_json().encode()),
            files={filename:sha(text.encode()) for filename,text in payloads.items()}))
    manifest = dict(schema_version=1, source_frozen_sha256=sha(frozen_bytes),
        source_report_sha256=snapshot.source_report_sha256, transport="codex_exec_stdin_task_text",
        native_system_message=False, response_is_placeholder=True,
        grading_prompt_sha256=sha(GRADING_PROMPT.encode()), runtime_policy_sha256=sha(RUNTIME_POLICY.encode()),
        scopes=entries,
        limitation="Textual task instructions, not native system-role messages. No student answers or reference scores exported. Rubric/rule strings exactly preserve the normalized frozen contracts, not raw-source formatting. Historical optimization model unknown.")
    with (destination/"manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", default=DEFAULT_FROZEN)
    parser.add_argument("--output-dir", default="exports/final-grader-prompts")
    args = parser.parse_args(argv)
    result = export_prompts(args.frozen, args.output_dir)
    print(json.dumps(dict(scopes=len(result["scopes"]), output=args.output_dir, live_calls=0)))


if __name__ == "__main__":
    main()
