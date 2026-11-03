# Frozen final Grader instruction export

`python -m krugradeai.prompt_export` reads the existing frozen best selection,
verifies it against its optimization report/manifest/code, and recovers unchanged
question/reference text from **Development only**. It does not grade, invoke a
provider, optimize, evaluate, or read Validation/Held-out record partitions.

The checked-in export is `exports/final-grader-prompts/`: Q5/Q7 use G1, all other
Q1–Q8 scopes use G0 (empty adaptation rules). Each Q number maps explicitly to the
original question/rubric ID in `manifest.json`. No guidelines were regenerated.

For regeneration into a **fresh** directory using the private source evidence:

```powershell
python -m krugradeai.prompt_export --frozen artifacts/real-final-evaluation/frozen-guidelines-full-001.json --output-dir exports/final-grader-prompts-new
```

Existing export directories and original experiment inputs cannot be overwritten.
The original artifact can contain local paths; generation requires that approved
evidence bundle at its recorded location. The exported per-question definitions
themselves have no absolute machine paths or dependency on that private bundle.

## Contents and message roles

- `Qn.grader.txt`: the **exact agent request template** assembled by
  `build_grading_prompt`: current `GRADING_PROMPT`, then `INPUT_JSON`, then the
  question, optional reference, immutable rubric, selected adaptation rules and
  an explicitly marked student-response placeholder.
- `Qn.codex-stdin.txt`: the exact **textual stdin template** used by the current
  provider: `RUNTIME_POLICY`, then `TASK_REQUEST_JSON`, then JSON wrapping the
  preceding agent request as `task_request`.
- `Qn.json`: portable `PromptDefinition` containing the original normalized
  question/reference, frozen rubric and exact selected rules. No response/label.
- `manifest.json`: selected version per path, rubric/rule/file hashes, prompt
  hashes and transport/placeholder limitations.

**These are not verified native system-role messages.** The current provider calls
`codex exec ... -` with a single stdin text request. Neither the runtime policy
nor the Grader instructions are passed through a system-role API. Codex's own
internal instructions/model configuration are outside the export's scope.
Copying this text into another provider's system field changes the transport
semantics and is not an exact reproduction of this implementation.

Rubric/rule strings, criterion IDs, maxima, ordering and rationale are preserved
exactly as stored in the frozen Pydantic contracts. JSON escaping/indentation is
serialization, not an edit to content. This is not a byte-for-byte export of raw
teacher documents: the existing adapter had already normalized those inputs.
The full question/reference/rubric is included because it is necessary task data.
No actual student answers, operational target scores, Development findings or
Validation metrics are included. Reference solutions may themselves describe the
correct mathematical answer; those are original supplied task material, not labels.

## Standalone use

Replace only the response by constructing the existing typed input; do not send
the placeholder as a real answer. No optimizer or private split files are needed:

```python
from pathlib import Path
from krugradeai.prompt_export import PromptDefinition, render_prompt
from krugradeai.models import StudentResponse, GraderInput

definition = PromptDefinition.model_validate_json(
    Path("exports/final-grader-prompts/Q5.json").read_bytes())
response = StudentResponse(id="new-response", question_id=definition.question.id,
                           text=original_student_answer)
request_text = render_prompt(definition, response)
# request_text is identical to what the existing LLMGrader builds.
# Prefer its strict scored-or-review parsing over accepting raw model JSON:
request = GraderInput(**definition.model_dump(), response=response)
# Explicitly configured provider and live authorization would be required:
# result = LLMGrader(provider).grade(request)
```

If using `CodexCLIProvider.generate`, pass `request_text`, **not** the already
wrapped `.codex-stdin.txt` template; the provider adds its runtime envelope once.
Other transports may use the plain agent request, but must still apply the
existing criterion/total validation and human-review protocol. The text export
alone cannot enforce runtime sandbox permissions or score validation.

File hashes identify the original export; changing the response naturally changes
request bytes. The exports correspond to the current verified grading code and
frozen selection. Historical optimization effective model remains unknown. The
official held-out result remains incomplete at 22/24 with identical aggregate
baseline/selected metrics. Exporting rules is not new evidence of grading gains.

No private student content is exported, but original exam questions, teacher
rubrics and reference solutions may have redistribution restrictions. Confirm
sharing rights/repository visibility before publishing the export bundle.
