# Reusing the three-agent optimization pipeline

This addition is an input/configuration adapter for the existing `OptimizationLoop`,
not a replacement algorithm, extra agent, or refactored Evaluation Pipeline.
Existing real-data and evaluation commands remain unchanged.

## Portable commands

Install Python >=3.11 and `python -m pip install -e ".[data]"` from the repository.
Only Pydantic is required for generic optimization; the data extra is required
for the complete test suite and existing workbook adapter.

The checked-in example is fully synthetic, includes two rubric paths and contains
no private student data:

```powershell
python -m krugradeai.run_optimization --config examples/optimization/synthetic.json --dry-run
python -m krugradeai.run_optimization --config examples/optimization/synthetic.json
```

The second command executes mock Grader/Reflector/Refiner calls only. It does not
load Validation or Held-out data. Its scripted outputs are not accuracy results.
The mock runtime only accepts the shipped synthetic Development fixtures: it
cannot grade arbitrary answers or serve as a general offline provider.

For a new dataset, copy `custom-codex.example.json` outside the example tree,
replace model/input/output values, and place private records in an ignored local
directory. First validate the plan:

```powershell
python -m krugradeai.run_optimization --config PATH_TO_CONFIG.json --dry-run
```

Only after intentional authorization/authentication, the corresponding live
command is `python -m krugradeai.run_optimization --config PATH_TO_CONFIG.json --live`.
No such command was executed during implementation. Use an existing compatible
Codex CLI authenticated with ChatGPT; an API key is not required. `executable`
optionally selects a CLI path. Both agent prompts and providers remain unchanged.

All input/output paths are relative to the configuration file, independent of the
shell's working directory. Executable paths containing `/` or `\` resolve the same
way; a bare executable name uses PATH. Unknown config fields are rejected. Live
mode requires an explicit model, timeout must be finite/positive, and iteration
counts are bounded from 1 through 10 for this convenience runner.

## Input contracts

`development` is UTF-8 JSONL: one existing `DevelopmentRecord` per nonblank line.
The file must contain Development only, never Validation or Held-out records.
Custom subjects and criterion IDs require no change to core source code:

```json
{
  "id": "SYNTHETIC-EXAMPLE-ONLY",
  "split": "development",
  "question": {"id": "toy-q", "text": "Solve 2x = 8.", "reference_solution": "x = 4"},
  "rubric": {
    "id": "toy-rubric-v1",
    "criteria": [
      {"id": "method", "description": "Valid equivalent transformation; partial credit allowed.", "max_score": "1"},
      {"id": "result", "description": "Correct stated result.", "max_score": "1"}
    ],
    "total_score": "2"
  },
  "response": {"id": "toy-response", "question_id": "toy-q", "text": "Divide both sides by 2; x = 4."},
  "ta_output": {
    "total_score": "2",
    "criterion_scores": [{"criterion_id": "method", "score": "1"}, {"criterion_id": "result", "score": "1"}]
  },
  "provenance": {"source": "fully-synthetic-example", "item_id": "toy-response", "label_origin": "human_derived"}
}
```

This is a fabricated **synthetic schema example**, not an actual human annotation.
For real records, declare the true provenance (`human_derived` or `ai_derived`),
never copy that example's provenance as a verification claim. `ta_output` is the
legacy numeric reference field name; it does not certify a TA author. Pydantic
allows provenance to be omitted for legacy fixtures; meaningful real reporting
requires explicit provenance. Decimal strings avoid floating-point ambiguity.

`ta_output` also accepts total-only `{"total_score":"1","criterion_scores":[]}`
or a partial criterion map through `TAReference`. Missing criterion scores are
never allocated from a total. Total-only records may stop refinement for insufficient
evidence. Question references are optional; rubric criteria must be nonempty with
unique IDs, positive maxima and a total equal to their sum. Scores must fit those
maxima and totals. Each rubric ID must refer to the same question and rubric for
the whole run; distinct questions require distinct rubric paths even if wording
of the criteria happens to match.

`initial_rules` uses `AdaptationRules`: `{"rules":[]}` or structured rules with
`id`, `instruction`, `criterion_ids`, and `rationale`. Initial rules in this CLI
are shared across paths, so criterion references must be valid in every path.
Use separate configs or the Python API for path-specific starting rules. Newly
generated rules remain subject to existing Refiner restrictions.

## Outputs, safety and boundaries

The fresh output directory receives:

- `configuration.json`: resolved configuration; no labels or answers.
- `optimization.json`: existing frozen optimization result/history including
  scored Development records, reflections, candidates, failures and review flags.
  **Private research output: it can contain answers and labels. Do not publish it.**
- `summary.json`: path stop reasons, latest versions, configuration and call count.

The directory must not exist; input files and `data/raw/` cannot be output targets.
This runner has no resume/retry. A partial directory from a process/setup failure
is not proof of completion; inspect it and do not treat missing history as success.
Agent failures/reviews follow existing orchestration behavior and appear in history.

The conservative reservation is `iterations × (2 × Development records + rubric
paths)`: at most one Grader and Reflector per record and one Refiner per path per
iteration. All agents share a counter that blocks calls above `max_calls`.
Agreement or early stopping uses fewer calls. No G1 is guaranteed.

This intentionally small entry point runs the **Development optimization loop
only**. It does not load Validation/Test, run final evaluation, choose a best
candidate, import the real mixed-source adapter, or enable AI adjudication. The
latest proposal is not best guidelines. For existing Validation/best selection,
the unchanged Python API accepts separate `validation_records` and `ValidationConfig`:

```python
from krugradeai.optimization import OptimizationLoop

# Inject existing Grader/Reflector/Refiner implementations or your own providers.
result = OptimizationLoop(grader=grader, reflector=reflector, refiner=refiner,
                          max_iterations=1).run(development,
                                                validation_records=validation)
# A selected result, if valid: result.paths[0].validation.best_rules / best_version
```

The original real-data runner continues to implement the approved provenance,
criterion-supplement, group checks, optional adjudication and human-only Validation
policy. Do not replace that completed experiment with this convenience runner.
For a new dataset, independently group related attempts/derived answers before
splitting. Generic records do not contain the original adapter's full ancestry
graph: the CLI cannot prove absence of unknown leakage. Core schemas cannot
authenticate labels or prove mathematical evidence correctness. New subjects
may need human interpretation of rubric scope, especially for ambiguous OCR,
unsupported point allocations or multimodal answers (the current input is text).

No Evaluation Pipeline was moved, separated or redesigned. Existing evaluation
tests, contracts and frozen experiment results remain intact.

## Verification

The full suite passes **399 tests** (386 existing plus 13 export/portability tests).
The checked-in synthetic config was also run offline: four Development records,
two rubric paths, ten mock-provider invocations, G1 proposals on both paths, and
no Validation/Held-out access. No live calls or real-data optimization were run.
The original full optimization report, final report and frozen-guideline hashes
are unchanged. No Git initialization, push or PR creation was performed.
