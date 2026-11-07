# KruGradeAI Pipeline

KruGradeAI is a Python implementation of a GradeOpt-inspired grading workflow.
It updates grading instructions from labeled Development examples without
fine-tuning the model or changing the teacher rubric.

The three agents have separate jobs:

- **Grader** scores each rubric criterion and explains the award. It receives the
  question, optional reference solution, rubric, student answer and adaptation
  rules, but no reference scores.
- **Reflector** compares a Development grade with its reference labels and looks
  for supported discrepancies. Missing criterion labels are not inferred from
  a total score.
- **Refiner** proposes adaptation rules from supported findings. It cannot edit
  the rubric, score bounds or source labels. Unresolved evidence can stop refinement.

Frozen Pydantic models and Decimal scores enforce criterion coverage, bounds and
total consistency. Each rubric has its own optimization path. See the
[architecture](docs/architecture.md) and the [agent contracts](docs/grader.md).

## Two independent pipelines

Both workflows use the existing Grader, contracts, provider, Decimal metrics and
score-only evaluation kernel. Run from the repository root after installation.
Installed CLI aliases are `krugradeai-optimize` and `krugradeai-evaluate`.

### Agent Optimization Pipeline

```powershell
python -m krugradeai.run_optimization --config examples/optimization/synthetic.json
python -m krugradeai.run_optimization --config examples/optimization/synthetic-validated.json
```

The first command preserves Development-only behavior. The second supplies separate
Validation data to select G0 or a generated candidate with the existing MAE,
coverage, tie-breaking, patience and stopping policies. Validation labels never
reach Reflector or Refiner. The Grader → Reflector → Refiner methodology is unchanged.

Inputs: JSON config with `development` (DevelopmentRecord JSONL), optional
`validation` (ValidationRecord JSONL), `validation_config`, `initial_rules`, optional
`prompt_file` (Grader instruction prefix), model/executable/timeout, iteration and
shared call-budget settings, and `output_directory`. Paths resolve relative to the
config. Reserve the existing Development call bound plus
`(max_iterations + 1) * validation_record_count` calls when supplying Validation.

Outputs: `configuration.json`, `status.json`, `optimization.json`, `summary.json`,
`grader-instructions.txt`, and per-scope `prompt-N.json` / `prompt-N.grader.txt`.
The JSON definition contains question, rubric and adaptation rules. The text has
an explicit response placeholder. With Validation, exports use the selected version,
which may be G0. Without Validation they are proposals marked `validated: false`.
No valid selection means no selected export for that scope and an incomplete run.
Exports contain no student answers or reference scores.

### Independent Evaluation Pipeline

```powershell
python -m krugradeai.run_evaluation --config examples/evaluation/synthetic.json
python -m krugradeai.run_evaluation --config examples/evaluation/synthetic.json --dry-run
```

Evaluation constructs only the existing Grader. It neither imports nor starts the
optimization loop, Reflector or Refiner, and requires no optimization report or
frozen selection. The example is entirely synthetic, including the label origins.

The JSON config specifies these separate inputs:

- `prompt_file`: UTF-8 instruction prefix. The runner appends `\nINPUT_JSON\n`
  and validated Grader input. Include the output JSON contract, as in the example
  `grader.txt`. Do not include a response placeholder or embedded input block.
- `rubric_file`: one teacher Rubric JSON with criterion IDs, descriptions, maxima
  and total. A run evaluates one rubric scope.
- `dataset`: JSONL with `id`, `question`, `response`, `ta_output`, `provenance`, and
  optional `split: "evaluation"`. Rubric is supplied separately. References accept
  the existing full or total-only schema. Provenance requires `source`, `item_id`
  and explicit `label_origin`: `human_derived` or `ai_derived`.
- `adaptation_rules`: optional existing rules JSON in the config. To evaluate an
  optimization export, use its rubric and adaptation rules with exported
  `grader-instructions.txt`. The complete `.grader.txt` is a manual assembly template,
  not an instruction-prefix input.
- `runtime`, `model`, `executable`, `timeout_seconds`, and `output_directory`.
  `runtime: "mock"` requires `mock_responses`, a JSON array with one scripted response
  per row. Mock responses test software behavior; they are not an offline grader.

Outputs: `report.json` and `status.json`. The report contains per-row outcomes and
failures/reviews, MAE, normalized MAE (mean per-record absolute error divided by
rubric maximum), Exact Match and Coverage. Missing or invalid grades reduce Coverage
and never become zero scores. Separate human-derived and AI-derived metrics include
empty groups. AI metrics measure pseudo-label agreement, not human grading accuracy;
mixed metrics are descriptive only.

Reports retain input and source-code SHA-256 hashes, resolved config, Python version,
requested model and provider call count. Keep the exact source inputs for replay;
hashes cannot recreate them. Mock runs are deterministic. Live LLM output is not
guaranteed deterministic; effective model identity remains unknown without evidence.

### Execution, compatibility and limitations

Both CLIs support `--dry-run`. Configured `codex_cli` execution uses explicit per-role model IDs and requires `--live`. No model was added and verification uses mocks only. Codex CLI
task-text instructions are **not necessarily native system-role messages**: the
existing provider transports policy and task text through CLI stdin.

Use fresh output directories; outputs are never overwritten or resumed. Reports
can contain private answers and labels. Keep outputs under Git-ignored `artifacts/`,
source data under `data/` or `private-data/`, and private configs under `local-config/`.
Only reviewed synthetic fixtures belong in `examples/`. Historical reports, frozen
guidelines and original rubric files remain preserved. Existing `run_optimization`,
`optimize_real`, `final_evaluation`, `prompt_export` and other module commands remain
available. The historical freeze/final-evaluation workflow retains its safeguards.

New datasets do **not** automatically produce validated final prompts. Users must
prepare schemas, immutable teacher rubrics, trustworthy labels and independent,
leakage-free partitions. Identity checks cannot discover unknown semantic or attempt
relationships. Mock optimization accepts only shipped Development fixtures. Real
optimization needs explicit live execution, sufficient coverage and a valid
selection; Validation selection alone is not held-out evidence or human approval.
Independent evaluation reports performance and never selects, refines or certifies
a prompt.

```text
krugradeai/
  run_optimization.py   # three-agent CLI, optional Validation and prompt export
  optimization.py       # unchanged loop and stopping policies
  run_evaluation.py     # independent score-only CLI
  evaluation.py         # external inputs, report and label-origin metrics
  prompt_transport.py   # shared external instruction-prefix adapter
  grader.py / models.py / providers.py / metrics.py / validation.py  # shared
examples/
  optimization/         # Development-only and validated synthetic workflows
  evaluation/           # prompt, rubric, labeled data and scripted outputs
```

Run `python -m unittest discover -s tests`. Regression tests execute both CLIs in
subprocesses and check label isolation, failed-grade Coverage, Validation exports
and that importing evaluation does not load optimization modules.

## Install

Requires Python 3.11+. From the repository root, on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[data]"
```

On macOS/Linux, activate the environment with `source .venv/bin/activate` instead.
Pydantic v2 is the core dependency. The `data` extra adds openpyxl for the workbook
adapter and full test suite.

Tests and mock examples do not need Codex CLI. Live runs currently use the
[Codex CLI provider](docs/codex-provider.md), authenticated through ChatGPT:

```console
npm install -g @openai/codex
codex login
codex login status
```

No OpenAI API key is required. The documented local CLI version was 0.166.0;
compatibility with another version must be checked. The provider uses
noninteractive, read-only, ephemeral execution, sends task data through stdin and
reads the separate final response file. Agents validate the returned JSON; invalid
output is not repaired or retried. An API-based provider is not implemented, but
another runtime can implement `LLMProvider.generate(request: str) -> str`.

To check the real Codex CLI provider with a synthetic grading example:

```console
python -m krugradeai.smoke_codex --live
```

This makes a live request; it is not a real-dataset evaluation.

## Run the Development loop

Start with the supplied synthetic example:

```console
python -m krugradeai.run_optimization --config examples/optimization/synthetic.json --dry-run
python -m krugradeai.run_optimization --config examples/optimization/synthetic.json
```

The first command validates the inputs and prints a call reservation. The second
runs the existing Grader–Reflector–Refiner loop with scripted mock providers. It
uses four synthetic records across two rubric paths; it does not measure grading
accuracy. The mock runtime accepts only the supplied fixtures.

For custom data, copy the
[Codex configuration template](examples/optimization/custom-codex.example.json),
replace its model and file-path placeholders, then run:

```console
python -m krugradeai.run_optimization --config PATH_TO_CONFIG.json --dry-run
python -m krugradeai.run_optimization --config PATH_TO_CONFIG.json --live
```

The second command makes live requests. Use a compatible installed CLI and choose
an explicit model before running it. Configuration supports `development`,
`output_directory`, `runtime`, `model`, `executable`, `timeout_seconds`,
`max_iterations`, `max_calls` and `initial_rules`. File paths resolve relative to
that configuration file, not the shell's working directory. Unknown fields are
rejected. This runner allows 1–10 iterations and reserves at most
`iterations × (2 × Development records + rubric paths)` provider calls. All three
agents share the configured call cap.

The output directory must be new. It contains:

| File | Contents |
| --- | --- |
| `configuration.json` | Resolved run configuration |
| `optimization.json` | Rule versions, grades, reflections, failures and review history |
| `summary.json` | Path stop reasons, latest versions and provider-call count |
| `status.json` | Execution state, last stage and call count; sanitized error type on failure |

A failed or interrupted run retains its partial files and cannot reuse the same
output path. Inspect those files before starting a separate run; there is no
resume or automatic retry. Agent failures or unresolved review outcomes produce
an `incomplete` status and a nonzero CLI exit. If storage itself fails, even the
status file may be missing or stale. Keep outputs under ignored `artifacts/`:
optimization history can contain private answers and labels.

**This entry point optimizes Development only.** It does not load Validation or
Held-out data, select best guidelines or enable AI adjudication. The latest
candidate is a proposal, not evidence of improvement.

## Data and rubrics

For a complete private source inventory, use the opt-in exhaustive workflow in
[full dataset preparation](docs/full_dataset_preparation.md). It retains every
physical source row and its disposition, keeps unknown label provenance separate,
and exercises both independent pipelines offline without the historical selection
policy. The historical `real_data` preparation remains unchanged for replay.
Use the documented `corrected_dataset` / `corrected_audit` workflow for the
owner-clarified TA-score and OCR-review provenance and the three-way split.

The portable runner reads UTF-8 JSONL with one `DevelopmentRecord` per line.
Each record contains `id`, `split: "development"`, `question`, `rubric`, `response`,
`ta_output` and optional `provenance`. See the
[synthetic records](examples/optimization/synthetic-development.jsonl) and
[full input format](docs/reusable_optimization.md).

Questions have an ID, text and optional reference solution. Rubrics have a stable
ID, uniquely identified criteria with descriptions and positive maxima, and a
`total_score` equal to their sum. Response `question_id` must match the question.
Use decimal strings for scores. A rubric path must refer to the same question and
rubric throughout a run. Custom subjects can use these schemas without changing
core source code; the existing workbook/ZIP adapter is examination-specific.

`ta_output` is the legacy name for reference scores, not proof that a TA supplied
them. It accepts totals with complete, partial or absent criterion labels. Retain
real label provenance as `human_derived` or `ai_derived`; AI labels are operational
pseudo-ground truth, not human verification. Initial adaptation rules must be
compatible with every path in the configuration.

Known record, response and source/item duplicates are rejected. For new datasets,
group related attempts and derived answers before splitting. Generic records do
not retain the real adapter's full attempt/ancestry metadata, so different IDs
cannot establish independence. Matching answer text alone does not prove overlap.
Do not upload private input files, credentials or generated history.

## Validation and full experiments

The Python `OptimizationLoop.run(..., validation_records=...)` interface supports
candidate evaluation and selection. It evaluates G0 and changed candidates on a
fixed Validation population, requires complete comparison coverage, and retains
the earlier eligible version on ties or regressions. Patience and minimum-delta
settings control stopping. Selected rules are stored in
`path.validation.best_rules` and `best_version`, separately from `final_version`.
See [optimization](docs/optimization.md) and [Validation](docs/validation.md).

The separate `optimize_real` runner handles the existing source-specific experiment:
95 Development records, 19 human-derived Validation records and eight rubric paths.
Its opt-in AI adjudication performs an independent score-blind assessment before
label comparison. Supported results remain AI evidence; unresolved records are
excluded from refinement. With the private prepared data and criterion supplement
available, inspect its single-iteration plan with:

```console
python -m krugradeai.optimize_real --full-development --ai-adjudication --dry-run --max-calls 332
```

This command cannot run from a source-only checkout without those inputs. See
[full-run configuration](docs/full_real_optimization.md) and
[data preparation](docs/real_data_preparation.md). Do not overwrite existing splits
or treat the convenience runner as a replacement for that experiment's policies.

`final_evaluation.py` freezes Validation-selected rules before final grading and
compares G0 with that selection on the same Held-out records. It checks source,
split and code hashes and claims a once-only execution receipt. Held-out feedback
never enters refinement or selection. Its `--prepare`, `--dry-run` and `--live`
modes require the corresponding completed experiment inputs; use
`python -m krugradeai.final_evaluation --help` for arguments. Do not reset receipts
or rerun a consumed experiment under another ID.

Private datasets, frozen guidelines, official reports and real-exam prompt exports
are not included in this repository. The separate full synthetic demonstration is
available through `python -m krugradeai.demo --help`; its default runtime is mocked.
Synthetic results are software checks, not measurements of real grading accuracy.

## Export grading instructions

Given a frozen selection and its matching private source evidence:

```console
python -m krugradeai.prompt_export --frozen PATH_TO_FROZEN_GUIDELINES.json --output-dir exports/new-prompts
```

The exporter validates the selection and uses Development question/reference text.
It writes per-scope JSON definitions, readable Grader request templates, Codex stdin
templates and a hash manifest. It preserves the normalized rubric and selected rules,
exports no actual student answers or reference scores, and refuses overwrite.

These are task-text templates, not native system-role messages. `PromptDefinition`
and `render_prompt` can assemble a new typed request without running the optimizer.
Pass the plain request to the provider; it adds its own runtime envelope. The text
alone does not enforce sandboxing or score validation. See
[prompt export](docs/final_prompt_export.md) for details. The historical Q1–Q8 bundle
is excluded; regeneration also needs the private artifact paths and hashes to match.
Confirm redistribution rights before sharing exam questions or reference solutions.

## Tests and metrics

```console
python -m unittest discover -s tests -v
```

The suite uses synthetic fixtures and mocked providers/subprocesses. It does not
run live grading or require the private experiment data. Tests cover contracts,
agent boundaries, orchestration, selection, exports and failure handling.

Evaluation reports MAE in rubric points, normalized MAE as the mean of each
absolute error divided by its rubric maximum, Exact Match as the fraction of
exactly matching totals, and coverage as scored/expected records. Error and Exact
Match use successful grades; review/failure stays in the coverage denominator and
is never assigned a zero score. QWK is available only when explicitly configured
ordinal categories and comparable score scales permit it. Real-data reporting
separates human-only, AI-only and mixed cohorts.

The previous Phase 1 documentation reports the following real experiment results.
The supporting JSON reports are private and not shipped here, so these figures
cannot be independently verified from this checkout:

| Measure | Reported result |
| --- | --- |
| Development / Validation / Held-out | 95 / 19 / 24 records |
| Development labels | 76 AI-derived, 19 human-derived |
| Held-out labels | 19 human-derived, 5 AI-derived |
| Full optimization | 178 provider calls; G1 evaluated for Q1/Q4/Q5/Q7/Q8 |
| Selected guidelines | G1 for Q5/Q7; G0 elsewhere |
| Q5 Validation MAE | 1.250 → 1.000 on two records |
| Q7 Validation MAE | 0.625 → 0.375 on two records |
| Mixed Held-out MAE | 0.500 → 0.500 |
| Official Held-out coverage | 22/24 (91.67%); incomplete |

No observed Held-out improvement is established. The historical optimization model
is unknown; Held-out grading reportedly used `gpt-6.1-sol`. Two Q1 review record IDs
could not be recovered, so supplementary adjudication was not performed. The
suggested relationship between “351+ raw items” and 138 active records is unverified.
Small grouped samples, uncertain transcription/label provenance and Validation
reuse limit conclusions about generalization. AI-label agreement is not human
grading accuracy. The original legacy system prompt is still missing, so the
baseline-comparison module has no verified real legacy-prompt result.

## Live model routing

The `models` object in the synthetic configs specifies all six routes. Grader, Validation, independent evaluation and held-out Test default to `gpt-6-luna`. Reflector and Refiner default to `gpt-6-astra`. Set evaluation `role` to `heldout_grader` for Test. A legacy `model` value overrides only the Grader for that pipeline. Each live invocation passes an explicit CLI model. The private `provider-invocations.jsonl` records role, requested ID, reported effective ID and invocation status. A successful Codex CLI turn records verification_status=requested_only. A missing effective model is recorded as null. The adapter checks turn completion, CLI exit status, reported routing conflicts and the output contract before returning a result. Exact CLI arguments are logged privately; prompts and credentials travel outside the argument list.

### Exploratory AI-assisted review

Set `review_policy` to `ai_assisted_pilot` in a Codex CLI optimization config to enable this experiment. The default is `strict`. A Grader review can trigger one independent Astra assessment of the answer, rubric, rules and review reason. A provisional score is labeled `ai_adjudicated` in the private review log and grade feedback. Abstention preserves the unresolved review. This applies to both Development and Validation grading. Any resulting Validation metric measures the assisted grading process.

A Development Reflector review can trigger one Astra diagnostic pass with the original findings and labeled Development evidence. The existing reflector checks must accept the quotes, arithmetic and criterion evidence before a supported adaptation can reach Refiner. The pilot follows the existing stop-path policy on unresolved Development reviews; an unresolved Validation grade reduces coverage. Each extra invocation charges the shared call budget. The private `assisted-reviews.jsonl` records the original decision and adjudication outcome. Strict and assisted experiment results should be identified by policy when reporting them.

### Standard experiment execution and reporting

The live optimizer claims a fresh directory under Git-ignored `artifacts/` and assigns an experiment ID. Dry-run the exact config first. Preflight reserves all ordinary calls, optional malformed-output retries, and one possible assessment or diagnostic per eligible review. Demand above `max_calls` rejects both dry-run and live execution. All agents share that cap. For subscription runs both CLI retry settings are fixed at zero. Set `retry_malformed_output: false` to disable the separate application retry. An unresolved review, invalid agent output after any permitted retry, provider error, or budget failure stops live execution. Inspect the fresh directory before starting another experiment.

`validation-history.jsonl` checkpoints each evaluated scope/version. The final `summary.json` includes generation history, patience and selected versions. Each breakdown reports the fixed population, label origins, direct-only and adjudicated-only coverage and MAE, and combined-policy MAE. MAE denominators count successful predictions in the corresponding cohort. Coverage uses the whole declared population. A direct-only MAE on a changing subset cannot substitute for selection MAE. Compare versions under the same review policy on the same original-TA Validation records, requiring complete coverage. The earliest minimum wins. Report any differences in adjudication incidence even when coverage is equal.

Use three iterations, patience two and minimum delta zero for the full experiment. Candidates derive from the latest rules. Export the selected best for covered scopes. Development-only proposals have no selected Validation version; use G0 for formal Test until compatible independent Validation evidence is available. Independent evaluation reports the same prediction breakdown and performs no selection. Supply `guideline_version` when evaluating a known version.
