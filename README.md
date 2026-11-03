# KruGradeAI Phase 1

KruGradeAI studies whether rubric-preserving adaptation rules can improve AI
grading agreement without fine-tuning. The pipeline is **Grader → Reflector →
Refiner → candidate Validation → best-guidelines selection → frozen Held-out
Evaluation**. A successful software workflow does not establish grading gains
or transfer to new questions.

## Architecture

- Frozen Pydantic models validate criterion scores, partial credit, totals and
  rubric identity. Grader inputs exclude reference labels.
- Provider-backed Grader, Reflector and Refiner use the runtime-neutral
  `LLMProvider`. Refiner returns adaptation rules only, never edited rubrics.
- `optimization.py` manages independent rubric paths, discrepancy evidence,
  explicit failures/reviews and iteration limits. Human-only Validation selects
  eligible candidates; ties retain the earlier version and incomplete coverage
  cannot qualify a candidate.
- Optional Development-only AI adjudication uses an independent score-blind
  assessment before comparison. Its findings are experimental AI evidence,
  not human verification. Unresolved cases remain excluded.
- `final_evaluation.py` freezes exact selected rules and checks split, source and
  code hashes before final evaluation. Held-out feedback never reaches Refiner.

See [architecture](docs/architecture.md), [experiment report](docs/phase1_experiment_report.md)
and [metrics summary](docs/phase1_evaluation_metrics_summary.md).

For developer reuse, start with [portable three-agent optimization](docs/reusable_optimization.md)
and [final prompt export](docs/final_prompt_export.md). These additions leave the
existing Evaluation Pipeline unchanged. Git initialization, push and PR creation
remain deferred until the destination repository URL is supplied.

## Installation and configuration

Requires Python 3.11+. From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[data]"
```

Pydantic v2 is the core dependency. The `data` extra installs openpyxl, required
for the workbook adapter and full test suite. No API key is required. Live
execution needs an installed compatible Codex CLI authenticated with ChatGPT
(`codex login`, `codex login status`). The locally verified CLI was 0.166.0;
other versions must pass the provider's option checks. Model, timeout and executable
are configurable on supported runners; select an explicit model for reproducibility.

The provider uses noninteractive read-only, ephemeral CLI execution, stdin task
data and separate final-output files. Compatible output schemas and local Pydantic
checks enforce structured responses. Prompts are task data, not permission to
execute embedded instructions. See [provider configuration](docs/codex-provider.md).

## Dataset/rubric format and portable optimization configuration

Custom datasets can use the existing `DevelopmentRecord` JSONL schema, one object
per line, with `id`, `split: "development"`, `question`, `rubric`, `response`,
`ta_output` and optional `provenance`. A rubric has a stable `id`, `criteria`
(`id`, `description`, `max_score`) and `total_score`. Questions contain `id`, `text`
and optional `reference_solution`; response `question_id` must match. The legacy
`ta_output` field accepts total-only or actual source-provided criterion scores;
it does not certify human verification. Preserve `provenance.label_origin` as
`human_derived` or `ai_derived` for real inputs. Do not invent allocations.

See the [complete format and boundaries](docs/reusable_optimization.md) and the
fully synthetic [sample records](examples/optimization/synthetic-development.jsonl).
No source-specific adapter changes are required for schema-compatible custom
questions/rubrics. Group related attempts before splitting; the generic format
cannot independently recover unknown ancestry or verify source labels.

Run only the existing three-agent loop with the safe sample configuration:

```powershell
python -m krugradeai.run_optimization --config examples/optimization/synthetic.json --dry-run
python -m krugradeai.run_optimization --config examples/optimization/synthetic.json
```

This is a scripted mock example, not live grading. Configuration controls the
input/output paths, runtime, model, timeout, executable, maximum iterations/calls
and initial adaptation rules. Paths resolve relative to the config file, not the
developer's machine or working directory. For custom live inputs, copy the
[configuration template](examples/optimization/custom-codex.example.json), replace
its placeholders, validate with `--dry-run` and use `--live` only after intentional
authorization. A shared call cap bounds all three agents; no retries/resume or
overwriting prior outputs is supported.

Outputs are `configuration.json`, `optimization.json` and `summary.json` under
the configured fresh output directory. History can contain private Development
answers/labels; keep it under ignored `artifacts/`. This entry point does not load
Validation/Test or call evaluation. Its latest proposal is **not** a best-guidelines
claim. Existing Python/real-runner Validation selection remains available unchanged.

## Export and independently use final grading prompts

The [final export bundle](exports/final-grader-prompts/manifest.json) contains Q1–Q8
definitions and readable instruction templates. Q5/Q7 use the exact frozen G1;
all other paths retain G0. No rules were regenerated. Each question has:

- `Qn.grader.txt`: exact Grader request template with question/reference, immutable
  rubric and selected rules, plus an explicitly marked answer placeholder.
- `Qn.codex-stdin.txt`: exact textual Codex runtime envelope around that template.
- `Qn.json`: portable question/rubric/rule definition without answers or labels.

**These are task-text instructions, not verified native system messages.** The
current provider sends one stdin string to `codex exec`; its runtime policy and
Grader prompt are text layers. Replace the response through the existing typed
input rather than treating the placeholder as an answer. `PromptDefinition` and
`render_prompt` support standalone use without the optimizer/private dataset.
Do not pass an already wrapped stdin template to the provider and wrap it twice.

To export again from the existing private frozen/source evidence into a fresh
directory, without LLM calls:

```powershell
python -m krugradeai.prompt_export --frozen artifacts/real-final-evaluation/frozen-guidelines-full-001.json --output-dir exports/final-grader-prompts-new
```

See [export details and standalone usage](docs/final_prompt_export.md). Hashes verify
exact normalized frozen rubric/rule content; no original student answers, scores
or Validation findings are exported. Original exam/reference/rubric redistribution
rights still require confirmation before GitHub publication. The exported model
instructions alone do not enforce sandboxing or local score validation.

## Tests and offline demonstration

```powershell
python -m unittest discover -s tests -v
```

Tests use synthetic fixtures and mocked subprocess/providers, with no live LLM
calls. A separate deterministic synthetic demonstration is available:

```powershell
python -m krugradeai.demo --experiment-id synthetic-new-demo --output-dir artifacts/synthetic-new-demo
```

Use a fresh declared synthetic experiment/output location; execution receipts
prevent accidental repeats. Synthetic metrics are software checks, not real accuracy.
See [demo instructions](docs/end-to-end.md).

## Real data and optimization

Original files in `data/raw/` and all generated `artifacts/` remain local and are
excluded from version control. Do not upload student data, local credentials or
execution transcripts. A fresh checkout does not contain prepared real records,
official JSON evidence or frozen guidelines. Obtain these through approved private
storage; do not fabricate replacements. Documentation links to local artifacts
require that private evidence bundle. Source hashes and lineage must match.

The deterministic adapter uses confirmed question/rubric scopes, groups related
attempts before splitting, preserves label provenance and quarantines unresolved
inputs. See [dataset audit](docs/real_dataset_audit.md), [scope reconciliation](docs/question_scope_reconciliation.md)
and [preparation](docs/real_data_preparation.md). Existing prepared outputs must not
be overwritten or resplit for reproducing this experiment.

Inspect a full optimization plan without providers:

```powershell
python -m krugradeai.optimize_real --full-development --ai-adjudication --dry-run --max-calls 332
```

For a separately authorized new experiment, the live command shape is:

```powershell
python -m krugradeai.optimize_real --live --full-development --ai-adjudication --model MODEL_ID --run-id NEW_EXPERIMENT_ID --max-calls 332 --timeout 120
```

Replace placeholders before use. This does not resume the completed experiment.
One iteration uses all 95 Development records and 19 human Validation records,
with a shared maximum of 332 provider calls when adjudication is enabled. Do not
rerun the completed experiment as submission preparation. See [full-run safeguards](docs/full_real_optimization.md).

## Frozen final evaluation and outputs

The official evaluation has already consumed its once-only attempt. Inspect the
frozen plan without grading:

```powershell
python -m krugradeai.final_evaluation --dry-run
```

For an independently authorized, previously unconsumed experiment, preparation
and execution use `--prepare` and `--live --model MODEL_ID`, respectively, with
explicit report/data/frozen inputs. See [final evaluation](docs/final_heldout_evaluation.md)
for the full command protocol. **Do not rerun `heldout-full-001`, reset receipts,
or reuse its test feedback for tuning.** Changing a run ID does not authorize
repeating the same consumed final evaluation.

Local output locations include:

| Location | Contents |
| --- | --- |
| `artifacts/real-mixed-v1/` | Prepared partitions, manifest and quarantine |
| `artifacts/development-criteria-v1/` | Opt-in criterion supplement and provenance |
| `artifacts/real-optimization/real-optimization-full-001/` | Completed optimization report, plan and call journal |
| `artifacts/real-final-evaluation/frozen-guidelines-full-001.json` | Exact frozen rules and integrity metadata |
| `artifacts/real-final-evaluation/heldout-full-001/` | Official final report and guideline exports |

Metrics include MAE, per-record normalized MAE, Exact Match and coverage. Error
metrics use scored cases; review/failure remains in the coverage denominator and
is never imputed as zero. Human-only, AI-only and mixed metrics are separate.

## Verified results and limitations

- Development: 95 (76 AI-derived, 19 human-derived); Validation: 19 human-derived;
  Held-out: 24 (19 human-derived, 5 AI-derived), across eight rubric paths.
- Full optimization used 178 provider calls. G1 was evaluated for Q1/Q4/Q5/Q7/Q8;
  human Validation selected G1 only for Q5 and Q7. Their MAE changed from
  1.250 → 1.000 and 0.625 → 0.375, respectively, on two examples per path.
- Official Held-out status is **incomplete**: 22/24 scored (91.67%). G0 and frozen
  best rules have identical aggregate mixed MAE, 0.500 → 0.500. No held-out gain
  or statistical equivalence is established.
- Historical optimization effective model is **unknown** (0/178 verified).
  Held-out explicitly requested `gpt-6.1-sol`; historical model matching is unverified.
- Two Q1 review IDs could not be recovered; supplementary adjudication was stopped.
  No 24/24 AI-assisted result exists. See [recovery audit](docs/supplementary_q1_recovery_audit.md).
- The proposed relationship between “351+ raw items” and 138 active records is
  **not verified**. Raw rows, augmented variants and assembled parent records have
  different units; do not present them as a proven attrition chain.
- AI-derived labels are operational pseudo-ground truth, not human verification.
  Small, grouped samples, transcription uncertainty and Validation reuse limit
  generalization. Raw-source review authentication is not independently established.
- Legacy-prompt comparison infrastructure is prepared, but the actual original
  prompt is missing. Phase 1 G0 is not that prompt; no legacy comparison results
  exist. See [comparison status](docs/old_prompt_baseline_comparison.md).

## Submission status

See [repository submission audit](docs/submission_readiness_audit.md) for checks,
preserved artifact hashes and unresolved publication blockers. Do not publish
private data or claim experimental limitations have been resolved.
