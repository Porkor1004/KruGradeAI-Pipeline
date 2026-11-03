# KruGradeAI Phase 1 architecture

This document records incremental implementation history. The Step 1–9 sections
below describe the scope at each historical step, not current missing features.
The current implementation includes concrete agents, real-data preparation,
optimization, human-only Validation and frozen final evaluation. Start with the
[README](../README.md) and [final experiment report](phase1_experiment_report.md)
for current scope, commands, verified outcomes and unresolved limitations.

At Step 1, contracts alone were implemented.

## Structure

- `krugradeai/models.py`: frozen Pydantic v2 data contracts and validation.
- `krugradeai/agents.py`: three independent abstract interfaces: Grader,
  Reflector, Refiner.
- `krugradeai/providers.py`: abstract runtime-neutral LLMProvider.
- `tests/test_contracts.py`: synthetic, dataset-independent unit tests.

## Data flow and trust boundaries

The intended later workflow is Grader → Reflector → Refiner → validation → early
stopping → best guidelines → held-out test. Only the first three interfaces and
supporting records exist now. No additional agents are defined.

GraderInput contains a question, original rubric, student response, and adaptation
rules. It has no TA-label field. All contracts reject unknown fields, including
nested input fields. TA scores can optionally enter ReflectorInput as ta_output;
they are not required, allowing later work on questions without labeled examples.
RefinerInput contains only current rules and reflector observations. RefinerOutput
is an alias for AdaptationRules, and Refiner.refine returns that type directly.
There is no rubric in its return schema.

These are data contracts, not a security sandbox: arbitrary text could contain
labels or instructions, and Python type annotations do not enforce future agent
implementations at runtime. Future orchestration must validate returned data at
its boundaries and must not put reference labels into Grader text fields.

## Immutability and scoring

All models use frozen=True, extra='forbid', and revalidate_instances='always'.
Rubrics and their criteria use tuples of frozen models, preventing ordinary
nested mutation. There are no mutable mapping or list fields. Changed adaptation
rules are new values; rubrics remain fixed throughout a run. Deliberate Python
validation bypasses such as model_construct or unvalidated model_copy(update=...)
are not supported input paths; use constructors/model_validate for new values.

Every rubric has one or more uniquely identified criteria, positive criterion
maxima, and a total equal to their sum. Scores are finite nonnegative Decimal
values, allowing partial credit without binary float summation tolerances. Pass
decimal strings for exact external values. Each score requires a rationale.
Student response text may be empty to represent a blank submission.

GraderOutput checks unique criterion IDs and exact total consistency on its own.
GradingRecord binds output to the original GraderInput and validates exact
criterion coverage and per-criterion maxima. Consumers must construct a validated
GradingRecord before accepting or persisting a grading result. ReflectorInput also
validates any TA output against that same rubric. A standalone output has no
rubric context and cannot check maxima or criterion membership by itself.

## Agent and provider interfaces

Grader.grade, Reflector.reflect, and Refiner.refine are synchronous abstract
methods with typed inputs/outputs. None inherits from another agent. No concrete
agent exists. Future implementations may receive an LLMProvider through dependency
injection; no runtime or provider is selected today.

LLMProvider.generate accepts serialized text and returns raw text. Future Codex
CLI and API adapters own transport, authentication, and transport errors; agents
will own serialization and schema validation. No API key or Codex installation is
needed for this step. The minimal text boundary avoids tying domain contracts to
a vendor SDK.

## Setup and tests

Requires Python 3.11+ and Pydantic v2. From the project root:

```console
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m unittest discover -s tests -v
```

With dependencies already installed, run:

```console
python -m unittest discover -s tests -v
```

Tests use the standard-library unittest runner and synthetic fixtures only. They
cover schema round trips, nested immutability, fractional credit, totals,
boundaries, criterion coverage, TA separation, refiner restrictions, and abstract
method signatures. They never access data/raw/. That directory is reserved and
must remain untouched. Package discovery includes only krugradeai packages.

## Step 2 extension

The provider-backed Grader is now implemented in `krugradeai/grader.py`, with a
reusable prompt in `prompts.py` and deterministic `mock_provider.py`. The Step 1
contract description above documents the initial baseline; see [grader.md](grader.md)
for the current implementation, additive fields, human-review exception protocol,
validation boundaries and limitations. Reflector and Refiner remain abstract.

## Step 3 extension

`krugradeai/reflector.py` now implements the provider-backed Reflector and
`reflector_prompt.py` provides its prompt. `TAReference` adds total-only and partial
criterion labels without removing legacy complete TA outputs. ReflectorOutput
adds signed differences, evidence-backed findings, suggestions and review fields;
legacy observation-only objects remain valid. The concrete implementation
requires TA labels and a complete structured response. See [reflector.md](reflector.md)
for the current contracts and validation limits. The Grader and provider interfaces
are unchanged, and Refiner remains abstract. The earlier sections describe the
Step 1 baseline and Step 2 changes, rather than the current implementation scope.

## Step 4 extension

The three existing agent interfaces now have concrete provider-backed implementations.
`krugradeai/refiner.py` implements LLMRefiner; `refiner_prompt.py` holds its prompt
and `rule_validation.py` contains explicit text guards. RefinerOutput remains
AdaptationRules only. Structured AdaptationRule objects add stable IDs, criterion
references and rationale while legacy string rules continue to work. RefinerInput
adds optional rubric context (required by LLMRefiner) and a tuple of additional
Reflector outputs. Candidates are never applied automatically.

The Refiner validates references and output structure, requires no change for
weak/unresolved findings, checks protected input integrity around the provider
call and rejects detectable overrides and copied answer content. See
[refiner.md](refiner.md) for the current contracts, tests and heuristic limitations.
Existing Grader, Reflector, provider and mock implementations remain unchanged.
No optimization loop, metrics, early stopping or dataset integration is present.

## Step 5 extension

`krugradeai/codex_provider.py` adds an opt-in CodexCLIProvider behind the unchanged
LLMProvider interface. It preserves installed PATH precedence (verified local CLI
0.166.0), runs noninteractive read-only ephemeral sessions in an empty temporary
working directory, sends prompts by stdin and reads only the separate final
response file. Compatible Pydantic models generate output-schema files; the
Grader's scored/review protocol retains explicit no-schema mode. All inference
subprocesses are mocked in tests. See [codex-provider.md](codex-provider.md) for
Windows resolution, cleanup, errors, schema compatibility and test instructions.
No existing agent or domain contract changed, and no live inference or work beyond
Step 5 was performed. The earlier runtime-integration statements describe the
historical Step 1–4 baseline.

The completed Step 5 configuration also accepts an optional model identifier and
explicit existing working directory. The provider wraps task data in a runtime
policy, strips API-key environment variables, and classifies authentication and
usage-limit failures into sanitized exceptions. `python -m krugradeai.smoke_codex
--live` is a separate opt-in synthetic smoke test; it is never invoked by the unit
suite or application imports. No live inference was performed during development.

## Step 6 extension

`optimization.py` connects the existing Grader, Reflector and Refiner interfaces
in a development-only sequential loop. `optimization_models.py` defines frozen
development records, rule versions and iteration history. Each rubric has an
independent path, with explicit failures/review flags and operational stopping
conditions. Changed candidates feed the next iteration; the latest version is
never labeled best, and a final candidate may remain unevaluated. Refiner's
existing candidate validation is now exposed as the shared pure helper
validate_candidate_rules without changing its behavior or interface.

See [optimization.md](optimization.md) for inputs, path grouping, dependency
injection, version tracking, error policy and test instructions. No metrics,
validation-based selection, held-out evaluation or live calls were added. Earlier
statements about the absence of orchestration describe the Step 1–5 baseline.

## Step 7 extension

The existing optimization loop now accepts a separate validation_records collection
and ValidationConfig. Baseline and every changed candidate are evaluated by the
Grader and deterministic metrics only; no validation content or per-record results
enter the Reflector or Refiner. Identity/provenance overlap is rejected before calls.
Independent rubric paths select their own best eligible MAE/normalized-MAE version,
with earlier-version tie-breaking and configurable patience/min_delta stopping.
The final proposed version is evaluated even at the iteration limit.

`metrics.py` provides Decimal MAE, normalized MAE, exact match, explicit-scale QWK
and coverage; `validation.py` performs fixed-population evaluation and selection.
Shared request guards moved unchanged to `execution.py` for development/validation
reuse. Frozen history models now include validation evaluations, exact selected
rules/version and transparent failures/reviews. Complete comparison coverage is
required; a restricted population must be declared up front. No valid evaluation
means explicit null selection, not a fallback to unevaluated rules.

The path's Step 6 final_version still identifies the latest proposal; the selected
rules are path.validation.best_rules/best_version. Omitting validation preserves
Step 6 behavior. See [validation.md](validation.md) for metrics conventions, split
boundaries, result fields and tests. No held-out final testing, new agent, second
optimization loop, live call or real dataset integration has been added.

## Step 8 extension

`experiment.py` runs the existing optimizer once with development/validation data,
then locks selection and performs one final baseline/selected held-out comparison.
`experiment_models.py` adds test-split records, partition/provenance integrity,
experiment results and versioned artifact contracts. A persistent exclusive-create
receipt prevents repeated use of an experiment ID within its ledger namespace.
The existing deterministic grading/metrics kernel is shared with validation;
held-out scores never enter selection, reflection or refinement.

`synthetic.py` provides explicitly synthetic multi-criterion fixtures and scripted
providers for the existing agents. `demo.py` runs the complete mock workflow by
default; Codex integration is a bounded --live opt-in. `guidelines_export.py` writes
selected rules with exact version, rubric identity/digest, metrics, configuration
and synthetic/real provenance, keeping full rubric definitions separate.
`artifact_paths.py` guards generated output locations. See [end-to-end.md](end-to-end.md)
for commands, one-shot guarantees, export fields and limitations. No actual dataset
or live inference was used to implement this step. Earlier absence-of-test-evaluation
statements describe the Step 1–7 baseline.

## Step 9 extension

`live_integration.py` prepares an explicitly --live opt-in, one-iteration integration
using one synthetic record per split and initial adaptation guidance. All three
existing agents are wired directly to CodexCLIProvider. Invocation instrumentation
and `integration_models.py` produce a sanitized report under artifacts/live-synthetic,
including candidate/selection metrics, final held-out coverage, failures and reviews.

An observational boundary delegates to the unchanged optimizer once. Skipped
Reflector/Refiner contracts can be probed on genuine development data only, before
held-out grading; probe proposals are never adopted or evaluated. Upstream failures
are not retried or fabricated into downstream inputs. The existing experiment
selection and one-shot held-out guard remain in force. See [live-integration.md](live-integration.md)
for the exact optional command, agreement behavior, report fields and offline tests.
No live Codex calls were made while implementing this step.

## Step 9.2 extension

The existing live integration runner adds an opt-in --optimization-path mode. It
uses the unchanged optimizer directly, with no diagnostic probes, and a fixed
synthetic consequential-error fixture in live_optimization_fixture.py. Reference
scores are justified from explicit immutable criteria and actual algebra; agreement
is not treated as a failure and no G1 is forced. Reports add stage execution kinds
and separate proposal, accepted rules, validation attempt/eligibility and best
selection. Proposal telemetry does not alter responses or weaken validation.
Default Step 9.1 contract-coverage behavior remains unchanged. See
[live-optimization-path.md](live-optimization-path.md) for commands and limitations.
No live calls were made during this implementation.
# Real mixed-source preparation

`real_data.py` is a deterministic, source-specific read-only adapter and preparation CLI. It does not import agents or providers. Frozen record provenance optionally records human/AI label origin; this does not change optimization algorithms. Complete reviewed subquestion covers map to unchanged canonical parent rubrics. Connected attempt/source ancestry is split before conversion to the existing Development, Validation and Heldout contracts. Validation is human-only; held-out reporting separates mixed, human and AI cohorts. See [real-data preparation](real_data_preparation.md) for counts, lineage, exclusions and execution boundaries. Preparation does not run the experiment.

## Real Development pilot

`pilot_real.py` adds an explicit `--live` diagnostic runner on four to eight normalized Development records. It reuses the existing Codex-backed agents and a single optimizer iteration without Validation or Held-out inputs. A timing provider wrapper records real latency; candidate rules are never exported or promoted. Existing total-only evidence guards remain intact. See [pilot instructions](real_data_pilot.md).

## Source-provided Development criterion labels

`development_criteria.py` optionally extracts audited human source-unit scores and explicit AI point-score maps into a separate Development artifact. Frozen per-score provenance distinguishes operational pseudo-ground truth from human-derived references. The default prepared split and algorithms remain unchanged. Reflector/Refiner prompts permit supported findings relative to opted-in operational targets while retaining rubric, evidence and review guards. See [policy and extraction instructions](mixed_source_criterion_labels.md).

## Bounded real optimization

`optimize_real.py` uses the existing optimization engine with source-enriched
Development and matching human-only Validation paths. No held-out partition is
loaded. One iteration and a reserved/enforced provider budget bound the run.
An explicit single-iteration `exclude_reviewed_records` policy can exclude
reviewed records and their reflections while refining from independent supported
evidence; default path-level review blocking is unchanged. Refiner and rubric
guards remain authoritative. Reports retain source-separated metrics, exclusions,
safe diagnostics and exact selected versions. See [first real optimization](first_real_optimization.md).

The real runner's explicit `--ai-adjudication` option adds a Development-only
policy adapter around Reflector. `adjudicator.py` and `adjudicator_prompt.py` use
two separate structured provider requests: a label/Grader-blind assessment, then
comparison against the locked result and original operational labels. Supported
rubric-consistent evidence is marked AI-adjudicated before Refiner; unresolved
records retain review status. The shared provider-call cap includes both phases.
No labels, rubrics, optimization algorithms, Validation policy or held-out
boundaries are changed. AI adjudication is never represented as human verification.
Policy `ai-adjudication-v2-independent-rubric` permits independently supported
rubric-based Grader errors even when the original source label is also disputed.
The independent score must contradict Grader, but need not agree with the source;
original labels and AI-minus-source arithmetic/metrics remain unchanged. A bad
label alone cannot justify refinement. See the [Optimization 002 investigation](optimization002_adjudication_investigation.md).

The existing real runner's `--full-development` mode processes all prepared
Development records in deterministic rubric/record order, including earlier pilot
records and total-only examples. It uses the existing single-iteration evidence
exclusion policy, aggregates supported path findings and evaluates G0/G1 on the
complete matching human Validation populations. `real_run_plan.py` supplies
data-only reservations/runtime estimates for `--dry-run`; all live roles share the
explicit call budget. Full runs write a pre-call plan and safe invocation journal.
Resume/replay is deliberately unsupported. See [full real optimization](full_real_optimization.md).

The final policy audit clarifies that supported operational-target interpretation
findings need specific rubric/student evidence, not proof of a uniquely correct
human-verified score. Demonstrated source-inconsistency claims require exact source
label and rubric citations; unresolved hypotheses remain excluded. Provider-backed
Reflector/Refiner integration and human-only Validation selection are verified with
synthetic tests. See [final verification and real source coordinates](final_reflector_refiner_verification.md).

`final_evaluation.py` freezes a completed real optimization selection before any
held-out records are read. It reuses the existing Grader, labeled-record evaluator,
source-separated metrics, final-path contracts and guidelines artifact schema;
it never invokes optimization agents. A fixed experiment/test receipt prevents
duplicate attempts across run IDs. The explicit evaluation model is shared by both
passes; the historical optimization effective model remains unknown and is recorded
as such. See [final held-out evaluation](final_heldout_evaluation.md).

