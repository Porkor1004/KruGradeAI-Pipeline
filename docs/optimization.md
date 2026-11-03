# Development optimization loop (Step 6)

`OptimizationLoop` connects the three existing agent interfaces:

Grader → discrepancy check → Reflector → tuple of reflections → Refiner →
candidate AdaptationRules → next development iteration.

There is no fourth agent, provider selection, live CLI call, held-out test or
fine-tuning. Step 7 adds optional isolated validation, metrics and selection to
this same loop; see [validation.md](validation.md). The development-only behavior
described below remains the default when validation_records is omitted.

## Interface

```python
from krugradeai.optimization import OptimizationLoop
from krugradeai.models import AdaptationRules

# Inject instances implementing the existing Grader, Reflector and Refiner ABCs.
optimizer = OptimizationLoop(grader=grader, reflector=reflector, refiner=refiner,
                             max_iterations=3)
result = optimizer.run(development_records, initial_rules=AdaptationRules())
```

Agents can use deterministic mock providers today or an explicitly selected
CodexCLIProvider later; orchestration imports no provider or CLI code. A run is
synchronous and processes records in supplied order. History contains no random
IDs or timestamps, so identical inputs and deterministic agent behavior produce
identical serialized results. Agents are responsible for resetting any state they
keep between runs. The loop itself keeps history local to each run.

`DevelopmentRecord` in `optimization_models.py` contains:

- `id`: unique identifier within the run.
- `question`: the existing Question, including optional reference solution.
- `rubric`: the existing immutable Rubric with one or more criteria.
- `response`: the existing StudentResponse matching the question ID.
- `ta_output`: TAReference (total-only or partial/full criterion labels), or the
  legacy complete GraderOutput label format.
- `split`: the literal development; validation/test values are rejected.

A record can be constructed from models or ordinary Pydantic-compatible mappings.
Inputs are validated before processing. TA totals must be feasible under the
rubric and any known criterion labels. Empty batches, duplicate record IDs,
incompatible initial criterion-scoped rules and conflicting rubric definitions
sharing the same rubric ID are rejected before any agent is called.

There is no data loader or splitting algorithm. The API accepts development data
only and has no validation/test inputs. The caller remains responsible for correct
provenance and ensuring that future validation/test examples are never relabeled
as development. A split field cannot prove that datasets are disjoint.

## Rubric paths and versions

The existing RefinerInput has one rubric. A run therefore groups records by rubric
ID and creates an independent path for each distinct immutable rubric. Paths are
processed in first-seen order; records within each path retain input order. No
findings or newly proposed rules transfer between paths. The maximum number of
iterations applies **per rubric path**. One initial rule set is copied to all
paths and must be compatible with each; run separate calls for different initial
sets when needed. There is no automatic cross-question/rubric generalization.

Each path begins at version 0. A changed, validated candidate creates the next
contiguous version with its parent version and creation iteration. It is used for
all records in the next iteration. A no-change response creates no new version.
Rule comparison uses full validated model equality, including ordering, rationale
and representation; the loop does not guess whether different prose is equivalent.

`final_version` means the latest candidate version, **not the best rules**. Every
version is retained. The candidate produced at the final permitted iteration may
not have been graded at all. Inspect history.rules_version to determine which
versions were actually used for grading. No candidate is selected based on
accuracy, validation performance or a claim of improvement.

## Per-iteration behavior

The Grader receives a new GraderInput with the current rules and no TA fields.
Every successful output is validated by GraderOutput and GradingRecord against
the unchanged rubric. Missing or failed grades remain absent; they are never zero.

A record is sent to the Reflector when the total differs from the TA total or any
available TA criterion score differs. This catches offsetting criterion errors
with equal totals. Total-only score agreement cannot reveal hidden criterion
errors, so those cases are not reflected. This is a per-record routing check,
not an evaluation metric or proof of correctness.

ReflectorInput supplies the validated grading record and verified labels only at
that stage. Reflections are schema-validated, numerical assertions are checked
against the labels when supplied, and criterion references are checked against
the rubric. The concrete LLMReflector additionally checks source quotations and
its other contextual invariants. Custom injected implementations must fulfill
those same semantic responsibilities; orchestration does not re-implement all
Reflector analysis or certify evidence truth.

Successful reflections are aggregated as the existing primary reflection plus
additional_reflections tuple. They are not summarized by a model or merged into
new claims. The Refiner receives that batch, the active rules and the same rubric.
Its candidate is checked with the shared `validate_candidate_rules` helper,
exposing its existing contextual safety validation without changing agent method
signatures or behavior. This also checks candidates returned by injected agents.

## Failures, reviews and termination

All records in the current path iteration are attempted so successful results and
failures can be inspected together. Grading/reflection errors are captured by
stage and record ID. Refiner errors are recorded at iteration scope. Failure
messages are fixed; arbitrary exception messages and provider diagnostics are not
copied into history. The exception class name remains available for diagnosis.
HumanReviewRequired exceptions and Reflector review flags preserve explicit
review reasons separately. History contains student/label evidence by design;
callers must treat serialized history as sensitive rather than public logs.

No retry or score/rule repair is performed. If any grade/reflection fails or
requires review, the path stops before refinement. Other rubric paths continue.
A Refiner failure/review likewise stops its path and retains the active version.
This conservative policy avoids adapting from an incomplete or unresolved batch.
If failures and reviews coexist, stop_reason is agent_failure and review flags
remain in history. Errors do not erase successful grades or reflections.

Terminal reasons are:

- `max_iterations`: the configured bound was reached; zero is allowed and makes
  no agent calls, retaining only initial version 0.
- `no_change`: the Refiner returned the current rules unchanged.
- `no_discrepancies`: all records were graded successfully with no observable
  score discrepancies; no Reflector/Refiner call is needed.
- `agent_failure`: an agent call or output validation failed.
- `human_review`: unresolved evidence requires human attention.

Without validation, these are operational path termination conditions. With
Step 7 validation enabled, an additional patience reason indicates validation-based
early stopping. An empty discrepancy set does not establish that the latest rules
are best, or that either the Grader or TA labels are correct.

## Integrity and history

Frozen nested models and tuple fields prevent ordinary mutation. Fresh validated
agent inputs separate caller data from processing. Each call also checks its
input's serialized state before/after, rejecting outputs if an agent bypasses
frozen-model protection and changes its request. This detects request mutation;
it is not a sandbox against arbitrary Python code with unrelated external
references, and it cannot undo deliberate mutation outside supplied requests.

`OptimizationResult.paths` contains each OptimizationPath with rule_versions,
history, final_version and stop_reason. IterationHistory records iteration,
active rules_version, per-record GradingRecord/reflection/discrepancy/review flag,
proposed_candidate, candidate_version, processing failures, review reasons and
any terminal reason. A null candidate denotes no accepted candidate; rejected
arbitrary model text is not retained. Objects are frozen and support Pydantic
JSON serialization/round trips. The loop does not persist files automatically.

## Tests

```console
python -m unittest discover -s tests -v
```

Optimization tests only:

```console
python -m unittest discover -s tests -p test_optimization.py -v
```

All optimization tests use deterministic mock agents or the existing concrete
agents with MockLLMProvider. They cover the full chain, multiple records/criteria,
known-criterion disagreement under equal totals, versions and candidate propagation,
no change, iteration bounds, total-only labels, schema boundaries, immutability,
request mutation detection, failures/review flags, path isolation and deterministic
history. No actual dataset or live LLM is used. data/raw/ remains untouched.

## Optional Step 7 validation

Supply validation_records to run(), and configure validation_config on the existing
OptimizationLoop. For example, ValidationConfig(patience=2, min_delta="0.01",
primary_metric="mae") enables baseline/candidate grading on a fixed separate set.
Use ValidationRecord with split=validation, distinct identities and optional stable
RecordProvenance. Every development rubric path needs matching validation data.

Baseline version zero is evaluated before development iterations. Each changed
candidate is evaluated before max-iteration or patience stopping. Incomplete
comparison coverage makes a version ineligible; metrics over successes remain
visible for diagnosis. Earlier versions win metric ties. Exact best rules/version
are returned under path.validation, while final_version retains its historical
meaning of latest proposal. No-change termination still creates no new version.
See [validation.md](validation.md) for full interfaces, fixed subsets, QWK handling,
min_delta semantics, failure policy and experimental split limitations.

## Step 8 final experiment boundary

Use Phase1Experiment around this optimizer to validate all three partition identities,
lock best rules, and perform the prespecified baseline/selected held-out evaluation
only after optimization. Held-out data is not an OptimizationLoop argument. The
experiment receipt prevents an automatic repeated final phase; test outcomes never
change rule versions. See [end-to-end.md](end-to-end.md) for the deterministic demo,
versioned guidelines export and bounded optional live integration.
