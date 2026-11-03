# Validation, metrics and best guidelines (Step 7)

Step 7 extends the existing OptimizationLoop. No additional optimization loop or
LLM agent is introduced. Existing calls without validation_records preserve Step 6
behavior; agent contracts/implementations and provider interfaces are unchanged.

## Usage

```python
from krugradeai.optimization import OptimizationLoop
from krugradeai.optimization_models import ValidationConfig

optimizer = OptimizationLoop(
    grader=grader, reflector=reflector, refiner=refiner,
    max_iterations=5,
    validation_config=ValidationConfig(
        primary_metric="mae",       # or "normalized_mae"
        patience=2,
        min_delta="0.01",
        qwk_categories=None,        # explicit ordinal categories, if appropriate
    ),
)
result = optimizer.run(development_records, initial_rules,
                       validation_records=validation_records)
for path in result.paths:
    selected = path.validation
    if selected.best_version is None:
        # Explicit no_valid_evaluation: do not substitute the latest candidate.
        continue
    best_rules = selected.best_rules
    best_version = selected.best_version
```

DevelopmentRecord remains frozen and development-only. ValidationRecord reuses
its question/rubric/answer/verified-label validation, with split fixed to validation.
Records can optionally carry RecordProvenance(source=..., item_id=...). Models,
including provenance, are immutable and forbid unknown fields.

ValidationConfig accepts MAE or normalized MAE for primary selection; exact match
and QWK are descriptive secondary metrics. Patience is a positive integer; min_delta
is finite and nonnegative, in the units of the selected metric. Normalized values
are fractions, not percentages. Configuration is included in returned results.

## Separation and overlap

Before any agent call, supplied validation data must be nonempty, have unique
record/response identities, and cover every development rubric path with exactly
matching rubric definitions. Unknown validation rubric paths are rejected. A
record ID shared across splits, a shared (question ID, response ID), or shared
(source, item_id) provenance rejects the run. Duplicate provenance within the
validation set is also rejected. Matching question IDs alone are allowed because
different student responses to one question can belong to different splits.

These checks depend on supplied identities/provenance. Renamed duplicates without
shared provenance cannot be reliably detected. Identical answer text alone does
not prove overlap: independent students can submit the same answer. Synthetic
fixtures are only software tests and do not establish an experimentally valid
split. Callers must supply real provenance and an appropriate experimental design.
There is no dataset import, automatic split or access to data/raw/.

Validation records are graded only by the existing Grader, with no TA fields in
its GraderInput. Verified totals enter deterministic ScorePair calculations outside
the Grader. Validation answers, labels, discrepancies and grading feedback never
enter ReflectorInput or RefinerInput. Development reflections remain the only
source of adaptation proposals. Scalar validation performance affects only rule
selection and stopping, not proposal content. Use stateless provider sessions;
an arbitrary injected Grader/provider that privately retains and leaks previous
requests is outside orchestration's data-flow guarantees. CodexCLIProvider uses
ephemeral sessions. No live calls were made for Step 7.

## Deterministic metrics

`metrics.py` exposes ScorePair(predicted, reference, maximum) and
calculate_metrics(samples, expected_count=..., qwk_categories=...). Score pairs
must contain finite nonnegative scores bounded by a positive rubric maximum.
Missing scores are not pairs. Decimal arithmetic uses a fixed precision of 50 with
standard half-even rounding, independent of the caller's Decimal context. Ratios
with repeating decimal expansions are rounded at that precision.

For n successful pairs, metrics are:

- MAE = sum(abs(predicted - reference)) / n.
- Normalized MAE = mean(abs(predicted - reference) / that record's maximum).
  It is always reported and supports heterogeneous rubric maximum scores.
- Exact match = count(predicted == reference) / n. Equality is exact; no rounding
  or tolerance changes partial-credit labels.
- Coverage = n / expected_count, with both counts reported. Empty evaluations have
  null error/agreement metrics and zero coverage, including expected_count zero.

MAE/accuracy over successes are descriptive when coverage is incomplete. Such
numbers must not be used to claim improvement without checking eligibility.
Zero is a genuine score and contributes to metrics; failed or human-review cases
remain missing and never become zero.

QWK uses explicit, strictly increasing, finite nonnegative ordinal categories.
Fractional categories are supported without converting scores to integers or
rounding. Quadratic weights are squared distances between category indices, not
between raw numeric values. The score is one minus weighted observed disagreement
divided by weighted expected disagreement from empirical marginals. The common
weight normalization factor cancels. This follows the weighted Cohen kappa
convention described in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html);
scikit-learn is not a runtime dependency.

QWK returns null with qwk_undefined_reason for empty data, absent categories,
unmapped score values, different rubric maxima without a shared ordinal scale,
or zero expected weighted disagreement (including constant identical labels).
No misleading zero/one is substituted for undefined QWK. Merely specifying
categories does not prove the underlying ordinal interpretation is appropriate.

## Fixed population and failure policy

Every version grades the same complete ordered validation set for its rubric.
Each ValidationEvaluation includes full-set metrics, per-record grades or missing
outcomes, sanitized processing failures, review reasons and selection eligibility.
There are no retries, repaired scores or dynamic intersections of successful rows.

By default, the comparison population is the entire validation set. Eligibility
requires every comparison record to have a validated grade. A candidate that fails
on difficult examples cannot beat a complete candidate by improving its
successful-only average. Incomplete evaluations remain visible but cannot be
selected. An invalid baseline does not prevent development optimization; a later
fully evaluated candidate may become the first valid choice.

An explicit `comparison_record_ids` tuple can predeclare a restricted population
before the run. It must contain known unique validation IDs and include a nonempty
population for every rubric path. All records are still graded and full-set
coverage/failures remain visible; comparison_metrics report the fixed subset.
Complete coverage is still required on that subset for every version. Failures
do not change which IDs are compared. Selecting a subset is a user decision that
must be justified in the experiment, not a way to hide observed failures.

If no baseline/candidate has an eligible evaluation, best_version and best_rules
are null and the validation summary reports no_valid_evaluation. The path's
operational stop_reason remains available separately. There is no fallback to
initial or latest rules without a valid evaluation. Summary failure/review counts
include all attempted validation versions, with full version context in evaluations.

## Selection and early stopping

Before any development iteration for a path, G0 is graded on validation records.
Each changed candidate is evaluated immediately after refinement, including at
the maximum iteration boundary. A candidate may be absent from development
history's active rules_version because it was proposed on the final iteration,
but it is still evaluated for validation selection. A no-change response creates
no new version, reuses that version's existing evaluation and terminates as before.
A zero iteration budget still evaluates and can select G0.

Selection keeps the lowest primary error among eligible versions. Exact ties
retain the earlier version. Independent rubric paths maintain independent
baselines, best versions, evaluations and patience counters. No pooled global
winner is inferred across incompatible rubrics.

Patience counts only after a changed candidate's validation attempt finishes.
A candidate resets the counter if its selected error improves over the previously
best valid error by **strictly more than** min_delta; the first valid candidate
after an invalid baseline also resets it. Otherwise the counter increments,
including incomplete/invalid validation attempts. Baseline evaluation itself does
not increment patience. Default patience 2 stops after two such non-improvements.

Selection and patience have different responsibilities: every strictly smaller
error becomes the best observed candidate even when its improvement is too small
to reset patience. This retains the true lowest observed error while allowing
min_delta to define a meaningful stopping threshold. Development failure/review,
no-change and no-discrepancy termination remain in force. Validation does not
revive a stopped development path. No Beam Search, UCB or additional search loop
is implemented.

## Result interfaces and history

OptimizationPath retains its Step 6 history, rule_versions, final_version and
operational stop_reason. final_version continues to identify the **latest**
proposed version for backward compatibility. For the returned best validated
rules, use **path.validation.best_rules** and **best_version**, which may identify
an earlier version. With no validation argument, path.validation is null and no
claim of best guidelines is made.

ValidationSummary includes configuration, primary metric, baseline_metrics, all
evaluations, exact best_rules/version, non-improvement count, failure/review totals
and validation stop_reason. Each evaluation includes its rule version, full and
comparison metrics/counts/coverage, fixed comparison IDs, eligibility/invalid
reason, grading outcomes and failures/reviews. Existing development iteration
history is not mixed with validation outcomes or discarded.

Result validators check evaluated-version coverage, stable comparison population,
selection eligibility and tie-breaking, and exact correspondence between selected
rules and version history. JSON round trips preserve those invariants. Results
contain student/label-related information and should be handled as sensitive data;
no history files are written automatically. Only exceptions' class names and fixed
failure messages are retained, rather than raw provider diagnostics.

## Tests and remaining scope

```console
python -m unittest discover -s tests -v
```

`tests/test_metrics.py` contains hand-calculated Decimal, normalization, coverage
and QWK tests. `tests/test_validation.py` exercises baseline/candidate selection,
patience/min_delta, final-candidate evaluation, no-change, incomplete coverage,
source isolation, overlap, immutable labels and independent paths. Existing
Step 6 and provider tests remain unchanged. All agent/provider execution in the
suite is mocked. No held-out final evaluation is implemented: selected validation
performance remains a selection result, not a final unbiased test estimate.
