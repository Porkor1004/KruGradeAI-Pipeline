# Full real Development optimization

This extends the existing real runner and existing optimization engine. It does
not create another pilot, agent or optimization algorithm. One iteration processes
**all 95 prepared Development records across eight rubric paths**, including
records previously used in limited experiments. Both source origins are preserved:
76 AI-derived operational pseudo-labels and 19 human-derived labels. There are
48 criterion-enriched records (184 AI-derived and 24 human-derived criterion
scores) and 47 total-only records. All 19 original human Validation records are
used for baseline evaluation and relevant candidate-path evaluation.

## Commands

From the project root, inspect the plan without providers, CLI calls, output
directories or report writes:

```powershell
python -m krugradeai.optimize_real --full-development --ai-adjudication --dry-run --max-calls 332
```

The explicitly opt-in live command is:

```powershell
python -m krugradeai.optimize_real --live --full-development --ai-adjudication --run-id real-optimization-full-001 --max-calls 332 --timeout 120
```

Neither implementation nor automated tests execute that live command. Dry-run
and live are mutually exclusive. `--ai-adjudication` remains optional. Without it,
the maximum reservation is 236 calls; omit that flag and use `--max-calls 236`.
Optional `--model` and `--executable` settings use the existing provider.

Full mode ignores sample-size/history-based exclusions and sorts every record by
rubric ID and record ID. This intentionally grades the complete fixed population,
not only fresh examples. Prior pilot outcomes and candidate rules are not imported
as evidence or initial guidelines. Each path starts at empty G0 adaptation rules.
History is used only for latency/rate estimation, never candidate selection.
The existing limited sampling mode remains available with its previous behavior.

## Execution and evidence boundaries

1. Validate prepared Development/supplement hashes, original-field preservation,
   split disjointness, teacher-rubric equality, human-only Validation provenance
   and Validation coverage for every path before constructing providers.
2. Evaluate G0 on each path's complete human Validation population. Grade every
   Development record under G0, preserving source labels and criterion provenance.
3. Send actual total or source-provided criterion discrepancies to Reflector.
   Total-only records contribute to total-score metrics; their missing criterion
   allocations are never inferred. They may produce uncertainty/review, but cannot
   support criterion-error refinement without criterion evidence.
4. With AI adjudication explicitly enabled, the existing score-blind independent
   assessment and locked comparison can examine reviewed, complete-criterion
   Development cases. At most the 48 complete-label records can invoke its two
   provider stages. The existing v2 rubric/evidence/uncertainty guards apply.
5. Full mode explicitly uses the existing one-iteration
   `exclude_reviewed_records` policy. Unresolved records and their entire reflections
   remain in history/reporting but are excluded from Refiner evidence. Supported
   findings are aggregated within their exact rubric path by the existing engine.
   If none survive, no G1 is forced. Agent failures continue to block their path.
6. Refiner may propose only adaptation rules. Existing rule validation rejects
   rubric changes, invented criteria/allocations, numeric score overrides and
   detectable student-specific memorization. A changed, valid candidate becomes G1.
7. Evaluate G1 on the exact same fixed human Validation population as G0 for that
   path. Incomplete coverage makes that evaluation ineligible. Existing MAE and
   earliest-best tie-breaking apply; an eligible G0 wins ties or regressions.

Selection never uses Development pseudo-label metrics. No held-out partition is
loaded, selected, graded or evaluated by this runner, even after path selection.
Held-out testing remains a separately authorized final experiment step.

## Calls and runtime

Let D=95 Development records, V=19 human Validation records, P=8 rubric paths,
and E=48 complete-criterion adjudication-capable records.

| Component | Maximum calls |
| --- | ---: |
| Baseline Validation Grader | 19 |
| Development G0 Grader | 95 |
| Reflector | 95 |
| AI independent adjudication | 48 |
| AI comparison adjudication | 48 |
| Refiner | 8 |
| G1 Validation Grader | 19 |
| **Total with adjudication** | **332** |

The bound is `2V + 2D + 2E + P`; without adjudication it is `2V + 2D + P = 236`.
Actual calls can be substantially lower due to agreement, unresolved evidence,
failures or unchanged candidates. There is never a second optimization iteration.

The current data-only dry-run gives these **conditional serial estimates**:

| Scenario | Calls | Estimated runtime |
| --- | ---: | ---: |
| All Development agrees; baseline + Development grading only | 114 | about 16 minutes |
| Historical 25% discrepancy rate, all eligible discrepancies adjudicated, all paths generate G1 | 189 | about 34 minutes |
| Maximum-call workload at observed mean call latencies | 332 | about 69 minutes |

These estimates use 12 unique previously sampled Development grades, not a
representative sample of the full population. Current observed mean call times
are approximately 8.24s Grader, 19.15s Reflector, 10.19s independent adjudication
and 10.72s comparison. Real Refiner latency is unobserved, so the plan explicitly
assumes 30s per call. Longer aggregated Refiner prompts may take more time.
G1 generation is an assumption in planning scenarios, not a predicted outcome.
There is no claimed grading improvement or new accuracy measurement.

At 120 seconds per call, the maximum workload's execution-timeout sum is
39,840 seconds (**11 hours 4 minutes**). This is not an overall wall-clock
deadline: local CLI setup, process shutdown and report overhead are additional.
Re-run the dry-run before live execution for updated timing observations.

## Budget, interruption and artifacts

Full mode has an explicit configurable global cap, with a hard ceiling of 512
provider calls. The recommended cap is the computed 332-call reservation, not the
ceiling. A smaller budget fails before provider construction; dry-run instead
reports `budget_sufficient=false` and the required reservation. Every role,
including both adjudication phases and Validation grading, shares the same runtime
counter. No provider invocation is made once that counter reaches the cap. No
automatic grading, validation or failed-adjudication retries are introduced.
The existing provider preserves read-only sandboxing and structured validation.

Output is exclusive under `artifacts/real-optimization/real-optimization-full-001/`:

- `run_plan.json`: pre-call plan, configuration and input snapshot hashes.
- `provider_calls.jsonl`: incrementally appended safe invocation metadata, timings
  and error categories; no prompts, raw answers or model responses.
- `report.json`: final outcomes, exact selected rules/versions and per-path
  G0/G1 human Validation metrics, eligibility, comparison identities, failures,
  reviewed/excluded Development records and no-candidate stop reasons.

Development G0 metrics are reported globally and per path, separately for AI and
human label origins. Original source labels remain metric references, including
after adjudication. G1 is evaluated on human Validation only; its Development
metrics are explicitly absent, rather than invented or adding another 95 calls.
All paths retain G0 metrics and a null G1 result when no candidate exists.

**Resume is not supported.** The journal is diagnostic, not a response cache or
transaction log suitable for replay. A process can stop after a remote call
finishes but before its result is recorded; automatically repeating that request
would risk duplicate calls. Existing run IDs are refused before any live work.
Handled interruptions preserve safe metadata and an execution-failure report;
abrupt process/OS termination may leave only the plan and completed-call journal.
Inspect partial outputs before deliberately starting another experiment with a
new ID. No automatic restart, overwrite or supposedly duplicate-free replay occurs.

Raw sources, original labels/rubrics, prepared partitions, criterion supplement,
split manifests and prior experiment reports are never written by this runner.

## Tests

```powershell
python -m unittest discover -s tests -v
```

New tests use a fully synthetic 95/19 population with eight paths, both label
origins, 48 criterion-enriched and 47 total-only records. They cover dry-run
isolation, reservations, prior-record inclusion, deterministic ordering,
aggregation/exclusion, human Validation coverage/ties, all 332 mocked calls
sharing a cap, interruption journaling, immutability and no held-out loading.
