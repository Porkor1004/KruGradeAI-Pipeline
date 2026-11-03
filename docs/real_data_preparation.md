# Mixed-source real-data preparation

One experiment replaces the cancelled A/B plan. Preparation is complete; no real-data grading, optimization or evaluation has run.

## Prepared artifacts

Directory: `artifacts/real-mixed-v1/`.

| Partition | AI-derived ZIP | Human-derived Excel | Total | Connected attempt groups | Source proportions |
|---|---:|---:|---:|---:|---|
| Development | 76 | 19 | 95 | 16 (10 ZIP + 6 human) | 80% AI / 20% human |
| Validation | 0 | 19 | 19 | 6 human | 100% human |
| Held-out Test | 5 | 19 | 24 | 7 (1 ZIP + 6 human) | 20.83% AI / 79.17% human |
| Eligible reserve | 211 | 0 | 211 | 28 ZIP | Not active |

All 57 eligible human parent records are used, across 18 human attempts. The 292 eligible ZIP records span 39 attempts; 81 enter active partitions and 211 remain in a documented reserve. Whole groups were selected to approximate the preferred source proportions. Reserved records are not defective and were not silently deleted. Their original locators and eligibility decisions are in the manifest; source content remains in the raw ZIP.

Files:

- `development.jsonl`: existing `DevelopmentRecord` schema, one record per line.
- `validation.jsonl`: existing `ValidationRecord` schema, human-derived labels only.
- `test.jsonl`: existing `HeldoutRecord` schema, with explicit label provenance.
- `manifest.json`: fixed seed, input hashes, partition hashes, immutable rubric definitions, group/component IDs, source files, response ancestry, source-row eligibility decisions, generation kinds and partition assignments. It contains no student answers or achieved labels; rubric maxima are retained.
- `quarantine.json`: source locators, IDs and reasons for exclusions, including review-log rows not promoted to active data.
- `dataset_provenance.json`: real-data description and source identities.

## Per-question coverage

Each entry below is **AI-derived / human-derived**. The canonical questions are `com-eng-math-ii-2026-1-p1` through `p8`.

| Parent | Development | Validation | Held-out Test |
|---|---:|---:|---:|
| 1 | 10 / 3 | 0 / 3 | 1 / 4 |
| 2 | 10 / 3 | 0 / 3 | 0 / 3 |
| 3 | 10 / 4 | 0 / 2 | 1 / 5 |
| 4 | 9 / 4 | 0 / 3 | 1 / 2 |
| 5 | 9 / 2 | 0 / 2 | 1 / 2 |
| 6 | 9 / 1 | 0 / 2 | 0 / 1 |
| 7 | 10 / 1 | 0 / 2 | 1 / 1 |
| 8 | 9 / 1 | 0 / 2 | 0 / 1 |

Every partition covers all eight identical parent rubric scopes. Human-only held-out coverage also includes all eight, but the AI-only held-out cohort covers only 1, 3, 4, 5 and 7. Report absent cohorts as unavailable, never as perfect agreement. Human held-out counts of one for questions 6–8 cannot support strong per-question conclusions.

## Eligibility and lineage

The adapter implements [the scope reconciliation](question_scope_reconciliation.md) and [audit](real_dataset_audit.md). It discovers the ZIP and workbook by their structures, then pins their SHA-256 values to the audited snapshot. Changed carriers fail closed and require a new audit. All source files are read-only; workbook rendering, formula authoring and source rewriting are unnecessary.

Only teacher-supplied questions 1–8 define authoritative parent rubrics. Both ZIP groups must agree on question text, references, criteria and maxima. Human criteria must match those definitions exactly. The official-reference hash identifies the previously audited carrier; it is not a claim that the absent PDF was reauthenticated.

Human records require a non-stale approved/corrected review join, matching score and answer, valid maxima and reference mapping. Excel Boolean answers become literal `True`/`False` text. Complete, unique, disjoint criterion covers authorize summing separately reviewed subpart scores within one attempt/PDF; missing parts are never assigned zero, and aggregate totals are never divided into invented criterion labels. All original response IDs and criterion lineage are preserved. Supplied question/reference/rubric text is not rewritten.

The audit's conservative ambiguity-marker screen excludes questionable answer text, including some cancelled writing that might later be recovered by review. The two score conflicts and unclear 7.2 answer remain excluded. Quarantine totals:

| Reason | Source entries |
|---|---:|
| Unsupported AI-generated reference scopes 9/10 | 78 |
| ZIP answer ambiguity markers | 20 |
| Incomplete human parent coverage | 47 |
| Unresolved human score/rubric conflicts | 2 |
| Previously flagged ambiguous human answer | 1 |
| Review-log-only entries outside audited usable sheet | 187 |
| **Total source decisions** | **335** |

These counts describe source entries, not 335 independent students. The 187 review-only entries include potentially recoverable labels and are not automatically invalid. Duplicate base sheets in the two Excel carriers are not imported. Quarantine and reserve are distinct.

The active ZIP pool includes 36 source-reviewed, 32 inferred-span, 10 assumed-transcription and 3 synthetic-full-response records. Each retains its generation kind and `ai_derived` label origin. None is promoted to a verified human label. Human labels are accepted on the user's identification of the review workflow plus the matching workbook evidence; reviewer identities and original PDFs/images remain unavailable.

## Split policy and reproducibility

Seed: `krugradeai-mixed-signal-v1`. Policy: `mixed-signal-v1`.

1. Connect explicit source attempt IDs, source files/PDFs and all response ancestry. A regenerated ID does not break a known ancestry link. Cross-source connected components require review rather than entering human-only Validation.
2. Search 20,000 deterministic SHA-256-ranked allocations of human groups into three approximately equal group counts. Require all eight parent scopes on every side; prefer at least two Validation observations per scope, then balanced human row counts. Selection uses IDs and coverage counts, never scores or model outputs.
3. Assign one intact ZIP group closest to one quarter of the human Test record count, approximating 20% AI within mixed Test.
4. Select whole remaining ZIP groups by deterministic subset-sum nearest four times the human Development count. Remaining eligible ZIP groups enter reserve. A and B are no longer separate experiments.

No attempt, source response ancestry or connected component crosses Development, Validation, Test or reserve. Unknown shared students across rooms cannot be proved absent without source identity evidence. Common short mathematical answers are not assumed to establish derivation; explicit source ancestry is the grouping authority.

Reproduction requires Python 3.11+, Pydantic v2 and the data extra:

```console
python -m pip install -e ".[data]"
python -m krugradeai.real_data --output-dir artifacts/real-mixed-reproduction
python -m unittest discover -s tests -v
```

This command performs preparation only and never imports a provider or invokes an agent. Choose an unused output directory. Existing nonempty destinations are identified and refused before source loading or output writes. During implementation only, the newly generated `real-mixed-v1/manifest.json` was identified and regenerated to remove constituent scores and add output hashes; its partition files were not overwritten. Future CLI runs cannot overwrite it.

## Safe pipeline and reporting interfaces

`RecordProvenance.label_origin` is an optional backward-compatible field (`human_derived` / `ai_derived`), required by the prepared-partition loader. Existing synthetic records remain valid. The legacy numeric field `ta_output` is reused without changing score bounds or algorithms; its name does **not** imply every target was verified by a TA. Always interpret it together with label provenance. Optimization reporting must retain this manifest and disclose mixed target reliability.

Load only the required partition:

```python
from krugradeai.real_data import load_prepared_partition

development = load_prepared_partition("artifacts/real-mixed-v1", "development")
validation = load_prepared_partition("artifacts/real-mixed-v1", "validation")
```

The loader verifies that partition's hash without opening other partition files. A later authorized optimization may pass only Development and Validation to the existing optimizer. Its Grader requests are label-blind; only Development discrepancies reach Reflector and Refiner. Never pass the manifest, raw workbook/ZIP, validation labels or held-out records to those agents. File separation is an application boundary, not an OS-level access-control claim.

Only after guideline selection is locked should a final evaluator load `test`. The new `heldout_metrics_by_origin(completed_evaluation, heldout_records)` helper summarizes **already obtained** outcomes into `mixed_source`, `human_only`, and `ai_labeled_only` metrics and coverage. It checks outcome/context identities, preserves failed requests in cohort denominators, and never calls an agent. Use it separately for baseline and selected-rule final passes. No real-data metric values exist yet. Mixed-source agreement must never be described as human grading accuracy; the five AI-labeled Test records are noisy-reference comparisons.

The optimizer and selection algorithms are unchanged. No automatic live execution or experiment runner was added.

## Verification

All **233 unit tests pass**: the existing 212 plus 21 new tests. New tests use synthetic ZIP/workbook fixtures, not real student data, and cover reproducibility, raw preservation, ancestry grouping, human-only Validation, immutable rubrics, score-independent group selection, complete-cover assembly, quarantine, provenance, partition integrity, exclusive output writes and stratified reporting. Input raw-file hashes remain unchanged.
