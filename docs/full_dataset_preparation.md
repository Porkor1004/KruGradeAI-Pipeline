# Full-source local preparation and offline verification

## Corrected owner-supported workflow

Use `corrected_dataset` and `corrected_audit` for the current dataset design.
The dataset owner clarified that original scores are TA grades, ZIP OCR was AI
reviewed and Untitled OCR was human reviewed. Missing reviewer identity fields
do not negate that evidence. Explicit augmented/generated score provenance still
takes precedence over any numerical agreement with an earlier TA score.

```powershell
python -m krugradeai.corrected_dataset --raw-dir ../data/raw --output-dir artifacts/full-dataset-corrected-provenance/prepared
python -m krugradeai.corrected_audit --prepared artifacts/full-dataset-corrected-provenance/prepared --output-dir artifacts/full-dataset-corrected-provenance/offline --historical ../artifacts/real-mixed-v1
```

Use fresh output paths for subsequent experiments. This workflow models answer,
score, OCR review and rubric origins separately; verifies row-level score fields;
preserves source and policy evidence; retains every physical source row; and
selects nonoverlapping original response/criterion representations for supervision.
Review-only entries may be usable when question scope, teacher rubric, actual
reviewed score and answer are available and consistent. Blank scores, unresolved
flags, stale/deferred/excluded reviews and missing rubrics remain explicitly
unscored or quarantined. Teacher-unavailable Q9/Q10 stay separate.

Partitions target 70/15/15 with the agreed source proportions. Validation accepts
only original TA-scored, human-reviewed Untitled student responses with compatible
teacher rubrics. Every eligible example is assigned once; attempts and derived
variants remain together. Common short answers do not prove response identity;
substantive exact matches are grouped conservatively across both sources. Every
duplicate candidate and grouping edge is recorded privately.

Some Development scopes may lack compatible independent Validation examples.
Those paths run in a separate Development-only batch and cannot claim a
Validation-selected candidate. Both mock pipeline checks cover the full assigned
population; mock results do not measure accuracy. Evaluation reports original TA
labels separately by OCR-review origin and separately from AI-generated scores.

The audit records overlap with historical input identities without loading old
predictions. A new version does not make previously exposed work pristine heldout
data. Keep all private outputs ignored by Git and review the dataset design before
any live execution. No push, merge or live call is part of preparation.

## Superseded first-audit workflow

The remaining instructions describe the earlier conservative diagnostic importer.
Its blanket unverified classification and empty Validation policy were superseded
by the owner clarification above. Keep it only to reproduce the earlier audit;
do not use it for the corrected experiment design.

Run from the PR checkout with the data extra installed. Supply a directory
containing the original named Signal ZIP and exactly one Untitled workbook. Use
fresh, Git-ignored output directories; these commands never overwrite a prior run.

```powershell
python -m krugradeai.full_dataset --raw-dir ../data/raw --output-dir artifacts/full-preparation-new
python -m krugradeai.audit_offline --prepared artifacts/full-preparation-new --output-dir artifacts/full-offline-new --historical ../artifacts/real-mixed-v1
```

Preparation reads the entire ZIP README, manifests and provenance before training
records and verifies the declared member hashes. CSV, JSONL and pipeline-case
files are alternate encodings, not additional examples. Every nonempty workbook
row, including review rows and alternate-sheet representations, has a master
ledger entry with a source locator, original payload, provenance, category and
reason. Blank formatting rows do not count as records. Original files stay read-only.

The preferred workbook representation is its prepared sheet. Other rows with the
same response ID remain in the ledger as quarantined alternate representations.
Unique original-sheet responses can supply unverified supervision when their
teacher criteria and score bounds match. Matching teacher criterion subsets may
be scored separately with explicit restricted scope, without fabricating missing
criterion scores or requiring full-parent assembly. Known unresolved historical
flags remain unscored. Review-only rows lacking question/rubric context remain
unscored; missing answers are unusable and quarantined.

Approved/corrected status without independent human-review evidence is not enough
to certify a human label. The workflow conservatively labels workbook supervision
`unverified`; ZIP labels are `ai_derived`. AI-generated reference scopes stay
unscored. Independently certified human labels require a new evidence-backed
import policy, not merely changing a label-origin string. Current preparation
cannot construct human-only Validation, so it leaves Validation empty and explicitly
blocks validated-prompt claims.

All rows participate in connected components using response ancestry, attempts,
source files and same-question exact normalized answer text. Common short answers
can connect many attempts conservatively. Unknown semantic links are not proven
absent by these checks. Components never cross supervised partitions. There is no
historical cap or source-balance reserve: all eligible supervision enters
Development or Test. With no certified Validation, the available supervised
population targets approximately 85% Development and 15% Test, seeking 80%
Untitled in Test subject to whole components. The requested 70/15/15 split cannot
be satisfied until independent human labels become available. Source percentages
are soft targets and never justify discarding data.

`RecordProvenance` retains optional `split_group`, source IDs, generation kind and
reference origin. Generic Validation rejects Development/component overlap when
those groups are supplied. Keep the full ledger: generic records alone cannot
discover undeclared relationships. Independent evaluation reports human-derived,
AI-derived and unverified label metrics separately, including empty cohorts.

The new optimizer runtime `schema_mock` is an explicit software script: Grader
returns zeros without reading labels; Reflector flags unsupported total-only
disagreements for review. Such stops are expected safeguards, not successful
optimization or dataset grading results. The existing synthetic mock workflow
continues to exercise supported refinement and Validation selection. Independent
evaluation uses explicit scripted mock responses per scope and partition. These
offline checks do not consume a live heldout experiment or produce accuracy claims.

The private report includes inventory reconciliation, per-question/provenance
counts, historical selection evidence, per-scope pipeline results, proposed live
configs and call reservations. A one-iteration optimization reserves
`2 * Development + rubric_paths` calls, plus `2 * Validation` when a trustworthy
Validation partition is later supplied. Independent evaluation uses one call per
row per prompt. Choose a model explicitly, establish human verification and lock
the heldout protocol before considering live execution. No live call, push or
merge is part of this workflow.
