# Reflector (Step 3)

`LLMReflector` implements the existing `Reflector.reflect(ReflectorInput)` interface.
It accepts the same injected `LLMProvider` abstraction as the Grader. Tests reuse
`MockLLMProvider`; there is no network, subprocess, dataset or Codex integration.

## Input and labels

`ReflectorInput.record` is the existing validated GradingRecord. It supplies the
question, optional reference solution, frozen rubric, student text, adaptation
rules, and Grader scores, rationales and feedback. `ta_output` accepts either:

- The existing GraderOutput, for complete criterion labels with explanations.
- TAReference(total_score=..., criterion_scores=...), for total-only labels or a
  partial set of TACriterionScore(criterion_id=..., score=...) labels.

TAReference does not fabricate explanations or missing labels. The caller is
responsible for verifying label provenance. Labels must be finite, nonnegative,
unique, known to the rubric and within its maxima. Their total must be feasible
for the known criterion scores and remaining criterion maxima. A complete set
must sum exactly to its total. Structurally impossible labels are rejected as
validation errors, preserving the existing strict input contract. Numerically
valid but substantively suspect TA labels can be discussed and flagged for human
review. Labels are never changed to reconcile a disagreement.

ReflectorInput still allows ta_output=None for compatibility with Step 1; the
concrete Reflector requires labels and rejects missing labels before calling the
provider. All input models remain frozen and forbid unknown fields.

## Output and validation

ReflectorOutput is extended without removing its legacy observations field.
Observation-only legacy objects remain constructible; the concrete Reflector
requires the complete new response. The output includes:

- `overall_score_difference`: exact Decimal AI total minus TA total.
- `direction`: agreement (zero), overgrading (positive), undergrading (negative).
  These describe comparison with the label, not which party is correct.
- `criterion_differences`: criterion ID and signed difference for exactly the
  supplied TA criterion labels. Total-only labels require an empty array.
- `findings`: criterion scope (or null), supported/hypothesis status, category,
  potential-root-cause claim, evidence citations and an optional adaptation
  suggestion. Categories are grader_error, ta_rubric_inconsistency,
  insufficient_evidence and no_discrepancy.
- `human_review_required` and `review_reasons`: a strict boolean and reasons.
- `observations`: optional-in-content summary strings (the array is required).

Evidence specifies a source, exact excerpt and optional criterion ID. Sources
are student_answer, rubric (criterion description), grader_rationale,
reference_solution and ta_label. Rubric/rationale citations require criterion IDs;
TA citations must quote the exact decimal label string, not a substring. Quotes
are checked against the original inputs. Unknown IDs, mismatched criterion scopes,
missing labels, fabricated quotes, incorrect differences, duplicate criterion
entries and contradictory direction/no-discrepancy assertions are rejected.

Supported grader-error findings require a corresponding TA criterion label,
student-answer evidence, evidence from that rubric criterion, and a nonempty
adaptation suggestion. With total-only labels, proposed criterion-level causes
remain hypotheses. Insufficient-evidence causes must use hypothesis status.
Hypotheses, suspected TA/rubric inconsistency and insufficient evidence require
human review with reasons. Blank student text also requires review. Total score
agreement does not suppress available criterion differences or imply correctness.

The reusable prompt lives in `reflector_prompt.py`. It discusses mathematical
reasoning, equivalent methods, partial credit, consequential errors, uncertainty,
TA fallibility and generalizable rule improvements. Suggestions are proposals
only: no Refiner behavior or rubric edits occur. The prompt serializes one
validated ReflectorInput, without external context lookup.

JSON parsing rejects Markdown fences, trailing content, duplicate object keys,
non-finite constants and malformed responses. Pydantic validates both the response
and its consistency with the input. Provider errors propagate unchanged. There
are no retries, repairs, score corrections or fallback findings.

## Usage

```python
from krugradeai.models import ReflectorInput, TAReference
from krugradeai.reflector import LLMReflector
from krugradeai.mock_provider import MockLLMProvider

# record is a validated GradingRecord; response_json is a configured JSON fixture.
request = ReflectorInput(record=record, ta_output=TAReference(total_score="1.5"))
reflection = LLMReflector(MockLLMProvider(response_json)).reflect(request)
```

Unlike the Grader's unscorable-answer exception, uncertainty is returned in the
Reflector's structured review fields. Callers must inspect those fields before
acting on findings. Numerical validation errors raise Pydantic ValidationError;
malformed JSON raises ValueError (including JSONDecodeError).

## Tests and limits

Run the full suite from the project root:

```console
python -m unittest discover -s tests -v
```

Run Reflector tests only:

```console
python -m unittest discover -s tests -p test_reflector.py -v
```

All fixtures are synthetic and all reflection calls use the deterministic mock.
Tests cover agreement, both discrepancy directions, total/partial/complete labels,
review flags, evidence validation, immutable inputs, malformed responses and
compatibility with actual LLMGrader results and the existing RefinerInput schema.

Arithmetic, labels, citation presence and structured claims are checked in code.
Verbatim evidence does not prove that a claim follows logically from the quote.
The system cannot prove mathematical correctness, judge whether a suggestion is
truly generalizable, detect every free-text contradiction, or certify an LLM's
supported/hypothesis classification. Those remain future runtime evaluation and
human-review responsibilities. No actual LLM quality is measured by mock tests.
No data/raw/ access, Refiner implementation, optimization loop, metrics or splitting
is included in this step.
