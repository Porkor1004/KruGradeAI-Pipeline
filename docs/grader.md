# Provider-backed Grader (Step 2)

`LLMGrader` implements the existing `Grader` interface. Inject any `LLMProvider`;
no API key, network connection or Codex CLI is needed for the supplied mock.

```python
from krugradeai.grader import LLMGrader, HumanReviewRequired
from krugradeai.mock_provider import MockLLMProvider

# request is a validated GraderInput; response_json is a configured JSON fixture.
grader = LLMGrader(MockLLMProvider(response_json))
try:
    grade = grader.grade(request)
except HumanReviewRequired as review:
    print(review.reason)  # Route to a human; do not substitute a zero score.
```

## Contracts and prompt

The existing Question model now has optional `reference_solution`. GraderOutput
has optional `overall_feedback` for compatibility with Step 1 records; every new
scored LLM response must supply nonempty feedback. Rubrics, criteria, agent method
signatures and the LLMProvider interface are unchanged.

`prompts.py` holds the reusable prompt and serializes only GraderInput. The prompt
requires criterion scores and explanations (`rationale`), an exact total and
feedback. It gives the rubric precedence over adaptation rules and the reference
solution. It instructs the model to accept equivalent valid methods and consider
partial work and consequential errors only when the rubric permits them. Student
work and all other input text are task data, not instructions to change scoring.
There is no label store or reference-score lookup. Unknown TA-score fields are
rejected before the provider is called. Free text must still be curated by the
caller: schemas cannot identify human scores hidden inside an answer or rule.

## Validation and review

The grader revalidates its input, calls the injected provider once, and parses a
single JSON object. It rejects Markdown fences, trailing text, duplicate keys and
non-finite JSON constants. Decimal parsing preserves fractional numeric values.
Pydantic validates the scored response, then the existing GradingRecord validates
exact criterion coverage, criterion maxima and totals against the original rubric.
Unknown fields (including proposed rubric modifications) are rejected. Exceptions
propagate; there are no retries, JSON repair, clamping or fallback grades.

An empty/whitespace answer raises `HumanReviewRequired` before any provider call.
For uncertain OCR or ambiguous content the prompt requests this alternative:

```json
{"human_review_required": true, "review_reason": "Equation is unreadable"}
```

That response is validated separately and raises HumanReviewRequired with no
score. It cannot include scores or other fields. This explicit exception preserves
the existing Grader return contract and avoids manufacturing numeric grades for
unscorable answers. Readable incomplete answers can receive rubric-supported
partial credit. Human-review detection for nonempty text is delegated to the LLM;
there is no heuristic OCR detector or mathematical correctness engine.

## Tests and limitations

From the project root, with the package dependencies installed:

```console
python -m unittest discover -s tests -v
```

For Grader tests only:

```console
python -m unittest discover -s tests -p test_grader.py -v
```

MockLLMProvider returns a configured JSON string unchanged and records prompts in
memory. All tests use synthetic data. They verify input transmission, validation,
partial-credit representation, review handling and errors without model calls.
Equivalent-method tests verify preservation of alternative student work and the
reference solution through the grading boundary; they do not prove an LLM's
mathematical judgment. Semantic rubric compliance and ambiguity detection need
future runtime evaluation. Prompt instructions cannot guarantee model obedience;
code enforces output structure and numerical rubric boundaries.

No Reflector/Refiner implementation, dataset access, optimization loop, metrics,
splitting, or runtime integration is included. data/raw/ is never accessed.
