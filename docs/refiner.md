# Refiner (Step 4)

`LLMRefiner` implements the existing `Refiner.refine(RefinerInput) -> AdaptationRules`
interface. It proposes a complete candidate rule set; it does not apply rules,
change scores or run an optimization iteration. The existing LLMProvider and
MockLLMProvider are reused unchanged.

## Inputs and backward compatibility

RefinerInput retains `current_rules` and `reflection`. Two defaulted fields extend
it: `rubric` (optional for legacy construction, required by LLMRefiner) and
`additional_reflections` (a tuple, empty by default). Each ReflectorOutput already
supports multiple findings; additional_reflections allows a batch of analyses for
the same rubric. Criterion references in rules and all reflection findings,
evidence and differences are checked against that rubric. Callers must ensure
batch provenance and that reflections came from the validated Reflector boundary;
the Refiner has no original records with which to reverify quotations or labels.

AdaptationRules still has exactly one top-level field, `rules`. Legacy string
rules remain valid throughout the Grader and Reflector pipeline. Structured rules
add the metadata needed for safe refinement:

```json
{
  "rules": [{
    "id": "check-equivalence",
    "instruction": "Check equivalent reasoning only as permitted by the rubric.",
    "criterion_ids": ["c"],
    "rationale": "Supported findings indicate inconsistent interpretation of reasoning."
  }]
}
```

IDs are case-sensitive, unique and limited to 1–64 ASCII letters/digits/underscore,
hyphen or period, beginning with a letter. Criterion IDs must be unique within a
rule and known to the supplied rubric; an empty tuple means a general rule.
Instructions and rationales must be nonempty. All models remain frozen, with
unknown fields forbidden. RefinerOutput remains an alias for AdaptationRules;
agent/provider method signatures and prior tests are unchanged.

## Proposal behavior

The reusable `refiner_prompt.py` asks the model to examine findings and evidence,
propose generalizable interpretation guidance and preserve useful rules unless
findings justify changing them. Revisions should retain the rule ID, additions
use new IDs, and removal means omitting a rule from the full candidate. New or
revised string rules are rejected; an existing legacy string may be retained
verbatim. Structured changes require a rationale. An empty candidate is allowed.

Code enforces a conservative evidence gate: changes require at least one supported
grader_error finding, and no reflection in the batch may require human review.
Otherwise the provider must return current_rules exactly, including representation
and order. An invalid changed response raises an error; it is never silently
replaced with the current rules. A TA disagreement, hypothesis, legacy observation
or suspected TA inconsistency alone does not authorize changes.

The evidence gate permits proposals; it does not prove that each removal or
revision is justified. Preservation of useful rules, justification quality,
recurrence and generalization are prompt responsibilities requiring later review.

## Validation and protected data

The Refiner revalidates input and requires rubric context before its one provider
call. It snapshots both the caller input and validated input before the call and
compares their serialized state afterward, including on provider failure. This
checks rubric definitions, maxima and all supplied context for mutation in
addition to frozen-model enforcement. Detected mutation raises an integrity error;
this detects bypasses rather than rolling them back or sandboxing provider code.

The output schema has no question, rubric, scoring weights, TA labels or score
fields. Attempts to return those fields, including within structured rules, are
rejected. The Refiner is not given raw question/TA objects; only rubric context
and Reflector analyses (which may quote evidence/labels) cross this boundary.

Strict JSON parsing rejects malformed JSON, duplicate object keys, Markdown
fences, trailing content and non-finite constants. Pydantic rejects missing rules,
unknown fields, duplicate rule IDs and invalid references. Contextual validation
then applies conservative lexical checks to instructions and rationales, including
retained legacy strings:

- Explicit points/full-credit prescriptions, selected score/weight/maxima changes,
  rubric overrides and mutations of question/reference/TA content.
- Selected student/response-ID rules and explicit answer-key patterns.
- Copied student-answer/reference-solution evidence excerpts (normalized excerpts
  of eight or more characters, or short mathematical excerpts of four or more).
- Unknown criterion IDs in structured references and explicit quoted/bracketed
  or `criterion ID` references in prose.

Immediate prohibitions such as “Never override the rubric” are permitted. Text
matching uses Unicode NFKC normalization and case folding. Checks reject rather
than rewrite content. A retained harmful legacy rule can therefore cause even a
no-change proposal to fail; the system does not silently remove it under weak
evidence. Provider errors propagate unchanged; no retries or repairs are made.

## Use and tests

```python
from krugradeai.models import RefinerInput
from krugradeai.refiner import LLMRefiner
from krugradeai.mock_provider import MockLLMProvider

# current_rules, reflection and rubric are validated domain models.
request = RefinerInput(current_rules=current_rules, reflection=reflection,
                       rubric=rubric, additional_reflections=())
candidate = LLMRefiner(MockLLMProvider(response_json)).refine(request)
```

Run all tests from the project root:

```console
python -m unittest discover -s tests -v
```

Run Refiner tests only:

```console
python -m unittest discover -s tests -p test_refiner.py -v
```

Tests use synthetic fixtures and deterministic mock providers. They cover
add/revise/remove/no-change candidates, preservation, multiple findings, structured
and legacy rules, real Reflector-output compatibility, Grader consumption,
malformed responses, references, overrides, copied content and mutation detection.
No real LLM, network, subprocess or data/raw/ access occurs.

## Limits

Lexical checks are deliberately limited, English-oriented heuristics. They can
reject benign wording and miss paraphrases, encoded content, implicit rubric
conflicts, unquoted criterion references, novel answer keys or student identities.
They do not provide semantic authorization or certify mathematical validity.
Only supplied evidence excerpts are available for copy detection. A nonempty
rationale does not prove a change is justified. Candidate rules require later
validation before adoption; this step implements no adoption loop, validation
metrics, early stopping, runtime integration or dataset splitting.
