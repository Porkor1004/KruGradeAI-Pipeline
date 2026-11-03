"""Synthetic tests for reflection and its validated evidence boundary."""
import json
import unittest
from decimal import Decimal
from pydantic import ValidationError
from krugradeai.models import GradingRecord, ReflectorInput, TAReference
from krugradeai.reflector import LLMReflector
from krugradeai.mock_provider import MockLLMProvider
from krugradeai.grader import LLMGrader
from test_grader import request, result


def reflection_input(ta="1", criteria=(), answer="2x = 4, so x = 2"):
    grade = LLMGrader(MockLLMProvider(result())).grade(request(answer, "Divide by two."))
    return ReflectorInput(record=GradingRecord(grader_input=request(answer, "Divide by two."), grader_output=grade),
                          ta_output=TAReference(total_score=ta, criterion_scores=criteria))


def finding(**changes):
    item = {"criterion_id": "c", "status": "hypothesis", "category": "grader_error",
            "claim": "Partial-credit interpretation may differ.",
            "evidence": [{"source": "student_answer", "quote": "2x = 4"},
                         {"source": "rubric", "criterion_id": "c", "quote": "Reasoning"}],
            "adaptation_suggestion": "When permitted by the rubric, check equivalent intermediate reasoning before awarding credit."}
    item.update(changes)
    return item


def reflection_json(difference="1", direction="overgrading", **changes):
    item = {"observations": [], "overall_score_difference": difference, "direction": direction,
            "criterion_differences": [], "findings": [finding()],
            "human_review_required": True, "review_reasons": ["Total-only labels do not establish a criterion-level cause."]}
    item.update(changes)
    return json.dumps(item)


def run_reflection(raw=None, item=None):
    return LLMReflector(MockLLMProvider(raw or reflection_json())).reflect(item or reflection_input())


class ReflectorTests(unittest.TestCase):
    def test_exact_agreement(self):
        output = run_reflection(reflection_json("0", "agreement", findings=[], human_review_required=False, review_reasons=[]), reflection_input("2"))
        self.assertEqual(output.overall_score_difference, Decimal(0))
        self.assertFalse(output.human_review_required)

    def test_overgrading(self):
        output = run_reflection()
        self.assertEqual(output.overall_score_difference, Decimal(1))
        self.assertEqual(output.direction, "overgrading")
        self.assertEqual(output.findings[0].status, "hypothesis")

    def test_undergrading(self):
        item = reflection_input("2")
        data = item.model_dump()
        data["record"]["grader_output"] = json.loads(result("0.5"))
        item = ReflectorInput.model_validate(data)
        output = run_reflection(reflection_json("-1.5", "undergrading"), item)
        self.assertEqual(output.overall_score_difference, Decimal("-1.5"))

    def test_total_only_has_no_criterion_differences(self):
        provider = MockLLMProvider(reflection_json())
        output = LLMReflector(provider).reflect(reflection_input())
        self.assertEqual(output.criterion_differences, ())
        payload = json.loads(provider.requests[0].split("\nINPUT_JSON\n")[1])
        self.assertEqual(payload["ta_output"]["criterion_scores"], [])
        self.assertEqual(payload["record"]["grader_input"]["question"]["reference_solution"], "Divide by two.")
        self.assertEqual(payload["record"]["grader_output"]["overall_feedback"], "The reasoning supports the result.")

    def test_criterion_labels_and_legacy_grader_output(self):
        item = reflection_input()
        legacy = json.loads(result("1"))
        item = ReflectorInput(record=item.record, ta_output=legacy)
        raw = reflection_json(criterion_differences=[{"criterion_id": "c", "difference": "1"}])
        self.assertEqual(run_reflection(raw, item).criterion_differences[0].difference, Decimal(1))

    def test_partial_ta_criteria_only_report_known_differences(self):
        data = reflection_input().model_dump()
        data["record"]["grader_input"]["rubric"]["criteria"] += ({"id": "d", "description": "Conclusion", "max_score": "1"},)
        data["record"]["grader_input"]["rubric"]["total_score"] = "3"
        data["record"]["grader_output"]["criterion_scores"] += ({"criterion_id": "d", "score": "0", "rationale": "No conclusion"},)
        data["ta_output"] = {"total_score": "1.5", "criterion_scores": [{"criterion_id": "c", "score": "1"}]}
        item = ReflectorInput.model_validate(data)
        raw = reflection_json("0.5", criterion_differences=[{"criterion_id": "c", "difference": "1"}])
        output = run_reflection(raw, item)
        self.assertEqual(len(output.criterion_differences), 1)
        self.assertEqual(output.overall_score_difference, Decimal("0.5"))

    def test_suspected_ta_inconsistency_is_reviewed(self):
        raw = reflection_json(findings=[finding(category="ta_rubric_inconsistency", claim="The TA label may reflect a different interpretation.")])
        output = run_reflection(raw)
        self.assertTrue(output.human_review_required)
        self.assertEqual(output.findings[0].category, "ta_rubric_inconsistency")

    def test_ambiguous_evidence_stays_uncertain(self):
        item = reflection_input(answer="[unreadable]")
        raw = reflection_json(findings=[finding(category="insufficient_evidence", evidence=[{"source": "student_answer", "quote": "[unreadable]"}], claim="OCR prevents a reliable diagnosis.")])
        self.assertTrue(run_reflection(raw, item).human_review_required)

    def test_invalid_json(self):
        for raw in ("not json", "[]", "{} trailing", "```json\n{}\n```", '{"direction":"agreement","direction":"overgrading"}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                run_reflection(raw)

    def test_invalid_criterion_references(self):
        for raw in (reflection_json(findings=[finding(criterion_id="fake")]), reflection_json(findings=[finding(evidence=[{"source": "rubric", "criterion_id": "fake", "quote": "Reasoning"}])])):
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                run_reflection(raw)

    def test_invalid_score_difference_or_direction(self):
        for raw in (reflection_json("-1"), reflection_json("NaN"), reflection_json("1", "undergrading"), reflection_json("99")):
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                run_reflection(raw)

    def test_fabricated_criterion_scores_from_total_rejected(self):
        with self.assertRaises(ValidationError):
            run_reflection(reflection_json(criterion_differences=[{"criterion_id": "c", "difference": "1"}]))
        with self.assertRaises(ValidationError):
            run_reflection(reflection_json(findings=[finding(status="supported")]))

    def test_missing_or_wrong_criterion_difference_rejected(self):
        item = reflection_input(criteria=({"criterion_id": "c", "score": "1"},))
        for differences in ([], [{"criterion_id": "c", "difference": "-1"}], [{"criterion_id": "c", "difference": "1"}] * 2):
            with self.subTest(differences=differences), self.assertRaises(ValidationError):
                run_reflection(reflection_json(criterion_differences=differences), item)

    def test_fabricated_quotes_and_missing_evidence_rejected(self):
        for evidence in ([], [{"source": "student_answer", "quote": "student made an arithmetic error"}]):
            with self.subTest(evidence=evidence), self.assertRaises(ValidationError):
                run_reflection(reflection_json(findings=[finding(evidence=evidence)]))

    def test_uncertainty_requires_review_and_reason(self):
        for changes in ({"human_review_required": False, "review_reasons": []}, {"review_reasons": []}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                run_reflection(reflection_json(**changes))

    def test_immutable_inputs_and_ta_labels(self):
        item = reflection_input()
        before = item.model_dump_json()
        run_reflection(item=item)
        self.assertEqual(item.model_dump_json(), before)
        with self.assertRaises(ValidationError):
            item.ta_output.total_score = Decimal(0)
        with self.assertRaises(ValidationError):
            item.record.grader_input.rubric.criteria[0].max_score = Decimal(99)

    def test_missing_ta_label_rejected_before_provider(self):
        provider = MockLLMProvider(reflection_json())
        item = ReflectorInput(record=reflection_input().record)
        with self.assertRaises(ValueError):
            LLMReflector(provider).reflect(item)
        self.assertEqual(provider.requests, [])

    def test_impossible_ta_labels_rejected(self):
        for label in ({"total_score": "3"}, {"total_score": "1", "criterion_scores": [{"criterion_id": "fake", "score": "1"}]}, {"total_score": "1", "criterion_scores": [{"criterion_id": "c", "score": "2"}]}):
            with self.subTest(label=label), self.assertRaises(ValidationError):
                ReflectorInput(record=reflection_input().record, ta_output=label)

class AdditionalReflectorTests(unittest.TestCase):
    def test_supported_criterion_finding_with_evidence_and_suggestion(self):
        item = reflection_input(criteria=({"criterion_id": "c", "score": "1"},))
        raw = reflection_json(criterion_differences=[{"criterion_id": "c", "difference": "1"}], findings=[finding(status="supported")], human_review_required=False, review_reasons=[])
        output = run_reflection(raw, item)
        self.assertEqual(output.findings[0].status, "supported")
        self.assertIn("rubric", output.findings[0].adaptation_suggestion)

    def test_supported_grader_error_requires_rubric_and_answer_evidence(self):
        item = reflection_input(criteria=({"criterion_id": "c", "score": "1"},))
        for evidence in ([{"source": "student_answer", "quote": "2x = 4"}], [{"source": "rubric", "criterion_id": "c", "quote": "Reasoning"}]):
            with self.subTest(evidence=evidence), self.assertRaises(ValidationError):
                run_reflection(reflection_json(criterion_differences=[{"criterion_id": "c", "difference": "1"}], findings=[finding(status="supported", evidence=evidence)]), item)

    def test_no_discrepancy_claim_cannot_contradict_scores(self):
        with self.assertRaises(ValidationError):
            run_reflection(reflection_json(findings=[finding(category="no_discrepancy", status="supported")]))

    def test_ta_evidence_must_match_actual_available_label(self):
        for evidence in ([{"source": "ta_label", "criterion_id": "c", "quote": "1"}], [{"source": "ta_label", "quote": "0"}]):
            with self.subTest(evidence=evidence), self.assertRaises(ValidationError):
                run_reflection(reflection_json(findings=[finding(evidence=evidence)]))
        output = run_reflection(reflection_json(findings=[finding(evidence=[{"source": "ta_label", "quote": "1"}])]))
        self.assertEqual(output.findings[0].evidence[0].quote, "1")

    def test_missing_fields_and_modified_inputs_rejected(self):
        baseline = json.loads(reflection_json())
        for field in baseline:
            data = dict(baseline)
            del data[field]
            with self.subTest(field=field), self.assertRaises(ValidationError):
                run_reflection(json.dumps(data))
        with self.assertRaises(ValidationError):
            run_reflection(reflection_json(rubric={"total_score": 99}))

    def test_duplicate_or_nonfinite_ta_labels_rejected(self):
        for labels in (({"criterion_id": "c", "score": "0.5"},) * 2, ({"criterion_id": "c", "score": "NaN"},)):
            with self.subTest(labels=labels), self.assertRaises(ValidationError):
                reflection_input(criteria=labels)

    def test_input_revalidated_before_provider_call(self):
        item = reflection_input()
        forged_label = item.ta_output.model_copy(update={"total_score": Decimal(99)})
        provider = MockLLMProvider(reflection_json())
        with self.assertRaises(ValidationError):
            LLMReflector(provider).reflect(item.model_copy(update={"ta_output": forged_label}))
        self.assertEqual(provider.requests, [])

    def test_adaptation_rules_preserved_and_reflection_roundtrips(self):
        item = reflection_input()
        data = item.model_dump()
        data["record"]["grader_input"]["adaptation_rules"] = {"rules": ["Consider equivalent methods where permitted."]}
        item = ReflectorInput.model_validate(data)
        provider = MockLLMProvider(reflection_json())
        output = LLMReflector(provider).reflect(item)
        payload = json.loads(provider.requests[0].split("\nINPUT_JSON\n")[1])
        self.assertEqual(payload["record"]["grader_input"]["adaptation_rules"]["rules"], ["Consider equivalent methods where permitted."])
        from krugradeai.models import ReflectorOutput, RefinerInput, AdaptationRules
        restored = ReflectorOutput.model_validate_json(output.model_dump_json())
        self.assertEqual(restored.overall_score_difference, Decimal(1))
        refinement = RefinerInput(current_rules=AdaptationRules(), reflection=restored)
        self.assertEqual(refinement.reflection.findings, restored.findings)

class ReflectionSuggestionTests(unittest.TestCase):
    def test_supported_error_requires_actionable_suggestion(self):
        item = reflection_input(criteria=({"criterion_id": "c", "score": "1"},))
        raw = reflection_json(criterion_differences=[{"criterion_id": "c", "difference": "1"}], findings=[finding(status="supported", adaptation_suggestion=None)], human_review_required=False, review_reasons=[])
        with self.assertRaises(ValidationError):
            run_reflection(raw, item)

    def test_insufficient_evidence_cannot_be_definitive(self):
        with self.assertRaises(ValidationError):
            run_reflection(reflection_json(findings=[finding(status="supported", category="insufficient_evidence")]))
