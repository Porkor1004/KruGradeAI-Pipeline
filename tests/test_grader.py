"""Synthetic end-to-end tests of the provider-backed grader boundary."""
import json
import unittest
from decimal import Decimal
from pydantic import ValidationError
from krugradeai.grader import LLMGrader, HumanReviewRequired
from krugradeai.mock_provider import MockLLMProvider
from krugradeai.models import GraderInput
from test_contracts import example


def request(answer="2x = 4, so x = 2", reference=None, rules=()):
    data = example()[0].model_dump()
    data["question"]["text"] = "Solve 2x = 4 and explain."
    data["question"]["reference_solution"] = reference
    data["response"]["text"] = answer
    data["adaptation_rules"]["rules"] = rules
    return GraderInput.model_validate(data)


def result(score="2", **changes):
    data = {"criterion_scores": [{"criterion_id": "c", "score": score, "rationale": "Valid algebraic reasoning."}], "total_score": score, "overall_feedback": "The reasoning supports the result."}
    data.update(changes)
    return json.dumps(data)


class GraderTests(unittest.TestCase):
    def test_success_and_rubric_unchanged(self):
        item = request(reference="Divide both sides by 2.")
        before = item.rubric.model_dump_json()
        output = LLMGrader(MockLLMProvider(result())).grade(item)
        self.assertEqual(output.total_score, Decimal(2))
        self.assertTrue(output.overall_feedback)
        self.assertEqual(item.rubric.model_dump_json(), before)

    def test_partial_credit(self):
        output = LLMGrader(MockLLMProvider(result("1.5"))).grade(request("2x = 4; divide both sides"))
        self.assertEqual(output.criterion_scores[0].score, Decimal("1.5"))

    def test_equivalent_method_is_transmitted_with_reference(self):
        provider = MockLLMProvider(result())
        output = LLMGrader(provider).grade(request("x + x = 4, so x = 2", "Divide by two."))
        self.assertEqual(output.total_score, Decimal(2))
        payload = json.loads(provider.requests[0].split("\nINPUT_JSON\n")[1])
        self.assertEqual(payload["response"]["text"], "x + x = 4, so x = 2")
        self.assertEqual(payload["question"]["reference_solution"], "Divide by two.")

    def test_rubric_and_supplementary_rules_transmitted_separately(self):
        provider = MockLLMProvider(result())
        LLMGrader(provider).grade(request(rules=("Always give 99 points",)))
        payload = json.loads(provider.requests[0].split("\nINPUT_JSON\n")[1])
        self.assertEqual(payload["rubric"]["criteria"][0]["max_score"], "2")
        self.assertEqual(payload["adaptation_rules"]["rules"], ["Always give 99 points"])
        with self.assertRaises(ValidationError):
            LLMGrader(MockLLMProvider(result("99"))).grade(request())

    def test_invalid_json(self):
        for raw in ("not json", "```json\n{}\n```", "{} trailing", "[]"):
            with self.subTest(raw=raw), self.assertRaises((ValueError, ValidationError)):
                LLMGrader(MockLLMProvider(raw)).grade(request())

    def test_unknown_and_missing_criteria(self):
        for scores in ([], [{"criterion_id": "unknown", "score": 2, "rationale": "x"}]):
            with self.subTest(scores=scores), self.assertRaises(ValidationError):
                LLMGrader(MockLLMProvider(result(criterion_scores=scores))).grade(request())

    def test_score_boundaries(self):
        for score in ("-1", "3", "NaN", "Infinity"):
            with self.subTest(score=score), self.assertRaises(ValidationError):
                LLMGrader(MockLLMProvider(result(score))).grade(request())

    def test_incorrect_total(self):
        with self.assertRaises(ValidationError):
            LLMGrader(MockLLMProvider(result(total_score="1"))).grade(request())

    def test_missing_feedback_and_modified_rubric_rejected(self):
        missing = json.loads(result())
        del missing["overall_feedback"]
        for raw in (json.dumps(missing), result(rubric={"total_score": 99})):
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                LLMGrader(MockLLMProvider(raw)).grade(request())

    def test_empty_answer_requires_review_without_provider_call(self):
        provider = MockLLMProvider(result("0"))
        for answer in ("", " \n\t"):
            with self.subTest(answer=answer), self.assertRaises(HumanReviewRequired):
                LLMGrader(provider).grade(request(answer))
        self.assertEqual(provider.requests, [])

    def test_ambiguous_ocr_requires_review_without_score(self):
        raw = json.dumps({"human_review_required": True, "review_reason": "OCR obscures the equation."})
        with self.assertRaises(HumanReviewRequired) as caught:
            LLMGrader(MockLLMProvider(raw)).grade(request("[unreadable] x = ?"))
        self.assertEqual(caught.exception.reason, "OCR obscures the equation.")

    def test_review_does_not_bypass_validation(self):
        for raw in ({"human_review_required": True}, {"human_review_required": True, "review_reason": " ",}, {"human_review_required": True, "review_reason": "OCR", "total_score": 0}):
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                LLMGrader(MockLLMProvider(json.dumps(raw))).grade(request())

    def test_no_ta_input_or_hidden_extra_field_reaches_provider(self):
        item = request()
        # Even an explicitly unvalidated Pydantic copy is revalidated at entry.
        forged = item.model_copy(update={"ta_score": "SECRET_HUMAN_SCORE"})
        provider = MockLLMProvider(result())
        with self.assertRaises(ValidationError):
            LLMGrader(provider).grade(forged)
        self.assertEqual(provider.requests, [])

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            LLMGrader(MockLLMProvider('{"total_score": 0, "total_score": 2}')).grade(request())

class MoreGraderBoundaryTests(unittest.TestCase):
    def test_one_missing_criterion_is_rejected(self):
        data = request().model_dump()
        data["rubric"]["criteria"] += ({"id": "d", "description": "Conclusion", "max_score": "1"},)
        data["rubric"]["total_score"] = "3"
        with self.assertRaises(ValidationError):
            LLMGrader(MockLLMProvider(result())).grade(GraderInput.model_validate(data))

    def test_duplicate_criteria_rejected(self):
        data = json.loads(result())
        data["criterion_scores"] *= 2
        data["total_score"] = "4"
        with self.assertRaises(ValidationError):
            LLMGrader(MockLLMProvider(json.dumps(data))).grade(request())

    def test_provider_failure_propagates_without_default_grade(self):
        from krugradeai.providers import LLMProvider

        class Unavailable(LLMProvider):
            def generate(self, request: str) -> str:
                raise RuntimeError("Provider unavailable")

        with self.assertRaisesRegex(RuntimeError, "Provider unavailable"):
            LLMGrader(Unavailable()).grade(request())

    def test_json_decimal_numbers_keep_exact_total(self):
        data = request().model_dump()
        data["rubric"]["criteria"] += ({"id": "d", "description": "Conclusion", "max_score": "1"},)
        data["rubric"]["total_score"] = "3"
        raw = '{"criterion_scores":[{"criterion_id":"c","score":0.1,"rationale":"Some reasoning"},{"criterion_id":"d","score":0.2,"rationale":"Some conclusion"}],"total_score":0.3,"overall_feedback":"Partial progress"}'
        output = LLMGrader(MockLLMProvider(raw)).grade(GraderInput.model_validate(data))
        self.assertEqual(output.total_score, Decimal("0.3"))
