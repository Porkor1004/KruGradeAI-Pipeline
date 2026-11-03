import unittest
from decimal import Decimal
from pydantic import ValidationError
from krugradeai.models import (Question, RubricCriterion, Rubric, StudentResponse,
    AdaptationRules, GraderInput, CriterionScore, GraderOutput, GradingRecord,
    ReflectorInput, ReflectorOutput, RefinerInput, RefinerOutput)
from krugradeai.agents import Grader, Reflector, Refiner
from krugradeai.providers import LLMProvider


def example():
    rubric = Rubric(id="r", criteria=(RubricCriterion(id="c", description="Reasoning", max_score="2"),), total_score="2")
    request = GraderInput(question=Question(id="q", text="Explain."), rubric=rubric,
        response=StudentResponse(id="s", question_id="q", text="Because."), adaptation_rules=AdaptationRules())
    output = GraderOutput(criterion_scores=(CriterionScore(criterion_id="c", score="1.5", rationale="Partial reasoning"),), total_score="1.5")
    return request, output


class SchemaTests(unittest.TestCase):
    def test_partial_credit_and_roundtrip(self):
        request, output = example()
        record = GradingRecord(grader_input=request, grader_output=output)
        self.assertEqual(record.grader_output.total_score, Decimal("1.5"))
        self.assertEqual(GradingRecord.model_validate_json(record.model_dump_json()), record)

    def test_rubric_deeply_immutable(self):
        request, _ = example()
        with self.assertRaises(ValidationError):
            request.rubric.total_score = Decimal(10)
        with self.assertRaises(ValidationError):
            request.rubric.criteria[0].max_score = Decimal(10)
        self.assertIsInstance(request.rubric.criteria, tuple)

    def test_grader_rejects_ta_labels(self):
        request, _ = example()
        for field in ("ta_score", "ta_scores", "reference_score"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                GraderInput.model_validate({**request.model_dump(), field: 1})
        with self.assertRaises(ValidationError):
            StudentResponse(id="s", question_id="q", text="x", ta_score=1)

    def test_score_boundaries(self):
        for score in ("-1", "NaN", "Infinity"):
            with self.subTest(score=score), self.assertRaises(ValidationError):
                CriterionScore(criterion_id="c", score=score, rationale="x")
        request, _ = example()
        output = GraderOutput(criterion_scores=(CriterionScore(criterion_id="c", score=3, rationale="x"),), total_score=3)
        with self.assertRaises(ValidationError):
            GradingRecord(grader_input=request, grader_output=output)

    def test_total_consistency(self):
        with self.assertRaises(ValidationError):
            Rubric(id="r", criteria=(RubricCriterion(id="c", description="x", max_score=2),), total_score=3)
        with self.assertRaises(ValidationError):
            GraderOutput(criterion_scores=(CriterionScore(criterion_id="c", score=1, rationale="x"),), total_score=2)

    def test_criteria_must_be_complete_unique_and_known(self):
        request, output = example()
        with self.assertRaises(ValidationError):
            GraderOutput(criterion_scores=(), total_score=0)
        with self.assertRaises(ValidationError):
            GraderOutput(criterion_scores=output.criterion_scores * 2, total_score=3)
        wrong = GraderOutput(criterion_scores=(CriterionScore(criterion_id="other", score=1, rationale="x"),), total_score=1)
        with self.assertRaises(ValidationError):
            GradingRecord(grader_input=request, grader_output=wrong)
        with self.assertRaises(ValidationError):
            Rubric(id="r", criteria=request.rubric.criteria * 2, total_score=4)

    def test_question_identity(self):
        request, _ = example()
        with self.assertRaises(ValidationError):
            GraderInput.model_validate({**request.model_dump(), "response": {"id": "s", "question_id": "other", "text": "x"}})

    def test_refiner_returns_rules_only(self):
        self.assertIs(RefinerOutput, AdaptationRules)
        with self.assertRaises(ValidationError):
            RefinerOutput(rules=("Be precise",), rubric=example()[0].rubric)
        self.assertEqual(RefinerOutput(rules=("Be precise",)).rules, ("Be precise",))

    def test_reflector_and_refiner_contracts(self):
        request, output = example()
        record = GradingRecord(grader_input=request, grader_output=output)
        reflection = ReflectorInput(record=record, ta_output=output)
        result = ReflectorOutput(observations=("Missing reasoning",))
        refinement = RefinerInput(current_rules=request.adaptation_rules, reflection=result)
        self.assertEqual(reflection.ta_output, output)
        self.assertEqual(refinement.reflection, result)
        bad = GraderOutput(criterion_scores=(CriterionScore(criterion_id="c", score=3, rationale="x"),), total_score=3)
        with self.assertRaises(ValidationError):
            ReflectorInput(record=record, ta_output=bad)

    def test_empty_and_extra_fields_rejected(self):
        for constructor, args in ((Question, dict(id="q", text=" ")), (Rubric, dict(id="r", criteria=(), total_score=0)), (RubricCriterion, dict(id="c", description="x", max_score=0))):
            with self.subTest(model=constructor.__name__), self.assertRaises(ValidationError):
                constructor(**args)
        with self.assertRaises(ValidationError):
            AdaptationRules(unexpected="x")


class ContractTests(unittest.TestCase):
    def test_interfaces_are_abstract_and_independent(self):
        for interface in (Grader, Reflector, Refiner, LLMProvider):
            with self.subTest(interface=interface.__name__), self.assertRaises(TypeError):
                interface()
        self.assertFalse(issubclass(Reflector, Grader))
        self.assertFalse(issubclass(Refiner, Reflector))

    def test_typed_methods(self):
        from typing import get_type_hints
        self.assertEqual(get_type_hints(Grader.grade), {"request": GraderInput, "return": GraderOutput})
        self.assertEqual(get_type_hints(Reflector.reflect), {"request": ReflectorInput, "return": ReflectorOutput})
        self.assertEqual(get_type_hints(Refiner.refine), {"request": RefinerInput, "return": AdaptationRules})
        self.assertEqual(get_type_hints(LLMProvider.generate), {"request": str, "return": str})

class AdditionalBoundaryTests(unittest.TestCase):
    def test_missing_criterion_rejected_at_record_boundary(self):
        request, output = example()
        rubric = Rubric(id="r", criteria=request.rubric.criteria + (RubricCriterion(id="d", description="Evidence", max_score=1),), total_score=3)
        complete_request = GraderInput(question=request.question, rubric=rubric, response=request.response)
        with self.assertRaises(ValidationError):
            GradingRecord(grader_input=complete_request, grader_output=output)

    def test_exact_decimal_sum(self):
        output = GraderOutput(criterion_scores=(CriterionScore(criterion_id="a", score="0.1", rationale="x"), CriterionScore(criterion_id="b", score="0.2", rationale="x")), total_score="0.3")
        self.assertEqual(output.total_score, Decimal("0.3"))
