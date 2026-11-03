"""Operational-target policy and provider-backed Refiner verification, synthetic only."""
import json
import unittest
from pydantic import ValidationError

from krugradeai.models import CriterionLabelProvenance, RefinerInput, AdaptationRules
from krugradeai.optimization_models import DevelopmentRecord, ValidationRecord, RecordProvenance
from krugradeai.optimization import OptimizationLoop
from krugradeai.reflector import LLMReflector
from krugradeai.refiner import LLMRefiner
from krugradeai.mock_provider import MockLLMProvider
from krugradeai.reflector_prompt import build_reflector_prompt
from krugradeai.adjudicator_prompt import comparison_prompt
from krugradeai.adjudicator import IndependentOutput
from test_ai_adjudication import assessment
from test_reflector import reflection_input, reflection_json, finding, run_reflection
import test_optimization as opt
import test_validation as vf


DESCRIPTION = "Reasoning: award one point for a valid intermediate equation and one point for a checked conclusion."


def operational_record(origin):
    data = opt.development().model_dump()
    data["rubric"]["criteria"][0]["description"] = DESCRIPTION
    data["provenance"] = dict(source="synthetic-development", item_id="synthetic-dev", label_origin=origin)
    data["ta_output"]["criterion_scores"][0]["provenance"] = CriterionLabelProvenance(
        source="synthetic", label_origin=origin,
        operational_role="pseudo_ground_truth" if origin == "ai_derived" else "human_reference",
        original_criterion_id="c", source_response_id="synthetic-dev", source_question_id=data["question"]["id"],
        source_locator="synthetic.jsonl:1", source_sha256="0"*64,
        extraction_evidence=("Explicit synthetic source score",)).model_dump()
    return DevelopmentRecord.model_validate(data)


def run_cycle(*, origin="ai_derived", scores=None, review=False):
    record = operational_record(origin)
    validations = []
    for i in (1, 2):
        data = vf.validation(f"val-{i}").model_dump()
        data["rubric"] = record.rubric.model_dump()
        data["provenance"] = dict(source="synthetic-human-validation", item_id=f"val-{i}", label_origin="human_derived")
        validations.append(ValidationRecord.model_validate(data))
    finding_data = finding(status="hypothesis" if review else "supported",
        claim="Operational method-credit target is supported without assuming the conclusion was checked.",
        evidence=[dict(source="student_answer", quote="2x = 4"),
                  dict(source="rubric", criterion_id="c", quote=DESCRIPTION)],
        adaptation_suggestion="Check evidence of conclusion verification separately from intermediate reasoning.")
    raw = reflection_json(criterion_differences=[dict(criterion_id="c", difference="1")],
        findings=[finding_data], human_review_required=review, review_reasons=["Unresolved reasoning"] if review else [])
    reflection_provider = MockLLMProvider(raw)
    candidate = dict(rules=[dict(id="v1", criterion_ids=["c"],
        instruction="Check for evidence of conclusion verification before treating it as shown.",
        rationale="Supported intermediate reasoning should be distinguished from checking a conclusion.")])
    refiner_provider = MockLLMProvider(json.dumps(candidate))
    before = [r.model_dump_json() for r in (record, *validations)]
    output = OptimizationLoop(grader=vf.ValidationGrader(scores),
        reflector=LLMReflector(reflection_provider), refiner=LLMRefiner(refiner_provider), max_iterations=1).run(
            (record,), validation_records=validations)
    assert before == [r.model_dump_json() for r in (record, *validations)]
    return output.paths[0], reflection_provider, refiner_provider


class FinalReflectorRefinerPolicyTests(unittest.TestCase):
    def test_both_operational_label_origins_reach_actual_llm_refiner_and_g1(self):
        for origin in ("ai_derived", "human_derived"):
            path, reflection, refinement = run_cycle(origin=origin)
            self.assertEqual(path.final_version, 1)
            self.assertEqual(path.validation.best_version, 1)
            self.assertEqual(path.validation.baseline_metrics.mae, 1)
            self.assertEqual(path.validation.evaluations[1].comparison_metrics.mae, 0)
            self.assertEqual(len(refinement.requests), 1)
            payload = json.loads(refinement.requests[0].split("\nINPUT_JSON\n")[1])
            self.assertEqual(payload["rubric"]["criteria"][0]["description"], DESCRIPTION)
            self.assertFalse(payload["reflection"]["human_review_required"])
            self.assertEqual(path.rule_versions[1].rules.rules[0].id, "v1")
            self.assertEqual(set(path.rule_versions[1].rules.model_dump()), {"rules"})
            source = json.loads(reflection.requests[0].split("\nINPUT_JSON\n")[1])
            self.assertEqual(source["ta_output"]["criterion_scores"][0]["provenance"]["label_origin"], origin)
            self.assertNotIn("val-1", refinement.requests[0])

    def test_tie_and_regression_keep_g0_despite_valid_candidate(self):
        for scores in ({0: "2", 1: "2"}, {0: "1", 1: "2"}):
            path, _, _ = run_cycle(scores=scores)
            self.assertEqual(path.final_version, 1)
            self.assertEqual(path.validation.best_version, 0)

    def test_incomplete_human_validation_cannot_select_g1(self):
        path, _, _ = run_cycle(scores={0: "2", 1: {
            "val-1-response": "1", "val-2-response": RuntimeError("PRIVATE")}})
        self.assertEqual(path.validation.best_version, 0)
        self.assertFalse(path.validation.evaluations[1].eligible)
        self.assertEqual(str(path.validation.evaluations[1].comparison_metrics.coverage), "0.5")
        self.assertNotIn("PRIVATE", path.model_dump_json())

    def test_unresolved_finding_never_forces_refiner(self):
        path, _, refiner = run_cycle(review=True)
        self.assertEqual(path.stop_reason, "human_review")
        self.assertEqual(path.final_version, 0)
        self.assertFalse(refiner.requests)

    def test_supported_data_quality_claim_needs_original_label_and_rubric_evidence(self):
        request = reflection_input(criteria=[dict(criterion_id="c", score="1")])
        for evidence in ([dict(source="rubric", criterion_id="c", quote="Reasoning")],
                         [dict(source="ta_label", criterion_id="c", quote="1")]):
            raw = reflection_json(criterion_differences=[dict(criterion_id="c", difference="1")],
                findings=[finding(category="ta_rubric_inconsistency", status="supported", evidence=evidence)])
            with self.assertRaises(ValidationError) as caught:
                run_reflection(raw, request)
            self.assertTrue(caught.exception.safe_diagnostics[0]["code"].startswith("supported_inconsistency_requires_"))

    def test_data_quality_observation_remains_review_required_and_cannot_change_rules(self):
        request = reflection_input(criteria=[dict(criterion_id="c", score="1")])
        raw = reflection_json(criterion_differences=[dict(criterion_id="c", difference="1")],
            findings=[finding(category="ta_rubric_inconsistency", status="supported", evidence=[
                dict(source="rubric", criterion_id="c", quote="Reasoning"),
                dict(source="ta_label", criterion_id="c", quote="1")])])
        output = run_reflection(raw, request)
        provider = MockLLMProvider(opt.candidate("v1").model_dump_json())
        with self.assertRaises(ValidationError):
            LLMRefiner(provider).refine(RefinerInput(current_rules=AdaptationRules(), reflection=output,
                rubric=request.record.grader_input.rubric))

    def test_prompt_prioritizes_operational_adaptation_without_rejecting_uncertainty_guards(self):
        request = reflection_input(criteria=[dict(criterion_id="c", score="1")])
        prompt = build_reflector_prompt(request)
        self.assertIn("Do not reject a target", prompt)
        self.assertIn("does not mean all-or-nothing", " ".join(prompt.split()))
        self.assertIn("Total-only labels cannot support it", prompt)
        self.assertNotIn("if a definitive cause cannot be", prompt)
        self.assertIn("not just the numerical disagreement", comparison_prompt(request,
            run_reflection(reflection_json(criterion_differences=[dict(criterion_id="c", difference="1")]), request),
            IndependentOutput(assessments=(assessment(),))))
