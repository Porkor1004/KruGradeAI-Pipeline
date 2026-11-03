"""Eligibility boundaries use synthetic labels only, with no live providers."""
import json
import unittest

from pydantic import ValidationError

from krugradeai.models import AdaptationRule, AdaptationRules, RefinerInput
from krugradeai.refiner import validate_candidate_rules
from krugradeai.refinement_diagnostics import evidence_summary
from test_reflector import finding, reflection_input, reflection_json, run_reflection


def candidate():
    return AdaptationRules(rules=(AdaptationRule(id="read-intermediates", criterion_ids=("c",),
        instruction="Assess intermediate reasoning against the supplied criterion before drawing conclusions.",
        rationale="Supported evidence indicates a general interpretation issue, subject to the unchanged rubric."),))


class TotalOnlyRefinementTests(unittest.TestCase):
    def test_total_only_discrepancy_is_eligible_for_analysis_but_not_refinement(self):
        request = reflection_input()
        output = run_reflection(item=request)
        summary = evidence_summary(request, output)
        self.assertTrue(summary["total_only_reference"])
        self.assertTrue(output.human_review_required)
        self.assertFalse(summary["refinement_evidence_eligible"])
        self.assertEqual(output.criterion_differences, ())
        refinement = RefinerInput(current_rules=AdaptationRules(), reflection=output,
                                  rubric=request.record.grader_input.rubric)
        self.assertEqual(validate_candidate_rules(refinement, AdaptationRules()), AdaptationRules())
        with self.assertRaises(ValidationError):
            validate_candidate_rules(refinement, candidate())

    def test_partial_criterion_annotation_makes_a_total_discrepancy_eligible(self):
        # Genuine synthetic annotation on one of TWO criteria. The total and
        # all source grading data are fixed; no missing criterion score is inferred.
        from krugradeai.models import ReflectorInput
        data = reflection_input().model_dump()
        data["record"]["grader_input"]["rubric"]["criteria"] += ({"id": "d", "description": "Conclusion", "max_score": "1"},)
        data["record"]["grader_input"]["rubric"]["total_score"] = "3"
        data["record"]["grader_output"]["criterion_scores"] += ({"criterion_id": "d", "score": "0", "rationale": "Synthetic rationale"},)
        data["ta_output"] = {"total_score": "1.5", "criterion_scores": [{"criterion_id": "c", "score": "1"}]}
        request = ReflectorInput.model_validate(data)
        before = request.model_dump_json()
        output = run_reflection(reflection_json("0.5", findings=[finding(status="supported")],
            criterion_differences=[{"criterion_id": "c", "difference": "1"}],
            human_review_required=False, review_reasons=[]), request)
        summary = evidence_summary(request, output)
        self.assertTrue(summary["refinement_evidence_eligible"])
        self.assertEqual(summary["reference_criterion_label_count"], 1)
        self.assertEqual([c.criterion_id for c in output.criterion_differences], ["c"])
        refinement = RefinerInput(current_rules=AdaptationRules(), reflection=output,
                                  rubric=request.record.grader_input.rubric)
        self.assertEqual(validate_candidate_rules(refinement, candidate()), candidate())
        self.assertEqual(request.model_dump_json(), before)

    def test_single_criterion_and_extreme_totals_do_not_invent_annotations(self):
        # Even mathematically constraining totals are not new source annotations.
        for total, difference in [("0", "2"), ("2", "0")]:
            request = reflection_input(ta=total)
            with self.subTest(total=total), self.assertRaises(ValidationError):
                run_reflection(reflection_json(difference, findings=[finding(status="supported")],
                    human_review_required=False, review_reasons=[]), request)
            self.assertEqual(request.ta_output.criterion_scores, ())

    def test_repeated_ambiguous_total_discrepancies_do_not_become_supported(self):
        request = reflection_input()
        output = run_reflection(item=request)
        refinement = RefinerInput(current_rules=AdaptationRules(), reflection=output,
            additional_reflections=(output, output), rubric=request.record.grader_input.rubric)
        with self.assertRaises(ValidationError):
            validate_candidate_rules(refinement, candidate())

    def test_suspected_label_error_is_not_a_grading_adjustment(self):
        request = reflection_input()
        output = run_reflection(reflection_json(findings=[finding(category="ta_rubric_inconsistency")]), request)
        summary = evidence_summary(request, output)
        self.assertIn("suspected_label_rubric_inconsistency", summary["blocker_codes"])
        self.assertFalse(summary["refinement_evidence_eligible"])

    def test_teacher_rubric_changes_remain_forbidden_even_with_supported_evidence(self):
        request = reflection_input(criteria=({"criterion_id": "c", "score": "1"},))
        output = run_reflection(reflection_json(findings=[finding(status="supported")],
            criterion_differences=[{"criterion_id": "c", "difference": "1"}],
            human_review_required=False, review_reasons=[]), request)
        refinement = RefinerInput(current_rules=AdaptationRules(), reflection=output,
            rubric=request.record.grader_input.rubric)
        override = AdaptationRules(rules=(AdaptationRule(id="bad", criterion_ids=("c",),
            instruction="Change the rubric maximum score to 4.", rationale="Synthetic forbidden action"),))
        with self.assertRaises(ValidationError):
            validate_candidate_rules(refinement, override)

    def test_pilot_exposes_safe_blocker_metadata_without_reason_or_quote_text(self):
        from test_pilot_real import RealPilotTests, synthetic_records
        case = RealPilotTests()
        case.setUp()
        try:
            case.write_data(synthetic_records(total_only=True))
            report, _ = case.execute(behavior="discrepancy")
            rows = [o.reflection_evidence for o in report.agent_outcomes if o.agent == "reflector"]
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r["total_only_reference"] and r["review_reason_count"] == 1 for r in rows))
            self.assertTrue(all("human_review_blocks_refiner" in r["blocker_codes"] for r in rows))
            self.assertFalse(any(i.agent == "refiner" for i in report.invocations))
            self.assertNotIn("PRIVATE", report.model_dump_json())
            self.assertNotIn("synthetic step", json.dumps(rows))
        finally:
            case.doCleanups()


if __name__ == "__main__":
    unittest.main()
