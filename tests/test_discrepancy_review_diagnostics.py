"""Synthetic policy/diagnostic regressions; no real answer content or live calls."""
import json
import unittest

from krugradeai.models import CriterionLabelProvenance, TAReference, TACriterionScore, RefinerInput, AdaptationRules
from krugradeai.refinement_diagnostics import evidence_summary
from krugradeai.refiner import validate_candidate_rules
from pydantic import ValidationError
from test_reflector import reflection_input, reflection_json, finding, run_reflection
from test_total_only_refinement import candidate


def pseudo_request():
    request = reflection_input()
    provenance = CriterionLabelProvenance(source="synthetic", label_origin="ai_derived",
        operational_role="pseudo_ground_truth", original_criterion_id="c", source_response_id="synthetic-source",
        source_question_id=request.record.grader_input.question.id, source_locator="synthetic.jsonl:1",
        source_sha256="0" * 64, extraction_evidence=("Explicit synthetic source field",))
    return type(request)(record=request.record, ta_output=TAReference(total_score=1,
        criterion_scores=(TACriterionScore(criterion_id="c", score=1, provenance=provenance),)))


class DiscrepancyReviewDiagnosticTests(unittest.TestCase):
    def test_complete_in_bounds_pseudo_labels_do_not_make_hypotheses_supported(self):
        request = pseudo_request()
        output = run_reflection(reflection_json(findings=[finding(category="ta_rubric_inconsistency")],
            criterion_differences=[dict(criterion_id="c", difference="1")]), request)
        summary = evidence_summary(request, output)
        self.assertTrue(summary["reference_structure"]["criterion_scores_within_bounds"])
        self.assertTrue(summary["reference_structure"]["total_consistent_with_available_labels"])
        self.assertTrue(summary["reference_structure"]["complete_criterion_coverage"])
        self.assertFalse(summary["refinement_evidence_eligible"])
        self.assertIn("suspected_label_rubric_inconsistency", summary["blocker_codes"])
        refinement = RefinerInput(current_rules=AdaptationRules(), reflection=output, rubric=request.record.grader_input.rubric)
        with self.assertRaises(ValidationError):
            validate_candidate_rules(refinement, candidate())

    def test_source_ai_provenance_alone_does_not_block_supported_findings(self):
        request = pseudo_request()
        before = request.model_dump_json()
        output = run_reflection(reflection_json(findings=[finding(status="supported")],
            criterion_differences=[dict(criterion_id="c", difference="1")],
            human_review_required=False, review_reasons=[]), request)
        summary = evidence_summary(request, output)
        self.assertFalse(summary["operational_policy"]["ai_origin_alone_requires_review"])
        self.assertTrue(summary["refinement_evidence_eligible"])
        refinement = RefinerInput(current_rules=AdaptationRules(), reflection=output, rubric=request.record.grader_input.rubric)
        self.assertEqual(validate_candidate_rules(refinement, candidate()), candidate())
        self.assertEqual(request.model_dump_json(), before)

    def test_per_finding_scope_and_evidence_types_are_recorded_without_private_text(self):
        request = pseudo_request()
        item = finding(category="ta_rubric_inconsistency", claim="PRIVATE CLAIM",
                       adaptation_suggestion="PRIVATE SUGGESTION")
        output = run_reflection(reflection_json(findings=[item],
            criterion_differences=[dict(criterion_id="c", difference="1")], review_reasons=["PRIVATE REASON"]), request)
        summary = evidence_summary(request, output)
        detail = summary["finding_diagnostics"][0]
        self.assertEqual(detail["criterion_id"], "c")
        self.assertTrue(detail["reference_criterion_label_available"])
        self.assertEqual(detail["evidence_sources"], ["rubric", "student_answer"])
        self.assertEqual(detail["rubric_evidence_criterion_ids"], ["c"])
        self.assertTrue(detail["finding_requires_review"])
        self.assertNotIn("PRIVATE", json.dumps(summary))
        self.assertNotIn("2x = 4", json.dumps(summary))

    def test_structural_checks_do_not_invent_a_discrete_partial_credit_grid(self):
        request = pseudo_request()
        data = request.model_dump()
        data["record"]["grader_input"]["rubric"]["criteria"][0]["description"] = "Reasoning. No further partial-credit allocation is supplied."
        request = type(request).model_validate(data)
        output = run_reflection(reflection_json(criterion_differences=[dict(criterion_id="c", difference="1")]), request)
        summary = evidence_summary(request, output)
        self.assertTrue(summary["reference_structure"]["checks_are_structural_not_semantic"])
        self.assertTrue(summary["reference_structure"]["criterion_scores_within_bounds"])
        self.assertTrue(output.human_review_required)


if __name__ == "__main__":
    unittest.main()
