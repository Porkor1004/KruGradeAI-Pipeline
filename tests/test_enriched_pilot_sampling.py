"""Source-evidence sampling and propagation tests; no real data or live calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from krugradeai.models import CriterionLabelProvenance, TACriterionScore, TAReference
from krugradeai.pilot_real import (select_sample, select_enriched_sample, source_evidence_availability,
                                  run_pilot, EvidencePropagationError, _Telemetry)
from krugradeai.grader import LLMGrader
from krugradeai.reflector import LLMReflector
from krugradeai.refiner import LLMRefiner
from krugradeai.mock_provider import MockLLMProvider
from test_pilot_real import synthetic_records, mocked_agents, PilotGrader, PilotReflector


def source_records():
    records = []
    for r in synthetic_records(total_only=True):
        scores = []
        if not r.question.id.endswith("3"):
            for criterion, value in (("method", 1), ("result", 0)):
                scores.append(TACriterionScore(criterion_id=criterion, score=value, provenance=CriterionLabelProvenance(
                    source=r.provenance.source, label_origin=r.provenance.label_origin,
                    operational_role="pseudo_ground_truth" if r.provenance.label_origin == "ai_derived" else "human_reference",
                    original_criterion_id=criterion, source_response_id=r.id, source_question_id=r.question.id,
                    source_locator="synthetic-source:1", source_sha256="0" * 64,
                    extraction_evidence=("Synthetic explicit source score only",))))
        records.append(type(r).model_validate(r.model_dump() | {"ta_output": TAReference(total_score=1, criterion_scores=tuple(scores))}))
    return tuple(records)


class EnrichedSamplingTests(unittest.TestCase):
    def test_enriched_sampling_is_deterministic_balanced_and_multiscope(self):
        records = source_records()
        before = [r.model_dump_json() for r in records]
        sample = select_enriched_sample(records)
        self.assertEqual(sample, select_enriched_sample(reversed(records)))
        self.assertEqual(len({r.rubric.id for r in sample}), 3)
        self.assertEqual(sum(r.provenance.label_origin == "ai_derived" for r in sample), 3)
        self.assertTrue(all(source_evidence_availability(r)["eligible_for_enriched_sampling"] for r in sample))
        self.assertEqual(before, [r.model_dump_json() for r in records])

    def test_default_selection_is_unchanged_when_enrichment_is_available(self):
        enriched = source_records()
        base = synthetic_records(total_only=True)
        self.assertEqual([r.id for r in select_sample(enriched)], [r.id for r in select_sample(base)])

    def test_no_silent_total_only_or_legacy_label_fallback(self):
        for records in (synthetic_records(total_only=True), synthetic_records()):
            with self.assertRaisesRegex(ValueError, "Insufficient source-criterion"):
                select_enriched_sample(records)

    def test_sampling_does_not_rank_by_scores_or_answers(self):
        records = source_records()
        expected = [r.id for r in select_enriched_sample(records)]
        modified = []
        for r in records:
            data = r.model_dump()
            data["response"]["text"] = "Different synthetic content"
            data["ta_output"]["total_score"] = "0"
            for score in data["ta_output"]["criterion_scores"]:
                score["score"] = "0"
            modified.append(type(r).model_validate(data))
        self.assertEqual(expected, [r.id for r in select_enriched_sample(modified)])

    def test_fallback_preserves_available_origins_and_multiple_paths(self):
        # Only one scope has human labels; other valid evidence is AI-only.
        records = [r for r in source_records() if r.provenance.label_origin == "ai_derived" or r.question.id.endswith("0")]
        sample = select_enriched_sample(records, size=8)
        self.assertEqual(len(sample), 8)
        self.assertGreater(len({r.rubric.id for r in sample}), 1)
        self.assertEqual({r.provenance.label_origin for r in sample}, {"ai_derived", "human_derived"})
        self.assertTrue(all(r.ta_output.criterion_scores for r in sample))

    def test_full_coverage_preferred_over_partial_within_origin_scope(self):
        records = list(source_records())
        chosen = select_enriched_sample(records, size=4)[0]
        data = chosen.model_dump()
        data["ta_output"]["criterion_scores"] = data["ta_output"]["criterion_scores"][:1]
        records[records.index(chosen)] = type(chosen).model_validate(data)
        self.assertNotIn(chosen.id, [r.id for r in select_enriched_sample(records, size=4)])

    def test_opt_in_requires_explicit_supplement_before_reading(self):
        with patch("krugradeai.pilot_real.load_prepared_partition") as load, self.assertRaises(ValueError):
            run_pilot(live=True, criterion_enriched_sampling=True)
        load.assert_not_called()


class EnrichedPropagationTests(unittest.TestCase):
    def setUp(self):
        from test_pilot_real import RealPilotTests
        self.case = RealPilotTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.records = source_records()
        self.overlay = self.case.root / "criteria"
        self.overlay.mkdir()
        (self.overlay / "development.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in self.records), encoding="utf-8")

    def execute(self, agents=None):
        with patch("krugradeai.pilot_real.load_enriched_development", return_value=self.records), patch(
                "krugradeai.pilot_real.build_agents", return_value=agents or mocked_agents("discrepancy", changed=True)):
            return run_pilot(live=True, data_directory=self.case.data, output_directory_path=self.case.root / "pilot",
                source_criteria_directory=self.overlay, criterion_enriched_sampling=True)

    def test_labels_values_and_provenance_reach_both_reflector_boundaries(self):
        report = self.execute()
        self.assertEqual(report.status, "completed")
        self.assertEqual(report.sampling_mode, "criterion_enriched")
        self.assertEqual(report.sampling_coverage["source_criterion_records"], 6)
        self.assertEqual(report.max_provider_invocations, 15)
        for row in report.invocations:
            if row.agent == "reflector":
                p = row.evidence_propagation
                self.assertTrue(p["labels_reached_reflector"] and p["reference_values_and_provenance_match"])
                self.assertEqual(p["expected_criterion_count"], 2)
                self.assertEqual(p["expected_reference_sha256"], p["received_reference_sha256"])
        for row in report.agent_outcomes:
            if row.agent == "reflector":
                self.assertTrue(row.evidence_propagation["reference_values_and_provenance_match"])
                self.assertTrue(row.reflection_evidence["refinement_evidence_eligible"])
        self.assertFalse(report.candidates_promoted)

    def test_enriched_agreement_does_not_force_reflection_or_candidates(self):
        report = self.execute(mocked_agents("agreement"))
        self.assertEqual(report.status, "completed")
        self.assertEqual(len(report.invocations), 6)
        self.assertTrue(all(i.agent == "grader" for i in report.invocations))
        self.assertTrue(all(p["stop_reason"] == "no_discrepancies" and not p["candidate_proposed"] for p in report.paths))

    def test_label_rubric_uncertainty_still_blocks_refiner_with_full_evidence(self):
        class Uncertain(PilotReflector):
            def generate(self, prompt):
                data = json.loads(super().generate(prompt))
                data["findings"][0].update(category="ta_rubric_inconsistency", status="hypothesis")
                data.update(human_review_required=True, review_reasons=["PRIVATE UNCERTAINTY"])
                return json.dumps(data)
        agents = (LLMGrader(PilotGrader("discrepancy")), LLMReflector(Uncertain("")), LLMRefiner(MockLLMProvider("")))
        report = self.execute(agents)
        self.assertEqual(report.status, "incomplete")
        self.assertFalse(any(i.agent == "refiner" for i in report.invocations))
        for row in report.agent_outcomes:
            if row.agent == "reflector":
                self.assertTrue(row.evidence_propagation["reference_values_and_provenance_match"])
                self.assertIn("suspected_label_rubric_inconsistency", row.reflection_evidence["blocker_codes"])
                self.assertFalse(row.reflection_evidence["refinement_evidence_eligible"])
        self.assertNotIn("PRIVATE", report.model_dump_json())

    def test_dropped_prompt_labels_fail_before_provider_execution(self):
        from krugradeai.reflector_prompt import build_reflector_prompt
        def dropped(request):
            data = request.model_dump(mode="json")
            data["ta_output"]["criterion_scores"] = []
            return "Synthetic prompt\nINPUT_JSON\n" + json.dumps(data)
        agents = mocked_agents("discrepancy", changed=True)
        with patch("krugradeai.reflector.build_reflector_prompt", side_effect=dropped):
            report = self.execute(agents)
        attempts = [r for r in report.invocations if r.agent == "reflector"]
        self.assertTrue(attempts)
        self.assertTrue(all(not r.provider_invoked for r in attempts))
        self.assertTrue(all(r.error_type == "EvidencePropagationError" for r in attempts))
        self.assertTrue(all(not r.evidence_propagation["reference_values_and_provenance_match"] for r in attempts))
        self.assertEqual(agents[1].provider.provider.requests, [])

    def test_dropped_agent_input_labels_fail_before_reflector_method(self):
        from test_reflector import reflection_input
        request = reflection_input(criteria=({"criterion_id": "c", "score": "1"},))
        telemetry = _Telemetry(lambda: 0.0, 3)
        rid = request.record.grader_input.response.id
        telemetry.expected_references[rid] = (request.ta_output, request.record.grader_input.rubric)
        bad = type(request).model_validate(dict(record=request.record, ta_output=TAReference(total_score=1)))
        from unittest.mock import Mock
        method = Mock()
        with self.assertRaises(EvidencePropagationError):
            telemetry.wrap_agent(method, "reflector")(bad)
        method.assert_not_called()
        self.assertFalse(telemetry.outcomes[0].evidence_propagation["reference_values_and_provenance_match"])


if __name__ == "__main__":
    unittest.main()
