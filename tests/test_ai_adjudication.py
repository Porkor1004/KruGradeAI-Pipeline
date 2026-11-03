"""Synthetic two-stage adjudication and real-runner integration; no live calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from krugradeai.adjudicator import (AIAdjudicator, AdjudicatingReflector,
    IndependentOutput, ComparisonOutput)
from krugradeai.codex_provider import generate_output_schema
from krugradeai.mock_provider import MockLLMProvider
from krugradeai.models import ReflectorInput, ReflectorOutput
from krugradeai.optimization_models import DevelopmentRecord, RecordProvenance
from krugradeai.optimize_real import execute_optimization, summarize, run_real_optimization
from krugradeai.pilot_real import _TimedProvider
from krugradeai.reflector import LLMReflector
from test_reflector import reflection_input, reflection_json, finding
import test_optimization as opt
import test_validation as val
from test_enriched_pilot_sampling import source_records
from test_real_optimization import human_validation
from test_pilot_real import mocked_agents, PilotGrader, PilotReflector


def assessment(**changes):
    data = dict(criterion_id="c", independent_score="1", rationale="Synthetic rubric-based assessment.",
        evidence=[dict(source="student_answer", quote="2x = 4", criterion_id=None),
                  dict(source="rubric", quote="Reasoning", criterion_id="c")],
        confidence="high", uncertainty_reasons=[], rubric_consistent=True, provenance="ai_adjudicated")
    data.update(changes)
    return data


def decision(**changes):
    data = dict(criterion_id="c", independent_score="1", assessment="grader_error",
        rationale="Synthetic comparison supports a general reasoning check.", confidence="high",
        uncertainty_reasons=[], supports_adaptation=True,
        adaptation_suggestion="Check intermediate reasoning against the rubric before awarding credit.",
        provenance="ai_adjudicated")
    data.update(changes)
    return data


def adjudicator(first=None, second=None, resolved=True):
    return AIAdjudicator(MockLLMProvider(json.dumps(dict(assessments=first or [assessment()]))),
        MockLLMProvider(json.dumps(dict(decisions=second or [decision()], all_review_concerns_resolved=resolved))))


def inputs():
    request = reflection_input(criteria=[dict(criterion_id="c", score="1")])
    raw = reflection_json(criterion_differences=[dict(criterion_id="c", difference="1")],
        findings=[finding(category="ta_rubric_inconsistency")])
    reflector = LLMReflector(MockLLMProvider(raw))
    return request, reflector.reflect(request), reflector


def development(request):
    source = request.record.grader_input
    return DevelopmentRecord(id="dev-adjudicated", question=source.question, rubric=source.rubric,
        response=source.response, ta_output=request.ta_output,
        provenance=RecordProvenance(source="synthetic-ai", item_id="dev-one", label_origin="ai_derived"))


class AIAdjudicationTests(unittest.TestCase):
    def test_score_blind_first_phase_then_locked_comparison_and_provenance(self):
        request, review, _ = inputs()
        service = adjudicator()
        before = request.model_dump_json()
        result = service.adjudicate(request, review)
        self.assertFalse(result.human_review_required)
        self.assertIn("ai_adjudicated", result.findings[0].claim)
        first = json.loads(service.independent_provider.requests[0].split("\nINPUT_JSON\n")[1])
        self.assertEqual(set(first), {"question", "rubric", "student_answer", "criterion_ids"})
        self.assertNotIn("ta_output", json.dumps(first))
        self.assertNotIn("grader_output", json.dumps(first))
        second = json.loads(service.comparison_provider.requests[0].split("\nINPUT_JSON\n")[1])
        self.assertEqual(second["locked_independent_assessment"]["assessments"][0]["independent_score"], "1")
        self.assertEqual(before, request.model_dump_json())
        self.assertTrue(service.events[0]["eligible"])
        self.assertNotIn("quote", json.dumps(service.events))

    def test_locked_score_cannot_be_revised_after_labels_are_seen(self):
        request, review, _ = inputs()
        service = adjudicator(second=[decision(independent_score="0.5")])
        self.assertEqual(service.adjudicate(request, review), review)
        self.assertEqual(service.events[0]["status"], "failure")

    def test_unknown_duplicate_missing_or_out_of_range_assessments_rejected(self):
        request, review, _ = inputs()
        for values in ([assessment(criterion_id="unknown")], [assessment(), assessment()],
                       [assessment(independent_score="3")]):
            with self.subTest(values=values):
                service = adjudicator(first=values)
                self.assertEqual(service.adjudicate(request, review), review)
                self.assertFalse(service.comparison_provider.requests)

    def test_unsupported_quotes_do_not_authorize_refinement(self):
        request, review, _ = inputs()
        service = adjudicator(first=[assessment(evidence=[dict(source="student_answer", quote="fabricated"),
            dict(source="rubric", criterion_id="c", quote="Reasoning")])])
        self.assertEqual(service.adjudicate(request, review), review)
        self.assertFalse(service.comparison_provider.requests)

    def test_uncertainty_rubric_inconsistency_or_unresolved_review_stays_excluded(self):
        request, review, _ = inputs()
        for service in (adjudicator(first=[assessment(confidence="medium")]),
                        adjudicator(first=[assessment(rubric_consistent=False)]),
                        adjudicator(first=[assessment(uncertainty_reasons=["Unclear step"])]),
                        adjudicator(resolved=False),
                        adjudicator(second=[decision(supports_adaptation=False, assessment="unresolved",
                            confidence="low", uncertainty_reasons=["Unclear"], adaptation_suggestion=None)])):
            self.assertEqual(service.adjudicate(request, review), review)
            self.assertFalse(service.events[0]["eligible"])

    def test_adjudicator_cannot_relabel_or_treat_disagreement_alone_as_error(self):
        request, review, _ = inputs()
        service = adjudicator(first=[assessment(independent_score="2")],
            second=[decision(independent_score="2", assessment="grader_supported", supports_adaptation=False,
                adaptation_suggestion=None)])
        self.assertEqual(service.adjudicate(request, review), review)
        self.assertEqual(request.ta_output.total_score, 1)
        contradictory = adjudicator(first=[assessment(independent_score="2")],
            second=[decision(independent_score="2")])
        self.assertEqual(contradictory.adjudicate(request, review), review)

    def test_supported_independent_error_need_not_agree_with_source_label(self):
        request, review, _ = inputs()
        before = request.model_dump_json()
        service = adjudicator(first=[assessment(independent_score="0.5")],
            second=[decision(independent_score="0.5")])
        output = service.adjudicate(request, review)
        self.assertFalse(output.human_review_required)
        self.assertEqual(output.criterion_differences[0].difference, 1)  # Original Grader 2 minus source 1.
        self.assertIn("Independent rubric-based criterion score: 0.5", output.findings[0].claim)
        self.assertIn("unchanged source reference criterion score: 1", output.findings[0].claim)
        self.assertTrue(service.events[0]["eligible"])
        self.assertFalse(service.events[0]["source_score_agreement_required"])
        self.assertEqual(before, request.model_dump_json())

    def test_optimization002_recorded_exclusions_are_not_reclassified_as_supported(self):
        request, review, _ = inputs()
        for second, expected_code in ((decision(independent_score="0", assessment="unresolved",
                confidence="medium", uncertainty_reasons=["Unclear allocation", "Unresolved interpretation"],
                supports_adaptation=False, adaptation_suggestion=None), "comparison_unresolved"),
                (decision(independent_score="0", assessment="reference_inconsistent",
                    supports_adaptation=False, adaptation_suggestion=None),
                    "reference_inconsistency_without_supported_grader_error")):
            service = adjudicator(first=[assessment(independent_score="0")], second=[second])
            self.assertEqual(service.adjudicate(request, review), review)
            event = service.events[0]
            self.assertTrue(event["independent_validated"])
            self.assertTrue(event["comparison_validated"])
            self.assertIn(expected_code, event["blocker_codes"])
            self.assertFalse(event["eligible"])

    def test_independent_target_drives_real_refiner_contract_without_relabeling(self):
        request, _, reflector = inputs()
        record = development(request)
        validation = val.validation().model_copy(update={"provenance": RecordProvenance(
            source="synthetic-human-validation", item_id="independent-target-validation", label_origin="human_derived")})
        service = adjudicator(first=[assessment(independent_score="0.5")],
            second=[decision(independent_score="0.5")])
        refiner = opt.ScriptedRefiner((opt.candidate("v1"),))
        result = execute_optimization((record,), (validation,), (val.ValidationGrader(),
            AdjudicatingReflector(reflector, service, (record,)), refiner))
        self.assertEqual(result.paths[0].final_version, 1)
        self.assertEqual(result.paths[0].validation.best_version, 1)
        self.assertEqual(record.ta_output.total_score, 1)
        self.assertIn("criterion score: 0.5", refiner.requests[0].reflection.findings[0].claim)
        self.assertEqual(result.paths[0].history[0].records[0].grading.grader_output.total_score, 2)
        metrics = summarize(result, (record,))["development_metrics_by_label_origin"]["ai_derived"]
        self.assertEqual(metrics["mae"], "1")  # Grader minus original source, not adjudicated target.

    def test_total_only_missing_evidence_makes_no_adjudication_calls(self):
        request, review, _ = inputs()
        request = ReflectorInput(record=request.record, ta_output=dict(total_score="1"))
        service = adjudicator()
        self.assertEqual(service.adjudicate(request, review), review)
        self.assertFalse(service.independent_provider.requests)

    def test_invalid_json_and_provider_failure_retain_original_review(self):
        request, review, _ = inputs()
        for raw in ("not-json", '{"assessments":[],"assessments":[]}'):
            service = adjudicator()
            service.independent_provider.response = raw
            self.assertEqual(service.adjudicate(request, review), review)
        service = adjudicator()
        with patch.object(service.comparison_provider, "generate", side_effect=RuntimeError("private")):
            self.assertEqual(service.adjudicate(request, review), review)
        self.assertNotIn("private", json.dumps(service.events))

    def test_wire_schemas_supported_without_weakening_local_validation(self):
        for model in (IndependentOutput, ComparisonOutput):
            schema = generate_output_schema(model)
            self.assertFalse(schema["additionalProperties"])

    def test_successful_actual_engine_g1_validation_and_best_selection(self):
        request, _, reflector = inputs()
        record = development(request)
        validation = val.validation().model_copy(update={"provenance": RecordProvenance(
            source="synthetic-human-validation", item_id="val-one", label_origin="human_derived")})
        service = adjudicator()
        wrapped = AdjudicatingReflector(reflector, service, (record,))
        refiner = opt.ScriptedRefiner((opt.candidate("v1"),))
        originals = [r.model_dump_json() for r in (record, validation)]
        result = execute_optimization((record,), (validation,), (val.ValidationGrader(), wrapped, refiner))
        report = summarize(result, (record,))
        path = report["paths"][0]
        self.assertEqual(path["g0_validation_mae"], "1")
        self.assertEqual(path["g1_validation_mae"], "0")
        self.assertEqual(path["selected_version"], 1)
        self.assertTrue(path["candidate_selected"])
        self.assertIn("ai_adjudicated", refiner.requests[0].reflection.findings[0].claim)
        self.assertEqual(service.events[0]["record_label_origin"], "ai_derived")
        self.assertEqual(originals, [r.model_dump_json() for r in (record, validation)])

    def test_validation_tie_or_regression_preserves_g0_and_unresolved_stops(self):
        request, _, reflector = inputs()
        record = development(request)
        validation = val.validation().model_copy(update={"provenance": RecordProvenance(
            source="synthetic-human-validation", item_id="val-one", label_origin="human_derived")})
        for scores in ({0: "2", 1: "2"}, {0: "1", 1: "2"}):
            result = execute_optimization((record,), (validation,), (val.ValidationGrader(scores),
                AdjudicatingReflector(reflector, adjudicator(), (record,)), opt.ScriptedRefiner((opt.candidate("v1"),))))
            self.assertEqual(result.paths[0].validation.best_version, 0)
        refiner = opt.ScriptedRefiner((opt.candidate("v1"),))
        result = execute_optimization((record,), (validation,), (val.ValidationGrader(),
            AdjudicatingReflector(reflector, adjudicator(resolved=False), (record,)), refiner))
        self.assertEqual(result.paths[0].stop_reason, "human_review")
        self.assertFalse(refiner.requests)

    def test_non_development_identity_and_changed_labels_rejected(self):
        request, _, reflector = inputs()
        record = development(request)
        wrapped = AdjudicatingReflector(reflector, adjudicator(), (record,))
        data = request.model_dump()
        data["record"]["grader_input"]["response"]["id"] = "validation-private"
        with self.assertRaises(ValueError):
            wrapped.reflect(ReflectorInput.model_validate(data))
        data = request.model_dump()
        data["ta_output"]["total_score"] = "2"
        data["ta_output"]["criterion_scores"][0]["score"] = "2"
        with self.assertRaises(ValueError):
            wrapped.reflect(ReflectorInput.model_validate(data))

    def test_real_runner_default_does_not_build_adjudicator_and_opt_in_reserves_calls(self):
        rows = source_records()
        with tempfile.TemporaryDirectory() as tmp:
            base, criteria, output = Path(tmp)/"base", Path(tmp)/"criteria", Path(tmp)/"output"
            base.mkdir(); criteria.mkdir()
            for p in (base/"development.jsonl", base/"validation.jsonl", base/"manifest.json",
                      criteria/"development.jsonl", criteria/"criterion_manifest.json"):
                p.write_text("synthetic")
            with patch("krugradeai.optimize_real.load_enriched_development", return_value=rows), patch(
                    "krugradeai.optimize_real.load_prepared_partition", return_value=human_validation(rows)) as load, patch(
                    "krugradeai.optimize_real.build_agents", side_effect=lambda **kw: mocked_agents("agreement")), patch(
                    "krugradeai.optimize_real.build_adjudicator", return_value=adjudicator()) as build:
                regular = run_real_optimization(live=True, data_directory=base, criteria_directory=criteria,
                    output_root=output, history_roots=(), run_id="default")
                build.assert_not_called()
                enriched = run_real_optimization(live=True, data_directory=base, criteria_directory=criteria,
                    output_root=Path(tmp)/"ai-output", history_roots=(), run_id="ai-enabled", ai_adjudication=True)
                self.assertEqual(enriched["reserved_call_upper_bound"], regular["reserved_call_upper_bound"]+8)
                self.assertTrue(enriched["ai_adjudication_enabled"])
                self.assertTrue(all(call.args[1] == "validation" for call in load.call_args_list))
                self.assertFalse(enriched["held_out_accessed"])

    def test_two_stages_share_global_provider_budget(self):
        from krugradeai.pilot_real import _Telemetry
        request, review, _ = inputs()
        service = adjudicator()
        telemetry = _Telemetry(lambda: 0, 1)
        telemetry.context = dict(record_id=request.record.grader_input.response.id, rubric_id="r")
        original_comparison = service.comparison_provider
        service.independent_provider = _TimedProvider(service.independent_provider, "adjudicator_independent", telemetry)
        service.comparison_provider = _TimedProvider(original_comparison, "adjudicator_comparison", telemetry)
        self.assertEqual(service.adjudicate(request, review), review)
        self.assertEqual(len(telemetry.invocations), 1)
        self.assertFalse(original_comparison.requests)

    def test_live_runner_mocked_end_to_end_reports_g1_and_human_validation(self):
        class GraderProvider(PilotGrader):
            def generate(self, prompt):
                payload = json.loads(prompt.split("\nINPUT_JSON\n")[1])
                self.behavior = "agreement" if payload["adaptation_rules"]["rules"] else "discrepancy"
                self.assertion_payloads.append(payload)
                return super().generate(prompt)

        class ReviewProvider(PilotReflector):
            def generate(self, prompt):
                payload = json.loads(super().generate(prompt))
                payload["findings"][0].update(status="hypothesis", category="ta_rubric_inconsistency")
                payload.update(human_review_required=True, review_reasons=["Synthetic allocation ambiguity"])
                return json.dumps(payload)

        def agents(**kwargs):
            result = mocked_agents("discrepancy", changed=True)
            result[0].provider = GraderProvider("discrepancy")
            result[0].provider.assertion_payloads = []
            result[1].provider = ReviewProvider("")
            return result

        def service(**kwargs):
            instance = adjudicator(first=[assessment(criterion_id="method", evidence=[
                dict(source="student_answer", quote="synthetic step"),
                dict(source="rubric", criterion_id="method", quote="Read method evidence")])],
                second=[decision(criterion_id="method")])
            instance.independent_provider = _TimedProvider(instance.independent_provider,
                "adjudicator_independent", kwargs["telemetry"])
            instance.comparison_provider = _TimedProvider(instance.comparison_provider,
                "adjudicator_comparison", kwargs["telemetry"])
            return instance

        rows = source_records()
        with tempfile.TemporaryDirectory() as tmp:
            base, criteria = Path(tmp)/"base", Path(tmp)/"criteria"
            base.mkdir(); criteria.mkdir()
            for p in (base/"development.jsonl", base/"validation.jsonl", base/"manifest.json",
                      criteria/"development.jsonl", criteria/"criterion_manifest.json"):
                p.write_text("synthetic")
            with patch("krugradeai.optimize_real.load_enriched_development", return_value=rows), patch(
                    "krugradeai.optimize_real.load_prepared_partition", return_value=human_validation(rows)) as load, patch(
                    "krugradeai.optimize_real.build_agents", side_effect=agents), patch(
                    "krugradeai.optimize_real.build_adjudicator", side_effect=service):
                report = run_real_optimization(live=True, ai_adjudication=True,
                    data_directory=base, criteria_directory=criteria, history_roots=(),
                    output_root=Path(tmp)/"output", max_calls=22)
                self.assertEqual(report["status"], "candidate_evaluated")
                self.assertEqual(report["reserved_call_upper_bound"], 22)
                self.assertEqual(len(report["provider_calls"]), 22)
                self.assertEqual(len(report["ai_adjudicated_findings"]), 4)
                self.assertTrue(all(e["eligible"] for e in report["ai_adjudicated_findings"]))
                self.assertTrue(all(p["selected_version"] == 1 for p in report["paths"]))
                self.assertTrue(all(p["g0_validation_mae"] == "1" and p["g1_validation_mae"] == "0"
                                    for p in report["paths"]))
                self.assertEqual({e["agent"] for e in report["provider_calls"]},
                    {"grader", "reflector", "refiner", "adjudicator_independent", "adjudicator_comparison"})
                self.assertTrue(all(e["evidence_propagation"]["reference_values_and_provenance_match"]
                    for e in report["provider_calls"] if e["agent"] == "reflector"))
                load.assert_called_once_with(base, "validation")
                saved = (Path(tmp)/"output"/"real-optimization-001"/"report.json").read_text()
                self.assertNotIn("synthetic step", saved)
