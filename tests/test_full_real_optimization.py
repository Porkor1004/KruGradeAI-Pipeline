"""Full-population runner tests with 95 synthetic records; never live CLI calls."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from krugradeai.models import TAReference
from krugradeai.optimization_models import DevelopmentRecord, ValidationRecord
from krugradeai.optimize_real import run_real_optimization, _JournaledProvider
from krugradeai.real_run_plan import estimate_plan
from krugradeai.pilot_real import _Telemetry
from krugradeai.mock_provider import MockLLMProvider
from test_enriched_pilot_sampling import source_records
from test_pilot_real import mocked_agents


def full_fixture():
    template = source_records()[0].model_dump(mode="json")
    records = []
    for i in range(95):
        data = json.loads(json.dumps(template))
        rid, scope = f"synthetic-development-{i:03}", i % 8
        origin = "human_derived" if i % 5 == 4 else "ai_derived"
        data.update(id=rid, provenance=dict(source="synthetic-development", item_id=rid, label_origin=origin))
        data["question"]["id"] = f"synthetic-question-{scope}"
        data["rubric"]["id"] = f"synthetic-rubric-{scope}"
        data["response"].update(id=rid, question_id=data["question"]["id"])
        if i >= 48:
            data["ta_output"]["criterion_scores"] = []
        for score in data["ta_output"]["criterion_scores"]:
            score["provenance"].update(label_origin=origin,
                operational_role="pseudo_ground_truth" if origin == "ai_derived" else "human_reference",
                source_response_id=rid, source_question_id=data["question"]["id"])
        records.append(DevelopmentRecord.model_validate(data))
    validation = []
    for i in range(19):
        data = records[i % 8].model_dump(mode="json")
        rid = f"synthetic-validation-{i:03}"
        data.update(id=rid, split="validation", provenance=dict(
            source="synthetic-human-validation", item_id=rid, label_origin="human_derived"))
        data["response"].update(id=rid, text="Validation-only synthetic content")
        data["ta_output"] = dict(total_score="1", criterion_scores=[])
        validation.append(ValidationRecord.model_validate(data))
    return tuple(records), tuple(validation)


class FullRealOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base, self.criteria = self.root/"base", self.root/"criteria"
        self.base.mkdir(); self.criteria.mkdir()
        self.records, self.validation = full_fixture()
        for path in (self.base/"development.jsonl", self.base/"validation.jsonl", self.base/"manifest.json",
                     self.criteria/"development.jsonl", self.criteria/"criterion_manifest.json"):
            path.write_text("synthetic fixture", encoding="utf-8")

    def run_mocked(self, *, agents=None, **kwargs):
        with patch("krugradeai.optimize_real.load_enriched_development", return_value=self.records), patch(
                "krugradeai.optimize_real.load_prepared_partition", return_value=self.validation) as load, patch(
                "krugradeai.optimize_real.build_agents", return_value=agents or mocked_agents()) as build:
            result = run_real_optimization(data_directory=self.base, criteria_directory=self.criteria,
                output_root=self.root/"output", history_roots=kwargs.pop("history_roots", ()), full_development=True, **kwargs)
            if build.called:
                load.assert_called_once_with(self.base, "validation")
            return result, build

    def test_dry_run_covers_all_95_and_19_with_332_call_bound_without_outputs(self):
        plan, build = self.run_mocked(dry_run=True, ai_adjudication=True, max_calls=332)
        build.assert_not_called()
        self.assertEqual((plan["development_count"], plan["validation_count"], plan["rubric_path_count"]), (95, 19, 8))
        self.assertEqual(plan["development_label_origins"], dict(ai_derived=76, human_derived=19))
        self.assertEqual((plan["criterion_enriched_count"], plan["total_only_count"]), (48, 47))
        self.assertEqual(plan["maximum_provider_calls"], 332)
        self.assertEqual(plan["grader_only"]["total_calls"], 114)
        self.assertTrue(plan["budget_sufficient"])
        self.assertFalse(plan["resume_supported"])
        self.assertFalse((self.root/"output").exists())

    def test_dry_run_reports_insufficient_budget_but_live_rejects_before_agents(self):
        plan, _ = self.run_mocked(dry_run=True, ai_adjudication=True, max_calls=331)
        self.assertFalse(plan["budget_sufficient"])
        with patch("krugradeai.optimize_real.build_agents") as build:
            with self.assertRaisesRegex(ValueError, "332"):
                self.run_mocked(live=True, ai_adjudication=True, max_calls=331)
            build.assert_not_called()
        self.assertFalse((self.root/"output").exists())

    def test_full_population_is_sorted_and_does_not_exclude_prior_pilot_records(self):
        history = self.root/"history"/"old"
        history.mkdir(parents=True)
        (history/"report.json").write_text(json.dumps(dict(
            development_sha256=hashlib.sha256((self.base/"development.jsonl").read_bytes()).hexdigest(),
            selected_records=[dict(record_id=r.id) for r in self.records], provider_calls=0)))
        self.records = tuple(reversed(self.records))
        with patch("krugradeai.optimize_real.history_exclusions") as exclude, patch(
                "krugradeai.optimize_real.select_optimization_sample") as select:
            result, _ = self.run_mocked(live=True, max_calls=236, history_roots=(self.root/"history",))
            exclude.assert_not_called(); select.assert_not_called()
        self.assertEqual(len(result["selected_records"]), 95)
        keys = [(r["rubric_id"], r["record_id"]) for r in result["selected_records"]]
        self.assertEqual(keys, sorted(keys))
        self.assertFalse(result["excluded_record_ids"])

    def test_agreement_grades_total_only_records_and_keeps_eight_no_candidate_paths(self):
        result, _ = self.run_mocked(live=True, max_calls=236)
        self.assertEqual(len(result["provider_calls"]), 114)
        self.assertEqual(len(result["paths"]), 8)
        self.assertTrue(all(p["stop_reason"] == "no_discrepancies" for p in result["paths"]))
        self.assertTrue(all(p["g1_validation_mae"] is None for p in result["paths"]))
        self.assertEqual(sum(p["validation"][0]["metrics"]["sample_count"] for p in result["paths"]), 19)
        self.assertEqual(result["development_metrics_by_label_origin"]["human_derived"]["sample_count"], 19)
        calls = [json.loads(line) for line in (self.root/"output"/"real-optimization-001"/"provider_calls.jsonl").read_text().splitlines()]
        self.assertEqual(calls, result["provider_calls"])
        self.assertTrue((self.root/"output"/"real-optimization-001"/"run_plan.json").is_file())

    def test_aggregate_supported_records_exclude_total_only_reviews_validate_g1_ties_keep_g0(self):
        agents = mocked_agents("discrepancy", changed=True)
        before = [r.model_dump_json() for r in (*self.records, *self.validation)]
        result, _ = self.run_mocked(live=True, agents=agents, max_calls=236)
        self.assertEqual(len(result["provider_calls"]), 236)
        self.assertEqual(sum(len(p["excluded_from_refinement"]) for p in result["paths"]), 47)
        self.assertTrue(all(p["candidate_evaluated"] and p["selected_version"] == 0 for p in result["paths"]))
        self.assertTrue(all(p["validation"][1]["eligible"] for p in result["paths"]))
        self.assertEqual(sum(p["validation"][1]["metrics"]["expected_count"] for p in result["paths"]), 19)
        refinement_requests = [json.loads(prompt.split("\nINPUT_JSON\n")[1])
            for prompt in agents[2].provider.provider.requests]
        self.assertEqual(len(refinement_requests), 8)
        self.assertEqual(sum(1+len(r["additional_reflections"]) for r in refinement_requests), 48)
        self.assertTrue(all(not reflection["human_review_required"] for r in refinement_requests
            for reflection in [r["reflection"], *r["additional_reflections"]]))
        self.assertNotIn("Validation-only synthetic content", json.dumps(refinement_requests))
        self.assertEqual(before, [r.model_dump_json() for r in (*self.records, *self.validation)])

    def test_missing_validation_or_overlap_preflight_fails_before_calls(self):
        self.validation = self.validation[:1]
        with self.assertRaises(ValueError):
            self.run_mocked(dry_run=True, max_calls=332)
        _, values = full_fixture()
        overlap = ValidationRecord.model_validate(self.records[4].model_dump() | {"split": "validation"})
        self.validation = (overlap, *values[1:])
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.run_mocked(dry_run=True, max_calls=332)

    def test_full_and_sample_limits_and_live_dry_mutual_exclusion(self):
        for kwargs in (dict(full_development=True, dry_run=True, max_calls=513),
                       dict(dry_run=True, max_calls=73), dict(live=True, dry_run=True)):
            with self.assertRaises(ValueError):
                run_real_optimization(**kwargs)

    def test_historical_scalar_counts_and_unobserved_refiner_latency_are_explicit(self):
        plan = estimate_plan(self.records, self.validation, ai_adjudication=False, timeout=120, max_calls=236,
            historical_reports=[dict(provider_calls=15), dict(provider_calls=[dict(agent="grader", seconds=10,
                transport_status="success", provider_invoked=True)])])
        self.assertEqual(plan["maximum_provider_calls"], 236)
        self.assertEqual(plan["latency_by_role"]["grader"]["seconds"], 10)
        self.assertEqual(plan["latency_by_role"]["refiner"]["basis"], "assumed_30_seconds_unobserved")

    def test_journal_and_final_failure_report_preserve_safe_interrupt_metadata(self):
        telemetry = _Telemetry(lambda: 0, 1)
        telemetry.context = dict(record_id="synthetic", rubric_id="synthetic")
        telemetry.journal_path = self.root/"journal.jsonl"
        provider = MockLLMProvider("private output")
        wrapped = _JournaledProvider(provider, "grader", telemetry)
        with patch.object(provider, "generate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                wrapped.generate("private prompt")
        data = json.loads(telemetry.journal_path.read_text())
        self.assertEqual(data["error_type"], "KeyboardInterrupt")
        self.assertEqual(data["transport_status"], "failure")
        self.assertNotIn("private", telemetry.journal_path.read_text())
        agents = mocked_agents()
        with patch.object(agents[0].provider, "generate", side_effect=KeyboardInterrupt):
            report, _ = self.run_mocked(live=True, agents=agents, max_calls=236)
        self.assertEqual(report["execution_error"], "KeyboardInterrupt")
        self.assertFalse(report["resume_supported"])
        with self.assertRaises(FileExistsError):
            self.run_mocked(live=True, max_calls=236)

    def test_full_ai_path_uses_at_most_48_adjudications_and_all_roles_share_332_budget(self):
        from krugradeai.adjudicator import AIAdjudicator
        from krugradeai.optimize_real import _timed_provider
        from test_pilot_real import PilotReflector
        from test_ai_adjudication import assessment, decision

        class ReviewProvider(PilotReflector):
            def generate(self, prompt):
                data = json.loads(super().generate(prompt))
                data["findings"][0].update(category="ta_rubric_inconsistency", status="hypothesis")
                data.update(human_review_required=True, review_reasons=["Synthetic original ambiguity"])
                return json.dumps(data)

        def build(**kwargs):
            first = MockLLMProvider(json.dumps(dict(assessments=[assessment(criterion_id="method",
                evidence=[dict(source="student_answer", quote="synthetic step"),
                          dict(source="rubric", criterion_id="method", quote="Read method evidence")])])))
            second = MockLLMProvider(json.dumps(dict(decisions=[decision(criterion_id="method")],
                                                    all_review_concerns_resolved=True)))
            return AIAdjudicator(_timed_provider(first, "adjudicator_independent", kwargs["telemetry"]),
                                 _timed_provider(second, "adjudicator_comparison", kwargs["telemetry"]))

        agents = mocked_agents("discrepancy", changed=True)
        agents[1].provider = ReviewProvider("")
        with patch("krugradeai.optimize_real.build_adjudicator", side_effect=build):
            report, _ = self.run_mocked(live=True, agents=agents, ai_adjudication=True, max_calls=332)
        self.assertEqual(len(report["provider_calls"]), 332)
        self.assertEqual(sum(c["agent"] == "adjudicator_independent" for c in report["provider_calls"]), 48)
        self.assertEqual(sum(c["agent"] == "adjudicator_comparison" for c in report["provider_calls"]), 48)
        self.assertEqual(sum(e["eligible"] for e in report["ai_adjudicated_findings"]), 48)
        self.assertEqual(sum(e.get("reason_code") == "incomplete_or_unattributed_evidence"
                             for e in report["ai_adjudicated_findings"]), 47)
        self.assertEqual(sum(p["candidate_evaluated"] for p in report["paths"]), 8)
        self.assertEqual(sum(len(p["excluded_from_refinement"]) for p in report["paths"]), 47)
        self.assertTrue(all(p["selected_version"] == 0 for p in report["paths"]))
        journal = self.root/"output"/"real-optimization-001"/"provider_calls.jsonl"
        self.assertEqual(len(journal.read_text().splitlines()), 332)
