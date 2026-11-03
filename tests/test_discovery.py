"""Synthetic, mocked Development-only discovery and continuation regressions."""
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from krugradeai.discover_development import (run_discovery, select_discovery_sample, score_bucket,
                                             history_exclusions, main)
from test_enriched_pilot_sampling import source_records
from test_pilot_real import mocked_agents, PilotGrader
from krugradeai.grader import LLMGrader


class DiscoverySelectionTests(unittest.TestCase):
    def test_deterministic_exclusion_diversity_and_input_preservation(self):
        records = source_records()
        before = [r.model_dump_json() for r in records]
        excluded = {records[0].id, records[1].id}
        sample = select_discovery_sample(records, excluded_ids=excluded, sample_size=6)
        self.assertEqual(sample, select_discovery_sample(reversed(records), excluded_ids=excluded, sample_size=6))
        self.assertFalse(excluded & {r.id for r in sample})
        self.assertEqual(len({r.question.id for r in sample}), 3)
        self.assertEqual({r.provenance.label_origin for r in sample}, {"ai_derived", "human_derived"})
        self.assertTrue(all(r.ta_output.criterion_scores for r in sample))
        self.assertEqual(before, [r.model_dump_json() for r in records])

    def test_exhaustion_never_repeats_or_uses_total_only(self):
        records = source_records()
        with self.assertRaisesRegex(ValueError, "Insufficient untested"):
            select_discovery_sample(records, excluded_ids={r.id for r in records}, sample_size=1)
        with self.assertRaises(ValueError):
            select_discovery_sample(records, sample_size=13)

    def test_existing_source_score_strata_are_diversified(self):
        records = []
        for index, r in enumerate(source_records()):
            data = r.model_dump()
            if data["ta_output"]["criterion_scores"]:
                value = index % 3
                data["ta_output"]["total_score"] = str(value)
                data["ta_output"]["criterion_scores"][0]["score"] = "1" if value else "0"
                data["ta_output"]["criterion_scores"][1]["score"] = "1" if value == 2 else "0"
            records.append(type(r).model_validate(data))
        sample = select_discovery_sample(records, sample_size=8)
        self.assertEqual({score_bucket(r) for r in sample}, {"zero", "full", "mid_partial"})


class DiscoveryRunnerTests(unittest.TestCase):
    def setUp(self):
        from test_pilot_real import RealPilotTests
        self.case = RealPilotTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.records = source_records()
        self.overlay = self.case.root / "criteria"
        self.overlay.mkdir()
        (self.overlay / "development.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in self.records), encoding="utf-8")
        self.history = self.case.root / "pilot005.json"
        self.excluded = [self.records[0].id, self.records[1].id]
        self.history.write_text(json.dumps(dict(
            development_sha256=hashlib.sha256((self.case.data / "development.jsonl").read_bytes()).hexdigest(),
            selected_records=[dict(record_id=rid) for rid in self.excluded])), encoding="utf-8")

    def execute(self, *, agents=None, **options):
        agents = agents or mocked_agents("discrepancy", changed=True)
        with patch("krugradeai.discover_development.load_enriched_development", return_value=self.records), patch(
                "krugradeai.discover_development.build_agents", return_value=agents):
            report = run_discovery(live=True, data_directory=self.case.data, source_criteria_directory=self.overlay,
                output_directory_path=self.case.root / "discovery", previous_reports=(self.history,), sample_size=6, **options)
        return report, agents

    def test_no_live_flag_precedes_all_files_and_providers(self):
        with patch("krugradeai.discover_development.load_enriched_development") as load, patch(
                "krugradeai.discover_development.build_agents") as build:
            with self.assertRaises(ValueError):
                run_discovery()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main([])
            self.assertEqual(error.exception.code, 2)
        load.assert_not_called()
        build.assert_not_called()

    def test_default_discovers_real_mock_differences_without_reflector_or_refiner(self):
        report, agents = self.execute()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["provider_calls"], 6)
        self.assertEqual(report["coverage"]["observed_discrepancies"], 6)
        self.assertTrue(all(i["agent"] == "grader" for i in report["invocations"]))
        self.assertEqual(report["continuation"], [])
        self.assertFalse(set(self.excluded) & {r["record_id"] for r in report["selected_records"]})
        for row in report["discrepancies"]:
            self.assertEqual(row["total_score_difference"], "-1")
            self.assertTrue(row["criterion_differences"][0]["reference_provenance"])
        self.assertNotIn("synthetic step", json.dumps(report))
        self.assertEqual(agents[1].provider.provider.requests, [])
        self.assertEqual(agents[2].provider.provider.requests, [])

    def test_optional_continuation_reuses_grades_and_existing_safety_checks(self):
        report, agents = self.execute(continue_discrepancies=True)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["cached_grades_reused"])
        self.assertEqual(len(agents[0].provider.provider.requests), 6)
        self.assertEqual(len([i for i in report["invocations"] if i["agent"] == "reflector"]), 6)
        self.assertTrue(any(p["candidate_proposed"] for p in report["continuation"]))
        self.assertFalse(report["candidates_promoted"])
        self.assertLessEqual(report["provider_calls"], report["max_provider_calls"])

    def test_full_agreement_does_not_force_continuation(self):
        report, _ = self.execute(agents=mocked_agents("agreement"), continue_discrepancies=True)
        self.assertEqual(report["coverage"]["observed_discrepancies"], 0)
        self.assertEqual(report["continuation"], [])
        self.assertEqual(report["provider_calls"], 6)

    def test_offsetting_criterion_discrepancies_detected_despite_total_agreement(self):
        class Offsetting(PilotGrader):
            def generate(self, prompt):
                value = json.loads(super().generate(prompt))
                value["total_score"] = 1
                value["criterion_scores"][0]["score"] = 0
                value["criterion_scores"][1]["score"] = 1
                return json.dumps(value)
        agents = list(mocked_agents("agreement"))
        agents[0] = LLMGrader(Offsetting("agreement"))
        report, _ = self.execute(agents=tuple(agents))
        self.assertEqual(len(report["discrepancies"]), 6)
        self.assertTrue(all(r["total_score_difference"] == "0" and len(r["criterion_differences"]) == 2 for r in report["discrepancies"]))

    def test_provider_call_limit_is_hard_and_never_weakens_review_guards(self):
        report, _ = self.execute(continue_discrepancies=True, max_calls=6)
        self.assertEqual(report["provider_calls"], 6)
        self.assertTrue(report["call_limit_reached"])
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(any(i["agent"] == "refiner" for i in report["invocations"]))

    def test_human_review_is_not_a_discrepancy_or_zero_score(self):
        report, _ = self.execute(agents=mocked_agents("review"), continue_discrepancies=True)
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(all(r["has_discrepancy"] is None and r["total_score_difference"] is None for r in report["observations"]))
        self.assertEqual(report["continuation"], [])
        self.assertNotIn("PRIVATE", json.dumps(report))

    def test_reflector_uncertainty_remains_review_not_forced_refinement(self):
        from test_pilot_real import PilotReflector
        from krugradeai.reflector import LLMReflector
        class Review(PilotReflector):
            def generate(self, prompt):
                data = json.loads(super().generate(prompt))
                data["findings"][0].update(status="hypothesis", category="insufficient_evidence")
                data.update(human_review_required=True, review_reasons=["PRIVATE REVIEW"])
                return json.dumps(data)
        agents = list(mocked_agents("discrepancy", changed=True))
        agents[1] = LLMReflector(Review(""))
        report, _ = self.execute(agents=tuple(agents), continue_discrepancies=True)
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(all(p["stop_reason"] == "human_review" for p in report["continuation"]))
        self.assertFalse(any(i["agent"] == "refiner" for i in report["invocations"]))
        self.assertNotIn("PRIVATE", json.dumps(report))

    def test_failed_grade_is_replayed_not_dropped_to_enable_refinement(self):
        failed = select_discovery_sample(self.records, excluded_ids=self.excluded, sample_size=6)[0]
        class OneFailure(PilotGrader):
            def generate(self, prompt):
                request = json.loads(prompt.split("\nINPUT_JSON\n")[1])
                if request["response"]["id"] == failed.response.id:
                    self.requests.append(prompt)
                    raise RuntimeError("PRIVATE FAILURE")
                return super().generate(prompt)
        agents = list(mocked_agents("discrepancy", changed=True))
        agents[0] = LLMGrader(OneFailure("discrepancy"))
        report, agents = self.execute(agents=tuple(agents), continue_discrepancies=True)
        self.assertEqual(len(agents[0].provider.provider.requests), 6)
        path = next(p for p in report["continuation"] if p["rubric_id"] == failed.rubric.id)
        self.assertEqual(path["stop_reason"], "agent_failure")
        self.assertFalse(path["candidate_proposed"])
        self.assertEqual(report["status"], "incomplete")
        self.assertNotIn("PRIVATE", json.dumps(report))

    def test_history_mismatch_or_partition_path_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "different Development"):
            history_exclusions((self.history,), development_sha256="other", development_ids=set())
        with self.assertRaisesRegex(ValueError, "JSON reports"):
            history_exclusions((self.case.root / "test.jsonl",), development_sha256="other", development_ids=set())

    def test_prior_discovery_records_automatically_excluded_and_outputs_exclusive(self):
        first, _ = self.execute(run_id="first")
        # 12 enriched minus 2 pilot exclusions minus 6 first-run selections leaves
        # only 4: requesting 6 again must fail instead of repeating examples.
        second, _ = self.execute(run_id="second")
        self.assertEqual(second["status"], "failed")
        self.assertEqual(second["provider_calls"], 0)
        self.assertEqual(second["selected_records"], [])
        with self.assertRaises(FileExistsError):
            self.execute(run_id="first")

    def test_missing_required_pilot005_report_fails_before_building_agents(self):
        with patch("krugradeai.discover_development.PILOT005", str(self.case.root / "missing005.json")), patch(
            "krugradeai.discover_development.load_enriched_development", return_value=self.records), patch(
            "krugradeai.discover_development.build_agents") as build:
            report = run_discovery(live=True, data_directory=self.case.data, source_criteria_directory=self.overlay,
                output_directory_path=self.case.root / "discovery", pilot_history_directory=self.case.root / "absent")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["provider_calls"], 0)
        build.assert_not_called()

    def test_only_development_metadata_and_history_files_are_opened(self):
        original = Path.open
        opened = []
        before = {p.name: p.read_bytes() for p in self.case.data.iterdir()}
        def observe(path, *args, **kwargs):
            opened.append(path.name)
            if path.name in ("validation.jsonl", "test.jsonl"):
                raise AssertionError("Forbidden partition")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", observe):
            self.execute()
        self.assertNotIn("validation.jsonl", opened)
        self.assertNotIn("test.jsonl", opened)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.case.data.iterdir()})

    def test_sample_and_call_bounds_fail_before_any_data_read(self):
        for arguments in (dict(sample_size=25), dict(sample_size=0), dict(sample_size=8, max_calls=7), dict(max_calls=73)):
            with self.subTest(arguments=arguments), patch("krugradeai.discover_development.load_enriched_development") as load:
                with self.assertRaises(ValueError):
                    run_discovery(live=True, **arguments)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
