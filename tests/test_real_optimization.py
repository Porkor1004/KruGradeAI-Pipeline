"""Real-run orchestration exercised exclusively with synthetic data and agents."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_optimization as old
import test_validation as vf
from test_enriched_pilot_sampling import source_records
from test_pilot_real import mocked_agents
from krugradeai.models import ReflectorOutput
from krugradeai.optimization import OptimizationLoop
from krugradeai.optimization_models import RecordProvenance, ValidationRecord
from krugradeai.optimize_real import (execute_optimization, run_real_optimization,
                                     select_optimization_sample, summarize)
from krugradeai.pilot_real import _Telemetry, _TimedProvider
from krugradeai.mock_provider import MockLLMProvider


def human_validation(records):
    result = []
    for index, r in enumerate({r.rubric.id: r for r in records}.values()):
        data = r.model_dump()
        data.update(id=f"validation-{index}", split="validation", provenance=dict(
            source="synthetic-human-validation", item_id=str(index), label_origin="human_derived"))
        data["response"]["id"] = f"validation-response-{index}"
        result.append(ValidationRecord.model_validate(data))
    return tuple(result)


class ReviewPolicyTests(unittest.TestCase):
    def run_case(self, policy="stop_path", all_review=False, failure=False):
        class Mixed(old.ScriptedReflector):
            def reflect(self, request):
                self.review = all_review or request.record.grader_input.response.id.startswith("review")
                return super().reflect(request)
        reflector = Mixed()
        refiner = old.ScriptedRefiner((old.candidate("v1"),))
        grader = vf.ValidationGrader()
        if failure:
            grader = old.ScriptedGrader(["2", RuntimeError("private")])
        loop = OptimizationLoop(grader=grader, reflector=reflector, refiner=refiner,
            max_iterations=1, review_policy=policy)
        output = loop.run((old.development("review"), old.development("supported")),
            validation_records=(vf.validation(),) if not failure else None)
        return output.paths[0], reflector, refiner

    def test_default_preserves_path_review_stop(self):
        path, _, refiner = self.run_case()
        self.assertEqual(path.stop_reason, "human_review")
        self.assertFalse(refiner.requests)

    def test_explicit_exclusion_allows_only_independent_supported_evidence(self):
        path, reflector, refiner = self.run_case("exclude_reviewed_records")
        self.assertEqual(path.final_version, 1)
        self.assertEqual(path.validation.best_version, 1)
        self.assertEqual(len(path.history[0].human_reviews), 1)
        request = refiner.requests[0]
        self.assertFalse(request.reflection.human_review_required)
        self.assertFalse(request.additional_reflections)
        self.assertEqual(len(reflector.requests), 2)

    def test_all_reviewed_cannot_refine(self):
        path, _, refiner = self.run_case("exclude_reviewed_records", all_review=True)
        self.assertEqual(path.stop_reason, "human_review")
        self.assertFalse(refiner.requests)

    def test_failures_remain_path_blockers(self):
        path, _, refiner = self.run_case("exclude_reviewed_records", failure=True)
        self.assertEqual(path.stop_reason, "agent_failure")
        self.assertFalse(refiner.requests)

    def test_policy_is_explicit_and_single_iteration_only(self):
        for policy, iterations in (("invalid", 1), ("exclude_reviewed_records", 2)):
            with self.assertRaises(ValueError):
                OptimizationLoop(grader=old.ScriptedGrader(), reflector=old.ScriptedReflector(),
                    refiner=old.ScriptedRefiner(), max_iterations=iterations, review_policy=policy)


class RealOptimizationTests(unittest.TestCase):
    def test_opt_in_precedes_data_or_provider_access(self):
        with patch("krugradeai.optimize_real.build_agents") as build, patch(
                "krugradeai.optimize_real.load_enriched_development") as load:
            with self.assertRaises(ValueError):
                run_real_optimization()
            build.assert_not_called()
            load.assert_not_called()

    def test_sampling_is_reproducible_fresh_multi_origin_and_path(self):
        rows = source_records()
        first = select_optimization_sample(rows)
        self.assertEqual(first, select_optimization_sample(tuple(reversed(rows))))
        self.assertEqual(len({r.rubric.id for r in first}), 2)
        self.assertEqual({r.provenance.label_origin for r in first}, {"ai_derived", "human_derived"})
        next_sample = select_optimization_sample(rows, (first[0].id,))
        self.assertNotIn(first[0].id, {r.id for r in next_sample})

    def test_human_only_validation_and_overlap_checks(self):
        sample = select_optimization_sample(source_records())
        values = human_validation(sample)
        data = values[0].model_dump()
        data["provenance"]["label_origin"] = "ai_derived"
        with self.assertRaises(ValueError):
            execute_optimization(sample, (ValidationRecord.model_validate(data),), mocked_agents())
        data = sample[0].model_dump() | {"split": "validation"}
        data["provenance"]["label_origin"] = "human_derived"
        with self.assertRaises(ValueError):
            execute_optimization(sample, (ValidationRecord.model_validate(data), *values), mocked_agents())

    def test_complete_candidate_validation_origin_metrics_and_immutable_sources(self):
        sample = select_optimization_sample(source_records())
        values = human_validation(sample)
        before = [r.model_dump_json() for r in (*sample, *values)]
        result = execute_optimization(sample, values, mocked_agents("discrepancy", changed=True))
        report = summarize(result, sample)
        self.assertTrue(all(p["candidate_evaluated"] for p in report["paths"]))
        # Unchanged validation predictions tie: retain G0, never auto-promote G1.
        self.assertTrue(all(p["selected_version"] == 0 for p in report["paths"]))
        self.assertTrue(all(m["coverage"] == "1" for m in report["development_metrics_by_label_origin"].values()))
        self.assertEqual(before, [r.model_dump_json() for r in (*sample, *values)])

    def test_budget_wrapper_never_invokes_provider_after_limit(self):
        provider = MockLLMProvider("{}")
        telemetry = _Telemetry(lambda: 0, 1)
        telemetry.context = dict(record_id="synthetic", rubric_id="synthetic")
        wrapped = _TimedProvider(provider, "grader", telemetry)
        wrapped.generate("synthetic")
        with self.assertRaises(RuntimeError):
            wrapped.generate("synthetic")
        self.assertEqual(len(telemetry.invocations), 1)

    def test_runner_loads_only_validation_and_saves_safe_report(self):
        rows = source_records()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, criteria = root / "base", root / "criteria"
            base.mkdir(); criteria.mkdir()
            for p in (base/"development.jsonl", base/"validation.jsonl", base/"manifest.json",
                      criteria/"development.jsonl", criteria/"criterion_manifest.json"):
                p.write_text("synthetic", encoding="utf-8")
            with patch("krugradeai.optimize_real.load_enriched_development", return_value=rows), patch(
                    "krugradeai.optimize_real.load_prepared_partition", return_value=human_validation(rows)) as load, patch(
                    "krugradeai.optimize_real.build_agents", return_value=mocked_agents("discrepancy", changed=True)):
                report = run_real_optimization(live=True, data_directory=base, criteria_directory=criteria,
                    output_root=root/"outputs", history_roots=(), exclude_reviewed_records=True)
                load.assert_called_once_with(base, "validation")
                self.assertFalse(report["held_out_accessed"])
                self.assertTrue(report["provider_calls"])
                text = (root/"outputs"/"real-optimization-001"/"report.json").read_text()
                self.assertNotIn('"response"', text)
                self.assertNotIn('"ta_output"', text)
                with self.assertRaises(FileExistsError):
                    run_real_optimization(live=True, data_directory=base, criteria_directory=criteria,
                        output_root=root/"outputs", history_roots=())

    def test_budget_preflight_rejects_before_provider_construction(self):
        rows = source_records()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)/"base"; base.mkdir()
            (base/"development.jsonl").write_text("synthetic")
            with patch("krugradeai.optimize_real.load_enriched_development", return_value=rows), patch(
                    "krugradeai.optimize_real.load_prepared_partition", return_value=human_validation(rows)), patch(
                    "krugradeai.optimize_real.build_agents") as build:
                with self.assertRaisesRegex(ValueError, "reserve"):
                    run_real_optimization(live=True, data_directory=base, criteria_directory=Path(tmp)/"criteria",
                        output_root=Path(tmp)/"output", history_roots=(), max_calls=1)
                build.assert_not_called()

    def test_incomplete_candidate_validation_retains_baseline(self):
        record = old.development().model_copy(update={"provenance": RecordProvenance(
            source="synthetic-development", item_id="one", label_origin="ai_derived")})
        val = vf.validation().model_copy(update={"provenance": RecordProvenance(
            source="synthetic-validation", item_id="one", label_origin="human_derived")})
        reflector = old.ScriptedReflector()
        refiner = old.ScriptedRefiner((old.candidate("v1"),))
        grader = vf.ValidationGrader({0: "2", 1: RuntimeError("private answer")})
        result = execute_optimization((record,), (val,), (grader, reflector, refiner))
        report = summarize(result, (record,))
        path = report["paths"][0]
        self.assertTrue(path["candidate_accepted"])
        self.assertTrue(path["candidate_evaluated"])
        self.assertFalse(path["validation"][1]["eligible"])
        self.assertEqual(path["selected_version"], 0)
        self.assertEqual([r.record.grader_input.response.id for r in reflector.requests], [record.response.id])
        self.assertNotIn("private answer", json.dumps(report))

    def test_no_discrepancies_never_forces_refiner(self):
        sample = select_optimization_sample(source_records())
        agents = mocked_agents("agreement")
        report = summarize(execute_optimization(sample, human_validation(sample), agents), sample)
        self.assertTrue(all(p["stop_reason"] == "no_discrepancies" for p in report["paths"]))
        self.assertTrue(all(not p["candidate_accepted"] for p in report["paths"]))
