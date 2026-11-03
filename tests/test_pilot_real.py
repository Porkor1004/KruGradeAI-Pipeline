"""Development-only pilot tests with synthetic records and mock providers."""
from contextlib import redirect_stderr
import hashlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from krugradeai.grader import LLMGrader
from krugradeai.mock_provider import MockLLMProvider
from krugradeai.models import (AdaptationRule, AdaptationRules, Question, Rubric, RubricCriterion,
                               StudentResponse, TAReference, TACriterionScore)
from krugradeai.optimization_models import DevelopmentRecord, RecordProvenance
from krugradeai.pilot_real import main, run_pilot, select_sample
from krugradeai.refiner import LLMRefiner
from krugradeai.reflector import LLMReflector


def synthetic_records(*, total_only=False):
    records = []
    for scope in range(4):
        question = Question(id=f"synthetic-question-{scope}", text="Synthetic question", reference_solution="Synthetic reference")
        rubric = Rubric(id=f"synthetic-rubric-{scope}", total_score=2, criteria=(
            RubricCriterion(id="method", description="Read method evidence", max_score=1),
            RubricCriterion(id="result", description="Read result evidence", max_score=1)))
        for origin in ("ai_derived", "human_derived"):
            for item in range(2):
                rid = f"synthetic-{scope}-{origin}-{item}"
                records.append(DevelopmentRecord(id=rid, question=question, rubric=rubric,
                    response=StudentResponse(id=rid, question_id=question.id, text="synthetic step"),
                    ta_output=TAReference(total_score=1, criterion_scores=() if total_only else (
                        TACriterionScore(criterion_id="method", score=1), TACriterionScore(criterion_id="result", score=0))),
                    provenance=RecordProvenance(source="synthetic", item_id=rid, label_origin=origin)))
    return tuple(records)


class PilotGrader(MockLLMProvider):
    def __init__(self, behavior):
        super().__init__("")
        self.behavior = behavior

    def generate(self, prompt):
        self.requests.append(prompt)
        if self.behavior == "failure":
            raise RuntimeError("PRIVATE ANSWER SECRET_TOKEN")
        if self.behavior == "bad_json":
            return "PRIVATE MALFORMED DATA"
        if self.behavior == "review":
            return json.dumps(dict(human_review_required=True, review_reason="PRIVATE ANSWER"))
        value = 1 if self.behavior == "agreement" else 0
        return json.dumps(dict(total_score=value, criterion_scores=[
            dict(criterion_id="method", score=value, rationale="Synthetic rationale"),
            dict(criterion_id="result", score=0, rationale="Synthetic rationale")], overall_feedback="Synthetic feedback"))


class PilotReflector(MockLLMProvider):
    def generate(self, prompt):
        self.requests.append(prompt)
        data = json.loads(prompt.split("\nINPUT_JSON\n")[1])
        total_only = not data["ta_output"]["criterion_scores"]
        finding = dict(criterion_id=None if total_only else "method", status="hypothesis" if total_only else "supported",
            category="insufficient_evidence" if total_only else "grader_error", claim="Synthetic analysis",
            evidence=[dict(source="student_answer", quote="synthetic step"),
                      dict(source="rubric", criterion_id="method", quote="Read method evidence")],
            adaptation_suggestion="Review reasoning against the supplied criterion")
        return json.dumps(dict(observations=[], overall_score_difference="-1", direction="undergrading",
            criterion_differences=[] if total_only else [dict(criterion_id="method", difference="-1"), dict(criterion_id="result", difference="0")],
            findings=[finding], human_review_required=total_only,
            review_reasons=["PRIVATE REVIEW CONTENT"] if total_only else []))


def mocked_agents(behavior="agreement", changed=False):
    candidate = AdaptationRules(rules=(AdaptationRule(id="read-reasoning", instruction="Consider intermediate reasoning against each supplied criterion.",
        criterion_ids=("method",), rationale="Supported development findings justify clearer evidence interpretation."),)) if changed else AdaptationRules()
    return LLMGrader(PilotGrader(behavior)), LLMReflector(PilotReflector("")), LLMRefiner(MockLLMProvider(candidate.model_dump_json()))


class RealPilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "prepared"
        self.data.mkdir()
        self.records = synthetic_records()
        self.write_data(self.records)

    def write_data(self, records):
        content = "".join(r.model_dump_json() + "\n" for r in records).encode()
        (self.data / "development.jsonl").write_bytes(content)
        (self.data / "manifest.json").write_text(json.dumps(dict(partition_sha256=dict(development=hashlib.sha256(content).hexdigest()))), encoding="utf-8")
        # No validation or held-out files exist; any attempt to load them fails.

    def execute(self, *, behavior="agreement", changed=False, **options):
        agents = mocked_agents(behavior, changed)
        ticks = itertools.count()
        with patch("krugradeai.pilot_real.build_agents", return_value=agents):
            report = run_pilot(live=True, data_directory=self.data, output_directory_path=self.root / "pilot",
                               clock=lambda: float(next(ticks)), **options)
        return report, agents

    def test_explicit_opt_in_precedes_files_and_provider_construction(self):
        with patch("krugradeai.pilot_real.load_prepared_partition") as load, patch("krugradeai.pilot_real.build_agents") as build:
            with self.assertRaises(ValueError):
                run_pilot(output_directory_path=self.root / "must-not-exist")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main([])
            self.assertEqual(error.exception.code, 2)
        load.assert_not_called()
        build.assert_not_called()
        self.assertFalse((self.root / "must-not-exist").exists())

    def test_sample_is_deterministic_balanced_across_multiple_scopes(self):
        selected = select_sample(self.records)
        self.assertEqual(selected, select_sample(reversed(self.records)))
        self.assertEqual(len(selected), 6)
        self.assertEqual(len({r.rubric.id for r in selected}), 3)
        for scope in {r.rubric.id for r in selected}:
            self.assertEqual({r.provenance.label_origin for r in selected if r.rubric.id == scope}, {"ai_derived", "human_derived"})
        for record in selected:
            self.assertEqual(record.model_dump_json(), next(r for r in self.records if r.id == record.id).model_dump_json())

    def test_no_discrepancies_stops_honestly_without_reflection_or_refinement(self):
        report, agents = self.execute()
        self.assertEqual(report.status, "completed")
        self.assertEqual(len(report.invocations), 6)
        self.assertTrue(all(p["stop_reason"] == "no_discrepancies" for p in report.paths))
        self.assertFalse(any(p["candidate_proposed"] for p in report.paths))
        self.assertEqual(agents[1].provider.provider.requests, [])
        self.assertEqual(agents[2].provider.provider.requests, [])

    def test_real_agents_can_reflect_refine_without_validation_or_promotion(self):
        report, _ = self.execute(behavior="discrepancy", changed=True)
        self.assertEqual(report.status, "completed")
        self.assertEqual([sum(i.agent == role for i in report.invocations) for role in ("grader", "reflector", "refiner")], [6, 6, 3])
        self.assertTrue(all(p["candidate_changed"] for p in report.paths))
        self.assertTrue(all(p["stop_reason"] == "max_iterations" for p in report.paths))
        self.assertFalse(report.candidates_promoted)
        self.assertTrue(all(not p["candidate_validated_on_validation"] and not p["candidate_selected_as_best"] for p in report.paths))
        self.assertEqual(list((self.root / "pilot" / report.run_id).iterdir()), [self.root / "pilot" / report.run_id / "report.json"])
        self.assertNotIn("read-reasoning", report.model_dump_json())

    def test_latency_is_measured_per_invocation_and_total(self):
        report, _ = self.execute(behavior="discrepancy")
        self.assertTrue(all(i.seconds == 1 for i in report.invocations))
        self.assertGreater(report.total_seconds, sum(i.seconds for i in report.invocations))
        self.assertEqual(report.runtime_estimate["provider_mean_seconds"], dict(grader=1, reflector=1, refiner=1))
        self.assertEqual(report.runtime_estimate["one_development_iteration_seconds"], len(self.records) * 2 + 4)
        self.assertFalse(report.runtime_estimate["includes_validation_or_heldout"])

    def test_transport_failure_is_sanitized_and_not_retried(self):
        report, _ = self.execute(behavior="failure")
        self.assertEqual(report.status, "incomplete")
        self.assertEqual(len(report.invocations), 6)
        self.assertTrue(all(i.transport_status == "failure" for i in report.invocations))
        self.assertIsNone(report.runtime_estimate["one_development_iteration_seconds"])
        self.assertNotIn("PRIVATE", report.model_dump_json())
        self.assertNotIn("SECRET_TOKEN", report.model_dump_json())

    def test_malformed_json_distinguishes_transport_from_agent_failure(self):
        report, _ = self.execute(behavior="bad_json")
        self.assertEqual(report.status, "incomplete")
        self.assertTrue(all(i.transport_status == "success" for i in report.invocations))
        self.assertTrue(all(o.status == "failure" for o in report.agent_outcomes))

    def test_grader_human_review_is_preserved_without_fabricating_scores(self):
        report, _ = self.execute(behavior="review")
        self.assertEqual(report.status, "incomplete")
        self.assertTrue(all(o.status == "human_review" for o in report.agent_outcomes))
        self.assertTrue(all(not r["graded"] and r["discrepancy"] is None for p in report.paths for r in p["records"]))
        self.assertNotIn("PRIVATE", report.model_dump_json())

    def test_total_only_discrepancies_keep_existing_evidence_guard(self):
        self.write_data(synthetic_records(total_only=True))
        report, _ = self.execute(behavior="discrepancy")
        self.assertEqual(report.status, "incomplete")
        self.assertTrue(all(p["stop_reason"] == "human_review" for p in report.paths))
        self.assertFalse(any(i.agent == "refiner" for i in report.invocations))
        self.assertTrue(any(o.status == "human_review" for o in report.agent_outcomes if o.agent == "reflector"))

    def test_unchanged_refiner_does_not_create_a_candidate(self):
        report, _ = self.execute(behavior="discrepancy")
        self.assertTrue(all(p["stop_reason"] == "no_change" for p in report.paths))
        self.assertTrue(all(p["refiner_returned_rules"] and not p["candidate_proposed"] for p in report.paths))

    def test_only_development_files_are_opened_and_inputs_remain_unchanged(self):
        before = {p.name: p.read_bytes() for p in self.data.iterdir()}
        opened = []
        original = Path.open

        def observe(path, *args, **kwargs):
            opened.append(path.name)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", observe):
            self.execute()
        self.assertNotIn("validation.jsonl", opened)
        self.assertNotIn("test.jsonl", opened)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.data.iterdir()})

    def test_existing_run_is_not_overwritten_or_reexecuted(self):
        report, _ = self.execute()
        path = self.root / "pilot" / report.run_id / "report.json"
        before = path.read_bytes()
        with patch("krugradeai.pilot_real.build_agents") as build, self.assertRaises(FileExistsError):
            run_pilot(live=True, data_directory=self.data, output_directory_path=self.root / "pilot")
        build.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_provider_setup_failure_is_reported_safely(self):
        with patch("krugradeai.pilot_real.build_agents", side_effect=RuntimeError("PRIVATE SECRET")):
            report = run_pilot(live=True, data_directory=self.data, output_directory_path=self.root / "pilot")
        self.assertEqual(report.status, "failed")
        self.assertEqual(report.setup_error_type, "RuntimeError")
        self.assertEqual(report.invocations, ())
        self.assertNotIn("PRIVATE", report.model_dump_json())

    def test_output_inside_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            run_pilot(live=True, data_directory=self.data, output_directory_path=self.data / "pilot")

    def test_size_and_timeout_bounds(self):
        for size in (0, 3, 5, 9, 95):
            with self.assertRaises(ValueError):
                select_sample(self.records, size=size)
        with self.assertRaises(ValueError):
            run_pilot(live=True, timeout=0)


if __name__ == "__main__":
    unittest.main()
