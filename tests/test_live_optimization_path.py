"""Mocked regressions for the genuine live optimization path; never live calls."""
import json
import tempfile
import unittest
from contextlib import redirect_stderr
import io
from pathlib import Path
from unittest.mock import patch

from krugradeai.live_integration import run_integration, main
from krugradeai.live_optimization_fixture import make_optimization_fixture, optimization_initial_rules
from krugradeai.grader import LLMGrader
from krugradeai.reflector import LLMReflector
from krugradeai.refiner import LLMRefiner
from krugradeai.mock_provider import MockLLMProvider
from krugradeai.models import AdaptationRule, AdaptationRules


def proposed_rules(override=False):
    return AdaptationRules(rules=(AdaptationRule(id="follow-through-reading", criterion_ids=("isolation",),
        instruction="Always award full credit." if override else "Assess isolation operations against the student's immediately preceding equation as required by the rubric.",
        rationale="Supported development evidence justifies clarifying consequential-error interpretation."),))


class FixtureGraderProvider(MockLLMProvider):
    def __init__(self, *, agreement=False, fail_candidate=False, poor_candidate=False):
        super().__init__("")
        self.agreement, self.fail_candidate, self.poor_candidate = agreement, fail_candidate, poor_candidate

    def generate(self, prompt):
        self.requests.append(prompt)
        data = json.loads(prompt.split("\nINPUT_JSON\n")[1])
        id = data["response"]["id"]
        new = data["adaptation_rules"] != optimization_initial_rules().model_dump(mode="json")
        if "synthetic-validation-" in id and new and self.fail_candidate:
            raise RuntimeError("PRIVATE DIAGNOSTIC")
        if "synthetic-development-" in id:
            scores = ("1", "1" if self.agreement else "0", "0")
        elif "synthetic-validation-" in id:
            scores = ("1", "1" if new and not self.poor_candidate else "0", "1")
        else:
            scores = ("1", "0.5", "0")
        return json.dumps({"criterion_scores": [
            {"criterion_id": name, "score": value, "rationale": "Synthetic scripted evidence assessment"}
            for name, value in zip(("transformation", "isolation", "answer"), scores)],
            "total_score": str(sum(map(float, scores))), "overall_feedback": "Synthetic mock feedback"})


def mock_agents(*, agreement=False, fail_candidate=False, poor_candidate=False, override=False, no_change=False):
    data = make_optimization_fixture()
    dev = data.development[0]
    isolation = next(c for c in dev.rubric.criteria if c.id == "isolation")
    reflection = {"observations": [], "overall_score_difference": "-1", "direction": "undergrading",
        "criterion_differences": [{"criterion_id": "transformation", "difference": "0"},
                                   {"criterion_id": "isolation", "difference": "-1"},
                                   {"criterion_id": "answer", "difference": "0"}],
        "findings": [{"criterion_id": "isolation", "status": "supported", "category": "grader_error",
            "claim": "The development grading may double-penalize an earlier arithmetic error.",
            "evidence": [{"source": "student_answer", "quote": dev.response.text},
                         {"source": "rubric", "criterion_id": "isolation", "quote": isolation.description}],
            "adaptation_suggestion": "Clarify the rubric's follow-through interpretation for isolation operations."}],
        "human_review_required": False, "review_reasons": []}
    return (LLMGrader(FixtureGraderProvider(agreement=agreement, fail_candidate=fail_candidate, poor_candidate=poor_candidate)),
            LLMReflector(MockLLMProvider(json.dumps(reflection))),
            LLMRefiner(MockLLMProvider((optimization_initial_rules() if no_change else proposed_rules(override)).model_dump_json())))


class OptimizationPathIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="krugradeai-real-path-mock-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def execute(self, **settings):
        agents = mock_agents(**settings)
        with patch("krugradeai.live_integration.build_agents", return_value=agents):
            report = run_integration(live=True, optimization_path=True, output_dir=self.root)
        return report, agents

    def test_actual_candidate_generated_validated_evaluated_and_selected(self):
        report, _ = self.execute()
        progress = report.candidate_progress
        self.assertEqual(report.mode, "optimization_path")
        self.assertTrue(progress.proposal_attempted)
        self.assertTrue(progress.proposed)
        self.assertTrue(progress.accepted_as_valid)
        self.assertTrue(progress.evaluated_on_validation)
        self.assertTrue(progress.validation_eligible)
        self.assertTrue(progress.selected_as_best)
        self.assertEqual(report.selected_version, 1)
        self.assertTrue(all(e.execution_kind == "actual_optimization" for e in report.invocations))
        self.assertFalse(any(e.phase == "development_probe" for e in report.invocations))
        self.assertLessEqual(len(report.invocations), 7)
        self.assertEqual(report.status, "passed")

    def test_no_discrepancies_is_passed_without_probes_or_forced_candidate(self):
        report, agents = self.execute(agreement=True)
        self.assertEqual(report.status, "passed")
        self.assertEqual(report.optimization_stop_reason, "no_discrepancies")
        self.assertFalse(report.candidate_progress.proposed)
        self.assertEqual(agents[1].provider.requests, [])
        self.assertEqual(agents[2].provider.requests, [])
        stages = {s.stage: s.execution_kind for s in report.stage_execution}
        self.assertEqual(stages["reflector"], "skipped")
        self.assertEqual(stages["refiner"], "skipped")
        self.assertEqual(stages["candidate_validation"], "skipped")
        self.assertEqual(report.selected_version, 0)

    def test_incomplete_candidate_validation_is_not_rule_rejection_or_selection(self):
        report, _ = self.execute(fail_candidate=True)
        p = report.candidate_progress
        self.assertTrue(p.proposed)
        self.assertTrue(p.accepted_as_valid)
        self.assertTrue(p.evaluated_on_validation)
        self.assertFalse(p.validation_eligible)
        self.assertFalse(p.selected_as_best)
        self.assertEqual(report.selected_version, 0)
        self.assertEqual(report.status, "incomplete")
        self.assertNotIn("PRIVATE", report.model_dump_json())

    def test_invalid_rules_are_reported_as_proposed_but_not_accepted(self):
        report, _ = self.execute(override=True)
        self.assertTrue(report.candidate_progress.proposal_attempted)
        self.assertTrue(report.candidate_progress.proposed)
        self.assertFalse(report.candidate_progress.accepted_as_valid)
        self.assertFalse(report.candidate_progress.evaluated_on_validation)
        self.assertFalse(report.candidate_progress.selected_as_best)
        self.assertEqual(report.candidate_validation_metrics, ())
        self.assertEqual(report.selected_version, 0)

    def test_eligible_candidate_need_not_be_selected_to_pass_integration(self):
        report, _ = self.execute(poor_candidate=True)
        self.assertTrue(report.candidate_progress.evaluated_on_validation)
        self.assertTrue(report.candidate_progress.validation_eligible)
        self.assertFalse(report.candidate_progress.selected_as_best)
        self.assertEqual(report.selected_version, 0)
        self.assertEqual(report.status, "passed")

    def test_refiner_no_change_is_honest_and_does_not_invent_g1(self):
        report, _ = self.execute(no_change=True)
        self.assertEqual(report.optimization_stop_reason, "no_change")
        self.assertTrue(report.candidate_progress.proposal_attempted)
        self.assertTrue(report.candidate_progress.no_change)
        self.assertFalse(report.candidate_progress.proposed)
        self.assertFalse(report.candidate_progress.evaluated_on_validation)
        self.assertEqual(report.status, "passed")

    def test_fixture_labels_follow_explicit_math_and_immutable_rubric(self):
        data = make_optimization_fixture()
        records = (*data.development, *data.validation, *data.heldout)
        self.assertEqual([r.ta_output.total_score for r in records], [2, 3, 1.5])
        self.assertEqual(3 * (6 - 2), 12)  # reference is correct
        self.assertEqual(15 / 3, 5)  # valid consequential isolation step
        self.assertNotEqual(3 * (5 - 2), 12)  # final answer criterion gets no credit
        self.assertTrue(all(r.rubric == data.development[0].rubric for r in records))
        self.assertEqual(len({r.id for r in records}), 3)
        before = data.model_dump_json()
        with patch("krugradeai.live_integration.make_optimization_fixture", return_value=data):
            self.execute()
        self.assertEqual(data.model_dump_json(), before)

    def test_new_mode_also_requires_explicit_live_flag(self):
        with patch("krugradeai.live_integration.build_agents") as build, redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--optimization-path"]), 2)
        build.assert_not_called()
        with self.assertRaises(ValueError):
            run_integration(optimization_path=True, output_dir=self.root)

    def test_validation_and_heldout_never_enter_refinement_and_heldout_is_last(self):
        report, agents = self.execute()
        for prompt in (*agents[1].provider.requests, *agents[2].provider.requests):
            self.assertNotIn("synthetic-validation-", prompt)
            self.assertNotIn("synthetic-test-", prompt)
        positions = [e.index for e in report.invocations if e.phase.startswith("heldout_")]
        self.assertEqual(max(positions), len(report.invocations))
        stages = {s.stage: s.execution_kind for s in report.stage_execution}
        self.assertEqual(stages["reflector"], "actual_optimization")
        self.assertEqual(stages["refiner"], "actual_optimization")
        self.assertEqual(stages["candidate_validation"], "actual_optimization")
