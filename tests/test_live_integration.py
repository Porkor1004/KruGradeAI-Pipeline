"""Mock-only tests of the opt-in live integration runner."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from krugradeai.live_integration import main, run_integration, make_live_fixture, initial_rules
from krugradeai.synthetic import build_mock_agents
from krugradeai.mock_provider import MockLLMProvider
from krugradeai.grader import LLMGrader
from krugradeai.reflector import LLMReflector
from krugradeai.refiner import LLMRefiner


class LiveIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="krugradeai-live-runner-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_opt_in_required_before_files_or_provider_construction(self):
        with patch("krugradeai.live_integration.build_agents") as build, redirect_stderr(io.StringIO()):
            self.assertEqual(main([]), 2)
        build.assert_not_called()
        with self.assertRaises(ValueError):
            run_integration(live=False, output_dir=self.root / "unused")
        self.assertFalse((self.root / "unused").exists())

    def test_full_mocked_pipeline_has_report_and_fixed_iteration_budget(self):
        agents = build_mock_agents()
        with patch("krugradeai.live_integration.build_agents", return_value=agents):
            report = run_integration(live=True, output_dir=self.root)
        self.assertEqual(report.status, "passed")
        self.assertEqual(report.max_refinement_iterations, 1)
        self.assertEqual(report.missing_stages, ())
        self.assertTrue(report.baseline_validation_metrics)
        self.assertEqual(len(report.candidate_validation_metrics), 1)
        self.assertIsNotNone(report.selected_version)
        self.assertEqual(report.heldout_selected_metrics.coverage, 1)
        self.assertTrue(all(i.status == "success" for i in report.invocations))
        self.assertLessEqual(len(report.invocations), 7)
        self.assertTrue((self.root / "live-synthetic-integration.integration.json").is_file())
        for prompt in (*agents[1].provider.requests, *agents[2].provider.requests):
            self.assertNotIn("synthetic-test-", prompt)
            self.assertNotIn("synthetic-validation-", prompt)

    def test_agreement_runs_development_only_probes_without_changing_rules(self):
        grade = '{"criterion_scores":[{"criterion_id":"method","score":"1","rationale":"Valid transformation"},{"criterion_id":"result","score":"0","rationale":"No final value"}],"total_score":"1","overall_feedback":"Synthetic"}'
        # Use a grader scripted by response identity to keep all three labels exact.
        from krugradeai.synthetic import SyntheticMockProvider
        class AgreeingProvider(SyntheticMockProvider):
            def generate(self, prompt):
                if "synthetic-development-2" in prompt:
                    self.requests.append(prompt)
                    return grade
                return super().generate(prompt)
        reflection = json.dumps({"observations": [], "overall_score_difference": "0", "direction": "agreement", "criterion_differences": [{"criterion_id":"method","difference":"0"},{"criterion_id":"result","difference":"0"}], "findings": [], "human_review_required": False, "review_reasons": []})
        agents = (LLMGrader(AgreeingProvider("grader")), LLMReflector(MockLLMProvider(reflection)), LLMRefiner(MockLLMProvider(initial_rules().model_dump_json())))
        with patch("krugradeai.live_integration.build_agents", return_value=agents):
            report = run_integration(live=True, output_dir=self.root)
        self.assertEqual(report.status, "passed")
        self.assertEqual(report.optimization_stop_reason, "no_discrepancies")
        self.assertEqual(report.selected_version, 0)
        self.assertEqual(report.candidate_validation_metrics, ())
        self.assertEqual([i.agent for i in report.invocations if i.phase == "development_probe"], ["reflector", "refiner"])

    def test_setup_failure_saved_without_raw_error_or_credentials(self):
        with patch("krugradeai.live_integration.build_agents", side_effect=RuntimeError("SECRET_TOKEN PRIVATE_PROMPT")):
            report = run_integration(live=True, output_dir=self.root)
        self.assertEqual(report.status, "failed")
        self.assertEqual(report.setup_error_type, "RuntimeError")
        text = (self.root / "live-synthetic-integration.integration.json").read_text(encoding="utf-8")
        self.assertNotIn("SECRET_TOKEN", text)
        self.assertNotIn("PRIVATE_PROMPT", text)
        self.assertTrue(report.missing_stages)

    def test_upstream_failure_is_not_retried_or_fabricated(self):
        agents = build_mock_agents()
        with patch.object(agents[0].provider, "generate", side_effect=RuntimeError("PRIVATE")) as generate, patch("krugradeai.live_integration.build_agents", return_value=agents):
            report = run_integration(live=True, output_dir=self.root)
        self.assertEqual(report.status, "incomplete")
        self.assertIsNone(report.selected_version)
        self.assertIsNone(report.heldout_selected_metrics)
        self.assertTrue(report.failures)
        self.assertIn("reflector", report.missing_stages)
        self.assertEqual(generate.call_count, 2)  # baseline validation and development
        self.assertNotIn("PRIVATE", report.model_dump_json())

    def test_factory_wires_all_three_codex_providers_without_executing_them(self):
        from krugradeai.live_integration import build_agents
        with patch("krugradeai.live_integration.CodexCLIProvider") as provider:
            agents = build_agents(model="chosen-model", timeout=60, executable=None)
        self.assertEqual(provider.call_count, 3)
        self.assertIsInstance(agents[0], LLMGrader)
        self.assertIsInstance(agents[1], LLMReflector)
        self.assertIsInstance(agents[2], LLMRefiner)
        self.assertTrue(all(call.kwargs["model"] == "chosen-model" for call in provider.call_args_list))
        provider.return_value.generate.assert_not_called()

    def test_initial_fixture_is_synthetic_disjoint_and_immutable(self):
        data = make_live_fixture()
        self.assertEqual([len(data.development), len(data.validation), len(data.heldout)], [1, 1, 1])
        self.assertEqual(data.provenance.kind, "synthetic")
        snapshot = data.model_dump_json()
        with patch("krugradeai.live_integration.make_live_fixture", return_value=data), patch("krugradeai.live_integration.build_agents", return_value=build_mock_agents()):
            run_integration(live=True, output_dir=self.root)
        self.assertEqual(data.model_dump_json(), snapshot)

    def test_duplicate_report_blocks_calls_and_heldout_is_last(self):
        agents = build_mock_agents()
        with patch("krugradeai.live_integration.build_agents", return_value=agents) as build:
            report = run_integration(live=True, output_dir=self.root)
            with self.assertRaises(FileExistsError):
                run_integration(live=True, output_dir=self.root)
        self.assertEqual(build.call_count, 1)
        heldout = [i.index for i in report.invocations if i.phase.startswith("heldout")]
        self.assertTrue(heldout)
        self.assertEqual(max(i.index for i in report.invocations), max(heldout))

class LiveReviewTests(unittest.TestCase):
    setUp = LiveIntegrationTests.setUp

    def test_human_review_is_recorded_without_fabricating_reflection(self):
        agents = build_mock_agents()
        generate = agents[0].provider.generate
        def respond(prompt):
            if "synthetic-development-2" in prompt:
                return '{"human_review_required":true,"review_reason":"Synthetic ambiguous work"}'
            return generate(prompt)
        with patch.object(agents[0].provider, "generate", side_effect=respond), patch("krugradeai.live_integration.build_agents", return_value=agents):
            report = run_integration(live=True, output_dir=self.root)
        self.assertEqual(report.status, "incomplete")
        self.assertTrue(report.human_reviews)
        self.assertTrue(any(e.status == "human_review" for e in report.invocations))
        self.assertIn("reflector", report.missing_stages)
        self.assertIn("refiner", report.missing_stages)
        self.assertEqual(report.candidate_validation_metrics, ())
        self.assertEqual(agents[1].provider.requests, [])

    def test_invalid_reflection_not_retried_or_silently_treated_as_success(self):
        agents = build_mock_agents()
        with patch.object(agents[1].provider, "generate", return_value="not JSON PRIVATE") as generate, patch("krugradeai.live_integration.build_agents", return_value=agents):
            report = run_integration(live=True, output_dir=self.root)
        self.assertEqual(report.status, "incomplete")
        self.assertEqual(generate.call_count, 1)
        self.assertIn("reflector", report.missing_stages)
        self.assertTrue(report.failures)
        self.assertNotIn("PRIVATE", report.model_dump_json())
        self.assertEqual(agents[2].provider.requests, [])
