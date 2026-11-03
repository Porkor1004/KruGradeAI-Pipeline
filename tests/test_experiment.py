"""Phase 1 experiment/held-out/export tests with fully synthetic data only."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError

from krugradeai.experiment import Phase1Experiment, ExperimentAlreadyRunError
from krugradeai.experiment_models import ExperimentDataset, ExperimentResult, GuidelinesArtifact
from krugradeai.guidelines_export import export_guidelines
from krugradeai.synthetic import make_synthetic_dataset, build_mock_agents
from krugradeai.optimization import OptimizationLoop
from krugradeai.optimization_models import ValidationConfig
from krugradeai.models import AdaptationRules


def components(maximum=5):
    grader, reflector, refiner = build_mock_agents()
    optimizer = OptimizationLoop(grader=grader, reflector=reflector, refiner=refiner,
                                 max_iterations=maximum, validation_config=ValidationConfig(patience=2, qwk_categories=("0", "0.5", "1", "1.5", "2")))
    return optimizer, grader, reflector, refiner


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="krugradeai-experiment-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def execute(self, dataset=None, id="synthetic-check", maximum=5):
        optimizer, grader, reflector, refiner = components(maximum)
        runner = Phase1Experiment(optimizer=optimizer, heldout_grader=grader, ledger_directory=self.root / "ledger")
        result = runner.run(id, dataset or make_synthetic_dataset(), runtime="mock")
        return result, runner, (optimizer, grader, reflector, refiner)

    def test_complete_pipeline_selection_early_stopping_and_heldout_metrics(self):
        result, _, _ = self.execute()
        path = result.optimization.paths[0]
        self.assertEqual(path.stop_reason, "patience")
        self.assertEqual(len(path.history), 3)
        self.assertEqual(path.validation.best_version, 1)
        self.assertEqual(result.heldout[0].selected_version, 1)
        self.assertEqual(result.heldout[0].baseline.metrics.mae, 0.5)
        self.assertEqual(result.heldout[0].selected.metrics.mae, 0)
        self.assertEqual(result.heldout[0].selected.metrics.exact_match, 1)
        self.assertEqual(result.heldout[0].selected.metrics.coverage, 1)
        self.assertEqual(result.status, "completed")
        self.assertEqual(ExperimentResult.model_validate_json(result.model_dump_json()), result)

    def test_fixtures_are_synthetic_distinct_multicriterion_partial_credit(self):
        data = make_synthetic_dataset()
        self.assertEqual(data.provenance.kind, "synthetic")
        records = (*data.development, *data.validation, *data.heldout)
        self.assertEqual(len({r.id for r in records}), len(records))
        self.assertTrue(all("synthetic" in r.provenance.source for r in records))
        self.assertTrue(all(len(r.rubric.criteria) >= 2 for r in records))
        self.assertTrue(any(r.ta_output.total_score % 1 for r in records))
        self.assertEqual({r.split for r in records}, {"development", "validation", "test"})

    def test_test_records_only_graded_after_selection_and_not_sent_upstream(self):
        data = make_synthetic_dataset()
        optimizer, grader, reflector, refiner = components()
        events = []
        original_run = optimizer.run
        def optimize(*args, **kwargs):
            result = original_run(*args, **kwargs)
            self.assertFalse(any("synthetic-test-" in p for p in grader.provider.requests))
            events.append("selected")
            return result
        original_grade = grader.grade
        def grade(request):
            if "synthetic-test-" in request.response.id:
                self.assertEqual(events[0], "selected")
                events.append("heldout")
            return original_grade(request)
        with patch.object(optimizer, "run", side_effect=optimize), patch.object(grader, "grade", side_effect=grade):
            result = Phase1Experiment(optimizer=optimizer, heldout_grader=grader, ledger_directory=self.root / "ledger").run("isolated", data, runtime="mock")
        for prompt in (*reflector.provider.requests, *refiner.provider.requests):
            self.assertNotIn("synthetic-test-", prompt)
            self.assertNotIn("synthetic-validation-", prompt)
        self.assertEqual(events.count("heldout"), 2 * len(data.heldout))
        self.assertEqual(result.optimization.paths[0].final_version, 3)
        self.assertEqual(result.optimization.paths[0].validation.best_version, 1)

    def test_same_experiment_cannot_run_again_even_with_new_runner(self):
        data = make_synthetic_dataset()
        _, runner, agents = self.execute(data)
        before = len(agents[1].provider.requests)
        with self.assertRaises(ExperimentAlreadyRunError):
            runner.run("synthetic-check", data, runtime="mock")
        replacement = Phase1Experiment(optimizer=agents[0], heldout_grader=agents[1], ledger_directory=self.root / "ledger")
        with self.assertRaises(ExperimentAlreadyRunError):
            replacement.run("synthetic-check", data, runtime="mock")
        self.assertEqual(len(agents[1].provider.requests), before)

    def test_all_three_partition_overlaps_rejected_before_calls(self):
        original = make_synthetic_dataset()
        mutations = []
        for left, right in (("development", "validation"), ("development", "heldout"), ("validation", "heldout")):
            for identity in ("id", "response", "provenance"):
                data = original.model_dump()
                if identity == "response":
                    data[right][0]["response"]["id"] = data[left][0]["response"]["id"]
                else:
                    data[right][0][identity] = data[left][0][identity]
                mutations.append(data)
        for data in mutations:
            with self.subTest(data=data), self.assertRaises((ValueError, ValidationError)):
                self.execute(ExperimentDataset.model_validate(data))

    def test_original_rubric_answers_and_labels_remain_immutable(self):
        data = make_synthetic_dataset()
        before = data.model_dump_json()
        self.execute(data)
        self.assertEqual(data.model_dump_json(), before)
        for record in (*data.development, *data.validation, *data.heldout):
            with self.assertRaises(ValidationError):
                record.rubric.criteria[0].max_score = 99
            with self.assertRaises(ValidationError):
                record.ta_output.total_score = 99

    def test_versioned_export_separates_rubric_and_preserves_selected_rules(self):
        result, _, _ = self.execute()
        paths = export_guidelines(result, self.root / "exports")
        self.assertEqual(len(paths), 1)
        artifact = GuidelinesArtifact.model_validate_json(paths[0].read_text(encoding="utf-8"))
        self.assertEqual(artifact.guideline_version, 1)
        self.assertEqual(artifact.rules, result.optimization.paths[0].validation.best_rules)
        self.assertEqual(artifact.provenance.kind, "synthetic")
        self.assertEqual(artifact.heldout_selected_metrics.mae, 0)
        self.assertEqual(len(artifact.rubric_scope.sha256), 64)
        raw = json.loads(paths[0].read_text(encoding="utf-8"))
        self.assertNotIn("rubric", raw)
        self.assertNotIn("student_answers", raw)
        with self.assertRaises(FileExistsError):
            export_guidelines(result, self.root / "exports")

    def test_missing_best_skips_heldout_and_exports_no_invented_guidelines(self):
        from krugradeai.grader import HumanReviewRequired
        optimizer, grader, reflector, refiner = components(1)
        original = grader.grade
        def grade(request):
            if "synthetic-validation-" in request.response.id:
                raise HumanReviewRequired("Synthetic unavailable label evidence")
            return original(request)
        with patch.object(grader, "grade", side_effect=grade):
            result = Phase1Experiment(optimizer=optimizer, heldout_grader=grader, ledger_directory=self.root / "ledger").run("no-best", make_synthetic_dataset(), runtime="mock")
        self.assertEqual(result.heldout[0].status, "no_valid_best")
        self.assertIsNone(result.heldout[0].baseline)
        self.assertIsNone(result.heldout[0].selected)
        self.assertFalse(any("synthetic-test-" in p for p in grader.provider.requests))
        self.assertEqual(export_guidelines(result, self.root / "exports"), ())

    def test_multiple_rubric_paths_and_reproducible_mock_results(self):
        data = make_synthetic_dataset(multiple_rubrics=True)
        first, _, _ = self.execute(data)
        optimizer, grader, _, _ = components()
        second = Phase1Experiment(optimizer=optimizer, heldout_grader=grader, ledger_directory=self.root / "another-ledger").run("synthetic-check", data, runtime="mock")
        self.assertEqual(first, second)
        self.assertEqual(len(first.heldout), 2)
        self.assertTrue(all(p.selected_version == 1 for p in first.heldout))

    def test_selected_baseline_reuses_single_final_grading_pass(self):
        result, _, agents = self.execute(maximum=0)
        self.assertEqual(result.heldout[0].selected_version, 0)
        self.assertEqual(result.heldout[0].baseline, result.heldout[0].selected)
        self.assertEqual(sum("synthetic-test-" in p for p in agents[1].provider.requests), 2)

    def test_heldout_failure_preserves_selection_and_reports_coverage(self):
        optimizer, grader, _, _ = components()
        original = grader.grade
        def grade(request):
            if "synthetic-test-1" in request.response.id:
                raise RuntimeError("PRIVATE")
            return original(request)
        with patch.object(grader, "grade", side_effect=grade):
            result = Phase1Experiment(optimizer=optimizer, heldout_grader=grader, ledger_directory=self.root / "ledger").run("failure", make_synthetic_dataset(), runtime="mock")
        self.assertEqual(result.heldout[0].selected_version, 1)
        self.assertEqual(result.heldout[0].selected.metrics.coverage, 0.5)
        self.assertEqual(result.heldout[0].status, "incomplete")
        self.assertNotIn("PRIVATE", result.model_dump_json())

    def test_demo_live_mode_is_opt_in_and_bounded(self):
        from krugradeai.demo import main
        import io
        from contextlib import redirect_stdout
        with patch("krugradeai.demo.build_live_agents") as live, redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--output-dir", str(self.root / "demo")]), 0)
        live.assert_not_called()
        with self.assertRaises(SystemExit):
            main(["--live", "--max-iterations", "2"])

class ExperimentBoundaryTests(unittest.TestCase):
    setUp = ExperimentTests.setUp

    def test_invalid_selection_rejected_before_heldout_calls(self):
        data = make_synthetic_dataset()
        optimizer, grader, _, _ = components(1)
        valid = optimizer.run(data.development, validation_records=data.validation)
        path = valid.paths[0]
        invalid_summary = path.validation.model_copy(update={"best_version": 999})
        invalid_path = path.model_copy(update={"validation": invalid_summary})
        invalid = valid.model_copy(update={"paths": (invalid_path,)})
        with patch.object(optimizer, "run", return_value=invalid):
            with self.assertRaises(ValidationError):
                Phase1Experiment(optimizer=optimizer, heldout_grader=grader, ledger_directory=self.root).run("invalid-selection", data, runtime="mock")
        self.assertFalse(any("synthetic-test-" in prompt for prompt in grader.provider.requests))

    def test_worse_heldout_performance_never_changes_selected_version(self):
        from krugradeai.models import GraderOutput
        optimizer, grader, _, _ = components()
        original = grader.grade
        def grade(request):
            if "synthetic-test-" in request.response.id and request.adaptation_rules.rules:
                return GraderOutput(criterion_scores=tuple({"criterion_id": c.id, "score": "0", "rationale": "Deliberately poor synthetic test result"} for c in request.rubric.criteria), total_score="0", overall_feedback="Synthetic failure scenario")
            return original(request)
        with patch.object(grader, "grade", side_effect=grade):
            result = Phase1Experiment(optimizer=optimizer, heldout_grader=grader, ledger_directory=self.root).run("worse-heldout", make_synthetic_dataset(), runtime="mock")
        self.assertGreater(result.heldout[0].selected.metrics.mae, result.heldout[0].baseline.metrics.mae)
        self.assertEqual(result.heldout[0].selected_version, 1)
        self.assertEqual(result.optimization.paths[0].validation.best_version, 1)

    def test_failed_attempt_remains_consumed_without_automatic_retry(self):
        optimizer, grader, _, _ = components()
        data = make_synthetic_dataset()
        runner = Phase1Experiment(optimizer=optimizer, heldout_grader=grader, ledger_directory=self.root)
        with patch.object(optimizer, "run", side_effect=RuntimeError("PRIVATE")):
            with self.assertRaises(RuntimeError):
                runner.run("failed-attempt", data, runtime="mock")
        receipt = (self.root / "failed-attempt.receipt.json").read_text(encoding="utf-8")
        self.assertNotIn("PRIVATE", receipt)
        self.assertEqual(json.loads(receipt)["state"], "failed-consumed")
        with self.assertRaises(ExperimentAlreadyRunError):
            runner.run("failed-attempt", data, runtime="mock")

    def test_opt_in_live_demo_path_is_bounded_with_mocked_providers(self):
        import io
        from contextlib import redirect_stdout
        from krugradeai.demo import main
        agents = build_mock_agents()
        with patch("krugradeai.demo.build_live_agents", return_value=agents) as builder, redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--live", "--output-dir", str(self.root / "mocked-live")]), 0)
        builder.assert_called_once()
        self.assertLessEqual(sum(len(agent.provider.requests) for agent in agents), 7)
