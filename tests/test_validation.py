"""Validation selection tests use synthetic, explicitly disjoint identities."""
import unittest
from decimal import Decimal
from pydantic import ValidationError
from krugradeai.agents import Grader
from krugradeai.grader import HumanReviewRequired
from krugradeai.models import AdaptationRules, GraderOutput
from krugradeai.optimization import OptimizationLoop
from krugradeai.optimization_models import ValidationRecord, ValidationConfig, RecordProvenance, OptimizationResult
import test_optimization as fixtures


def validation(id="val-1", rubric_id="r"):
    data = fixtures.development(id, rubric_id).model_dump()
    data["split"] = "validation"
    return ValidationRecord.model_validate(data)


class ValidationGrader(Grader):
    def __init__(self, scores=None):
        self.scores = scores or {0: "2", 1: "1"}
        self.requests = []

    def grade(self, request):
        self.requests.append(request)
        if request.response.id.startswith("val-"):
            version = 0 if not request.adaptation_rules.rules else int(request.adaptation_rules.rules[0].id[1:])
            value = self.scores.get(version, "2")
            if isinstance(value, dict):
                value = value[request.response.id]
            if isinstance(value, Exception):
                raise value
        else:
            value = "2"
        return GraderOutput(criterion_scores=({"criterion_id": "c", "score": value, "rationale": "Synthetic reasoning"},), total_score=value, overall_feedback="Synthetic feedback")


def run(scores=None, maximum=3, config=None, values=None, refiner=None, grader=None):
    grader = grader or ValidationGrader(scores)
    refiner = refiner or fixtures.ScriptedRefiner([fixtures.candidate("v" + str(i)) for i in range(1, maximum + 1)])
    optimizer = OptimizationLoop(grader=grader, reflector=fixtures.ScriptedReflector(), refiner=refiner,
                                 max_iterations=maximum, validation_config=config or ValidationConfig())
    return optimizer.run([fixtures.development()], validation_records=values if values is not None else [validation()])


class ValidationOptimizationTests(unittest.TestCase):
    def test_baseline_and_final_candidate_evaluated_and_best_returned(self):
        output = run(maximum=1)
        path = output.paths[0]
        self.assertEqual([e.version for e in path.validation.evaluations], [0, 1])
        self.assertEqual(path.validation.baseline_metrics.mae, 1)
        self.assertEqual(path.validation.best_version, 1)
        self.assertEqual(path.validation.best_rules, fixtures.candidate("v1"))
        self.assertEqual(path.validation.evaluations[-1].metrics.mae, 0)
        self.assertEqual(path.stop_reason, "max_iterations")
        self.assertEqual(OptimizationResult.model_validate_json(output.model_dump_json()), output)

    def test_tie_prefers_earlier_version_and_patience_two(self):
        output = run({0: "2", 1: "2", 2: "2", 3: "1"}, maximum=5)
        path = output.paths[0]
        self.assertEqual(path.stop_reason, "patience")
        self.assertEqual(len(path.history), 2)
        self.assertEqual(path.validation.best_version, 0)
        self.assertEqual(path.validation.best_rules, AdaptationRules())
        self.assertEqual([e.version for e in path.validation.evaluations], [0, 1, 2])

    def test_strict_best_is_retained_but_small_improvements_count_towards_patience(self):
        config = ValidationConfig(patience=2, min_delta="0.2")
        output = run({0: "2", 1: "1.9", 2: "1.8", 3: "1"}, config=config, maximum=4)
        self.assertEqual(output.paths[0].stop_reason, "patience")
        self.assertEqual(output.paths[0].validation.best_version, 2)
        self.assertEqual(output.paths[0].validation.evaluations[-1].metrics.mae, Decimal("0.8"))

    def test_significant_improvement_resets_patience(self):
        output = run({0: "2", 1: "2", 2: "1.5", 3: "1.5"}, maximum=3)
        self.assertEqual(len(output.paths[0].history), 3)
        self.assertEqual(output.paths[0].stop_reason, "max_iterations")
        self.assertEqual(output.paths[0].validation.best_version, 2)

    def test_no_change_reuses_evaluated_version_and_stops(self):
        output = run(maximum=4, refiner=fixtures.ScriptedRefiner())
        self.assertEqual(output.paths[0].stop_reason, "no_change")
        self.assertEqual(len(output.paths[0].validation.evaluations), 1)

    def test_incomplete_candidate_cannot_beat_complete_baseline(self):
        scores = {0: "2", 1: {"val-1-response": "1", "val-2-response": RuntimeError("PRIVATE")}}
        output = run(scores, maximum=1, values=[validation(), validation("val-2")])
        summary = output.paths[0].validation
        self.assertEqual(summary.best_version, 0)
        candidate = summary.evaluations[-1]
        self.assertEqual(candidate.metrics.mae, 0)
        self.assertEqual(candidate.metrics.coverage, Decimal("0.5"))
        self.assertFalse(candidate.eligible)
        self.assertEqual(len(candidate.failures), 1)
        self.assertNotIn("PRIVATE", output.model_dump_json())

    def test_no_valid_candidate_is_explicit(self):
        output = run({0: RuntimeError("private"), 1: HumanReviewRequired("Confirm OCR")}, maximum=1)
        summary = output.paths[0].validation
        self.assertIsNone(summary.best_version)
        self.assertIsNone(summary.best_rules)
        self.assertEqual(summary.stop_reason, "no_valid_evaluation")
        self.assertEqual(len(summary.evaluations[-1].human_reviews), 1)

    def test_valid_candidate_can_replace_invalid_baseline(self):
        output = run({0: RuntimeError("private"), 1: "1"}, maximum=1)
        self.assertEqual(output.paths[0].validation.best_version, 1)

    def test_zero_iterations_still_evaluates_baseline(self):
        output = run(maximum=0)
        self.assertEqual(output.paths[0].validation.best_version, 0)
        self.assertEqual(output.paths[0].history, ())
        self.assertEqual(len(output.paths[0].validation.evaluations), 1)

    def test_validation_data_never_reaches_reflector_or_refiner(self):
        reflector, refiner, grader = fixtures.ScriptedReflector(), fixtures.ScriptedRefiner([fixtures.candidate("v1")]), ValidationGrader()
        item = validation().model_dump()
        item["response"]["text"] = "VALIDATION_ANSWER_SENTINEL"
        item["question"]["reference_solution"] = "VALIDATION_REFERENCE_SENTINEL"
        val = ValidationRecord.model_validate(item)
        before = val.model_dump_json()
        optimizer = OptimizationLoop(grader=grader, reflector=reflector, refiner=refiner, max_iterations=1)
        optimizer.run([fixtures.development()], validation_records=[val])
        for request in (*reflector.requests, *refiner.requests):
            self.assertNotIn("VALIDATION_", request.model_dump_json())
            self.assertNotIn("val-1", request.model_dump_json())
        self.assertEqual(val.model_dump_json(), before)
        self.assertEqual([r.response.id for r in grader.requests], ["val-1-response", "dev-1-response", "val-1-response"])
        self.assertTrue(all("ta_output" not in r.model_dump() for r in grader.requests))

    def test_overlap_id_response_identity_and_provenance_rejected(self):
        dev = fixtures.development()
        alternatives = []
        data = validation().model_dump(); data["id"] = dev.id; alternatives.append(data)
        data = validation().model_dump(); data["response"]["id"] = dev.response.id; alternatives.append(data)
        data = dev.model_dump(); data["provenance"] = {"source": "synthetic-source", "item_id": "original-row"}
        dev = type(dev).model_validate(data)
        data = validation().model_dump(); data["provenance"] = {"source": "synthetic-source", "item_id": "original-row"}; alternatives.append(data)
        for data in alternatives:
            grader = ValidationGrader()
            optimizer = OptimizationLoop(grader=grader, reflector=fixtures.ScriptedReflector(), refiner=fixtures.ScriptedRefiner())
            with self.subTest(data=data), self.assertRaises(ValueError):
                optimizer.run([dev], validation_records=[ValidationRecord.model_validate(data)])
            self.assertEqual(grader.requests, [])

    def test_fixed_explicit_comparison_subset_is_not_failure_dependent(self):
        config = ValidationConfig(comparison_record_ids=("val-1",))
        scores = {0: "2", 1: {"val-1-response": "1", "val-2-response": RuntimeError("unavailable")}}
        output = run(scores, maximum=1, config=config, values=[validation(), validation("val-2")])
        summary = output.paths[0].validation
        self.assertEqual(summary.best_version, 1)
        self.assertEqual(summary.evaluations[-1].comparison_metrics.coverage, 1)
        self.assertEqual(summary.evaluations[-1].metrics.coverage, Decimal("0.5"))
        self.assertEqual(summary.evaluations[0].comparison_record_ids, ("val-1",))
        self.assertEqual(summary.evaluations[1].comparison_record_ids, ("val-1",))

    def test_different_rubric_paths_select_independently(self):
        optimizer = OptimizationLoop(grader=ValidationGrader(), reflector=fixtures.ScriptedReflector(),
                                     refiner=fixtures.ScriptedRefiner([fixtures.candidate("v1"), AdaptationRules()]), max_iterations=1,
                                     validation_config=ValidationConfig(primary_metric="normalized_mae"))
        output = optimizer.run([fixtures.development(rubric_id="a"), fixtures.development("dev-2", "b")], validation_records=[validation(rubric_id="a"), validation("val-2", "b")])
        self.assertEqual([p.validation.best_version for p in output.paths], [1, 0])
        self.assertEqual(output.paths[0].validation.primary_metric, "normalized_mae")

    def test_missing_validation_path_and_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            run(values=[])
        for data in ({"patience": 0}, {"min_delta": "-1"}, {"min_delta": "NaN"}, {"primary_metric": "qwk"}, {"comparison_record_ids": []}):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                ValidationConfig(**data)

class AdditionalValidationTests(unittest.TestCase):
    def test_min_delta_equal_threshold_does_not_reset_patience(self):
        output = run({0: "2", 1: "1.8"}, maximum=3, config=ValidationConfig(patience=1, min_delta="0.2"))
        self.assertEqual(output.paths[0].stop_reason, "patience")
        self.assertEqual(output.paths[0].validation.best_version, 1)

    def test_immutable_sources_and_labels_preserved(self):
        dev, val = fixtures.development(), validation()
        snapshots = dev.model_dump_json(), val.model_dump_json()
        optimizer = OptimizationLoop(grader=ValidationGrader(), reflector=fixtures.ScriptedReflector(),
                                     refiner=fixtures.ScriptedRefiner([fixtures.candidate("v1")]), max_iterations=1)
        optimizer.run([dev], validation_records=[val])
        self.assertEqual((dev.model_dump_json(), val.model_dump_json()), snapshots)
        for item in (dev, val):
            with self.assertRaises(ValidationError):
                item.ta_output.total_score = 0
            with self.assertRaises(ValidationError):
                item.rubric.criteria[0].max_score = 99

    def test_same_answers_without_shared_identity_do_not_prove_overlap(self):
        self.assertEqual(fixtures.development().response.text, validation().response.text)
        self.assertEqual(run(maximum=0).paths[0].validation.best_version, 0)

    def test_unknown_comparison_ids_and_duplicate_validation_rows_rejected(self):
        with self.assertRaises(ValueError):
            run(maximum=0, config=ValidationConfig(comparison_record_ids=("unknown",)))
        with self.assertRaises(ValueError):
            run(maximum=0, values=[validation(), validation()])

    def test_wrong_rubric_definition_rejected_before_evaluation(self):
        data = validation().model_dump()
        data["rubric"]["criteria"][0]["description"] = "Different rubric definition"
        with self.assertRaises(ValueError):
            run(values=[ValidationRecord.model_validate(data)])

    def test_unknown_best_version_in_serialized_results_rejected(self):
        data = run(maximum=1).model_dump()
        data["paths"][0]["validation"]["best_version"] = 99
        with self.assertRaises(ValidationError):
            OptimizationResult.model_validate(data)

class BestGuidelinesRegressionTests(unittest.TestCase):
    def test_best_rules_are_returned_even_when_latest_candidate_is_worse(self):
        path = run({0: "2", 1: "1.2", 2: "1.8"}, maximum=2).paths[0]
        self.assertEqual(path.final_version, 2)
        self.assertEqual(path.validation.best_version, 1)
        self.assertEqual(path.validation.best_rules, fixtures.candidate("v1"))

    def test_validation_ta_explanations_and_feedback_never_enter_development_agents(self):
        class FeedbackGrader(ValidationGrader):
            def grade(self, request):
                output = super().grade(request)
                if request.response.id.startswith("val-"):
                    return GraderOutput.model_validate({**output.model_dump(), "overall_feedback": "VALIDATION_FEEDBACK_SENTINEL"})
                return output
        data = validation().model_dump()
        data["ta_output"] = {"criterion_scores": [{"criterion_id": "c", "score": "1", "rationale": "VALIDATION_LABEL_SENTINEL"}], "total_score": "1"}
        val = ValidationRecord.model_validate(data)
        reflector = fixtures.ScriptedReflector()
        refiner = fixtures.ScriptedRefiner([fixtures.candidate("v1")])
        optimizer = OptimizationLoop(grader=FeedbackGrader(), reflector=reflector, refiner=refiner, max_iterations=1)
        result = optimizer.run([fixtures.development()], validation_records=[val])
        self.assertIn("VALIDATION_FEEDBACK_SENTINEL", result.model_dump_json())
        for request in (*reflector.requests, *refiner.requests):
            self.assertNotIn("VALIDATION_", request.model_dump_json())

    def test_duplicate_development_response_rejected_before_baseline_calls(self):
        first = fixtures.development()
        data = first.model_dump()
        data['id'] = 'different-development-id'
        duplicate = type(first).model_validate(data)
        grader = ValidationGrader()
        optimizer = OptimizationLoop(grader=grader, reflector=fixtures.ScriptedReflector(),
                                     refiner=fixtures.ScriptedRefiner(), max_iterations=1)
        with self.assertRaises(ValueError):
            optimizer.run([first, duplicate], validation_records=[validation()])
        self.assertEqual(grader.requests, [])

    def test_duplicate_development_source_item_rejected_even_with_new_response_id(self):
        data = fixtures.development().model_dump()
        data['provenance'] = dict(source='synthetic-source', item_id='same-source-row')
        first = fixtures.DevelopmentRecord.model_validate(data)
        data['id'] = 'different-id'
        data['response']['id'] = 'different-response-id'
        second = fixtures.DevelopmentRecord.model_validate(data)
        grader = ValidationGrader()
        optimizer = OptimizationLoop(grader=grader, reflector=fixtures.ScriptedReflector(),
                                     refiner=fixtures.ScriptedRefiner(), max_iterations=1)
        with self.assertRaises(ValueError):
            optimizer.run([first, second])
        self.assertEqual(grader.requests, [])
        from krugradeai.validation import prepare_validation
        with self.assertRaises(ValueError):
            prepare_validation((first, second), [validation()], ValidationConfig())

    def test_source_item_ids_are_scoped_by_source_not_guessed_from_answer_text(self):
        data = fixtures.development('dev-one').model_dump()
        data['provenance'] = dict(source='source-a', item_id='row-1')
        first = fixtures.DevelopmentRecord.model_validate(data)
        data['id'] = 'dev-two'
        data['response']['id'] = 'other-response'
        data['provenance']['source'] = 'source-b'
        second = fixtures.DevelopmentRecord.model_validate(data)
        from krugradeai.validation import prepare_validation
        groups = prepare_validation((first, second), [validation()], ValidationConfig())
        self.assertEqual(len(groups), 1)
