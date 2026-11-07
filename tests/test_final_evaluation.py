"""Final-only frozen evaluation using synthetic sources and mocked transport."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from krugradeai.experiment import ExperimentAlreadyRunError
from krugradeai.experiment_models import HeldoutRecord
from krugradeai.final_evaluation import (prepare, plan, freeze_selection, evaluate_final, sha, _FinalProvider)
from krugradeai.metrics import ScorePair, calculate_metrics
from krugradeai.mock_provider import MockLLMProvider
from test_full_real_optimization import full_fixture


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root/"prepared"; self.base.mkdir()
        dev, val = full_fixture()
        scope_examples = {r.question.id:r for r in dev}
        self.test = []
        for scope, count in enumerate((5,3,6,3,3,1,2,1)):
            for i in range(count):
                data = scope_examples[f"synthetic-question-{scope}"].model_dump(mode="json")
                rid = f"synthetic-test-{scope}-{i}"
                origin = "ai_derived" if scope in (0,2,3,4,6) and i == 0 else "human_derived"
                data.update(id=rid, split="test", provenance=dict(source="synthetic-test", item_id=rid, label_origin=origin))
                data["response"].update(id=rid, text="Synthetic held-out answer")
                data["ta_output"] = dict(total_score="1", criterion_scores=[])
                self.test.append(HeldoutRecord.model_validate(data))
        partitions = dict(development=dev, validation=val, test=tuple(self.test))
        hashes, lineage = {}, []
        for split, records in partitions.items():
            content = "".join(r.model_dump_json()+"\n" for r in records).encode()
            (self.base/(split+".jsonl")).write_bytes(content)
            hashes[split] = sha(content)
            for record in records:
                lineage.append(dict(id=record.id, partition=split, source=record.provenance.source,
                    question_id=record.question.id, label_origin=record.provenance.label_origin,
                    group_id=record.id, connected_group_id=record.id, response_ids=[record.response.id],
                    lineage=[record.id], source_attempt_ids=[record.id]))
        manifest = dict(partition_sha256=hashes, lineage=lineage,
            rubric_definitions={r.question.id:r.rubric.model_dump(mode="json") for r in dev})
        self.manifest = self.base/"manifest.json"
        self.manifest.write_text(json.dumps(manifest))
        self.report = self.root/"optimization.json"
        paths = []
        for i in range(8):
            rubric = scope_examples[f"synthetic-question-{i}"].rubric
            candidate = dict(rules=[dict(id="reasoning", instruction="Inspect intermediate reasoning under the supplied criterion.",
                criterion_ids=["method"], rationale="Supported synthetic findings.")])
            versions = [dict(version=0,parent_version=None,created_iteration=0,rules=dict(rules=[]))]
            chosen = 1 if i in (4,6) else 0
            if i in (0,3,4,6,7):
                versions.append(dict(version=1,parent_version=0,created_iteration=1,rules=candidate))
            ids = [r.id for r in val if r.rubric.id == rubric.id]
            evaluations = []
            for version in range(len(versions)):
                prediction = 1 if chosen == version == 1 else 0
                metrics = calculate_metrics([ScorePair(predicted=prediction, reference=1, maximum=2) for _ in ids], expected_count=len(ids))
                evaluations.append(dict(version=version, eligible=True, metrics=metrics.model_dump(mode="json"),
                    comparison_metrics=metrics.model_dump(mode="json"), comparison_record_ids=ids))
            paths.append(dict(rubric_id=rubric.id, guideline_versions=versions, validation=evaluations,
                selected_version=chosen, selected_rules=versions[chosen]["rules"]))
        report = dict(execution_error=None, held_out_accessed=False, mode="full_development",
            validation_label_origin="human_derived", model=None, provider_calls=[],
            development_sha256=hashes["development"], input_sha256={str(self.manifest):sha(self.manifest.read_bytes()),
                str(self.base/"validation.jsonl"):hashes["validation"]},
            selected_records=[dict(record_id=r.id) for r in dev], paths=paths)
        self.report.write_text(json.dumps(report))
        self.frozen = self.root/"frozen"/"guidelines.json"
        self.output = self.root/"outputs"

    def freeze(self):
        return prepare(self.frozen, report_path=self.report, data_directory=self.base)

    def execute(self, provider=None, **kwargs):
        class ContextProvider(MockLLMProvider):
            def generate(self, prompt):
                self.requests.append(prompt)
                data = json.loads(prompt.split("\nINPUT_JSON\n")[1])
                value = 1 if data["adaptation_rules"]["rules"] else 0
                return json.dumps(dict(total_score=value, criterion_scores=[
                    dict(criterion_id="method", score=value, rationale="Synthetic reasoning"),
                    dict(criterion_id="result", score=0, rationale="Synthetic result")], overall_feedback="Synthetic feedback"))
        provider = provider or ContextProvider("")
        with patch("krugradeai.final_evaluation.CodexCLIProvider", return_value=provider) as build:
            result = evaluate_final(live=True, frozen_path=self.frozen, output_root=self.output,
                model="synthetic-model", **kwargs)
        self.assertEqual(build.call_args.kwargs["role"], "heldout_grader")
        self.assertEqual(build.call_args.kwargs["model"], "synthetic-model")
        return result, provider

    def test_prepare_reads_report_manifest_only_and_freezes_exact_selection(self):
        original = Path.read_bytes
        def guarded(path):
            if path.name.endswith(".jsonl"):
                raise AssertionError("Preparation must not open answer/label partitions")
            return original(path)
        with patch.object(Path, "read_bytes", guarded):
            snapshot = self.freeze()
            details = plan(snapshot)
        self.assertEqual([p.selected_version for p in snapshot.paths], [0,0,0,0,1,0,1,0])
        self.assertEqual(details["maximum_calls"], 29)
        self.assertEqual(details["label_origins"], dict(ai_derived=5,human_derived=19))
        self.assertIsNone(snapshot.optimization_effective_model)
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_frozen_baseline_and_selected_same_model_all_cohorts_correct_counts_and_no_optimizers(self):
        snapshot = self.freeze()
        before = self.frozen.read_bytes()
        result, provider = self.execute()
        self.assertEqual(len(provider.requests), 29)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["baseline"]["mixed_source"]["mae"], "1")
        self.assertEqual(result["selected"]["human_only"]["sample_count"], 19)
        self.assertEqual(result["selected"]["ai_labeled_only"]["sample_count"], 5)
        self.assertEqual(result["selected"]["mixed_source"]["coverage"], "1")
        self.assertEqual(self.frozen.read_bytes(), before)
        versions = {p.rubric.id:p.selected_version for p in snapshot.paths}
        self.assertEqual({p["rubric_id"]:p["selected_version"] for p in result["paths"]}, versions)
        for prompt in provider.requests:
            data = json.loads(prompt.split("\nINPUT_JSON\n")[1])
            self.assertNotIn("ta_output", data)
            self.assertNotIn("label_origin", json.dumps(data))
        self.assertEqual(len(list((self.output/"heldout-full-001").glob("*_g*.json"))), 8)
        self.assertNotIn("Synthetic held-out answer", (self.output/"heldout-full-001"/"report.json").read_text())

    def test_same_experiment_cannot_repeat_under_new_run_id_or_output_directory(self):
        self.freeze(); self.execute()
        with self.assertRaises(ExperimentAlreadyRunError), patch("krugradeai.final_evaluation.load_prepared_partition") as load:
            evaluate_final(live=True,frozen_path=self.frozen,output_root=self.root/"different",
                run_id="different",model="synthetic-model")
        load.assert_not_called()

    def test_failures_preserve_coverage_not_zero_and_g0_selection_stays_frozen(self):
        self.freeze()
        provider = MockLLMProvider("invalid JSON")
        result, _ = self.execute(provider=provider)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["selected"]["mixed_source"]["sample_count"], 0)
        self.assertEqual(result["selected"]["mixed_source"]["coverage"], "0")
        self.assertIsNone(result["selected"]["mixed_source"]["mae"])
        self.assertFalse(result["selection_changed"])

    def test_model_and_budget_are_explicit_before_test_read_or_claim(self):
        self.freeze()
        for kwargs in (dict(model=None,max_calls=29), dict(model="synthetic-model",max_calls=28)):
            with patch("krugradeai.final_evaluation.load_prepared_partition") as load, self.assertRaises(ValueError):
                evaluate_final(live=True,frozen_path=self.frozen,output_root=self.output,**kwargs)
            load.assert_not_called()
        self.assertFalse((self.root/"final-heldout-ledger").exists())

    def test_changed_selection_or_code_rejected_before_test_is_opened(self):
        self.freeze()
        data=json.loads(self.report.read_text()); data["paths"][4]["selected_version"]=0
        self.report.write_text(json.dumps(data))
        with patch("krugradeai.final_evaluation.load_prepared_partition") as load, self.assertRaises(ValueError):
            self.execute()
        load.assert_not_called()

    def test_missing_best_and_changed_validation_population_refused_at_prepare(self):
        data=json.loads(self.report.read_text())
        data["paths"][0]["selected_version"]=None
        self.report.write_text(json.dumps(data))
        with self.assertRaises(ValueError): self.freeze()
        self.assertFalse(self.frozen.exists())

    def test_partition_group_overlap_blocks_freeze(self):
        data=json.loads(self.manifest.read_text()); active=data["lineage"]
        test=next(r for r in active if r["partition"]=="test")
        test["connected_group_id"]=active[0]["connected_group_id"]
        self.manifest.write_text(json.dumps(data))
        report=json.loads(self.report.read_text()); report["input_sha256"][str(self.manifest)]=sha(self.manifest.read_bytes())
        self.report.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,"overlap"): self.freeze()

    def test_validation_source_origin_cannot_be_hidden_by_report_flag(self):
        data=json.loads(self.manifest.read_text())
        next(r for r in data["lineage"] if r["partition"]=="validation")["label_origin"]="ai_derived"
        self.manifest.write_text(json.dumps(data))
        report=json.loads(self.report.read_text())
        report["input_sha256"][str(self.manifest)]=sha(self.manifest.read_bytes())
        self.report.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,"human-only"):
            self.freeze()

    def test_failure_after_claim_consumes_experiment_without_retry(self):
        self.freeze()
        with patch("krugradeai.final_evaluation.load_prepared_partition",side_effect=ValueError("PRIVATE")), self.assertRaises(ValueError):
            evaluate_final(live=True,frozen_path=self.frozen,output_root=self.output,model="synthetic-model")
        with self.assertRaises(ExperimentAlreadyRunError):
            evaluate_final(live=True,frozen_path=self.frozen,output_root=self.output,
                run_id="second",model="synthetic-model")
        self.assertNotIn("PRIVATE", (self.output/"heldout-full-001"/"failure.json").read_text())

    def test_transport_budget_never_calls_provider_above_limit(self):
        provider=MockLLMProvider("{}")
        wrapper=_FinalProvider(provider,1,self.root/"calls.jsonl")
        wrapper.generate("synthetic")
        with self.assertRaises(RuntimeError): wrapper.generate("synthetic")
        self.assertEqual(len(provider.requests),1)

    def test_frozen_selection_remains_even_when_heldout_selected_rules_perform_worse(self):
        class WorseProvider(MockLLMProvider):
            def generate(self, prompt):
                self.requests.append(prompt)
                data=json.loads(prompt.split("\nINPUT_JSON\n")[1])
                value=0 if data["adaptation_rules"]["rules"] else 1
                return json.dumps(dict(total_score=value,criterion_scores=[
                    dict(criterion_id="method",score=value,rationale="Synthetic"),
                    dict(criterion_id="result",score=0,rationale="Synthetic")],overall_feedback="Synthetic"))
        self.freeze()
        result,_=self.execute(provider=WorseProvider(""))
        self.assertEqual(result["baseline"]["mixed_source"]["mae"],"0")
        self.assertNotEqual(result["selected"]["mixed_source"]["mae"],"0")
        self.assertEqual([p["selected_version"] for p in result["paths"]],[0,0,0,0,1,0,1,0])

    def test_changed_grading_code_or_frozen_rules_is_rejected_before_loading_test(self):
        self.freeze()
        with patch("krugradeai.final_evaluation.code_hashes",return_value={"grader.py":"changed"}), patch(
                "krugradeai.final_evaluation.load_prepared_partition") as load, self.assertRaises(ValueError):
            self.execute()
        load.assert_not_called()
        data=json.loads(self.frozen.read_text()); data["paths"][4]["selected_rules"]["rules"][0]["instruction"]="Changed interpretation"
        self.frozen.write_text(json.dumps(data))
        with patch("krugradeai.final_evaluation.load_prepared_partition") as load, self.assertRaises(ValueError):
            self.execute()
        load.assert_not_called()
