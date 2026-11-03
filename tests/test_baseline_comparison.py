"""Separate legacy prompt vs frozen rules; all data/providers are synthetic."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from krugradeai.baseline_comparison import (MissingOriginalPromptError, read_original_prompt,
    _OriginalPromptProvider, compare_records, run_comparison, markdown_report)
from krugradeai.experiment_models import HeldoutRecord
from krugradeai.final_evaluation import FrozenPath, FrozenGuidelines
from krugradeai.metrics import ScorePair, calculate_metrics
from krugradeai.mock_provider import MockLLMProvider
from krugradeai.models import AdaptationRule, AdaptationRules, GraderInput
from krugradeai.prompts import GRADING_PROMPT, build_grading_prompt
from test_full_real_optimization import full_fixture

LEGACY = "Synthetic legacy grader instructions.\r\nPreserve this prompt exactly.\r\n"


def fixture():
    dev, _ = full_fixture()
    samples = []
    paths = []
    for i in range(2):
        data=dev[i].model_dump()
        data.update(id=f"test-{i}",split="test",provenance=dict(source="synthetic-heldout",item_id=str(i),label_origin="human_derived"))
        data["response"]["id"]=f"test-{i}"
        data["ta_output"]=dict(total_score=1,criterion_scores=[])
        record=HeldoutRecord.model_validate(data); samples.append(record)
        metric=calculate_metrics([ScorePair(predicted=1,reference=1,maximum=2)],expected_count=1)
        paths.append(FrozenPath(rubric=record.rubric,baseline_rules=AdaptationRules(),selected_version=1,
            selected_rules=AdaptationRules(rules=(AdaptationRule(id="evidence",criterion_ids=("method",),
                instruction="Inspect intermediate reasoning under the supplied criterion.",rationale="Synthetic supported evidence."),)),
            baseline_validation=metric,selected_validation=metric,selected_validation_comparison=metric))
    snapshot=FrozenGuidelines(source_report="synthetic",source_report_sha256="0"*64,data_directory="synthetic",
        manifest_sha256="0"*64,heldout_sha256="0"*64,configured_model=None,model_evidence="synthetic",
        grading_code_sha256={},paths=tuple(paths))
    return tuple(samples),snapshot


class ConditionalProvider(MockLLMProvider):
    def __init__(self, *, old_review=False, malformed_old=False):
        super().__init__(""); self.old_review=old_review; self.malformed_old=malformed_old

    def generate(self,prompt):
        self.requests.append(prompt)
        payload=json.loads(prompt.rsplit("\nINPUT_JSON\n",1)[1])
        old=prompt.startswith(LEGACY)
        if old and self.malformed_old:return '{"score":1}'
        if old and self.old_review and payload['response']['id']=='test-0':
            return json.dumps(dict(human_review_required=True,review_reason="PRIVATE CONTENT"))
        value=0 if old else 1
        return json.dumps(dict(total_score=value,criterion_scores=[
            dict(criterion_id="method",score=value,rationale="Synthetic evidence"),
            dict(criterion_id="result",score=0,rationale="Synthetic result")],overall_feedback="Synthetic feedback"))


class BaselineComparisonTests(unittest.TestCase):
    def test_missing_legacy_prompt_never_defaults_to_g0_or_constructs_providers(self):
        rows,snapshot=fixture(); factory=Mock()
        with self.assertRaises(MissingOriginalPromptError):
            compare_records(rows,snapshot,"",factory,maximum_calls=4)
        factory.assert_not_called()
        with self.assertRaises(MissingOriginalPromptError):read_original_prompt(None)

    def test_legacy_file_and_instruction_prefix_are_byte_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"old.txt"; path.write_bytes(LEGACY.encode())
            text,_=read_original_prompt(path)
            rows,_=fixture(); r=rows[0]
            request=GraderInput(question=r.question,rubric=r.rubric,response=r.response)
            provider=MockLLMProvider("{}"); wrapped=_OriginalPromptProvider(provider,text)
            wrapped.generate(build_grading_prompt(request))
            self.assertEqual(provider.requests[0],LEGACY+"\nINPUT_JSON\n"+request.model_dump_json())
            self.assertEqual(path.read_bytes(),LEGACY.encode())

    def test_old_prompt_cannot_receive_optimized_rules(self):
        rows,snapshot=fixture(); r=rows[0]
        request=GraderInput(question=r.question,rubric=r.rubric,response=r.response,
            adaptation_rules=snapshot.paths[0].selected_rules)
        with self.assertRaises(ValueError):
            _OriginalPromptProvider(MockLLMProvider("{}"),LEGACY).generate(build_grading_prompt(request))

    def test_same_records_rubrics_and_blind_labels_with_valid_improvement(self):
        rows,snapshot=fixture(); providers=[]
        def factory():
            p=ConditionalProvider(); providers.append(p); return p
        before=[r.model_dump_json() for r in rows],snapshot.model_dump_json()
        result=compare_records(rows,snapshot,LEGACY,factory,maximum_calls=4)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['metrics']['old_prompt']['human_only']['mae'],'1')
        self.assertEqual(result['metrics']['optimized_guidelines']['human_only']['mae'],'0')
        self.assertEqual(result['improvement']['human_only']['mae'],'1')
        self.assertEqual(result['improvement']['human_only']['exact_match'],'1')
        self.assertTrue(result['same_success_population'])
        self.assertEqual(len(result['provider_calls']),4)
        a=[json.loads(s.rsplit('\nINPUT_JSON\n',1)[1]) for s in providers[0].requests]
        b=[json.loads(s.rsplit('\nINPUT_JSON\n',1)[1]) for s in providers[1].requests]
        self.assertEqual([x['response'] for x in a],[x['response'] for x in b])
        self.assertEqual([x['rubric'] for x in a],[x['rubric'] for x in b])
        self.assertTrue(all(not x['adaptation_rules']['rules'] for x in a))
        self.assertTrue(all(x['adaptation_rules']['rules'] for x in b))
        self.assertTrue(all('ta_output' not in x for x in a+b))
        self.assertEqual(before,([r.model_dump_json() for r in rows],snapshot.model_dump_json()))

    def test_review_and_different_success_sets_have_paired_metrics_not_zero_imputation(self):
        rows,snapshot=fixture()
        result=compare_records(rows,snapshot,LEGACY,lambda:ConditionalProvider(old_review=True),maximum_calls=4)
        self.assertEqual(result['status'],'incomplete')
        self.assertFalse(result['same_success_population'])
        self.assertEqual(result['metrics']['old_prompt']['human_only']['coverage'],'0.5')
        self.assertEqual(result['metrics']['old_prompt']['human_only']['sample_count'],1)
        self.assertEqual(result['paired_success_metrics']['human_only']['old_prompt']['sample_count'],1)
        self.assertEqual(result['paired_success_metrics']['human_only']['optimized_guidelines']['sample_count'],1)
        reviews=[x for x in result['outcomes'] if x['status']=='human_review']
        self.assertEqual(reviews[0]['record_id'],'test-0')
        self.assertEqual(len(reviews[0]['provider_call_indices']),1)
        self.assertNotIn('PRIVATE CONTENT',json.dumps(result))

    def test_incompatible_old_output_protocol_is_failure_not_repaired_or_claimed_accuracy(self):
        rows,snapshot=fixture()
        result=compare_records(rows,snapshot,LEGACY,lambda:ConditionalProvider(malformed_old=True),maximum_calls=4)
        self.assertEqual(result['failures']['old_prompt'],2)
        self.assertIsNone(result['metrics']['old_prompt']['human_only']['mae'])
        self.assertIsNone(result['improvement']['human_only']['mae'])

    def test_budget_and_scope_mismatch_fail_before_providers(self):
        rows,snapshot=fixture(); factory=Mock()
        for records,budget in ((rows,3),(rows[:1],4),(rows+rows,8)):
            with self.assertRaises(ValueError):compare_records(records,snapshot,LEGACY,factory,maximum_calls=budget)
        factory.assert_not_called()

    def test_nonheldout_data_is_rejected(self):
        rows,snapshot=fixture(); dev,_=full_fixture()
        with self.assertRaises(ValueError):
            compare_records(dev[:2],snapshot,LEGACY,ConditionalProvider,maximum_calls=4)

    def test_markdown_reports_values_differences_and_limitations(self):
        rows,snapshot=fixture()
        report=compare_records(rows,snapshot,LEGACY,ConditionalProvider,maximum_calls=4)
        report['model']='synthetic-model'
        text=markdown_report(report)
        self.assertIn('| MAE | 1.000 | 0.000 | 1.000 |',text)
        self.assertIn('post-hoc',text)
        self.assertIn('not a new independent confirmatory test',text)
        self.assertNotIn('PRIVATE',text)

    def test_real_entrypoint_is_opt_in_and_missing_prompt_blocks_data_access(self):
        with patch('krugradeai.baseline_comparison.load_prepared_partition') as load, patch(
                'krugradeai.baseline_comparison.CodexCLIProvider') as build:
            with self.assertRaises(ValueError):run_comparison()
            with self.assertRaises(MissingOriginalPromptError):run_comparison(live=True,model='synthetic-model')
            load.assert_not_called(); build.assert_not_called()

    def test_mocked_entrypoint_pins_identical_provider_settings_and_preserves_original_files(self):
        # Reuse the existing synthetic completed-experiment fixture; do not import
        # its TestCase at module scope (which would duplicate unittest discovery).
        from test_final_evaluation import FinalEvaluationTests
        fixture_case=FinalEvaluationTests('test_prepare_reads_report_manifest_only_and_freezes_exact_selection')
        fixture_case.setUp(); self.addCleanup(fixture_case.doCleanups)
        fixture_case.freeze()
        old=fixture_case.root/'legacy.txt'; old.write_bytes(LEGACY.encode())
        paths=(old,fixture_case.frozen,fixture_case.report,fixture_case.manifest,
               fixture_case.base/'test.jsonl',fixture_case.base/'validation.jsonl')
        before=[p.read_bytes() for p in paths]
        providers=[]
        def build(**settings):
            provider=ConditionalProvider(); providers.append(provider); return provider
        with patch('krugradeai.baseline_comparison.CodexCLIProvider',side_effect=build) as constructor:
            result=run_comparison(live=True,old_prompt_path=old,frozen_path=fixture_case.frozen,
                model='synthetic-model',timeout=120,output_root=fixture_case.root/'comparison',max_calls=48)
        self.assertEqual(len(constructor.call_args_list),2)
        self.assertEqual(constructor.call_args_list[0],constructor.call_args_list[1])
        self.assertEqual(constructor.call_args_list[0].kwargs,
            dict(model='synthetic-model',timeout_seconds=120,executable=None))
        self.assertEqual(len(result['provider_calls']),48)
        self.assertEqual(result['metrics']['old_prompt']['human_only']['expected_count'],19)
        self.assertEqual(result['metrics']['optimized_guidelines']['human_only']['expected_count'],19)
        self.assertTrue(result['posthoc_test_reuse'])
        self.assertFalse(result['selection_changed'])
        self.assertEqual(before,[p.read_bytes() for p in paths])
        reportfile=fixture_case.root/'comparison'/'old-prompt-comparison-001'/'comparison.md'
        self.assertTrue(reportfile.is_file())
        with patch('krugradeai.baseline_comparison.CodexCLIProvider') as constructor, self.assertRaises(FileExistsError):
            run_comparison(live=True,old_prompt_path=old,frozen_path=fixture_case.frozen,
                model='synthetic-model',output_root=fixture_case.root/'other-comparison',run_id='second',max_calls=48)
        constructor.assert_not_called()
