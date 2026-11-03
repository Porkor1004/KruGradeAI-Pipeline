"""Synthetic export checks; no real records or model calls."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from krugradeai.codex_provider import CodexCLIProvider
from krugradeai.final_evaluation import sha
from krugradeai.grader import LLMGrader
from krugradeai.models import GraderInput, StudentResponse
from krugradeai.prompt_export import PromptDefinition, codex_stdin, export_prompts, render_prompt
from krugradeai.prompts import build_grading_prompt
from krugradeai.mock_provider import MockLLMProvider


class PromptExportTests(unittest.TestCase):
    def setUp(self):
        from test_final_evaluation import FinalEvaluationTests
        self.fixture=FinalEvaluationTests('test_prepare_reads_report_manifest_only_and_freezes_exact_selection')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.snapshot=self.fixture.freeze()
        self.destination=self.fixture.root/'export'

    def test_all_scopes_exact_rubrics_rules_versions_no_answers_or_labels(self):
        before=self.fixture.frozen.read_bytes(),self.fixture.report.read_bytes()
        manifest=export_prompts(self.fixture.frozen,self.destination)
        self.assertEqual([r['selected_version'] for r in manifest['scopes']],[0,0,0,0,1,0,1,0])
        self.assertFalse(manifest['native_system_message'])
        for row,path in zip(manifest['scopes'],self.snapshot.paths):
            definition=PromptDefinition.model_validate_json((self.destination/(row['name']+'.json')).read_bytes())
            self.assertEqual(definition.rubric,path.rubric)
            self.assertEqual(definition.adaptation_rules,path.selected_rules)
            text=(self.destination/(row['name']+'.grader.txt')).read_text(encoding='utf-8')
            payload=json.loads(text.split('\nINPUT_JSON\n',1)[1])
            self.assertNotIn('ta_output',payload)
            self.assertEqual(payload['response']['text'],'REPLACE_WITH_ORIGINAL_STUDENT_ANSWER')
            self.assertNotIn('Synthetic held-out answer',text)
            for filename,digest in row['files'].items():
                self.assertEqual(sha((self.destination/filename).read_bytes()),digest)
        self.assertEqual(before,(self.fixture.frozen.read_bytes(),self.fixture.report.read_bytes()))

    def test_only_development_partition_is_read(self):
        from krugradeai.real_data import load_prepared_partition
        calls=[]
        def load(directory,split):
            calls.append(split); self.assertEqual(split,'development')
            return load_prepared_partition(directory,split)
        with patch('krugradeai.prompt_export.load_prepared_partition',side_effect=load):
            export_prompts(self.fixture.frozen,self.destination)
        self.assertEqual(calls,['development'])

    def test_standalone_render_is_exact_grader_request(self):
        export_prompts(self.fixture.frozen,self.destination)
        definition=PromptDefinition.model_validate_json((self.destination/'Q5.json').read_bytes())
        response=StudentResponse(id='new',question_id=definition.question.id,text='Synthetic new answer')
        rendered=render_prompt(definition,response)
        request=GraderInput(**definition.model_dump(),response=response)
        self.assertEqual(rendered,build_grading_prompt(request))
        provider=MockLLMProvider('{"human_review_required":true,"review_reason":"Synthetic review"}')
        from krugradeai.grader import HumanReviewRequired
        with self.assertRaises(HumanReviewRequired): LLMGrader(provider).grade(request)
        self.assertEqual(provider.requests,[rendered])

    def test_exported_stdin_matches_real_provider_subprocess_input(self):
        export_prompts(self.fixture.frozen,self.destination)
        request=(self.destination/'Q7.grader.txt').read_text(encoding='utf-8')
        expected=(self.destination/'Q7.codex-stdin.txt').read_text(encoding='utf-8')
        def execute(args,**kwargs):
            self.assertEqual(kwargs['input'],expected)
            self.assertEqual(expected,codex_stdin(request))
            self.assertEqual(args[args.index('--sandbox')+1],'read-only')
            Path(args[args.index('--output-last-message')+1]).write_text('{}')
            return subprocess.CompletedProcess(args,0,stderr='')
        with patch('krugradeai.codex_provider.resolve_codex_command',return_value=['codex.exe']),patch(
                'krugradeai.codex_provider.subprocess.run',side_effect=execute):
            self.assertEqual(CodexCLIProvider().generate(request),'{}')

    def test_no_overwrite_or_changed_frozen_rules(self):
        export_prompts(self.fixture.frozen,self.destination)
        with self.assertRaises(FileExistsError): export_prompts(self.fixture.frozen,self.destination)
        data=json.loads(self.fixture.frozen.read_bytes())
        data['paths'][4]['selected_rules']={'rules':[]}
        self.fixture.frozen.write_text(json.dumps(data))
        with self.assertRaises(ValueError): export_prompts(self.fixture.frozen,self.fixture.root/'other')

    def test_protected_destination_and_mismatched_response_rejected(self):
        with self.assertRaises(ValueError): export_prompts(self.fixture.frozen,self.fixture.base/'export')
        path=self.snapshot.paths[0]
        from krugradeai.models import Question
        definition=PromptDefinition(question=Question(id='x',text='Synthetic'),rubric=path.rubric,
                                    adaptation_rules=path.selected_rules)
        with self.assertRaises(ValueError): render_prompt(definition,StudentResponse(id='r',question_id='wrong',text='a'))
