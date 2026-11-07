"""Exercise CLI structured grading without a model invocation."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from krugradeai.codex_provider import CodexCLIProvider, CodexOutputError
from krugradeai.run_optimization import _Budget, _BoundedProvider
from krugradeai.grader import CLIGraderResponse, LLMGrader, HumanReviewRequired
from krugradeai.synthetic import make_synthetic_dataset
from krugradeai.models import GraderInput


class StructuredGraderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        exe = self.root/'codex.exe'
        exe.touch()
        self.log = self.root/'invocations.jsonl'
        self.provider = CodexCLIProvider(role='grader',model='gpt-6-luna',invocation_log=self.log,executable=exe)
        self.grade = {'criterion_scores':[{'criterion_id':'method','score':'1','rationale':'Example.'},
                                          {'criterion_id':'result','score':'1','rationale':'Example.'}],
                      'total_score':'2','overall_feedback':'Synthetic feedback.'}

    def transport(self, *responses):
        responses = iter(responses)
        def run(argv, **opts):
            self.assertEqual(argv[argv.index('--model')+1],'gpt-6-luna')
            schema = json.loads(Path(argv[argv.index('--output-schema')+1]).read_text())
            self.assertEqual(schema['required'],['result'])
            self.assertFalse(schema['additionalProperties'])
            raw = next(responses)
            if raw is not None:
                Path(argv[argv.index('--output-last-message')+1]).write_text(raw)
            events=[{'type':'turn.started'}, {'type':'item.completed','item':{'text':'WRONG CHANNEL'}},
                    {'type':'turn.completed','usage':{}}]
            return subprocess.CompletedProcess(argv,0,'\n'.join(map(json.dumps,events)),'')
        return patch('krugradeai.codex_provider.subprocess.run',side_effect=run)

    def test_valid_grade_comes_from_final_file(self):
        with self.transport(json.dumps({'result':self.grade})):
            self.assertEqual(json.loads(self.provider.generate('grade')),self.grade)

    def test_invalid_outputs_rejected_with_safe_diagnostics(self):
        for raw in ['{private',None,'',json.dumps({'result':{'total_score':'0'}}),
                    json.dumps({'result':dict(self.grade,unexpected=True)})]:
            with self.transport(raw), self.assertRaises(CodexOutputError):
                self.provider.generate('grade')
            event=json.loads(self.log.read_text().splitlines()[-1])
            self.assertTrue(event['parser_error'])
            self.assertFalse(event['diagnostic_file'])
            self.assertNotIn('{private',self.log.read_text())

    def test_retry_counts_and_cannot_overrun_budget(self):
        for cap in (1,2):
            budget=_Budget(cap)
            provider=_BoundedProvider(self.provider,budget,True)
            with self.transport('bad',json.dumps({'result':self.grade})):
                if cap==1:
                    with self.assertRaisesRegex(RuntimeError,'exhausted'):
                        provider.generate('grade')
                else:
                    self.assertEqual(json.loads(provider.generate('grade')),self.grade)
            self.assertEqual(budget.calls,cap)

    def test_review_uses_existing_review_gate_without_retry(self):
        row=make_synthetic_dataset().development[0]
        budget=_Budget(2)
        with self.transport(json.dumps({'result':{'human_review_required':True,'review_reason':'Evidence unavailable'}})):
            with self.assertRaises(HumanReviewRequired):
                LLMGrader(_BoundedProvider(self.provider,budget,True)).grade(GraderInput(question=row.question,rubric=row.rubric,response=row.response))
        self.assertEqual(budget.calls,1)
