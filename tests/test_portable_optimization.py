"""Portable runner uses synthetic fixtures and mocked transports only."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from krugradeai.mock_provider import MockLLMProvider
from krugradeai.run_optimization import OptimizationConfig, _BoundedProvider, _Budget, load_config, run
from krugradeai.synthetic import make_synthetic_dataset


class PortableOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.config=self.root/'config.json'
        self.records=make_synthetic_dataset(multiple_rubrics=True).development
        self.input=self.root/'development.jsonl'
        self.input.write_text(''.join(r.model_dump_json()+'\n' for r in self.records),encoding='utf-8')
        self.write_config()

    def write_config(self,**overrides):
        self.settings=dict(development='development.jsonl',output_directory='output',
                           runtime='synthetic_mock',max_iterations=1,max_calls=10)
        self.settings.update(overrides)
        self.config.write_text(json.dumps(self.settings),encoding='utf-8')

    def test_config_paths_are_portable_from_another_working_directory(self):
        original=Path.cwd()
        try:
            os.chdir(self.root.parent)
            config=load_config(self.config)
            self.assertEqual(config.development,str(self.input))
            self.assertEqual(config.output_directory,str(self.root/'output'))
            plan=run(self.config,dry_run=True)
        finally: os.chdir(original)
        self.assertEqual(plan['records'],4)
        self.assertEqual(plan['reserved_maximum_calls'],10)
        self.assertFalse((self.root/'output').exists())

    def test_complete_mock_three_agent_loop_preserves_data_no_evaluation(self):
        before=self.input.read_bytes()
        result=run(self.config)
        self.assertEqual(len(result['paths']),2)
        self.assertTrue(all(p['latest_version']==1 for p in result['paths']))
        self.assertFalse(result['validation_loaded']); self.assertFalse(result['heldout_loaded'])
        self.assertEqual(before,self.input.read_bytes())
        history=json.loads((self.root/'output'/'optimization.json').read_bytes())
        self.assertTrue(history['paths'][0]['history'][0]['records'][0]['reflection'])
        with self.assertRaises(FileExistsError): run(self.config)

    def test_live_requires_opt_in_and_explicit_model(self):
        with self.assertRaises(ValueError): self.write_config(runtime='codex_cli'); load_config(self.config)
        self.write_config(runtime='codex_cli',model='synthetic-model')
        with patch('krugradeai.run_optimization.CodexCLIProvider') as provider:
            with self.assertRaises(ValueError): run(self.config)
            run(self.config,dry_run=True)
            provider.assert_not_called()

    def test_invalid_config_split_and_budget_fail_before_provider(self):
        for overrides in ({'max_calls':1},{'max_iterations':0},{'unexpected':True}):
            self.write_config(**overrides)
            with self.assertRaises(ValueError): run(self.config,dry_run=True)
        self.write_config()
        data=self.records[0].model_dump(mode='json'); data['split']='test'
        self.input.write_text(json.dumps(data))
        with self.assertRaises(ValueError): run(self.config,dry_run=True)

    def test_mock_is_not_a_generic_offline_grader(self):
        data=self.records[0].model_dump(mode='json'); data['question']['text']='A different question'
        self.input.write_text(json.dumps(data))
        with self.assertRaises(ValueError): run(self.config)

    def test_custom_rubric_without_core_changes_and_same_provider_settings(self):
        data=self.records[0].model_dump(mode='json')
        data['question'].update(id='custom',text='A new subject',reference_solution=None)
        data['rubric']={'id':'custom-rubric','criteria':[{'id':'custom-part','description':'Custom evidence','max_score':'3'}],'total_score':'3'}
        data['response']['question_id']='custom'
        data['ta_output']={'total_score':'2','criterion_scores':[]}
        self.input.write_text(json.dumps(data))
        self.write_config(runtime='codex_cli',model='test-model',executable='tool/codex.exe',max_calls=3)
        provider=MockLLMProvider(json.dumps(dict(criterion_scores=[dict(criterion_id='custom-part',score='2',rationale='Synthetic')],
                                                total_score='2',overall_feedback='Synthetic')))
        with patch('krugradeai.run_optimization.CodexCLIProvider',return_value=provider) as build:
            result=run(self.config,live=True)
        self.assertEqual(result['paths'][0]['stop_reason'],'no_discrepancies')
        self.assertEqual(result['provider_calls'],1)
        for call in build.call_args_list:
            self.assertEqual(call.kwargs['model'],'test-model')
            self.assertEqual(call.kwargs['executable'],str(self.root/'tool'/'codex.exe'))
        self.assertNotIn('ta_output',json.loads(provider.requests[0].split('\nINPUT_JSON\n')[1]))

    def test_budget_wrapper_enforces_shared_cap(self):
        source=MockLLMProvider('{}'); budget=_Budget(1)
        a,b=_BoundedProvider(source,budget),_BoundedProvider(source,budget)
        a.generate('one')
        with self.assertRaises(RuntimeError): b.generate('two')
        self.assertEqual(len(source.requests),1)

    def test_provider_setup_failure_records_safe_status_and_preserves_output(self):
        before = self.input.read_bytes()
        with patch('krugradeai.run_optimization.build_mock_agents', side_effect=RuntimeError('PRIVATE ANSWER')):
            with self.assertRaises(RuntimeError):
                run(self.config)
        output = self.root/'output'
        status = json.loads((output/'status.json').read_bytes())
        self.assertEqual(status, dict(status='failed', stage='providers', provider_calls=0, error_type='RuntimeError'))
        self.assertTrue((output/'configuration.json').exists())
        self.assertNotIn('PRIVATE', (output/'status.json').read_text())
        saved = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(FileExistsError):
            run(self.config)
        self.assertEqual(saved, {p.name: p.read_bytes() for p in output.iterdir()})
        self.assertEqual(before, self.input.read_bytes())

    def test_interruption_and_unexpected_loop_error_have_distinct_safe_status(self):
        for number, error in enumerate((KeyboardInterrupt(), RuntimeError('PRIVATE PROMPT'))):
            self.write_config(output_directory=f'output-{number}')
            with patch('krugradeai.run_optimization.OptimizationLoop.run', side_effect=error):
                with self.assertRaises(type(error)):
                    run(self.config)
            status = json.loads((self.root/f'output-{number}'/'status.json').read_bytes())
            self.assertEqual(status['status'], 'interrupted' if number == 0 else 'failed')
            self.assertEqual(status['stage'], 'optimization')
            self.assertEqual(status['provider_calls'], 0)
            self.assertNotIn('PRIVATE', json.dumps(status))

    def test_artifact_write_failure_retains_configuration_and_call_count(self):
        original = Path.open
        def failing_open(path, *args, **kwargs):
            if path.name == 'optimization.json':
                raise OSError('PRIVATE STORAGE PATH')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', failing_open):
            with self.assertRaises(OSError):
                run(self.config)
        output = self.root/'output'
        status = json.loads((output/'status.json').read_bytes())
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['stage'], 'artifacts')
        self.assertEqual(status['provider_calls'], 10)
        self.assertTrue((output/'configuration.json').exists())
        self.assertFalse((output/'summary.json').exists())
        self.assertNotIn('PRIVATE', json.dumps(status))

    def test_captured_agent_failure_is_incomplete_not_completed(self):
        from krugradeai.synthetic import build_mock_agents
        agents = build_mock_agents()
        agents[0].provider = MockLLMProvider('{}')  # Invalid synthetic grade, caught by the engine.
        with patch('krugradeai.run_optimization.build_mock_agents', return_value=agents):
            summary = run(self.config)
        self.assertEqual(summary['status'], 'incomplete')
        self.assertTrue(all(p['stop_reason'] == 'agent_failure' for p in summary['paths']))
        status = json.loads((self.root/'output'/'status.json').read_bytes())
        self.assertEqual(status['status'], 'incomplete')
        self.assertEqual(status['provider_calls'], 4)
        self.assertTrue((self.root/'output'/'optimization.json').exists())

    def test_cli_incomplete_run_has_nonzero_exit_without_live_calls(self):
        from krugradeai.run_optimization import main
        with patch('krugradeai.run_optimization.run', return_value={'status':'incomplete'}), patch('builtins.print'):
            self.assertEqual(main(['--config',str(self.config)]), 1)
        with patch('krugradeai.run_optimization.run', return_value={'status':'completed'}), patch('builtins.print'):
            self.assertEqual(main(['--config',str(self.config)]), 0)

    def test_duplicate_source_item_is_rejected_during_preflight(self):
        rows = [r.model_dump(mode='json') for r in self.records]
        rows[1]['provenance'] = rows[0]['provenance']
        self.input.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with patch('krugradeai.run_optimization.build_mock_agents') as build:
            with self.assertRaises(ValueError):
                run(self.config, dry_run=True)
        build.assert_not_called()
        self.assertFalse((self.root/'output').exists())
