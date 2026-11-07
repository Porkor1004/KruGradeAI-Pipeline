"""Exercise the real separated entry points with scripted role providers."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from krugradeai.run_optimization import run as optimize
from krugradeai.evaluation import run as evaluate, ScriptedProvider
from krugradeai.synthetic import SyntheticMockProvider
from krugradeai.model_routing import ModelRoutes, ModelRoutingError
from krugradeai.codex_provider import CodexCLIProvider

ROOT = Path(__file__).resolve().parents[1]

class RoutingTests(unittest.TestCase):
    def test_roles_reach_their_own_providers(self):
        artifacts=ROOT/'artifacts'; artifacts.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts) as tmp:
            root = Path(tmp)
            calls = []
            def provider(role, model, **kwargs):
                self.assertEqual(model, getattr(ModelRoutes(), role))
                agent = SyntheticMockProvider('grader' if role=='validation_grader' else role)
                original = agent.generate
                def generate(prompt):
                    calls.append(role)
                    return original(prompt)
                agent.generate = generate
                return agent
            path = ROOT/'examples/optimization/synthetic-validated.json'
            config = json.loads(path.read_text())
            for key in ('development','validation'):
                config[key] = str(path.parent/config[key])
            config.update(runtime='codex_cli', output_directory=str(root/'opt'))
            cp = root/'config.json'
            cp.write_text(json.dumps(config))
            with patch('krugradeai.run_optimization.CodexCLIProvider',side_effect=provider):
                optimize(cp,live=True)
            self.assertEqual(set(calls),{'grader','validation_grader','reflector','refiner'})
            for role in ('evaluation_grader','heldout_grader'):
                path = ROOT/'examples/evaluation/synthetic.json'
                config = json.loads(path.read_text())
                script = ScriptedProvider(json.loads((path.parent/config.pop('mock_responses')).read_text()))
                script.effective_model = 'gpt-6-luna'
                for key in ('dataset','rubric_file','prompt_file'):
                    config[key] = str(path.parent/config[key])
                config.update(runtime='codex_cli',role=role,output_directory=str(root/role))
                cp.write_text(json.dumps(config))
                with patch('krugradeai.evaluation.CodexCLIProvider',return_value=script) as build:
                    self.assertEqual(evaluate(cp,live=True)['effective_model'],'gpt-6-luna')
                self.assertEqual((build.call_args.kwargs['role'],build.call_args.kwargs['model']),(role,'gpt-6-luna'))

    def test_transport_requires_matching_reported_identity(self):
        artifacts=ROOT/'artifacts'; artifacts.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts) as tmp:
            root = Path(tmp)
            exe = root/'codex.exe'
            exe.touch()
            log = root/'calls.jsonl'
            provider = CodexCLIProvider(executable=exe,role='reflector',model='gpt-6-astra',invocation_log=log)
            for observed in (None,'gpt-6-luna','gpt-6-astra'):
                def runner(argv, **kwargs):
                    self.assertEqual(argv[argv.index('--model')+1],'gpt-6-astra')
                    Path(argv[argv.index('--output-last-message')+1]).write_text('{}')
                    return subprocess.CompletedProcess(argv,0,json.dumps(dict(type='turn.started'))+'\n'+json.dumps(dict(type='turn.completed',effective_model=observed)),'')
                with patch('krugradeai.codex_provider.subprocess.run',side_effect=runner):
                    if observed in (None,'gpt-6-astra'):
                        self.assertEqual(provider.generate('private input'),'{}')
                    else:
                        with self.assertRaises(ModelRoutingError):
                            provider.generate('private input')
            entries = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(entries[1]['verification_status'],'requested_only')
            self.assertIsNone(entries[1]['effective_model'])
            self.assertEqual(entries[1]['exit_status'],0)
            self.assertEqual(entries[1]['completion_status'],'completed')
            self.assertNotIn('private input',log.read_text())

    def test_failed_incomplete_and_rerouted_events_are_rejected(self):
        from krugradeai.codex_provider import _completed_turn
        for stream in ('', 'malformed', '{"type":"turn.started"}', '{"type":"turn.failed"}',
                       '{"type":"model/rerouted","toModel":"other"}'):
            with self.assertRaises(ModelRoutingError):
                _completed_turn(stream, 'gpt-6-luna')
