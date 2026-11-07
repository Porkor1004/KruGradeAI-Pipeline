"""Both CLIs run offline; evaluation cannot construct optimization agents."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from krugradeai.evaluation import run
from krugradeai.run_optimization import run as optimize

ROOT = Path(__file__).resolve().parents[1]


class IndependentPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self, example, output='output', **overrides):
        source = ROOT / 'examples' / example
        data = json.loads(source.read_text())
        for key in ('dataset', 'rubric_file', 'prompt_file', 'mock_responses', 'development', 'validation'):
            if data.get(key):
                data[key] = str((source.parent / data[key]).resolve())
        data['output_directory'] = str(self.root / output)
        data.update(overrides)
        target = self.root / (output + '.json')
        target.write_text(json.dumps(data), encoding='utf-8')
        return target

    def test_external_prompt_reaches_grader_without_labels_or_optimizer(self):
        prompt = self.root / 'external.txt'
        prompt.write_text('EXTERNAL GRADER INSTRUCTIONS: retain rubric scoring authority.\n', encoding='utf-8')
        config = self.config('evaluation/synthetic.json', prompt_file=str(prompt))
        from krugradeai.evaluation import ScriptedProvider
        captured = []
        original = ScriptedProvider.generate
        def capture(provider, request):
            captured.append(request)
            return original(provider, request)
        with patch('krugradeai.optimization.OptimizationLoop.run', side_effect=AssertionError('optimizer called')), \
             patch('krugradeai.reflector.LLMReflector.reflect', side_effect=AssertionError('reflector called')), \
             patch('krugradeai.refiner.LLMRefiner.refine', side_effect=AssertionError('refiner called')), \
             patch.object(ScriptedProvider, 'generate', capture):
            result = run(config)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['provider_calls'], 2)
        self.assertEqual(result['metrics']['mae'], '0.5')
        self.assertEqual(result['metrics']['normalized_mae'], '0.25')
        self.assertEqual(result['metrics']['exact_match'], '0')
        self.assertEqual(result['metrics']['coverage'], '1')
        self.assertEqual(len(result['input_sha256']), 5)
        for origin in ('human_derived', 'ai_derived'):
            self.assertEqual(result['metrics_by_label_origin'][origin]['expected_count'], 1)
        instructions = Path(json.loads(config.read_text())['prompt_file']).read_bytes().decode('utf-8')
        for request in captured:
            self.assertTrue(request.startswith(instructions))
            payload = json.loads(request.split('\nINPUT_JSON\n')[1])
            self.assertNotIn('ta_output', payload)
            self.assertNotIn('provenance', payload)
        with self.assertRaises(FileExistsError):
            run(config)

    def test_failed_grade_reduces_coverage_and_never_becomes_zero(self):
        response_file = self.root / 'responses.json'
        source = json.loads((ROOT/'examples/evaluation/responses.json').read_text())
        source[0] = {'human_review_required': True, 'review_reason': 'Unreadable work'}
        response_file.write_text(json.dumps(source))
        result = run(self.config('evaluation/synthetic.json', mock_responses=str(response_file)))
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['metrics']['coverage'], '0.5')
        self.assertEqual(result['metrics_by_label_origin']['human_derived']['coverage'], '0')
        self.assertIsNone(result['metrics_by_label_origin']['human_derived']['mae'])

    def test_unknown_origin_and_invalid_labels_fail_before_provider(self):
        rows = [json.loads(line) for line in (ROOT/'examples/evaluation/synthetic.jsonl').read_text().splitlines()]
        for mutation in ('origin', 'bounds', 'duplicate'):
            data = json.loads(json.dumps(rows))
            if mutation == 'origin':
                data[0]['provenance']['label_origin'] = None
            elif mutation == 'bounds':
                data[0]['ta_output']['total_score'] = '999'
            else:
                data.append(data[0])
            dataset = self.root / 'dataset.jsonl'
            dataset.write_text(''.join(json.dumps(r)+'\n' for r in data))
            config = self.config('evaluation/synthetic.json', dataset=str(dataset))
            with patch('krugradeai.evaluation.ScriptedProvider') as provider:
                with self.assertRaises(ValueError):
                    run(config)
                provider.assert_not_called()

    def test_live_opt_in_dry_run_and_relative_paths(self):
        config = self.config('evaluation/synthetic.json', runtime='codex_cli', model='gpt-6-luna')
        with patch('krugradeai.evaluation.CodexCLIProvider') as provider:
            with self.assertRaises(ValueError):
                run(config)
            self.assertEqual(run(config, dry_run=True)['maximum_calls'], 2)
            provider.assert_not_called()
        self.assertFalse((self.root/'output').exists())

    def test_validation_selection_export_and_unchanged_default(self):
        plain = optimize(self.config('optimization/synthetic.json', 'plain'))
        selected = optimize(self.config('optimization/synthetic-validated.json', 'selected'))
        self.assertFalse(plain['validation_loaded'])
        self.assertTrue(selected['validation_loaded'])
        self.assertEqual(plain['paths'], selected['paths'])
        history = json.loads((self.root/'selected/optimization.json').read_text())
        for index, path in enumerate(history['paths'], 1):
            best = path['validation']['best_version']
            definition = json.loads((self.root/f'selected/prompt-{index}.json').read_text())
            self.assertEqual(definition['adaptation_rules'], path['rule_versions'][best]['rules'])
            self.assertNotIn('ta_output', definition)
            self.assertTrue(selected['prompt_exports'][index-1]['validated'])

    def test_both_cli_entry_points_in_subprocesses(self):
        for module, example in [('run_evaluation', 'evaluation/synthetic.json'),
                                ('run_optimization', 'optimization/synthetic.json'),
                                ('run_optimization', 'optimization/synthetic-validated.json')]:
            config = self.config(example, module + str(len(list(self.root.iterdir()))))
            result = subprocess.run([sys.executable, '-m', 'krugradeai.' + module, '--config', str(config)],
                                    cwd=ROOT, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'], 'completed')

    def test_evaluation_import_has_no_optimization_dependency(self):
        code = "import sys; import krugradeai.run_evaluation; assert not any('krugradeai.'+n in sys.modules for n in ('optimization','reflector','refiner','run_optimization'))"
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
