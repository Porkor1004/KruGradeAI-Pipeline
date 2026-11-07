"""Offline checks for the production CLI controls and cohort metrics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from krugradeai.codex_provider import CodexCLIProvider
from krugradeai.grader import AIAdjudicatedGrade, HumanReviewRequired
from krugradeai.models import AdaptationRules
from krugradeai.prediction_reporting import prediction_report
from krugradeai.run_optimization import OptimizationConfig, run
from krugradeai.synthetic import make_synthetic_dataset, build_mock_agents
from krugradeai.optimization_models import ValidationConfig
from krugradeai.validation import evaluate_labeled_records, ValidationTracker

ROOT = Path(__file__).resolve().parents[1]


class StandardExperimentTests(unittest.TestCase):
    def test_retry_settings_on_cli_and_forbidden_nonzero_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe=Path(tmp)/'codex.exe'
            exe.touch()
            p=CodexCLIProvider(executable=exe,disable_cli_retries=True)
            for kind in ('request','stream'):
                self.assertIn('model_providers.experiment_subscription.'+kind+'_max_retries=0',p.command)
            with self.assertRaises(ValueError):
                OptimizationConfig(development='d',output_directory='o',cli_request_retries=1)

    def test_true_projection_rejected_before_provider_or_output(self):
        path=ROOT/'examples/optimization/synthetic-validated.json'
        cfg=json.loads(path.read_text())
        for key in ('development','validation'):
            cfg[key]=str(path.parent/cfg[key])
        with tempfile.TemporaryDirectory() as tmp:
            cfg.update(runtime='codex_cli',review_policy='ai_assisted_pilot',max_calls=20,
                       max_iterations=1,output_directory=str(Path(tmp)/'out'))
            cp=Path(tmp)/'cfg.json';cp.write_text(json.dumps(cfg))
            with patch('krugradeai.run_optimization.CodexCLIProvider') as provider:
                with self.assertRaisesRegex(ValueError,'Projected maximum'):
                    run(cp,live=True)
                provider.assert_not_called()
            self.assertFalse(Path(cfg['output_directory']).exists())

    def test_reporting_uses_typed_provenance_and_fixed_denominators(self):
        rows=make_synthetic_dataset().validation
        base=build_mock_agents()[0]
        class Mixed:
            def grade(self,req):
                direct=base.grade(req)
                if req.response.id==rows[1].response.id:
                    return AIAdjudicatedGrade.model_validate(direct.model_dump())
                return direct.model_copy(update={'overall_feedback':'ai_adjudicated is ordinary text here'})
        ev=evaluate_labeled_records(Mixed(),rows,AdaptationRules(),0,ValidationConfig())
        r=prediction_report(ev,rows)
        self.assertEqual(r['direct_only']['denominator'],1)
        self.assertEqual(r['adjudicated_only']['denominator'],1)
        self.assertEqual(r['direct_only']['coverage'],'0.5')
        self.assertEqual(r['assisted_policy_total']['denominator'],2)
        self.assertEqual(r['assisted_policy_total']['mae'],ev.metrics.model_dump(mode='json')['mae'])
        self.assertNotIn(rows[0].response.text,json.dumps(r))

    def test_validation_stops_at_first_failure_or_unresolved_review(self):
        from krugradeai.models import AdaptationRules
        rows=make_synthetic_dataset().validation
        for error in (RuntimeError('synthetic failure'),HumanReviewRequired('synthetic uncertainty')):
            seen=[]
            class Fails:
                def grade(self,request):
                    seen.append(request.response.id)
                    raise error
            reports=[]
            tracker=ValidationTracker(Fails(),rows,ValidationConfig(),fail_fast=True,
                observer=lambda t,e: reports.append(prediction_report(e,rows)))
            with self.assertRaisesRegex(RuntimeError,'Validation stopped'):
                tracker.observe(0,AdaptationRules())
            self.assertEqual(len(seen),1)
            self.assertFalse(reports[0]['full_coverage'])
            self.assertEqual(reports[0]['assisted_policy_total']['population'],len(rows))
            self.assertEqual(reports[0]['unresolved_review_count'],int(isinstance(error,HumanReviewRequired)))
