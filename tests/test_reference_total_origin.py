"""An OCR reviewer does not determine authorship of a reference total."""
import json
import unittest
from krugradeai.synthetic import make_synthetic_dataset, build_mock_agents
from krugradeai.optimization import OptimizationLoop
from krugradeai.models import AdaptationRules


class TotalOriginTests(unittest.TestCase):
    def test_actual_grader_and_reflector_requests(self):
        for origin in ('original_ta', 'ai_generated', None):
            row = make_synthetic_dataset().development[0]
            row = row.model_copy(update={'provenance':row.provenance.model_copy(update={
                'score_origin':origin, 'ocr_review_origin':'ai'})})
            agents = build_mock_agents()
            OptimizationLoop(grader=agents[0], reflector=agents[1], refiner=agents[2],
                             max_iterations=1).run((row,), AdaptationRules())
            for prompt in agents[0].provider.requests:
                payload=json.loads(prompt.split('\nINPUT_JSON\n')[1])
                self.assertEqual(set(payload), {'question','rubric','response','adaptation_rules'})
            self.assertTrue(agents[1].provider.requests)
            for prompt in agents[1].provider.requests:
                context=json.loads(prompt.split('\nCOMPUTED_CONTRACT_CONTEXT\n')[1].split('\nINPUT_JSON\n')[0])
                self.assertEqual(context['reference_total_score_origin'],origin or 'unknown')
                label=json.loads(prompt.split('\nINPUT_JSON\n')[1])['ta_output']
                self.assertEqual(label,row.ta_output.model_dump(mode='json'))
