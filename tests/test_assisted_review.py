import json
import tempfile
import unittest
from pathlib import Path
from krugradeai.assisted_review import AssistedGrader
from krugradeai.grader import HumanReviewRequired
from krugradeai.models import GraderInput
from krugradeai.synthetic import make_synthetic_dataset, build_mock_agents
from krugradeai.mock_provider import MockLLMProvider


class NeedsReview:
    def grade(self, request):
        raise HumanReviewRequired("No confident award")


class AssistedReviewTests(unittest.TestCase):
    def test_provisional_assessment_is_blind_and_audited(self):
        row=make_synthetic_dataset().development[0]
        request=GraderInput(question=row.question,response=row.response,rubric=row.rubric)
        grade=build_mock_agents()[0].grade(request)
        provider=MockLLMProvider(json.dumps(dict(result=dict(provenance='ai_adjudicated',
            grade=grade.model_dump(mode='json'),uncertainty='Synthetic uncertainty'))))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audit.jsonl'
            output=AssistedGrader(NeedsReview(),provider,path,'validation').grade(request)
            self.assertIn('ai_adjudicated',output.overall_feedback)
            payload=json.loads(provider.requests[0].split('\nINPUT_JSON\n')[1])
            self.assertEqual(set(payload),{'question','answer','rubric','rules','review_reason'})
            events=list(map(json.loads,path.read_text().splitlines()))
            self.assertEqual([e['decision'] for e in events],['original_review','ai_adjudicated'])
            self.assertEqual(events[-1]['partition'],'validation')

    def test_abstention_remains_unresolved(self):
        row=make_synthetic_dataset().development[0]
        request=GraderInput(question=row.question,response=row.response,rubric=row.rubric)
        provider=MockLLMProvider('{"result":{"abstain":true,"reason":"Missing required visual evidence."}}')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audit.jsonl'
            with self.assertRaises(HumanReviewRequired):
                AssistedGrader(NeedsReview(),provider,path,'development').grade(request)
            self.assertEqual(len(provider.requests),1)
            self.assertEqual(json.loads(path.read_text().splitlines()[-1])['decision'],'abstained')
