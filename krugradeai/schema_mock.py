"""Offline schema exercise only. Zero awards are scripts, never grading results."""
import json
from krugradeai.providers import LLMProvider


class SchemaMockProvider(LLMProvider):
    def __init__(self, role):
        self.role = role

    def generate(self, request):
        item = json.loads(request.split('\nINPUT_JSON\n', 1)[1])
        if self.role == 'grader':
            if 'ta_output' in item or 'provenance' in item:
                raise ValueError('Reference leaked into Grader')
            return json.dumps(dict(criterion_scores=[dict(criterion_id=c['id'], score='0', rationale='Offline scripted zero; not a grading assessment.') for c in item['rubric']['criteria']], total_score='0', overall_feedback='Schema mock only.'))
        if self.role == 'reflector':
            from decimal import Decimal
            delta = Decimal(item['record']['grader_output']['total_score']) - Decimal(item['ta_output']['total_score'])
            findings = [dict(status='hypothesis', category='insufficient_evidence', claim='Scripted zeros do not support causal adaptation evidence from total-only labels.', evidence=[dict(source='ta_label', quote=str(item['ta_output']['total_score']))])] if delta else []
            predicted = {c['criterion_id']: Decimal(c['score']) for c in item['record']['grader_output']['criterion_scores']}
            differences = [dict(criterion_id=c['criterion_id'], difference=str(predicted[c['criterion_id']]-Decimal(c['score']))) for c in item['ta_output'].get('criterion_scores', [])]
            return json.dumps(dict(observations=['Offline scripted reflection; no adaptation evidence.'], overall_score_difference=str(delta), direction='agreement' if delta == 0 else 'undergrading' if delta < 0 else 'overgrading', criterion_differences=differences, findings=findings, human_review_required=bool(delta), review_reasons=['Mock discrepancy is not evidence of a grading defect.'] if delta else []))
        return '{"rules":[]}'
