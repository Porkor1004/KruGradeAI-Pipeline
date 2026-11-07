"""Coverage and MAE for the fixed scoring population, without exposing answers."""
from collections import Counter
from decimal import Decimal, localcontext


def prediction_report(evaluation, records):
    by_id = {r.id: r for r in records}
    direct, assisted = [], []
    origins = Counter()
    errors_by_origin = {}
    for outcome in evaluation.records:
        record = by_id[outcome.record_id]
        origin = record.provenance.score_origin if record.provenance else None
        origin = origin or 'unspecified'
        origins[origin] += 1
        groups = errors_by_origin.setdefault(origin, {'direct': [], 'adjudicated': []})
        if outcome.grading is not None:
            error = abs(outcome.grading.grader_output.total_score - record.ta_output.total_score)
            (assisted if outcome.ai_adjudicated else direct).append(error)
            groups['adjudicated' if outcome.ai_adjudicated else 'direct'].append(error)

    def metric(errors, population):
        with localcontext() as context:
            context.prec = 50
            return dict(count=len(errors), denominator=len(errors), population=population,
                        coverage=str(Decimal(len(errors))/population) if population else None,
                        mae=str(sum(errors)/len(errors)) if errors else None)

    n = len(records)
    return dict(version=evaluation.version, sample_count=n, reference_score_origins=dict(origins),
                direct_only=metric(direct,n), adjudicated_only=metric(assisted,n),
                assisted_policy_total=metric(direct+assisted,n),
                by_reference_origin={origin:{kind:metric(errors, origins[origin])
                    for kind, errors in groups.items()} for origin, groups in errors_by_origin.items()},
                unresolved_review_count=len(evaluation.human_reviews), failure_count=len(evaluation.failures),
                full_coverage=len(direct)+len(assisted)==n,
                selection_eligible=evaluation.eligible, selection_mae=evaluation.comparison_metrics.model_dump(mode='json')['mae'])
