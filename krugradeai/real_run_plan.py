"""Data-only call reservations and conditional serial-runtime estimates."""
from collections import Counter, defaultdict
import math
import statistics

from krugradeai.optimization_models import ValidationConfig
from krugradeai.validation import prepare_validation

ROLES = ("grader", "reflector", "refiner", "adjudicator_independent", "adjudicator_comparison")


def adjudication_capable(record):
    """Same complete-label/nonblank precondition as the adjudication gate."""
    return bool(record.response.text.strip()) and {c.criterion_id for c in record.ta_output.criterion_scores} == {
        c.id for c in record.rubric.criteria}


def estimate_plan(development, validation, *, ai_adjudication, timeout, max_calls, historical_reports=()):
    # Fail before any provider construction on overlap, rubric or coverage errors.
    prepare_validation(tuple(development), tuple(validation), ValidationConfig())
    d, v = len(development), len(validation)
    p = len({r.rubric.id for r in development})
    e = sum(adjudication_capable(r) for r in development) if ai_adjudication else 0
    baseline = dict(grader=d+v, reflector=0, refiner=0,
                    adjudicator_independent=0, adjudicator_comparison=0)
    worst = dict(grader=d+2*v, reflector=d, refiner=p,
                 adjudicator_independent=e, adjudicator_comparison=e)
    timings = defaultdict(list)
    seen = {}
    for report in historical_reports:
        # Older pilot reports store only a scalar provider-call count. They are
        # valid history but cannot supply per-role timing observations.
        calls = report.get("provider_calls", ())
        for call in calls if isinstance(calls, (list, tuple)) else ():
            if not isinstance(call, dict):
                continue
            seconds = call.get("seconds")
            if (call.get("agent") in ROLES and call.get("transport_status") == "success"
                    and call.get("provider_invoked") is True and type(seconds) in (int, float)
                    and math.isfinite(seconds) and seconds >= 0):
                timings[call["agent"]].append(seconds)
        for path in report.get("paths", ()):
            for row in path.get("development", ()):
                if row.get("graded") is True and type(row.get("discrepancy")) is bool:
                    seen.setdefault(row["record_id"], row["discrepancy"])
    rate = sum(seen.values())/len(seen) if seen else 0.25
    latencies = {role: dict(seconds=statistics.mean(timings[role]) if timings[role] else 30.0,
                            observations=len(timings[role]),
                            basis="historical_success_mean" if timings[role] else "assumed_30_seconds_unobserved")
                 for role in ROLES}
    # Conditional planning scenario, NOT a prediction that any G1 will exist.
    planning = dict(grader=d+2*v, reflector=math.ceil(d*rate), refiner=p,
                    adjudicator_independent=math.ceil(e*rate), adjudicator_comparison=math.ceil(e*rate))
    scenario = lambda calls: dict(calls_by_role=calls, total_calls=sum(calls.values()),
        estimated_seconds=sum(calls[role]*latencies[role]["seconds"] for role in ROLES))
    per_path = []
    for rubric_id in sorted({r.rubric.id for r in development}):
        rows = tuple(r for r in development if r.rubric.id == rubric_id)
        per_path.append(dict(rubric_id=rubric_id, development_count=len(rows),
            validation_count=sum(r.rubric.id == rubric_id for r in validation),
            label_origins=dict(Counter(r.provenance.label_origin for r in rows)),
            criterion_enriched_count=sum(bool(r.ta_output.criterion_scores) for r in rows),
            total_only_count=sum(not r.ta_output.criterion_scores for r in rows)))
    return dict(development_count=d, validation_count=v, rubric_path_count=p,
        development_label_origins=dict(Counter(r.provenance.label_origin for r in development)),
        criterion_enriched_count=sum(bool(r.ta_output.criterion_scores) for r in development),
        total_only_count=sum(not r.ta_output.criterion_scores for r in development),
        adjudication_capable_count=e, per_path=per_path,
        grader_only=scenario(baseline), conditional_planning=scenario(planning), worst_case=scenario(worst),
        maximum_provider_calls=sum(worst.values()), configured_max_calls=max_calls,
        budget_sufficient=max_calls >= sum(worst.values()),
        historical_unique_development_grades=len(seen), assumed_discrepancy_rate=rate,
        discrepancy_rate_basis="historical_selected_examples_not_representative" if seen else "assumed_25_percent",
        planning_assumptions="All discrepant eligible records adjudicated; every path refines and validates G1. Neither is guaranteed.",
        latency_by_role=latencies, timeout_sum_seconds=sum(worst.values())*timeout,
        runtime_limitations="Serial estimates exclude local overhead. Per-call timeout sum is not an overall deadline. Historical small samples are biased; Refiner latency may be unobserved.",
        max_iterations=1, resume_supported=False, held_out_accessed=False)
