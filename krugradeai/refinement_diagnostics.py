"""Privacy-safe evidence summaries, not refinement authorization or scoring."""
from collections import Counter
from krugradeai.models import ReflectorInput, ReflectorOutput


def evidence_summary(request: ReflectorInput, output: ReflectorOutput) -> dict:
    """Summarize validated reflection metadata without quotes, labels or reasons.

    Existing Reflector/Refiner validators and orchestration remain authoritative.
    A total-only discrepancy is eligible for uncertain analysis, not for a
    supported criterion-error claim or automatic rule changes.
    """
    labeled = {c.criterion_id for c in request.ta_output.criterion_scores}
    maxima = {c.id: c.max_score for c in request.record.grader_input.rubric.criteria}
    label_scores = {c.criterion_id: c.score for c in request.ta_output.criterion_scores}
    known_sum = sum(label_scores.values())
    remaining = sum(maximum for cid, maximum in maxima.items() if cid not in labeled)
    supported = any(f.category == "grader_error" and f.status == "supported" and f.criterion_id in labeled
                    for f in output.findings)
    categories = sorted({f.category for f in output.findings})
    statuses = sorted({f.status for f in output.findings})
    blockers = []
    if not labeled and output.overall_score_difference != 0:
        blockers.append("total_only_discrepancy_unattributed")
    if "hypothesis" in statuses:
        blockers.append("hypothesis_requires_review")
    if "insufficient_evidence" in categories:
        blockers.append("insufficient_evidence")
    if "ta_rubric_inconsistency" in categories:
        blockers.append("suspected_label_rubric_inconsistency")
    if not supported:
        blockers.append("no_supported_criterion_error")
    if output.human_review_required:
        blockers.append("human_review_blocks_refiner")
    finding_diagnostics = []
    for index, finding in enumerate(output.findings):
        # These are typed categories and known IDs, never model-supplied claims,
        # quotations, numerical labels, suggestions or free-text review reasons.
        cid = finding.criterion_id
        finding_diagnostics.append(dict(finding_index=index,
            criterion_id=cid if cid is None or cid in maxima else "unknown_criterion",
            category=finding.category, status=finding.status,
            reference_criterion_label_available=cid in labeled,
            evidence_sources=sorted({e.source for e in finding.evidence}),
            rubric_evidence_criterion_ids=sorted({e.criterion_id for e in finding.evidence
                if e.source == "rubric" and e.criterion_id in maxima}),
            adaptation_suggestion_present=finding.adaptation_suggestion is not None,
            finding_requires_review=finding.status == "hypothesis" or finding.category in (
                "insufficient_evidence", "ta_rubric_inconsistency")))
    return dict(total_only_reference=not labeled, reference_criterion_label_count=len(labeled),
                reference_structure=dict(criterion_ids_known=labeled <= maxima.keys(),
                    criterion_scores_within_bounds=all(cid in maxima and 0 <= score <= maxima[cid] for cid, score in label_scores.items()),
                    complete_criterion_coverage=labeled == maxima.keys(),
                    total_consistent_with_available_labels=known_sum <= request.ta_output.total_score <= known_sum + remaining,
                    checks_are_structural_not_semantic=True),
                operational_policy=dict(ai_origin_alone_requires_review=False,
                    rubric_authority_over_operational_targets=True,
                    source_labels_do_not_prove_grader_error=True),
                finding_diagnostics=finding_diagnostics,
                criterion_label_origins=dict(Counter(c.provenance.label_origin if getattr(c, "provenance", None) else "legacy_unspecified"
                                                     for c in request.ta_output.criterion_scores)),
                operational_roles=dict(Counter(c.provenance.operational_role if getattr(c, "provenance", None) else "legacy_unspecified"
                                                for c in request.ta_output.criterion_scores)),
                finding_categories=categories, finding_statuses=statuses,
                review_reason_count=len(output.review_reasons), human_review_required=output.human_review_required,
                supported_criterion_error=supported,
                refinement_evidence_eligible=supported and not output.human_review_required,
                blocker_codes=blockers, raw_reason_or_evidence_retained=False)
