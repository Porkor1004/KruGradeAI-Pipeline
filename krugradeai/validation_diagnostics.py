"""Allowlisted validation metadata; never persist rejected values or messages."""

RULES = {
    "Insufficient evidence cannot establish a definitive cause": "insufficient_evidence_requires_hypothesis",
    "Supported grading errors require an adaptation improvement suggestion": "supported_error_requires_suggestion",
    "Difference and direction must be supplied together": "difference_direction_pair_required",
    "Direction contradicts the AI-minus-TA difference": "direction_sign_mismatch",
    "Criterion differences must be unique": "duplicate_criterion_differences",
    "Human review requires reasons; reasons require human review": "review_flag_reasons_mismatch",
    "Uncertain findings and suspected label inconsistencies require human review": "uncertainty_requires_review",
    "Overall difference must equal AI total minus verified TA total": "total_difference_mismatch",
    "Overall difference must equal AI total minus reference total": "total_difference_mismatch",
    "Supported source inconsistency requires the cited criterion label evidence": "supported_inconsistency_requires_label_evidence",
    "Supported source inconsistency requires the cited rubric evidence": "supported_inconsistency_requires_rubric_evidence",
    "Criterion differences must match exactly the available TA labels": "criterion_difference_label_mismatch",
    "A discrepancy requires evidence-backed findings or uncertainty analysis": "discrepancy_requires_findings",
    "Blank student work requires human review": "blank_answer_requires_review",
    "Unknown finding criterion ID": "unknown_finding_criterion",
    "No-discrepancy claims require score agreement at the cited scope": "unsupported_agreement_claim",
    "Supported criterion error requires the corresponding TA criterion label": "supported_error_requires_criterion_label",
    "Supported grading errors require student-answer evidence": "supported_error_requires_student_evidence",
    "Supported grading errors require evidence from the cited rubric criterion": "supported_error_requires_rubric_evidence",
    "Unknown evidence criterion ID": "unknown_evidence_criterion",
    "Evidence criterion contradicts finding criterion": "evidence_finding_criterion_mismatch",
    "Rubric evidence requires a criterion ID": "rubric_evidence_requires_criterion",
    "Grader rationale evidence requires a criterion ID": "rationale_evidence_requires_criterion",
    "Cannot cite an unavailable TA criterion label": "unavailable_criterion_label_citation",
    "TA evidence must quote the exact label value": "label_quote_mismatch",
    "Evidence quote does not occur in the cited source": "evidence_quote_not_found",
}
ERROR_TYPES = {"missing", "extra_forbidden", "literal_error", "bool_type", "string_type", "string_too_short",
               "decimal_parsing", "finite_number", "tuple_type", "list_type", "too_short", "model_type",
               "greater_than_equal", "greater_than", "value_error", "none_required", "dict_type"}


def validation_details(error, model, stage):
    fields = set()

    def visit(node):
        if isinstance(node, dict):
            fields.update(node.get("properties", {}))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(model.model_json_schema())
    details = []
    for issue in error.errors(include_input=False, include_context=False, include_url=False)[:32]:
        kind = issue["type"] if issue["type"] in ERROR_TYPES else "validation_error"
        rule = RULES.get(issue["msg"].removeprefix("Value error, "))
        location = [value if isinstance(value, int) or value in fields else "unknown_field" for value in issue["loc"]]
        details.append(dict(stage=stage, category="cross_field" if rule else "structure_or_type" if kind != "value_error" else "unclassified_validator",
                            code=rule or kind, location=location))
    return tuple(details)


def annotate(error, model, stage):
    error.safe_diagnostics = validation_details(error, model, stage)
    return error
