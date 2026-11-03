"""Opt-in two-stage Development adjudication; never changes reference labels."""
import hashlib
import json
from decimal import Decimal
from typing import Literal

from pydantic import Field, StrictBool
from krugradeai.adjudicator_prompt import independent_prompt, comparison_prompt
from krugradeai.agents import Reflector
from krugradeai.models import Contract, Score, Text, ReflectionEvidence, ReflectorInput, ReflectorOutput
from krugradeai.reflector import (_ValidatedReflection, _ReflectionResponse,
                                  _unique_object, _reject_constant)

POLICY_VERSION = "ai-adjudication-v2-independent-rubric"


class IndependentAssessment(Contract):
    criterion_id: Text
    independent_score: Score
    rationale: Text
    evidence: tuple[ReflectionEvidence, ...] = Field(min_length=2)
    confidence: Literal["high", "medium", "low"]
    uncertainty_reasons: tuple[Text, ...]
    rubric_consistent: StrictBool
    provenance: Literal["ai_adjudicated"]


class IndependentOutput(Contract):
    assessments: tuple[IndependentAssessment, ...] = Field(min_length=1)


class AdjudicationDecision(Contract):
    criterion_id: Text
    independent_score: Score
    assessment: Literal["grader_error", "grader_supported", "reference_inconsistent", "unresolved"]
    rationale: Text
    confidence: Literal["high", "medium", "low"]
    uncertainty_reasons: tuple[Text, ...]
    supports_adaptation: StrictBool
    adaptation_suggestion: Text | None
    provenance: Literal["ai_adjudicated"]


class ComparisonOutput(Contract):
    decisions: tuple[AdjudicationDecision, ...] = Field(min_length=1)
    all_review_concerns_resolved: StrictBool


def _parse(provider, prompt, model):
    return model.model_validate(json.loads(provider.generate(prompt), parse_float=Decimal,
        object_pairs_hook=_unique_object, parse_constant=_reject_constant))


def _exact_ids(items, expected):
    ids = [x.criterion_id for x in items]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Adjudication must cover exactly the disputed rubric criteria")


class AIAdjudicator:
    """Separate provider requests and locked scores prevent reference-score leakage.

    No provider conversation state is passed between stages; Codex runs ephemeral
    processes. Same-model AI assessments can still have correlated errors.
    """
    def __init__(self, independent_provider, comparison_provider):
        self.independent_provider = independent_provider
        self.comparison_provider = comparison_provider
        self.events = []

    def adjudicate(self, request, review):
        request = ReflectorInput.model_validate(request)
        source = request.record.grader_input
        labels = {c.criterion_id: c.score for c in request.ta_output.criterion_scores}
        grade = {c.criterion_id: c.score for c in request.record.grader_output.criterion_scores}
        disputed = tuple(cid for cid in labels if labels[cid] != grade[cid])
        event = dict(response_id=source.response.id, rubric_id=source.rubric.id,
            provenance="ai_adjudicated", status="excluded", eligible=False,
            policy_version=POLICY_VERSION, source_score_agreement_required=False,
            independent_validated=False, comparison_validated=False,
            source_and_grader_scores=[dict(criterion_id=cid, source_score=str(labels[cid]),
                grader_score=str(grade[cid]), grader_minus_source=str(grade[cid]-labels[cid])) for cid in disputed],
            original_human_review=True, original_review_reason_count=len(review.review_reasons),
            original_finding_categories=sorted({f.category for f in review.findings}),
            original_finding_statuses=sorted({f.status for f in review.findings}),
            criterion_label_origins=sorted({c.provenance.label_origin if c.provenance else "legacy_unspecified"
                for c in request.ta_output.criterion_scores}), assessments=[], decisions=[])
        self.events.append(event)
        if not review.human_review_required:
            event["reason_code"] = "not_a_review_case"
            return review
        # Conservative entire-record policy: no unattributed total-score component.
        if not disputed or set(labels) != set(grade) or not source.response.text.strip():
            event["reason_code"] = "incomplete_or_unattributed_evidence"
            return review
        stage = "independent_output"
        try:
            first = _parse(self.independent_provider, independent_prompt(request, disputed), IndependentOutput)
            stage = "independent_context"
            _exact_ids(first.assessments, disputed)
            rubric = {c.id: c for c in source.rubric.criteria}
            for item in first.assessments:
                if item.independent_score > rubric[item.criterion_id].max_score:
                    raise ValueError("Independent score exceeds rubric bounds")
                if not any(e.source == "student_answer" for e in item.evidence) or not any(
                        e.source == "rubric" and e.criterion_id == item.criterion_id for e in item.evidence):
                    raise ValueError("Independent assessment requires answer and criterion evidence")
                for evidence in item.evidence:
                    if evidence.criterion_id not in (None, item.criterion_id):
                        raise ValueError("Adjudication evidence criterion mismatch")
                    if evidence.source == "student_answer":
                        text = source.response.text
                    elif evidence.source == "rubric" and evidence.criterion_id == item.criterion_id:
                        text = rubric[item.criterion_id].description
                    else:
                        raise ValueError("Independent evidence must be student answer or rubric")
                    if evidence.quote not in text:
                        raise ValueError("Unsupported adjudication evidence quotation")
            # The complete score-blind result is now frozen before phase 2 exists.
            event["independent_sha256"] = hashlib.sha256(first.model_dump_json().encode()).hexdigest()
            event["independent_validated"] = True
            event["assessments"] = [dict(criterion_id=i.criterion_id, independent_score=str(i.independent_score),
                confidence=i.confidence, rubric_consistent=i.rubric_consistent,
                uncertainty_count=len(i.uncertainty_reasons), evidence_sources=sorted({e.source for e in i.evidence}),
                rationale_sha256=hashlib.sha256(i.rationale.encode()).hexdigest(),
                provenance=i.provenance) for i in first.assessments]
            stage = "comparison_output"
            second = _parse(self.comparison_provider, comparison_prompt(request, review, first), ComparisonOutput)
            stage = "comparison_context"
            _exact_ids(second.decisions, disputed)
            independent = {i.criterion_id: i for i in first.assessments}
            for decision in second.decisions:
                if decision.independent_score != independent[decision.criterion_id].independent_score:
                    raise ValueError("Comparison attempted to change locked independent assessment")
                if decision.supports_adaptation and (
                        decision.assessment != "grader_error" or decision.confidence != "high" or
                        decision.uncertainty_reasons or not decision.adaptation_suggestion):
                    raise ValueError("Contradictory adjudication adaptation claim")
                if decision.assessment == "grader_error" and decision.supports_adaptation and (
                        decision.independent_score == grade[decision.criterion_id]):
                    raise ValueError("Adjudicated grader error requires disagreement with Grader")
            event["comparison_validated"] = True
            event["all_review_concerns_resolved"] = second.all_review_concerns_resolved
            event["decisions"] = [dict(criterion_id=d.criterion_id, independent_score=str(d.independent_score),
                assessment=d.assessment, confidence=d.confidence, uncertainty_count=len(d.uncertainty_reasons),
                rationale_sha256=hashlib.sha256(d.rationale.encode()).hexdigest(),
                supports_adaptation=d.supports_adaptation, provenance=d.provenance) for d in second.decisions]
            blockers = []
            if not second.all_review_concerns_resolved:
                blockers.append("review_concerns_unresolved")
            for d in second.decisions:
                i = independent[d.criterion_id]
                if d.assessment == "unresolved":
                    blockers.append("comparison_unresolved")
                elif d.assessment == "reference_inconsistent":
                    blockers.append("reference_inconsistency_without_supported_grader_error")
                elif d.assessment == "grader_supported":
                    blockers.append("adjudication_supports_grader")
                if not d.supports_adaptation:
                    blockers.append("adaptation_not_supported")
                if d.confidence != "high" or d.uncertainty_reasons:
                    blockers.append("comparison_uncertain")
                if i.confidence != "high" or i.uncertainty_reasons:
                    blockers.append("independent_assessment_uncertain")
                if not i.rubric_consistent:
                    blockers.append("independent_assessment_not_rubric_consistent")
            event["blocker_codes"] = sorted(set(blockers))
            eligible = second.all_review_concerns_resolved and all(
                d.supports_adaptation and d.assessment == "grader_error" and d.confidence == "high"
                and not d.uncertainty_reasons and independent[d.criterion_id].confidence == "high"
                and not independent[d.criterion_id].uncertainty_reasons
                and independent[d.criterion_id].rubric_consistent for d in second.decisions)
            if not eligible:
                event["reason_code"] = "unresolved_or_unsupported_adjudication"
                return review
            total_diff = request.record.grader_output.total_score - request.ta_output.total_score
            # Original arithmetic and evidence guards still apply. No TAReference is
            # replaced; adjudication is an evidence source, never a dataset label.
            replacement = ReflectorOutput(observations=("AI-adjudicated operational evidence; not human verification.",),
                overall_score_difference=total_diff,
                direction="overgrading" if total_diff > 0 else "undergrading" if total_diff < 0 else "agreement",
                criterion_differences=tuple(dict(criterion_id=cid, difference=grade[cid]-labels[cid]) for cid in labels),
                findings=tuple(dict(criterion_id=d.criterion_id, status="supported", category="grader_error",
                    claim=("[ai_adjudicated; not human-verified] Independent rubric-based criterion score: "
                           + str(d.independent_score) + "; original Grader criterion score: "
                           + str(grade[d.criterion_id]) + "; unchanged source reference criterion score: "
                           + str(labels[d.criterion_id]) + ". " + d.rationale),
                    evidence=independent[d.criterion_id].evidence,
                    adaptation_suggestion=d.adaptation_suggestion) for d in second.decisions))
            stage = "refinement_conversion"
            replacement = _ValidatedReflection(request=request,
                output=_ReflectionResponse.model_validate(replacement.model_dump())).output
            event.update(status="supported", eligible=True, reason_code="supported_ai_adjudication")
            event["refiner_finding_count"] = len(replacement.findings)
            return replacement
        except Exception as error:
            # Malformed, inconsistent or failed adjudications retain original review.
            event.update(status="failure", reason_code="adjudication_validation_or_provider_failure",
                         error_type=type(error).__name__, failure_stage=stage)
            return review


class AdjudicatingReflector(Reflector):
    """Runner-only policy adapter; optimizer and default Reflector are unchanged."""
    def __init__(self, reflector, adjudicator, development):
        self.reflector, self.adjudicator = reflector, adjudicator
        self.allowed = {r.response.id: r for r in development}

    def reflect(self, request):
        source = request.record.grader_input
        record = self.allowed.get(source.response.id)
        if record is None or (source.question, source.rubric, source.response, request.ta_output) != (
                record.question, record.rubric, record.response, record.ta_output):
            raise ValueError("Adjudication accepts only unchanged selected Development evidence")
        review = self.reflector.reflect(request)
        if not review.human_review_required:
            return review
        result = self.adjudicator.adjudicate(request, review)
        self.adjudicator.events[-1]["record_label_origin"] = record.provenance.label_origin if record.provenance else "legacy_unspecified"
        return result
