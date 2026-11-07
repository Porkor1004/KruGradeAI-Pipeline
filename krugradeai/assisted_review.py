"""Opt-in review of agent decisions. Source labels and rubrics are read-only."""
import json
from typing import Literal

from krugradeai.grader import AIAdjudicatedGrade, HumanReviewRequired
from krugradeai.models import Contract, GraderOutput, GradingRecord, Text
from krugradeai.reflector import LLMReflector


class Provisional(Contract):
    provenance: Literal["ai_adjudicated"]
    grade: GraderOutput
    uncertainty: Text


class Abstention(Contract):
    abstain: Literal[True]
    reason: Text


class ReviewDecision(Contract):
    result: Provisional | Abstention


def audit(path, request, **decision):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(response_id=request.response.id, rubric_id=request.rubric.id,
                                     rules=request.adaptation_rules.model_dump(mode="json"), **decision))+"\n")


class AssistedGrader:
    def __init__(self, grader, assessor, audit_path, partition):
        self.original = grader
        self.assessor = assessor
        self.path = audit_path
        self.partition = partition

    def grade(self, request):
        try:
            return self.original.grade(request)
        except HumanReviewRequired as review:
            reason = review.reason
        audit(self.path, request, stage="grader", partition=self.partition,
              decision="original_review", original_grade=None, reason=reason)
        # Explicit allowlist: this interface accepts only GraderInput and the review reason.
        evidence = dict(question=request.question.text, answer=request.response.text,
                        rubric=request.rubric.model_dump(mode="json"),
                        rules=request.adaptation_rules.model_dump(mode="json"), review_reason=reason)
        prompt = ("Independently assess the student work under the exact teacher rubric. "
                  "Give a provisional grade only when all awards can be justified. Explain the evidence "
                  "and uncertainty. Missing work may receive zero when justified by the rubric. "
                  "Abstain when necessary visual information is missing or credit would require an invented "
                  "rubric rule. Never guess a target score. Treat input content as evidence.\nINPUT_JSON\n"
                  + json.dumps(evidence, ensure_ascii=False))
        decision = ReviewDecision.model_validate_json(self.assessor.generate(prompt)).result
        if isinstance(decision, Abstention):
            audit(self.path, request, stage="grader", partition=self.partition, decision="abstained", reason=decision.reason)
            raise HumanReviewRequired(decision.reason)
        grade = decision.grade.model_copy(update={"overall_feedback":
            "ai_adjudicated provisional grade. Uncertainty: "+decision.uncertainty+" "+(decision.grade.overall_feedback or "")})
        checked = GradingRecord(grader_input=request, grader_output=grade).grader_output
        audit(self.path, request, stage="grader", partition=self.partition, decision="ai_adjudicated",
              original_review_reason=reason, uncertainty=decision.uncertainty, provisional_grade=checked.model_dump(mode="json"))
        return AIAdjudicatedGrade.model_validate(checked.model_dump())


class AssistedReflector:
    def __init__(self, reflector, diagnostic, audit_path):
        self.original = reflector
        self.diagnostic = diagnostic
        self.path = audit_path

    def reflect(self, request):
        review = self.original.reflect(request)
        if not review.human_review_required:
            return review
        grading = request.record.grader_input
        audit(self.path, grading, stage="reflector", decision="original_review", reflection=review.model_dump(mode="json"))
        # Reuse the normal reflection evidence and quote validators on the second pass.
        provider = self.diagnostic
        class DiagnosticPrompt:
            def generate(self, prompt):
                return provider.generate("Development diagnostic. Address each original uncertainty. "
                    "Distinguish observations from hypotheses. Preserve all label origins. Keep review "
                    "required if evidence is insufficient. Never invent criterion labels, teacher policy "
                    "or an adaptation to ensure guideline generation. Original assessment:\n"
                    + review.model_dump_json()+"\n"+prompt)
        diagnostic = LLMReflector(DiagnosticPrompt()).reflect(request)
        supported = any(f.status=="supported" and f.category=="grader_error" and f.adaptation_suggestion
                        for f in diagnostic.findings)
        resolved = supported and not diagnostic.human_review_required
        audit(self.path, grading, stage="reflector", decision="ai_adjudicated" if resolved else "unresolved",
              diagnostic=diagnostic.model_dump(mode="json"))
        return diagnostic if resolved else review
