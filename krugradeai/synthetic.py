"""Explicitly synthetic fixtures and scripted providers, never real accuracy data."""
import json
import re
from decimal import Decimal

from krugradeai.experiment_models import DatasetProvenance, ExperimentDataset, HeldoutRecord
from krugradeai.grader import LLMGrader
from krugradeai.mock_provider import MockLLMProvider
from krugradeai.models import (
    AdaptationRule, AdaptationRules, GraderInput, Question, ReflectorInput,
    RefinerInput, Rubric, RubricCriterion, StudentResponse, TAReference, TACriterionScore,
)
from krugradeai.optimization_models import DevelopmentRecord, RecordProvenance, ValidationRecord
from krugradeai.reflector import LLMReflector
from krugradeai.refiner import LLMRefiner

SOURCE = "krugradeai-fully-synthetic-phase1-v1"
# These scripts are toy fixtures, not a mathematical grading algorithm. They are
# intentionally independent from the record objects containing human labels.
ANSWERS = (
    "x + x = 8; hence x = 4.",
    "Divide both sides by 2; I did not complete the calculation.",
    "8 = 2x; dividing both sides by 2 gives x = 4.",
    "I would divide by 2, but stopped before giving a result.",
    "Halving the two equal sides gives x = 4.",
    "I would halve the left side, but have not transformed the right side.",
)


def make_synthetic_dataset(*, multiple_rubrics: bool = False) -> ExperimentDataset:
    development, validation, heldout = [], [], []
    for scope in range(1, 3 if multiple_rubrics else 2):
        question = Question(id=f"synthetic-question-{scope}", text="Solve 2x = 8 and show reasoning.",
                            reference_solution="Divide both sides by 2 to obtain x = 4.")
        rubric = Rubric(id=f"synthetic-linear-{scope}", criteria=(
            RubricCriterion(id="method", description="Credit valid equivalent transformations. Partial credit is allowed for incomplete valid reasoning.", max_score="1"),
            RubricCriterion(id="result", description="Credit a correct final value supported by the stated work. Do not invent a missing final value.", max_score="1"),
        ), total_score="2")
        for partition, record_type, destination, offsets in (
            ("development", DevelopmentRecord, development, (0, 1)),
            ("validation", ValidationRecord, validation, (2, 3)),
            ("test", HeldoutRecord, heldout, (4, 5)),
        ):
            for number, offset in enumerate(offsets, 1):
                id = f"synthetic-{partition}-{number}-scope-{scope}"
                method = "0.5" if offset in (3, 5) else "1"
                final = "1" if offset in (0, 2, 4) else "0"
                destination.append(record_type(
                    id=id, question=question, rubric=rubric,
                    response=StudentResponse(id=id + "-response", question_id=question.id, text=ANSWERS[offset]),
                    ta_output=TAReference(total_score=Decimal(method) + Decimal(final), criterion_scores=(
                        TACriterionScore(criterion_id="method", score=method), TACriterionScore(criterion_id="result", score=final))),
                    provenance=RecordProvenance(source=SOURCE, item_id=id),
                ))
    return ExperimentDataset(provenance=DatasetProvenance(kind="synthetic", sources=(SOURCE,),
                             description="Fully synthetic software demonstration; not a valid real-world accuracy experiment."),
                             development=tuple(development), validation=tuple(validation), heldout=tuple(heldout))


class SyntheticMockProvider(MockLLMProvider):
    """Scripted transport for existing agents; no access to a dataset/label store."""

    def __init__(self, role: str):
        super().__init__("")
        if role not in ("grader", "reflector", "refiner"):
            raise ValueError("Unknown synthetic provider role")
        self.role = role

    def generate(self, request: str) -> str:
        self.requests.append(request)
        payload = request.split("\nINPUT_JSON\n", 1)[1]
        if self.role == "grader":
            item = GraderInput.model_validate_json(payload)
            try:
                offset = ANSWERS.index(item.response.text)
            except ValueError:
                raise ValueError("Synthetic provider received an unknown toy answer") from None
            adapted = bool(item.adaptation_rules.rules)
            # Leave one development discrepancy unresolved to exercise subsequent
            # proposals and validation patience; test/validation outcomes are fixed.
            method = ("0.5" if offset in (0, 1, 2, 4) else "0")
            if adapted and offset != 1:
                method = "0.5" if offset in (3, 5) else "1"
            final = "1" if offset in (0, 2, 4) else "0"
            return json.dumps({"criterion_scores": [
                {"criterion_id": "method", "score": method, "rationale": "Synthetic scripted assessment of transformation evidence."},
                {"criterion_id": "result", "score": final, "rationale": "Synthetic scripted assessment of stated final value."}],
                "total_score": str(Decimal(method) + Decimal(final)), "overall_feedback": "Synthetic scripted feedback, not a real grading assessment."})
        if self.role == "reflector":
            item = ReflectorInput.model_validate_json(payload)
            ai = {c.criterion_id: c.score for c in item.record.grader_output.criterion_scores}
            difference = item.record.grader_output.total_score - item.ta_output.total_score
            method = next(c for c in item.record.grader_input.rubric.criteria if c.id == "method")
            return json.dumps({"observations": ["Synthetic development analysis only."],
                "overall_score_difference": str(difference),
                "direction": "undergrading" if difference < 0 else "overgrading" if difference > 0 else "agreement",
                "criterion_differences": [{"criterion_id": c.criterion_id, "difference": str(ai[c.criterion_id] - c.score)} for c in item.ta_output.criterion_scores],
                "findings": [{"criterion_id": "method", "status": "supported", "category": "grader_error",
                    "claim": "Synthetic discrepancy suggests inconsistent interpretation of transformation evidence.",
                    "evidence": [{"source": "student_answer", "quote": item.record.grader_input.response.text, "criterion_id": None},
                                 {"source": "rubric", "quote": method.description, "criterion_id": "method"}],
                    "adaptation_suggestion": "Interpret equivalent transformations only as allowed by the rubric."}],
                "human_review_required": False, "review_reasons": []})
        item = RefinerInput.model_validate_json(payload)
        previous = next((r for r in item.current_rules.rules if isinstance(r, AdaptationRule)), None)
        match = re.search(r"Synthetic revision (\d+)", previous.rationale) if previous is not None else None
        revision = int(match.group(1)) + 1 if match else 1
        instruction = "Recognize equivalent mathematical transformations only as permitted by the rubric."
        if revision >= 2:
            instruction += " Inspect intermediate reasoning before deciding rubric compliance."
        if revision >= 3:
            instruction += " Distinguish incomplete reasoning from invalid transformations when the rubric permits partial reasoning."
        rules = AdaptationRules(rules=(AdaptationRule(id="equivalent-methods", instruction=instruction,
                    criterion_ids=("method",), rationale=f"Synthetic revision {revision}: supported development evidence motivates clearer interpretation."),))
        return rules.model_dump_json()


def build_mock_agents():
    """Return existing agent classes with deterministic synthetic providers."""
    return (LLMGrader(SyntheticMockProvider("grader")), LLMReflector(SyntheticMockProvider("reflector")),
            LLMRefiner(SyntheticMockProvider("refiner")))
