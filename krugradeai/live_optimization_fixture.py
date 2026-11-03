"""Fixed synthetic consequential-error fixture with rubric-derived labels.

No discrepancy is guaranteed or required. The reference scores are justified by
explicit criteria, not chosen to disagree with any model output.
"""
from decimal import Decimal
from krugradeai.experiment_models import DatasetProvenance, ExperimentDataset, HeldoutRecord
from krugradeai.models import (
    AdaptationRule, AdaptationRules, Question, Rubric, RubricCriterion,
    StudentResponse, TAReference, TACriterionScore,
)
from krugradeai.optimization_models import DevelopmentRecord, ValidationRecord, RecordProvenance

SOURCE = "krugradeai-synthetic-consequential-error-v1"


def optimization_initial_rules() -> AdaptationRules:
    return AdaptationRules(rules=(AdaptationRule(
        id="read-criterion-evidence", criterion_ids=("transformation", "isolation", "answer"),
        instruction="Apply each criterion independently to the student's stated evidence without inventing missing work.",
        rationale="Initial general interpretation guidance for the synthetic integration."),))


def make_optimization_fixture() -> ExperimentDataset:
    question = Question(id="synthetic-consequential-question", text="Solve 3(x - 2) = 12, showing your algebraic reasoning.",
                        reference_solution="Divide by 3: x - 2 = 4. Add 2 to obtain x = 6. Alternatively expand to 3x - 6 = 12, obtain 3x = 18, then divide by 3.")
    rubric = Rubric(id="synthetic-consequential-rubric", total_score="3", criteria=(
        RubricCriterion(id="transformation", max_score="1", description="Award 1 for correctly expanding the original equation or correctly dividing both sides to obtain an equivalent equation. Award 0.5 if a correct transformation is identified but not performed. Otherwise award 0. Accept mathematically equivalent methods."),
        RubricCriterion(id="isolation", max_score="1", description="Award 1 for correctly performing a stated operation that leaves x alone, judged against the student's immediately preceding equation. Credit valid follow-through even if an earlier arithmetic error made that preceding equation incorrect for the original problem. Award 0.5 if a correct operation to leave x alone is identified but not performed. Otherwise award 0. Merely reaching an equation such as 3x = a number does not leave x alone. Correctness of the final value for the original problem is assessed only by the answer criterion."),
        RubricCriterion(id="answer", max_score="1", description="Award 1 only if the student explicitly states x = 6, the correct value for the original equation. Otherwise award 0. An incorrect consequential value or an unstated value receives no credit here; do not infer a missing final answer."),
    ))
    cases = (
        (DevelopmentRecord, "synthetic-development-consequential", "3x - 6 = 12. Then 3x = 15. Dividing both sides by 3 gives x = 5.", ("1", "1", "0")),
        (ValidationRecord, "synthetic-validation-alternative", "Divide both sides by 3: x - 2 = 4. Add 2 to both sides: x = 6.", ("1", "1", "1")),
        (HeldoutRecord, "synthetic-test-intermediate", "3x - 6 = 12, so 3x = 18. I would divide by 3 next.", ("1", "0.5", "0")),
    )
    records = []
    for record_type, id, answer, scores in cases:
        records.append(record_type(id=id, question=question, rubric=rubric,
            response=StudentResponse(id=id + "-response", question_id=question.id, text=answer),
            ta_output=TAReference(total_score=sum((Decimal(s) for s in scores), Decimal(0)), criterion_scores=tuple(
                TACriterionScore(criterion_id=c.id, score=score) for c, score in zip(rubric.criteria, scores))),
            provenance=RecordProvenance(source=SOURCE, item_id=id)))
    return ExperimentDataset(provenance=DatasetProvenance(kind="synthetic", sources=(SOURCE,),
                             description="Synthetic rubric-derived labels: development follow-through 2/3; alternative validation solution 3/3; incomplete held-out work 1.5/3. No model disagreement is presumed."),
                             development=(records[0],), validation=(records[1],), heldout=(records[2],))
