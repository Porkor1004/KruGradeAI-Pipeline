"""Decimal-safe total-score metrics; missing grades are never score pairs."""
from collections.abc import Iterable
from decimal import Context, Decimal, localcontext
from typing import Annotated, Self

from pydantic import Field, model_validator
from krugradeai.models import Contract, PositiveScore, Score, ScoreDifference, Text


class ScorePair(Contract):
    predicted: Score
    reference: Score
    maximum: PositiveScore

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.predicted > self.maximum or self.reference > self.maximum:
            raise ValueError("Scores must not exceed the rubric maximum")
        return self


class EvaluationMetrics(Contract):
    sample_count: Annotated[int, Field(ge=0, strict=True)]
    expected_count: Annotated[int, Field(ge=0, strict=True)]
    coverage: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
    mae: Score | None
    normalized_mae: Score | None
    exact_match: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)] | None
    qwk: ScoreDifference | None
    qwk_undefined_reason: Text | None


def validate_categories(categories: Iterable[Decimal] | None) -> tuple[Decimal, ...] | None:
    if categories is None:
        return None
    try:
        values = tuple(Decimal(str(value)) for value in categories)
    except Exception:
        raise ValueError("QWK categories must be finite ordered decimal scores") from None
    if len(values) < 2 or any(not v.is_finite() or v < 0 for v in values):
        raise ValueError("QWK needs at least two finite nonnegative categories")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("QWK categories must be strictly increasing")
    return values


def _qwk(samples: tuple[ScorePair, ...], categories: tuple[Decimal, ...] | None) -> tuple[Decimal | None, str | None]:
    if not samples:
        return None, "No successful score pairs"
    if len({s.maximum for s in samples}) != 1:
        return None, "Different rubric maxima do not define a shared ordinal scale"
    if categories is None:
        return None, "Explicit ordinal score categories were not configured"
    index = {score: i for i, score in enumerate(categories)}
    if any(s.predicted not in index or s.reference not in index for s in samples):
        return None, "A score is outside the configured categories; scores were not rounded"
    observed = Decimal(0)
    predicted = [0] * len(categories)
    reference = [0] * len(categories)
    for sample in samples:
        i, j = index[sample.predicted], index[sample.reference]
        predicted[i] += 1
        reference[j] += 1
        observed += (i - j) ** 2
    expected = sum((Decimal((i - j) ** 2 * a * b) for i, a in enumerate(predicted)
                    for j, b in enumerate(reference)), Decimal(0)) / Decimal(len(samples))
    if expected == 0:
        return None, "Expected weighted disagreement is zero (degenerate labels)"
    return Decimal(1) - observed / expected, None


def calculate_metrics(samples: Iterable[ScorePair], *, expected_count: int,
                      qwk_categories: Iterable[Decimal] | None = None) -> EvaluationMetrics:
    pairs = tuple(ScorePair.model_validate(sample) for sample in samples)
    if type(expected_count) is not int or expected_count < len(pairs) or expected_count < 0:
        raise ValueError("expected_count must be a nonnegative integer at least as large as sample_count")
    categories = validate_categories(qwk_categories)
    with localcontext(Context(prec=50)):
        size = len(pairs)
        qwk, reason = _qwk(pairs, categories)
        return EvaluationMetrics(
            sample_count=size, expected_count=expected_count,
            coverage=Decimal(size) / expected_count if expected_count else Decimal(0),
            mae=sum((abs(p.predicted - p.reference) for p in pairs), Decimal(0)) / size if size else None,
            normalized_mae=sum((abs(p.predicted - p.reference) / p.maximum for p in pairs), Decimal(0)) / size if size else None,
            exact_match=Decimal(sum(p.predicted == p.reference for p in pairs)) / size if size else None,
            qwk=qwk, qwk_undefined_reason=reason,
        )
