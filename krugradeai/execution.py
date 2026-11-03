"""Shared agent-boundary integrity and sanitized failure recording."""
from collections.abc import Callable
from typing import TypeVar
from krugradeai.models import Contract
from krugradeai.optimization_models import ProcessingFailure

T = TypeVar("T")


class _IntegrityViolation(RuntimeError):
    pass


def _invoke(method: Callable[..., T], request: Contract) -> T:
    before = request.model_dump_json()
    try:
        return method(request)
    finally:
        if request.model_dump_json() != before:
            raise _IntegrityViolation("Agent mutated its frozen input")


def _failure(stage: str, record_id: str | None, error: Exception) -> ProcessingFailure:
    return ProcessingFailure(stage="integrity" if isinstance(error, _IntegrityViolation) else stage,
                             record_id=record_id, error_type=type(error).__name__,
                             message=f"{stage.capitalize()} failed; no output from that call was accepted.")
