"""Independent agent interfaces; no implementations or prompts."""
from abc import ABC, abstractmethod
from krugradeai.models import (
    AdaptationRules, GraderInput, GraderOutput, ReflectorInput,
    ReflectorOutput, RefinerInput,
)


class Grader(ABC):
    @abstractmethod
    def grade(self, request: GraderInput) -> GraderOutput:
        """Return criterion scores without access to TA labels."""
        raise NotImplementedError


class Reflector(ABC):
    @abstractmethod
    def reflect(self, request: ReflectorInput) -> ReflectorOutput:
        """Return observations about a validated grading record."""
        raise NotImplementedError


class Refiner(ABC):
    @abstractmethod
    def refine(self, request: RefinerInput) -> AdaptationRules:
        """Return adaptation rules only."""
        raise NotImplementedError
