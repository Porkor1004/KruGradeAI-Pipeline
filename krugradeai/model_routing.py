"""Explicit experiment routes shared by optimization and evaluation."""
from krugradeai.models import Contract, Text

LUNA = "gpt-6-luna"
ASTRA = "gpt-6-astra"


class ModelRoutes(Contract):
    grader: Text = LUNA
    reflector: Text = ASTRA
    refiner: Text = ASTRA
    validation_grader: Text = LUNA
    evaluation_grader: Text = LUNA
    heldout_grader: Text = LUNA


class ModelRoutingError(RuntimeError):
    """Abort the experiment when the requested model cannot be verified."""
