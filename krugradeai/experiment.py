"""Single-use Phase 1 experiment: optimize first, then one final test phase."""
import hashlib
import json
import re
from pathlib import Path

from krugradeai.agents import Grader
from krugradeai.artifact_paths import output_directory
from krugradeai.experiment_models import (
    ExperimentConfig, ExperimentDataset, ExperimentResult, HeldoutPathResult, RubricScope,
)
from krugradeai.models import AdaptationRules
from krugradeai.optimization import OptimizationLoop
from krugradeai.optimization_models import OptimizationResult, ValidationConfig
from krugradeai.validation import evaluate_labeled_records


class ExperimentAlreadyRunError(RuntimeError):
    """An experiment ID is consumed before any optimization or test calls."""


class Phase1Experiment:
    def __init__(self, *, optimizer: OptimizationLoop, heldout_grader: Grader,
                 ledger_directory: str | Path):
        self.optimizer, self.heldout_grader = optimizer, heldout_grader
        self.ledger_directory = Path(ledger_directory)

    def run(self, experiment_id: str, dataset: ExperimentDataset, *,
            initial_rules: AdaptationRules | None = None, runtime: str = "custom") -> ExperimentResult:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", experiment_id):
            raise ValueError("Experiment ID must be a safe identifier of at most 64 characters")
        dataset = ExperimentDataset.model_validate(dataset)
        config = ExperimentConfig(max_iterations=self.optimizer.max_iterations,
                                  validation=self.optimizer.validation_config,
                                  initial_rules=initial_rules if initial_rules is not None else AdaptationRules(), runtime=runtime)
        digest = hashlib.sha256((dataset.model_dump_json() + "\n" + config.model_dump_json()).encode("utf-8")).hexdigest()
        directory = output_directory(self.ledger_directory)
        receipt = directory / (experiment_id + ".receipt.json")
        claim = {"experiment_id": experiment_id, "input_sha256": digest, "state": "claimed"}
        try:
            with receipt.open("x", encoding="utf-8") as stream:
                json.dump(claim, stream, sort_keys=True)
        except FileExistsError:
            raise ExperimentAlreadyRunError("Experiment ID is already consumed; held-out evaluation will not be repeated.") from None
        try:
            # This is the only optimizer invocation. Held-out objects are never
            # passed to it, and no optimizer/refiner is called after this point.
            optimization = OptimizationResult.model_validate(self.optimizer.run(
                dataset.development, config.initial_rules, validation_records=dataset.validation))
            selection_snapshot = optimization.model_dump_json()
            final_config = ValidationConfig(primary_metric=config.validation.primary_metric,
                                            patience=config.validation.patience, min_delta=config.validation.min_delta,
                                            qwk_categories=config.validation.qwk_categories)
            results = []
            for path in optimization.paths:
                selection = path.validation
                if selection is None or selection.best_version is None:
                    results.append(HeldoutPathResult(rubric_id=path.rubric_id, selected_version=None,
                                                     status="no_valid_best"))
                    continue
                test_records = tuple(r for r in dataset.heldout if r.rubric.id == path.rubric_id)
                baseline = evaluate_labeled_records(self.heldout_grader, test_records,
                                                    path.rule_versions[0].rules, 0, final_config)
                selected = baseline if selection.best_version == 0 else evaluate_labeled_records(
                    self.heldout_grader, test_records, selection.best_rules, selection.best_version, final_config)
                status = "completed" if baseline.eligible and selected.eligible else "incomplete"
                results.append(HeldoutPathResult(rubric_id=path.rubric_id, selected_version=selection.best_version,
                                                 status=status, baseline=baseline, selected=selected))
            if optimization.model_dump_json() != selection_snapshot:
                raise ValueError("Protected optimization selection changed during held-out evaluation")
            scopes = {}
            for record in dataset.development:
                scopes[record.rubric.id] = RubricScope(rubric_id=record.rubric.id,
                    sha256=hashlib.sha256(record.rubric.model_dump_json().encode("utf-8")).hexdigest())
            result = ExperimentResult(experiment_id=experiment_id, config=config, provenance=dataset.provenance,
                                      rubric_scopes=tuple(scopes.values()), optimization=optimization,
                                      heldout=tuple(results), status="completed" if all(r.status == "completed" for r in results) else "incomplete")
        except Exception as error:
            receipt.write_text(json.dumps({**claim, "state": "failed-consumed", "error_type": type(error).__name__}, sort_keys=True), encoding="utf-8")
            raise
        receipt.write_text(json.dumps({**claim, "state": "completed"}, sort_keys=True), encoding="utf-8")
        return result
