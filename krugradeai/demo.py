"""Deterministic synthetic demo; live Codex use requires --live explicitly."""
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from krugradeai.experiment import ExperimentAlreadyRunError, Phase1Experiment
from krugradeai.experiment_models import ExperimentDataset
from krugradeai.guidelines_export import export_guidelines
from krugradeai.optimization import OptimizationLoop
from krugradeai.optimization_models import ValidationConfig
from krugradeai.synthetic import build_mock_agents, make_synthetic_dataset


def build_live_agents(model: str | None, timeout: float):
    # Called only after explicit CLI opt-in. No provider invocation occurs merely
    # by importing this module or displaying --help.
    from krugradeai.codex_provider import CodexCLIProvider
    from krugradeai.grader import LLMGrader
    from krugradeai.reflector import LLMReflector
    from krugradeai.refiner import LLMRefiner
    from krugradeai.models import ReflectorOutput, AdaptationRules
    return (LLMGrader(CodexCLIProvider(model=model, timeout_seconds=timeout)),
            LLMReflector(CodexCLIProvider(model=model, timeout_seconds=timeout, output_model=ReflectorOutput)),
            LLMRefiner(CodexCLIProvider(model=model, timeout_seconds=timeout, output_model=AdaptationRules)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 synthetic demonstration. Metrics are scripted toy results, never real grading accuracy.")
    parser.add_argument("--live", action="store_true", help="Explicitly opt in to bounded real Codex calls with synthetic data only.")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--experiment-id")
    parser.add_argument("--output-dir", default="artifacts/synthetic-phase1")
    args = parser.parse_args(argv)
    maximum = args.max_iterations if args.max_iterations is not None else (1 if args.live else 5)
    if maximum < 0 or (args.live and maximum > 1):
        parser.error("Live demo allows at most one optimization iteration; iteration counts must be nonnegative")
    data = make_synthetic_dataset()
    if args.live:
        # One record per split; at most seven generate calls across baseline,
        # optimization, selection and the final baseline/selected comparison.
        data = ExperimentDataset(provenance=data.provenance, development=data.development[:1],
                                 validation=data.validation[:1], heldout=data.heldout[:1])
    experiment_id = args.experiment_id or ("synthetic-live-phase1" if args.live else "synthetic-mock-phase1")
    root = Path(args.output_dir)
    try:
        grader, reflector, refiner = build_live_agents(args.model, args.timeout) if args.live else build_mock_agents()
        optimizer = OptimizationLoop(grader=grader, reflector=reflector, refiner=refiner,
                                     max_iterations=maximum, validation_config=ValidationConfig(
                                         patience=2, qwk_categories=("0", "0.5", "1", "1.5", "2")))
        runner = Phase1Experiment(optimizer=optimizer, heldout_grader=grader, ledger_directory=root / ".experiments")
        result = runner.run(experiment_id, data, runtime="codex_cli" if args.live else "mock")
        exports = export_guidelines(result, root / "guidelines")
    except ExperimentAlreadyRunError:
        print("Experiment ID already consumed. Held-out evaluation was not repeated; use a new ID only for a new declared experiment.", file=sys.stderr)
        return 1
    except Exception as error:
        # Do not print prompts, student content, credentials or raw provider errors.
        print(f"Synthetic demonstration failed ({type(error).__name__}); no result is claimed.", file=sys.stderr)
        return 1
    summary = {
        "experiment_id": result.experiment_id, "data_kind": "synthetic", "runtime": result.config.runtime,
        "warning": "Synthetic software demonstration only; these metrics do not represent real grading accuracy.",
        "status": result.status, "paths": [], "guidelines_exports": [str(p.resolve()) for p in exports],
    }
    final = {h.rubric_id: h for h in result.heldout}
    for path in result.optimization.paths:
        selection, heldout = path.validation, final[path.rubric_id]
        selected = selection.evaluations[selection.best_version] if selection.best_version is not None else None
        summary["paths"].append({
            "rubric_id": path.rubric_id, "stop_reason": path.stop_reason,
            "iterations": len(path.history), "latest_version": path.final_version,
            "selected_version": selection.best_version,
            "validation_baseline": selection.baseline_metrics.model_dump(mode="json"),
            "validation_selected": selected.metrics.model_dump(mode="json") if selected is not None else None,
            "heldout_status": heldout.status,
            "heldout_baseline": heldout.baseline.metrics.model_dump(mode="json") if heldout.baseline is not None else None,
            "heldout_selected": heldout.selected.metrics.model_dump(mode="json") if heldout.selected is not None else None,
        })
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
