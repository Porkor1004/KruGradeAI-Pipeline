"""Explicit opt-in live integration, preserving the established optimizer."""
import argparse
import json
from functools import wraps
from pathlib import Path
import re
import sys

from krugradeai.artifact_paths import output_directory
from krugradeai.codex_provider import CodexCLIProvider, _unique_object, _reject_constant
from krugradeai.execution import _failure, _invoke
from krugradeai.experiment import Phase1Experiment
from krugradeai.experiment_models import ExperimentDataset
from krugradeai.grader import HumanReviewRequired, LLMGrader
from krugradeai.integration_models import AgentInvocation, CandidateMetrics, IntegrationReport, CandidateProgress, StageExecution
from krugradeai.live_optimization_fixture import make_optimization_fixture, optimization_initial_rules
from krugradeai.models import AdaptationRule, AdaptationRules, ReflectorInput, ReflectorOutput, RefinerInput
from krugradeai.optimization import OptimizationLoop
from krugradeai.optimization_models import HumanReviewFlag, ValidationConfig
from krugradeai.refiner import LLMRefiner
from krugradeai.reflector import LLMReflector
from krugradeai.synthetic import make_synthetic_dataset

_REQUIRED = ("development_grader", "reflector", "refiner", "validation_grader", "heldout_grader")
_SAFE_STAGES = {"transport", "invocation", "schema", "configuration", "authentication", "usage_limit", "output_read", "output_json", "output_validation", "process"}


def make_live_fixture() -> ExperimentDataset:
    source = make_synthetic_dataset()
    return ExperimentDataset(provenance=source.provenance, development=source.development[1:2],
                             validation=source.validation[:1], heldout=source.heldout[:1])


def initial_rules() -> AdaptationRules:
    return AdaptationRules(rules=(AdaptationRule(
        id="initial-evidence", instruction="Read intermediate reasoning before applying the immutable rubric.",
        criterion_ids=("method", "result"), rationale="Initial synthetic integration guidance."),))


def build_agents(*, model: str | None, timeout: float, executable: str | None):
    options = dict(model=model, timeout_seconds=timeout, executable=executable)
    return (LLMGrader(CodexCLIProvider(**options)),
            LLMReflector(CodexCLIProvider(**options, output_model=ReflectorOutput)),
            LLMRefiner(CodexCLIProvider(**options, output_model=AdaptationRules)))


class _Recorder:
    def __init__(self, rules):
        self.rules = rules
        self.events = []
        self.probe = False
        self.proposed = False
        self.no_change = False

    def note_proposal(self, rules):
        if not self.probe:
            self.proposed = rules != self.rules
            self.no_change = rules == self.rules

    def observe_proposal_output(self, provider):
        """Observe typed proposals before agent safety validation; never repair.

        Retain only booleans, not response text or generated instructions. The
        unchanged raw output still passes through all existing agent validators.
        """
        generate = provider.generate
        @wraps(generate)
        def call(prompt):
            raw = generate(prompt)
            if not self.probe:
                try:
                    data = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
                    self.note_proposal(AdaptationRules.model_validate(data))
                except (ValueError, TypeError):
                    pass  # Unparseable output is not a typed candidate proposal.
            return raw
        provider.generate = call

    def wrap(self, method, role):
        @wraps(method)
        def call(request):
            record_id = None
            if self.probe:
                phase = "development_probe"
            elif role == "grader":
                record_id = request.response.id
                if "synthetic-validation-" in record_id:
                    phase = "validation_baseline" if request.adaptation_rules == self.rules else "validation_candidate"
                elif "synthetic-test-" in record_id:
                    phase = "heldout_baseline" if request.adaptation_rules == self.rules else "heldout_selected"
                else:
                    phase = "development"
            else:
                phase = "development"
            status, error_type, error_stage, review = "success", None, None, False
            try:
                output = method(request)
                review = bool(getattr(output, "human_review_required", False))
                if role == "refiner" and isinstance(output, AdaptationRules):
                    self.note_proposal(output)
                return output
            except HumanReviewRequired:
                status, error_type, review = "human_review", "HumanReviewRequired", True
                raise
            except Exception as error:
                status, error_type = "failure", type(error).__name__
                stage = getattr(error, "stage", None)
                error_stage = stage if stage in _SAFE_STAGES else None
                raise
            finally:
                self.events.append(AgentInvocation(index=len(self.events) + 1, agent=role, phase=phase,
                                                   status=status, record_id=record_id, error_type=error_type,
                                                   error_stage=error_stage, human_review_required=review,
                                                   execution_kind="diagnostic_probe" if self.probe else "actual_optimization"))
        return call


class _IntegrationBoundary:
    """Delegate once, then probe skipped contracts on development data only.

    This is instrumentation, not another optimizer/agent. Probe outputs are never
    adopted, selected or propagated to subsequent grading.
    """
    def __init__(self, optimizer, reflector, refiner, recorder):
        self.optimizer, self.reflector, self.refiner, self.recorder = optimizer, reflector, refiner, recorder
        self.max_iterations, self.validation_config = optimizer.max_iterations, optimizer.validation_config
        self.failures, self.reviews = [], []

    def run(self, records, initial, *, validation_records):
        result = self.optimizer.run(records, initial, validation_records=validation_records)
        outcomes = [r for p in result.paths for h in p.history for r in h.records]
        outcome = next((r for r in outcomes if r.grading is not None), None)
        if outcome is None:
            return result  # No invented grading record and no retries.
        record = next(r for r in records if r.id == outcome.record_id)
        reflection = outcome.reflection
        self.recorder.probe = True
        try:
            if not any(e.agent == "reflector" for e in self.recorder.events):
                try:
                    reflection = _invoke(self.reflector.reflect, ReflectorInput(record=outcome.grading, ta_output=record.ta_output,
                        reference_total_score_origin=(record.provenance.score_origin if record.provenance else None) or "unknown"))
                    if reflection.human_review_required:
                        self.reviews.append(HumanReviewFlag(stage="reflector", record_id=record.id, reasons=("Development probe requires human review.",)))
                except HumanReviewRequired:
                    self.reviews.append(HumanReviewFlag(stage="reflector", record_id=record.id, reasons=("Development probe requires human review.",)))
                except Exception as error:
                    self.failures.append(_failure("reflector", record.id, error))
            if reflection is not None and not any(e.agent == "refiner" for e in self.recorder.events):
                try:
                    _invoke(self.refiner.refine, RefinerInput(current_rules=outcome.grading.grader_input.adaptation_rules,
                                                             reflection=reflection, rubric=record.rubric))
                except HumanReviewRequired:
                    self.reviews.append(HumanReviewFlag(stage="refiner", reasons=("Development probe requires human review.",)))
                except Exception as error:
                    self.failures.append(_failure("refiner", None, error))
        finally:
            self.recorder.probe = False
        return result


def _missing(events):
    covered = set()
    for event in events:
        if event.status != "success":
            continue
        if event.agent != "grader":
            covered.add(event.agent)
        elif event.phase == "development":
            covered.add("development_grader")
        elif event.phase.startswith("validation_"):
            covered.add("validation_grader")
        elif event.phase.startswith("heldout_"):
            covered.add("heldout_grader")
    return tuple(stage for stage in _REQUIRED if stage not in covered)


def _stage_execution(events):
    stages = []
    for stage in (*_REQUIRED, "candidate_validation"):
        relevant = [e for e in events if
                    (stage in ("reflector", "refiner") and e.agent == stage) or
                    (e.agent == "grader" and
                     ((stage == "development_grader" and e.phase == "development") or
                      (stage == "validation_grader" and e.phase.startswith("validation_")) or
                      (stage == "heldout_grader" and e.phase.startswith("heldout_")) or
                      (stage == "candidate_validation" and e.phase == "validation_candidate")))]
        kind = "actual_optimization" if any(e.execution_kind == "actual_optimization" for e in relevant) else "diagnostic_probe" if relevant else "skipped"
        status = "failure" if any(e.status == "failure" for e in relevant) else "human_review" if any(e.human_review_required for e in relevant) else "success" if relevant else "skipped"
        stages.append(StageExecution(stage=stage, execution_kind=kind, status=status))
    return tuple(stages)


def run_integration(*, live: bool = False, output_dir: str | Path = "artifacts/live-synthetic",
                    experiment_id: str | None = None, model: str | None = None,
                    timeout: float = 60.0, executable: str | None = None,
                    optimization_path: bool = False) -> IntegrationReport:
    if live is not True:
        raise ValueError("Live integration requires explicit --live opt-in")
    if experiment_id is None:
        experiment_id = "live-synthetic-optimization-path" if optimization_path else "live-synthetic-integration"
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", experiment_id):
        raise ValueError("Use a safe experiment identifier")
    root = output_directory(output_dir)
    report_path = root / (experiment_id + ".integration.json")
    # Reserve before constructing providers. Never replace a previous report.
    with report_path.open("x", encoding="utf-8") as stream:
        starting_rules = optimization_initial_rules() if optimization_path else initial_rules()
        recorder = _Recorder(starting_rules)
        result, boundary, setup_type, setup_stage = None, None, None, None
        try:
            dataset = make_optimization_fixture() if optimization_path else make_live_fixture()
            grader, reflector, refiner = build_agents(model=model, timeout=timeout, executable=executable)
            if optimization_path and hasattr(refiner, "provider"):
                recorder.observe_proposal_output(refiner.provider)
            grader.grade = recorder.wrap(grader.grade, "grader")
            reflector.reflect = recorder.wrap(reflector.reflect, "reflector")
            refiner.refine = recorder.wrap(refiner.refine, "refiner")
            optimizer = OptimizationLoop(grader=grader, reflector=reflector, refiner=refiner, max_iterations=1,
                                         validation_config=ValidationConfig(patience=2, qwk_categories=("0", "0.5", "1", "1.5", "2", "2.5", "3") if optimization_path else ("0", "0.5", "1", "1.5", "2")))
            if not optimization_path:
                boundary = _IntegrationBoundary(optimizer, reflector, refiner, recorder)
            result = Phase1Experiment(optimizer=optimizer if optimization_path else boundary, heldout_grader=grader, ledger_directory=root / ".experiments").run(
                experiment_id, dataset, initial_rules=starting_rules, runtime="codex_cli")
        except Exception as error:
            setup_type = type(error).__name__
            stage = getattr(error, "stage", None)
            setup_stage = stage if stage in _SAFE_STAGES else None
        failures, reviews = list(boundary.failures if boundary else ()), list(boundary.reviews if boundary else ())
        baseline, candidates, selected, stop, heldout_baseline, heldout_selected, heldout_status = None, (), None, None, None, None, None
        accepted, evaluated, eligible, candidate_selected = False, False, None, False
        if result is not None:
            path = result.optimization.paths[0]
            selection, final = path.validation, result.heldout[0]
            baseline, selected, stop = selection.baseline_metrics, selection.best_version, path.stop_reason
            candidates = tuple(CandidateMetrics(version=e.version, metrics=e.metrics, eligible=e.eligible) for e in selection.evaluations[1:])
            accepted = any(h.candidate_version is not None for h in path.history)
            evaluated = bool(candidates)
            eligible = candidates[0].eligible if candidates else None
            candidate_selected = bool(candidates and selected == candidates[0].version)
            heldout_baseline = final.baseline.metrics if final.baseline else None
            heldout_selected = final.selected.metrics if final.selected else None
            heldout_status = final.status
            for history in path.history:
                failures.extend(history.failures); reviews.extend(history.human_reviews)
            evaluations = [*selection.evaluations]
            if final.baseline is not None:
                evaluations.append(final.baseline)
            if final.selected is not None and final.selected.version != 0:
                evaluations.append(final.selected)
            for evaluation in evaluations:
                failures.extend(evaluation.failures); reviews.extend(evaluation.human_reviews)
        # Human-review metadata is retained without copying model-generated free
        # text into a report that may be shared publicly.
        reviews = tuple(HumanReviewFlag(stage=r.stage, record_id=r.record_id, reasons=("Human review requested; no uncertain score was invented.",)) for r in reviews)
        missing = _missing(recorder.events)
        healthy = result is not None and result.status == "completed" and (optimization_path or not missing) and not failures and not reviews and not any(e.human_review_required for e in recorder.events)
        attempted = any(e.agent == "refiner" and e.execution_kind == "actual_optimization" for e in recorder.events)
        progress = CandidateProgress(proposal_attempted=attempted, proposed=recorder.proposed or accepted,
                                     accepted_as_valid=accepted, evaluated_on_validation=evaluated,
                                     validation_eligible=eligible, selected_as_best=candidate_selected,
                                     no_change=recorder.no_change and attempted)
        report = IntegrationReport(experiment_id=experiment_id, status="failed" if setup_type else "passed" if healthy else "incomplete",
                    invocations=tuple(recorder.events), missing_stages=missing, baseline_validation_metrics=baseline,
                    candidate_validation_metrics=candidates, selected_version=selected, optimization_stop_reason=stop,
                    heldout_baseline_metrics=heldout_baseline, heldout_selected_metrics=heldout_selected,
                    heldout_status=heldout_status, failures=tuple(failures), human_reviews=reviews,
                    setup_error_type=setup_type, setup_error_stage=setup_stage,
                    mode="optimization_path" if optimization_path else "contract_coverage",
                    stage_execution=_stage_execution(recorder.events), candidate_progress=progress,
                    notes=("Synthetic integration only; metrics do not measure real grading accuracy.",
                           "Actual optimization only: no diagnostic probes or forced discrepancies; skipped refinement is a legitimate outcome." if optimization_path else "Skipped Reflector/Refiner contracts are probed on genuine development data only; probe proposals are never adopted.",
                           "An empty candidate metrics list means no new candidate version was created; no validation metrics were fabricated."))
        stream.write(report.model_dump_json(indent=2) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Opt-in Codex integration on one synthetic record per split.")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--optimization-path", action="store_true", help="Use the consequential-error fixture and actual optimization only, without diagnostic probes.")
    parser.add_argument("--model")
    parser.add_argument("--executable")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--experiment-id")
    parser.add_argument("--output-dir", default="artifacts/live-synthetic")
    args = parser.parse_args(argv)
    if not args.live:
        print("No live calls made. Pass --live explicitly to run the synthetic integration.", file=sys.stderr)
        return 2
    try:
        report = run_integration(live=True, output_dir=args.output_dir, experiment_id=args.experiment_id,
                                 model=args.model, timeout=args.timeout, executable=args.executable,
                                 optimization_path=args.optimization_path)
    except Exception as error:
        print(f"Integration not started ({type(error).__name__}); existing reports/receipts are not overwritten.", file=sys.stderr)
        return 1
    print(f"Synthetic integration: {report.status}; report: {Path(args.output_dir) / (report.experiment_id + '.integration.json')}")
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
