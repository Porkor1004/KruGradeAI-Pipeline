"""Explicitly opt-in, bounded real Development optimization with human Validation."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

from krugradeai.artifact_paths import output_directory
from krugradeai.development_criteria import load_enriched_development
from krugradeai.discover_development import history_exclusions, select_discovery_sample
from krugradeai.live_integration import build_agents
from krugradeai.metrics import ScorePair, calculate_metrics
from krugradeai.models import AdaptationRules
from krugradeai.optimization import OptimizationLoop
from krugradeai.pilot_real import _Telemetry, _TimedProvider, source_evidence_availability
from krugradeai.real_data import load_prepared_partition
from krugradeai.adjudicator import (AIAdjudicator, AdjudicatingReflector,
                                   IndependentOutput, ComparisonOutput, POLICY_VERSION)
from krugradeai.codex_provider import CodexCLIProvider
from krugradeai.real_run_plan import estimate_plan

SEED = "krugradeai-first-real-optimization-v1"


class _JournaledProvider(_TimedProvider):
    """Privacy-safe completed-call journal, deliberately NOT a replay cache."""
    def generate(self, request):
        start = len(self.telemetry.invocations)
        try:
            return super().generate(request)
        except KeyboardInterrupt:
            for index in range(start, len(self.telemetry.invocations)):
                self.telemetry.invocations[index] = self.telemetry.invocations[index].model_copy(
                    update=dict(transport_status="failure", error_type="KeyboardInterrupt"))
            raise
        finally:
            with self.telemetry.journal_path.open("a", encoding="utf-8") as stream:
                for call in self.telemetry.invocations[start:]:
                    stream.write(call.model_dump_json()+"\n")


def _timed_provider(provider, role, telemetry):
    cls = _JournaledProvider if getattr(telemetry, "journal_path", None) is not None else _TimedProvider
    return cls(provider, role, telemetry)


def build_adjudicator(*, model, timeout, executable, telemetry):
    """Two separately invoked ephemeral Codex tasks under the same global budget."""
    options = dict(model=model, timeout_seconds=timeout, executable=executable)
    return AIAdjudicator(
        _timed_provider(CodexCLIProvider(**options, output_model=IndependentOutput),
                       "adjudicator_independent", telemetry),
        _timed_provider(CodexCLIProvider(**options, output_model=ComparisonOutput),
                       "adjudicator_comparison", telemetry))


def select_optimization_sample(records, excluded_ids=(), sample_size=4, seed=SEED):
    """Prefer fresh source-backed records; retain at least two rubric paths."""
    if sample_size not in (4, 6, 8):
        raise ValueError("Sample size must be 4, 6 or 8")
    pool = tuple(r for r in records if r.id not in set(excluded_ids)
                 and source_evidence_availability(r)["eligible_for_enriched_sampling"])
    counts = Counter(r.rubric.id for r in pool)
    # Require enough records in each of two paths to permit independent evidence
    # alongside a reviewed record. No selection based on observed model errors.
    eligible = tuple(r for r in pool if counts[r.rubric.id] >= sample_size // 2)
    ranked = select_discovery_sample(eligible, sample_size=2, seed=seed)
    scopes = {r.rubric.id for r in ranked}
    if len(scopes) != 2:
        raise ValueError("Two sufficiently populated fresh rubric paths required")
    return select_discovery_sample(tuple(r for r in eligible if r.rubric.id in scopes),
                                   sample_size=sample_size, seed=seed)


def execute_optimization(development, validation, agents, *, review_policy="stop_path"):
    """Single existing-engine cycle. Only Development enters Reflector/Refiner."""
    scopes = {r.rubric.id for r in development}
    validation = tuple(r for r in validation if r.rubric.id in scopes)
    if any(r.provenance is None or r.provenance.label_origin != "human_derived" for r in validation):
        raise ValueError("Human-only Validation required")
    return OptimizationLoop(grader=agents[0], reflector=agents[1], refiner=agents[2],
        max_iterations=1, review_policy=review_policy).run(
            development, AdaptationRules(), validation_records=validation)


def summarize(result, development):
    rows = {r.id: r for r in development}
    origin_metrics = {}
    outcomes = [r for p in result.paths for h in p.history for r in h.records]
    for origin in ("ai_derived", "human_derived"):
        expected = [r for r in development if r.provenance.label_origin == origin]
        pairs = [ScorePair(predicted=r.grading.grader_output.total_score,
                           reference=rows[r.record_id].ta_output.total_score,
                           maximum=rows[r.record_id].rubric.total_score)
                 for r in outcomes if r.grading is not None and rows[r.record_id].provenance.label_origin == origin]
        origin_metrics[origin] = calculate_metrics(pairs, expected_count=len(expected)).model_dump(mode="json")
    paths = []
    for path in result.paths:
        h = path.history[0]
        val = path.validation
        path_metrics = {}
        for origin in ("ai_derived", "human_derived"):
            cohort = [r for r in development if r.rubric.id == path.rubric_id and r.provenance.label_origin == origin]
            pairs = [ScorePair(predicted=r.grading.grader_output.total_score,
                reference=rows[r.record_id].ta_output.total_score, maximum=rows[r.record_id].rubric.total_score)
                for r in h.records if r.grading is not None and rows[r.record_id].provenance.label_origin == origin]
            path_metrics[origin] = calculate_metrics(pairs, expected_count=len(cohort)).model_dump(mode="json")
        paths.append(dict(rubric_id=path.rubric_id, stop_reason=path.stop_reason,
            g0_development_metrics_by_label_origin=path_metrics,
            g1_development_metrics=None, g1_development_metrics_reason="Candidate evaluation uses human Validation only",
            candidate_proposed=h.proposed_candidate is not None,
            candidate_accepted=h.candidate_version is not None,
            candidate_version=h.candidate_version,
            candidate_evaluated=any(e.version == 1 for e in val.evaluations),
            candidate_selected=val.best_version == 1,
            selected_version=val.best_version,
            selected_rules=val.best_rules.model_dump(mode="json") if val.best_rules is not None else None,
            guideline_versions=[v.model_dump(mode="json") for v in path.rule_versions],
            g0_validation_mae=str(val.evaluations[0].comparison_metrics.mae) if val.evaluations[0].comparison_metrics.mae is not None else None,
            g1_validation_mae=next((str(e.comparison_metrics.mae) if e.comparison_metrics.mae is not None else None
                                    for e in val.evaluations if e.version == 1), None),
            validation=[dict(version=e.version, eligible=e.eligible,
                metrics=e.metrics.model_dump(mode="json"),
                comparison_metrics=e.comparison_metrics.model_dump(mode="json"),
                comparison_record_ids=list(e.comparison_record_ids),
                failures=[f.model_dump(mode="json") for f in e.failures],
                human_review_count=len(e.human_reviews)) for e in val.evaluations],
            development=[dict(record_id=r.record_id, discrepancy=r.discrepancy,
                graded=r.grading is not None, human_review=r.human_review_required) for r in h.records],
            excluded_from_refinement=[r.record_id for r in h.records if r.human_review_required],
            failures=[f.model_dump(mode="json") for f in h.failures],
            human_reviews=[dict(stage=r.stage, record_id=r.record_id, reason_count=len(r.reasons)) for r in h.human_reviews]))
    return dict(paths=paths, development_metrics_by_label_origin=origin_metrics,
                interpretation="AI-derived metrics measure agreement with pseudo-ground truth, not human accuracy")


def run_real_optimization(*, live=False, data_directory="artifacts/real-mixed-v1",
        criteria_directory="artifacts/development-criteria-v1", output_root="artifacts/real-optimization",
        run_id="real-optimization-001", sample_size=4, max_calls=40, timeout=120.0,
        model=None, executable=None, exclude_reviewed_records=False, ai_adjudication=False,
        full_development=False, dry_run=False,
        history_roots=("artifacts/real-pilot", "artifacts/real-discovery")):
    if any(type(value) is not bool for value in (full_development, dry_run)):
        raise ValueError("Full Development and dry-run modes require explicit Boolean flags")
    if dry_run and live:
        raise ValueError("Use either --dry-run or --live")
    if live is not True and dry_run is not True:
        raise ValueError("Real optimization requires explicit --live")
    if type(ai_adjudication) is not bool:
        raise ValueError("AI adjudication requires an explicit Boolean opt-in")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Use a unique simple run ID")
    ceiling = 512 if full_development else 72
    if type(max_calls) is not int or not 1 <= max_calls <= ceiling:
        raise ValueError(f"Provider budget must be 1..{ceiling}")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be positive and finite")
    destination = Path(output_root).resolve() / run_id
    for directory in (data_directory, criteria_directory):
        source = Path(directory).resolve()
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError("Output must be separate from prepared inputs")
    if destination.exists() and not dry_run:
        raise FileExistsError("Run already exists; nothing overwritten")
    development = load_enriched_development(data_directory, criteria_directory, enable_source_criteria=True)
    digest = hashlib.sha256((Path(data_directory) / "development.jsonl").read_bytes()).hexdigest()
    reports = []
    for root in (*history_roots, output_root):
        for path in Path(root).glob("*/report.json"):
            if json.loads(path.read_text(encoding="utf-8")).get("development_sha256") == digest:
                reports.append(path)
    if full_development:
        # Include earlier diagnostic examples: this is a complete experiment, not
        # more discovery sampling. Sorting fixes path and within-path order.
        excluded, exclusions = set(), []
        sample = tuple(sorted(development, key=lambda r: (r.rubric.id, r.id)))
    else:
        excluded, exclusions = history_exclusions(reports, development_sha256=digest,
            development_ids={r.id for r in development})
        sample = select_optimization_sample(development, excluded, sample_size)
    # Only this named partition is loaded; no held-out loader is used.
    validation = tuple(r for r in load_prepared_partition(data_directory, "validation")
                       if r.rubric.id in {r.rubric.id for r in sample})
    # Reserve worst-case baseline + development + reflection + refinement + G1
    # validation calls before any provider invocation. Runtime cap remains active.
    plan = estimate_plan(sample, validation, ai_adjudication=ai_adjudication, timeout=timeout,
        max_calls=max_calls, historical_reports=[json.loads(p.read_text(encoding="utf-8")) for p in reports])
    plan.update(mode="full_development" if full_development else "sample", dry_run=dry_run,
                development_sha256=digest, resume_supported=False)
    upper_bound = plan["maximum_provider_calls"]
    if dry_run:
        return plan
    if max_calls < upper_bound:
        raise ValueError(f"Budget must reserve at least {upper_bound} calls for this fixed population")
    output_directory(destination.parent)
    destination.mkdir(exist_ok=False)
    telemetry = _Telemetry(time.perf_counter, max_calls)
    if full_development:
        telemetry.journal_path = destination / "provider_calls.jsonl"
    telemetry.expected_references = {r.response.id: (r.ta_output, r.rubric) for r in sample}
    policy = "exclude_reviewed_records" if exclude_reviewed_records or full_development else "stop_path"
    input_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(data_directory)/"manifest.json", Path(data_directory)/"validation.jsonl",
        Path(criteria_directory)/"criterion_manifest.json", Path(criteria_directory)/"development.jsonl")}
    if full_development:
        # Establish writable, exclusive outputs before any billable invocation.
        telemetry.journal_path.open("x", encoding="utf-8").close()
        with (destination / "run_plan.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(plan=plan, input_sha256=input_hashes, model=model,
                           timeout_seconds=timeout, review_policy=policy,
                           ai_adjudication_policy=POLICY_VERSION if ai_adjudication else None), stream, indent=2)
    started = time.perf_counter()
    result, execution_error, adjudicator = None, None, None
    try:
        agents = build_agents(model=model, timeout=timeout, executable=executable)
        for role, agent in zip(("grader", "reflector", "refiner"), agents):
            agent.provider = _timed_provider(agent.provider, role, telemetry)
        if ai_adjudication:
            adjudicator = build_adjudicator(model=model, timeout=timeout, executable=executable, telemetry=telemetry)
            agents = (agents[0], AdjudicatingReflector(agents[1], adjudicator, sample), agents[2])
        for role, agent, method in zip(("grader", "reflector", "refiner"), agents, ("grade", "reflect", "refine")):
            setattr(agent, method, telemetry.wrap_agent(getattr(agent, method), role))
        result = execute_optimization(sample, validation, agents, review_policy=policy)
    except (Exception, KeyboardInterrupt) as error:
        execution_error = type(error).__name__
    report = dict(development_sha256=digest, seed=SEED, max_iterations=1, max_calls=max_calls,
        mode=plan["mode"], run_plan=plan, resume_supported=False,
        reserved_call_upper_bound=upper_bound, review_policy=policy,
        ai_adjudication_enabled=ai_adjudication,
        ai_adjudication_policy=POLICY_VERSION if ai_adjudication else None,
        ai_adjudicated_findings=adjudicator.events if adjudicator is not None else [],
        validation_label_origin="human_derived",
        selected_development_origin_counts=dict(Counter(r.provenance.label_origin for r in sample)),
        model=model, timeout_seconds=timeout,
        selected_records=[dict(record_id=r.id, rubric_id=r.rubric.id,
            label_origin=r.provenance.label_origin, criterion_count=len(r.ta_output.criterion_scores)) for r in sample],
        historical_exclusions=exclusions, excluded_record_ids=sorted(excluded),
        input_sha256=input_hashes,
        elapsed_seconds=time.perf_counter()-started,
        provider_calls=[r.model_dump(mode="json") for r in telemetry.invocations],
        agent_outcomes=[r.model_dump(mode="json") for r in telemetry.outcomes],
        held_out_accessed=False, execution_error=execution_error,
        **(summarize(result, sample) if result is not None else dict(paths=[], development_metrics_by_label_origin={})))
    report["status"] = ("execution_failure" if result is None else
        "candidate_evaluated" if any(p["candidate_evaluated"] for p in report["paths"]) else "no_candidate_evaluated")
    with (destination / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print a data-only plan; no providers or output files")
    parser.add_argument("--full-development", action="store_true", help="Use every prepared Development record, including prior pilot records")
    parser.add_argument("--exclude-reviewed-records", action="store_true")
    parser.add_argument("--ai-adjudication", action="store_true")
    parser.add_argument("--run-id", default="real-optimization-001")
    parser.add_argument("--sample-size", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=40, help="Global provider cap; must reserve the plan maximum (ceiling 512 full / 72 sample)")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--model")
    parser.add_argument("--executable")
    args = parser.parse_args(argv)
    try:
        report = run_real_optimization(**vars(args))
    except Exception as error:
        # Never print prompts, answers or provider exception messages.
        parser.exit(1, f"Real optimization could not complete: {type(error).__name__}; check configuration and reserved call budget.\n")
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    print(json.dumps(dict(report=f"artifacts/real-optimization/{args.run_id}/report.json",
                          status=report["status"], provider_calls=len(report["provider_calls"]))))
    if report["execution_error"] is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
