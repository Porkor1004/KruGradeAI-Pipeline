"""Opt-in Development discrepancy discovery, with optional guarded continuation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

from krugradeai.agents import Grader
from krugradeai.artifact_paths import output_directory
from krugradeai.development_criteria import load_enriched_development
from krugradeai.execution import _invoke
from krugradeai.grader import HumanReviewRequired
from krugradeai.live_integration import build_agents
from krugradeai.models import AdaptationRules, GraderInput, GraderOutput, GradingRecord
from krugradeai.optimization import OptimizationLoop
from krugradeai.optimization_models import DevelopmentRecord
from krugradeai.pilot_real import _Telemetry, _TimedProvider, source_evidence_availability

SEED = "krugradeai-discrepancy-discovery-v1"
PILOT005 = "artifacts/real-pilot/real-development-pilot-005-enriched/report.json"


def score_bucket(record):
    """Coarse source-label strata, not a prediction of errors or difficulty."""
    value, maximum = record.ta_output.total_score, record.rubric.total_score
    if value == 0:
        return "zero"
    if value == maximum:
        return "full"
    ratio = value / maximum
    return "low_partial" if ratio <= 0.25 else "mid_partial" if ratio <= 0.75 else "high_partial"


def select_discovery_sample(records, *, excluded_ids=(), sample_size=8, seed=SEED):
    if type(sample_size) is not int or not 1 <= sample_size <= 24:
        raise ValueError("Discovery sample size must be between 1 and 24")
    rows = tuple(DevelopmentRecord.model_validate(r) for r in records)
    if len({r.id for r in rows}) != len(rows):
        raise ValueError("Duplicate Development identities")
    if any(r.provenance is None or r.provenance.label_origin is None for r in rows):
        raise ValueError("Explicit Development label provenance required")
    excluded = set(excluded_ids)
    pool = [r for r in rows if r.id not in excluded and source_evidence_availability(r)["eligible_for_enriched_sampling"]]
    if len(pool) < sample_size:
        raise ValueError("Insufficient untested source-criterion records; no repeat or total-only fallback")
    chosen = []
    questions, origins, buckets, combinations = Counter(), Counter(), Counter(), Counter()
    rank = lambda value: hashlib.sha256((seed + "\0" + value).encode()).hexdigest()
    while len(chosen) < sample_size:
        record = min(pool, key=lambda r: (
            questions[r.rubric.id], origins[r.provenance.label_origin], buckets[score_bucket(r)],
            combinations[(r.rubric.id, r.provenance.label_origin, score_bucket(r))],
            not source_evidence_availability(r)["complete_criterion_coverage"], rank(r.id)))
        chosen.append(record)
        pool.remove(record)
        questions[record.rubric.id] += 1
        origins[record.provenance.label_origin] += 1
        buckets[score_bucket(record)] += 1
        combinations[(record.rubric.id, record.provenance.label_origin, score_bucket(record))] += 1
    return tuple(chosen)


def history_exclusions(paths, *, development_sha256, development_ids):
    """Read only safe report metadata; no historical grading text is needed."""
    excluded, inspected = set(), []
    for path in sorted({Path(p).resolve() for p in paths}):
        if path.suffix.lower() != ".json":
            raise ValueError("Exclusions must be JSON reports, not dataset partitions")
        content = path.read_bytes()
        report = json.loads(content)
        if report.get("development_sha256") != development_sha256:
            raise ValueError("Exclusion report belongs to a different Development snapshot")
        selected = report.get("selected_records")
        if not isinstance(selected, list) or any(not isinstance(r, dict) or not isinstance(r.get("record_id"), str) for r in selected):
            raise ValueError("Malformed exclusion report identities")
        ids = {r["record_id"] for r in selected}
        # Failed/ambiguous attempts remain excluded: no repeated calls to fish for
        # a particular model result. Never promote report IDs into dataset rows.
        excluded.update(ids & set(development_ids))
        inspected.append(dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(),
                              matching_development_ids=len(ids & set(development_ids))))
    return excluded, inspected


class DiscoveryCallLimitError(RuntimeError):
    pass


class _DiscoveryProvider(_TimedProvider):
    def generate(self, request):
        if len(self.telemetry.invocations) >= self.telemetry.budget:
            raise DiscoveryCallLimitError("Discovery provider-call limit reached; no retry")
        return super().generate(request)


class _ReplayGrader(Grader):
    """Transport cache for the existing Grader role; never invents a new grade."""
    def __init__(self, cache):
        self.cache = cache

    def grade(self, request):
        key = request.model_dump_json()
        if key not in self.cache:
            raise ValueError("Continuation request differs from the observed grading input")
        result = self.cache[key]
        if isinstance(result, Exception):
            raise result
        return result


def observed_difference(record, grading):
    ai = {c.criterion_id: c.score for c in grading.grader_output.criterion_scores}
    total_difference = grading.grader_output.total_score - record.ta_output.total_score
    differences = [dict(criterion_id=c.criterion_id, difference=str(ai[c.criterion_id] - c.score),
                        reference_provenance=c.provenance.model_dump(mode="json"))
                   for c in record.ta_output.criterion_scores if ai[c.criterion_id] != c.score]
    return dict(record_id=record.id, question_id=record.question.id, rubric_id=record.rubric.id,
                label_origin=record.provenance.label_origin,
                operational_role="pseudo_ground_truth" if record.provenance.label_origin == "ai_derived" else "human_reference",
                status="graded", has_discrepancy=total_difference != 0 or bool(differences),
                total_score_difference=str(total_difference), criterion_differences=differences,
                evidence_availability=source_evidence_availability(record),
                interpretation="Operational-reference disagreement, not proof of actual grading error")


def run_discovery(*, live=False, data_directory="artifacts/real-mixed-v1",
                  source_criteria_directory="artifacts/development-criteria-v1",
                  output_directory_path="artifacts/real-discovery", run_id="development-discovery-001",
                  sample_size=8, max_calls=None, continue_discrepancies=False, seed=SEED,
                  previous_reports=None, pilot_history_directory="artifacts/real-pilot",
                  model=None, timeout=120.0, executable=None, clock=None, additional_reports=()):
    if live is not True:
        raise ValueError("Development discovery requires explicit --live opt-in")
    if type(continue_discrepancies) is not bool:
        raise ValueError("Continuation must be explicitly enabled with a Boolean flag")
    if type(sample_size) is not int or not 1 <= sample_size <= 24:
        raise ValueError("Discovery sample size must be between 1 and 24")
    max_calls = (3 * sample_size if continue_discrepancies else sample_size) if max_calls is None else max_calls
    if type(max_calls) is not int or not sample_size <= max_calls <= 72:
        raise ValueError("Call limit must cover the sample and be at most 72")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Use a simple unique discovery run ID")
    output = Path(output_directory_path).resolve()
    destination = output / run_id
    for source in (Path(data_directory).resolve(), Path(source_criteria_directory).resolve()):
        if destination == source or source in destination.parents or destination in source.parents:
            raise ValueError("Discovery output must be separate from all input artifacts")
    if destination.exists():
        raise FileExistsError("Discovery run already exists; nothing executed or overwritten")
    output_directory(output)
    destination.mkdir(exist_ok=False)
    clock = clock or time.perf_counter
    started = clock()
    telemetry = _Telemetry(clock, max_calls)
    records, sample, observations, continuation, exclusions = (), (), [], [], []
    base_hash, criteria_hash, setup_error = None, None, None
    try:
        records = load_enriched_development(data_directory, source_criteria_directory, enable_source_criteria=True)
        base_hash = hashlib.sha256((Path(data_directory) / "development.jsonl").read_bytes()).hexdigest()
        criteria_hash = hashlib.sha256((Path(source_criteria_directory) / "development.jsonl").read_bytes()).hexdigest()
        if previous_reports is None:
            # Pilot 005 must be present. Other same-snapshot real pilot reports
            # are also excluded, without reading any partition's content.
            paths = {Path(PILOT005)}
            for path in Path(pilot_history_directory).glob("*/report.json"):
                if json.loads(path.read_text(encoding="utf-8")).get("development_sha256") == base_hash:
                    paths.add(path)
        else:
            paths = {Path(p) for p in previous_reports}
        paths.update(Path(p) for p in additional_reports)
        # Repeated discovery commands automatically avoid earlier discovery IDs.
        for path in output.glob("*/report.json"):
            if path.parent != destination and json.loads(path.read_text(encoding="utf-8")).get("development_sha256") == base_hash:
                paths.add(path)
        excluded, exclusions = history_exclusions(paths, development_sha256=base_hash, development_ids={r.id for r in records})
        sample = select_discovery_sample(records, excluded_ids=excluded, sample_size=sample_size, seed=seed)
        snapshots = {r.id: r.model_dump_json() for r in records}
        telemetry.expected_references = {r.response.id: (r.ta_output, r.rubric) for r in sample}
        agents = build_agents(model=model, timeout=timeout, executable=executable)
        for role, agent, method_name in zip(("grader", "reflector", "refiner"), agents, ("grade", "reflect", "refine")):
            agent.provider = _DiscoveryProvider(agent.provider, role, telemetry)
            setattr(agent, method_name, telemetry.wrap_agent(getattr(agent, method_name), role))
        cache = {}
        for record in sample:
            request = GraderInput(question=record.question, rubric=record.rubric, response=record.response,
                                  adaptation_rules=AdaptationRules())
            key = request.model_dump_json()
            try:
                grade = GradingRecord(grader_input=request,
                    grader_output=GraderOutput.model_validate(_invoke(agents[0].grade, request)))
                cache[key] = grade.grader_output
                observations.append(observed_difference(record, grade))
            except Exception as error:
                cache[key] = error
                observations.append(dict(record_id=record.id, question_id=record.question.id,
                    label_origin=record.provenance.label_origin, status="human_review" if isinstance(error, HumanReviewRequired) else "failure",
                    has_discrepancy=None, total_score_difference=None, criterion_differences=[], error_type=type(error).__name__,
                    evidence_availability=source_evidence_availability(record)))
        if continue_discrepancies and any(o["has_discrepancy"] for o in observations):
            # Reuse the validated observations; do not regrade to seek a different
            # answer, drop failed/review records, or change any safety guard.
            result = OptimizationLoop(grader=_ReplayGrader(cache), reflector=agents[1], refiner=agents[2], max_iterations=1).run(
                sample, AdaptationRules())
            for path in result.paths:
                h = path.history[0]
                continuation.append(dict(rubric_id=path.rubric_id, stop_reason=path.stop_reason,
                    candidate_proposed=h.candidate_version is not None, candidate_version=h.candidate_version,
                    candidate_validated=False, candidate_promoted=False,
                    failures=[dict(stage=f.stage, record_id=f.record_id, error_type=f.error_type) for f in h.failures],
                    human_reviews=[dict(stage=r.stage, record_id=r.record_id) for r in h.human_reviews]))
        if any(r.model_dump_json() != snapshots[r.id] for r in records):
            raise ValueError("Development input integrity violation")
    except Exception as error:
        setup_error = type(error).__name__
    limit_hit = any(o.error_type == "DiscoveryCallLimitError" for o in telemetry.outcomes)
    status = "failed" if setup_error else "incomplete" if (
        any(o["status"] != "graded" for o in observations) or
        any(p["failures"] or p["human_reviews"] for p in continuation)) else "completed"
    report = dict(schema_version=1, data_scope="development_only", run_id=run_id, status=status,
        seed=seed, development_sha256=base_hash, source_criteria_sha256=criteria_hash, model=model,
        timeout_seconds=timeout, sample_limit=sample_size, max_provider_calls=max_calls,
        provider_calls=sum(i.provider_invoked for i in telemetry.invocations), call_limit_reached=limit_hit,
        continue_discrepancies=continue_discrepancies, cached_grades_reused=bool(continuation), max_refinement_iterations=1,
        total_seconds=max(0, clock() - started), setup_error_type=setup_error, exclusion_reports=exclusions,
        selected_records=[dict(record_id=r.id, question_id=r.question.id, rubric_id=r.rubric.id,
            provenance=r.provenance.model_dump(mode="json"), source_score_bucket=score_bucket(r),
            evidence_availability=source_evidence_availability(r),
            input_sha256=hashlib.sha256(r.model_dump_json().encode()).hexdigest()) for r in sample],
        coverage=dict(selected=len(sample), graded=sum(o["status"] == "graded" for o in observations),
            observed_discrepancies=sum(o["has_discrepancy"] is True for o in observations),
            rubric_paths=len({r.rubric.id for r in sample}),
            label_origins=dict(Counter(r.provenance.label_origin for r in sample)),
            source_score_buckets=dict(Counter(score_bucket(r) for r in sample))),
        observations=observations, discrepancies=[o for o in observations if o["has_discrepancy"]],
        invocations=[i.model_dump(mode="json") for i in telemetry.invocations],
        agent_outcomes=[o.model_dump(mode="json") for o in telemetry.outcomes], continuation=continuation,
        candidates_promoted=False,
        notes=["Development only; no Validation/Held-out inputs or raw files were opened.",
               "AI-derived criterion labels are authorized operational pseudo-ground truth, not human verification.",
               "Observed disagreements do not independently establish actual grading error.",
               "Selection diversifies source-label strata; it does not modify prompts, labels, answers or rubrics.",
               "No repeats/total-only fallback on exhaustion; prior selected records remain excluded even after failed calls.",
               "Continuation is opt-in within this run, using cached validated grades and existing review guards.",
               "Pilot candidates are never validated, exported as guidelines or promoted."])
    with (destination / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Opt-in Development-only discrepancy discovery")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--continue-discrepancies", action="store_true")
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--data-dir", default="artifacts/real-mixed-v1")
    parser.add_argument("--source-criteria-dir", default="artifacts/development-criteria-v1")
    parser.add_argument("--output-dir", default="artifacts/real-discovery")
    parser.add_argument("--run-id", default="development-discovery-001")
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--exclude-report", action="append", default=[])
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--executable")
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("Explicit --live is required; nothing loaded or executed")
    try:
        report = run_discovery(live=True, data_directory=args.data_dir, source_criteria_directory=args.source_criteria_dir,
            output_directory_path=args.output_dir, run_id=args.run_id, sample_size=args.sample_size, max_calls=args.max_calls,
            continue_discrepancies=args.continue_discrepancies, seed=args.seed, additional_reports=args.exclude_report,
            model=args.model, timeout=args.timeout, executable=args.executable)
    except Exception as error:
        print(f"Discovery not started ({type(error).__name__}); no raw diagnostic data printed.", file=sys.stderr)
        return 1
    print(f"Discovery {report['status']}: {report['coverage']['graded']} graded, {report['coverage']['observed_discrepancies']} disagreements, {report['provider_calls']} provider calls.")
    print(f"Report: {Path(args.output_dir).resolve() / args.run_id / 'report.json'}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
