"""Explicitly opt-in, development-only real-data latency pilot.

Runs the existing optimizer once per sampled rubric path, without validation,
held-out access, guideline export or adoption of pilot candidates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time

from pydantic import Field

from krugradeai.artifact_paths import output_directory
from krugradeai.grader import HumanReviewRequired
from krugradeai.live_integration import build_agents
from krugradeai.models import AdaptationRules, Contract, GraderInput
from krugradeai.optimization import OptimizationLoop
from krugradeai.optimization_models import DevelopmentRecord
from krugradeai.providers import LLMProvider
from krugradeai.real_data import load_prepared_partition
from krugradeai.refinement_diagnostics import evidence_summary
from krugradeai.development_criteria import load_enriched_development

PILOT_SEED = "krugradeai-real-pilot-v1"


class InvocationTiming(Contract):
    index: int
    agent: str
    record_id: str | None
    rubric_id: str | None
    seconds: float = Field(ge=0)
    transport_status: str
    error_type: str | None = None
    validation_diagnostics: tuple[dict, ...] = ()
    evidence_propagation: dict | None = None
    provider_invoked: bool = True


class AgentOutcome(Contract):
    agent: str
    record_id: str | None
    rubric_id: str | None
    status: str
    seconds: float = Field(ge=0)
    error_type: str | None = None
    validation_diagnostics: tuple[dict, ...] = ()
    reflection_evidence: dict | None = None
    evidence_propagation: dict | None = None


class PilotReport(Contract):
    schema_version: str = "1"
    run_id: str
    status: str
    development_sha256: str | None
    seed: str
    model: str | None
    timeout_seconds: float
    sample_size: int
    full_development_size: int
    max_iterations: int = 1
    max_provider_invocations: int
    total_seconds: float = Field(ge=0)
    selected_records: tuple[dict, ...]
    invocations: tuple[InvocationTiming, ...]
    agent_outcomes: tuple[AgentOutcome, ...]
    paths: tuple[dict, ...]
    runtime_estimate: dict
    setup_error_type: str | None = None
    candidates_promoted: bool = False
    source_criteria_enabled: bool = False
    source_criteria_sha256: str | None = None
    criterion_label_summary: dict = Field(default_factory=dict)
    sampling_mode: str = "default"
    sampling_coverage: dict = Field(default_factory=dict)
    notes: tuple[str, ...]


def select_sample(records, size=6, seed=PILOT_SEED):
    """Pair one record of each label origin within each selected rubric scope.

    Uses only identifiers/provenance, never reference scores or answer content.
    Existing frozen records are returned intact; input ordering is irrelevant.
    """
    if size not in (4, 6, 8):
        raise ValueError("Pilot sample size must be 4, 6 or 8")
    records = tuple(DevelopmentRecord.model_validate(r) for r in records)
    if len({r.id for r in records}) != len(records):
        raise ValueError("Duplicate development identities")
    for record in records:
        GraderInput(question=record.question, rubric=record.rubric, response=record.response)
        if record.provenance is None or record.provenance.label_origin is None:
            raise ValueError("Pilot requires explicit real-data label provenance")
    rank = lambda text: hashlib.sha256((seed + "\0" + text).encode()).hexdigest()
    scopes = sorted({r.rubric.id for r in records}, key=rank)
    sample = []
    for scope in scopes:
        by_origin = {origin: sorted((r for r in records if r.rubric.id == scope and r.provenance.label_origin == origin),
                                    key=lambda r: rank(r.id)) for origin in ("ai_derived", "human_derived")}
        if all(by_origin.values()):
            sample.extend(by_origin[origin][0] for origin in ("ai_derived", "human_derived"))
            if len(sample) == size:
                return tuple(sample)
    raise ValueError("Not enough compatible rubric scopes containing both label origins")


def source_evidence_availability(record):
    """Report source-backed coverage, never a promise of supported findings."""
    scores = record.ta_output.criterion_scores
    known = {c.id for c in record.rubric.criteria}
    source_valid = bool(scores) and all(getattr(c, "provenance", None) is not None
        and c.provenance.original_criterion_id == c.criterion_id
        and c.provenance.label_origin == record.provenance.label_origin
        and c.provenance.source == record.provenance.source for c in scores)
    return dict(reference_criterion_label_count=len(scores), source_provided_criteria_valid=source_valid,
                criterion_ids=[c.criterion_id for c in scores],
                rubric_criterion_count=len(known), complete_criterion_coverage={c.criterion_id for c in scores} == known,
                eligible_for_enriched_sampling=source_valid,
                refinement_eligibility="requires_supported_reflection_and_no_review")


def select_enriched_sample(records, size=6, seed=PILOT_SEED):
    """Select only source-backed records; maximize paired scope/origin coverage.

    Scores/answer content and model outcomes are never ranking features. Prefer
    fully covered records over partial labels within each scope/origin. If paired
    scopes are insufficient, fill deterministically from the enriched pool while
    preferring new paths and then underrepresented origins. Never fill from totals.
    """
    if size not in (4, 6, 8):
        raise ValueError("Pilot sample size must be 4, 6 or 8")
    records = tuple(DevelopmentRecord.model_validate(r) for r in records)
    if len({r.id for r in records}) != len(records):
        raise ValueError("Duplicate development identities")
    if any(r.provenance is None or r.provenance.label_origin is None for r in records):
        raise ValueError("Pilot requires explicit real-data label provenance")
    eligible = [r for r in records if source_evidence_availability(r)["eligible_for_enriched_sampling"]]
    if len(eligible) < size or len({r.rubric.id for r in eligible}) < 2:
        raise ValueError("Insufficient source-criterion records across multiple rubric paths; no total-only fallback")
    rank = lambda value: hashlib.sha256((seed + "\0" + value).encode()).hexdigest()
    scopes = sorted({r.rubric.id for r in eligible}, key=rank)
    sample = []
    def quality(r):
        return not source_evidence_availability(r)["complete_criterion_coverage"], rank(r.id)
    for scope in scopes:
        pair = {origin: sorted((r for r in eligible if r.rubric.id == scope and r.provenance.label_origin == origin), key=quality)
                for origin in ("ai_derived", "human_derived")}
        if all(pair.values()):
            sample.extend(pair[origin][0] for origin in ("ai_derived", "human_derived"))
            if len(sample) == size:
                return tuple(sample)
    while len(sample) < size:
        selected = {r.id for r in sample}
        represented = {r.rubric.id for r in sample}
        origins = Counter(r.provenance.label_origin for r in sample)
        remaining = [r for r in eligible if r.id not in selected]
        choice = min(remaining, key=lambda r: (r.rubric.id in represented, origins[r.provenance.label_origin],
                                               *quality(r), rank(r.rubric.id)))
        sample.append(choice)
    return tuple(sample)


class EvidencePropagationError(RuntimeError):
    """Pilot-only fail-closed check; no reference values appear in the error."""


def _propagation(expected, received, rubric):
    """Compare exact values AND provenance; persist only safe metadata/hashes."""
    expected_data = expected.model_dump(mode="json")
    actual = received if isinstance(received, dict) else {}
    scores = actual.get("criterion_scores", [])
    scores = scores if isinstance(scores, list) else []
    known = {c.id for c in rubric.criteria}
    canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return dict(evidence_available=bool(expected.criterion_scores),
                expected_criterion_count=len(expected.criterion_scores), received_criterion_count=len(scores),
                expected_criterion_ids=[c.criterion_id for c in expected.criterion_scores],
                received_criterion_ids=[c.get("criterion_id") if isinstance(c, dict) and c.get("criterion_id") in known else "unknown_criterion" for c in scores],
                labels_reached_reflector=bool(scores),
                reference_values_and_provenance_match=actual == expected_data,
                expected_reference_sha256=hashlib.sha256(canonical(expected_data)).hexdigest(),
                received_reference_sha256=hashlib.sha256(canonical(actual)).hexdigest())


class _Telemetry:
    def __init__(self, clock, budget):
        self.clock, self.budget = clock, budget
        self.invocations, self.outcomes = [], []
        self.context = {}
        self.expected_references = {}

    def wrap_agent(self, method, role):
        def call(request):
            if role == "grader":
                record_id, rubric_id = request.response.id, request.rubric.id
            elif role == "reflector":
                request_input = request.record.grader_input
                record_id, rubric_id = request_input.response.id, request_input.rubric.id
            else:
                record_id, rubric_id = None, request.rubric.id
            previous = self.context
            self.context = dict(record_id=record_id, rubric_id=rubric_id)
            start, status, error_type, diagnostics = self.clock(), "success", None, ()
            reflection_evidence = None
            propagation = None
            try:
                if role == "reflector" and record_id in self.expected_references:
                    expected, rubric = self.expected_references[record_id]
                    propagation = _propagation(expected, request.ta_output.model_dump(mode="json"), rubric)
                    propagation["boundary"] = "reflector_agent_input"
                    if not propagation["reference_values_and_provenance_match"]:
                        raise EvidencePropagationError("Reflector reference differs from selected Development evidence")
                output = method(request)
                if getattr(output, "human_review_required", False):
                    status = "human_review"
                if role == "reflector":
                    reflection_evidence = evidence_summary(request, output)
                return output
            except HumanReviewRequired as error:
                status, error_type = "human_review", type(error).__name__
                raise
            except Exception as error:
                status, error_type = "failure", type(error).__name__
                diagnostics = getattr(error, "safe_diagnostics", ())
                raise
            finally:
                self.outcomes.append(AgentOutcome(agent=role, record_id=record_id, rubric_id=rubric_id,
                    status=status, seconds=max(0, self.clock() - start), error_type=error_type,
                    validation_diagnostics=diagnostics, reflection_evidence=reflection_evidence,
                    evidence_propagation=propagation))
                self.context = previous
        return call


class _TimedProvider(LLMProvider):
    def __init__(self, provider, role, telemetry):
        self.provider, self.role, self.telemetry = provider, role, telemetry

    def generate(self, request):
        telemetry = self.telemetry
        if len(telemetry.invocations) >= telemetry.budget:
            raise RuntimeError("Pilot provider invocation budget exhausted")
        start, status, error_type, diagnostics = telemetry.clock(), "success", None, ()
        propagation = None
        provider_invoked = False
        try:
            record_id = telemetry.context.get("record_id")
            if self.role == "reflector" and record_id in telemetry.expected_references:
                expected, rubric = telemetry.expected_references[record_id]
                try:
                    payload = json.loads(request.split("\nINPUT_JSON\n", 1)[1])
                    received = payload.get("ta_output") if isinstance(payload, dict) else None
                except (ValueError, IndexError, TypeError):
                    received = None
                propagation = _propagation(expected, received, rubric)
                propagation["boundary"] = "reflector_provider_prompt"
                if not propagation["reference_values_and_provenance_match"]:
                    raise EvidencePropagationError("Reflector prompt omitted or changed Development reference evidence")
            provider_invoked = True
            return self.provider.generate(request)
        except Exception as error:
            status, error_type = "failure", type(error).__name__
            diagnostics = getattr(error, "safe_diagnostics", ())
            raise
        finally:
            telemetry.invocations.append(InvocationTiming(index=len(telemetry.invocations) + 1,
                agent=self.role, **telemetry.context, seconds=max(0, telemetry.clock() - start),
                transport_status=status, error_type=error_type, validation_diagnostics=diagnostics,
                evidence_propagation=propagation, provider_invoked=provider_invoked))


def estimate_runtime(invocations, outcomes, result, full_records):
    """Conditional, development-only extrapolation, not a completion guarantee."""
    means = {role: statistics.mean(e.seconds for e in invocations if e.agent == role and e.transport_status == "success")
             for role in ("grader", "reflector", "refiner")
             if any(e.agent == role and e.transport_status == "success" for e in invocations)}
    answer = dict(status="insufficient_evidence", full_development_size=len(full_records),
                  provider_mean_seconds=means, grader_only_seconds=None,
                  one_development_iteration_seconds=None, observed_discrepancy_rate=None,
                  expected_reflector_calls=None, refiner_call_upper_bound=len({r.rubric.id for r in full_records}),
                  includes_validation_or_heldout=False, includes_multiple_iterations=False)
    if "grader" in means:
        answer["grader_only_seconds"] = len(full_records) * means["grader"]
    if result is None:
        return answer
    history = [h for p in result.paths for h in p.history]
    records = [r for h in history for r in h.records]
    graded = [r for r in records if r.grading is not None]
    if not graded:
        return answer
    rate = sum(r.discrepancy is True for r in graded) / len(graded)
    answer["observed_discrepancy_rate"] = rate
    answer["expected_reflector_calls"] = len(full_records) * rate
    if any(o.status != "success" for o in outcomes) or any(h.failures or h.human_reviews for h in history) or len(graded) != len(records):
        return answer
    if "grader" not in means or (rate and not {"reflector", "refiner"} <= means.keys()):
        return answer
    answer["status"] = "conditional_estimate"
    # At most one Refiner call per full-development rubric path, assuming
    # discrepancies and sufficient evidence there; not inferred as certain.
    answer["one_development_iteration_seconds"] = answer["grader_only_seconds"] + (
        answer["expected_reflector_calls"] * means["reflector"] + answer["refiner_call_upper_bound"] * means["refiner"] if rate else 0)
    return answer


def run_pilot(*, live=False, data_directory="artifacts/real-mixed-v1", output_directory_path="artifacts/real-pilot",
              run_id="real-development-pilot-001", sample_size=6, seed=PILOT_SEED,
              model=None, timeout=120.0, executable=None, clock=None, source_criteria_directory=None,
              criterion_enriched_sampling=False):
    if live is not True:
        raise ValueError("Real-data pilot requires explicit --live opt-in")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Use a simple unique pilot run ID")
    if sample_size not in (4, 6, 8) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid pilot size or timeout")
    if criterion_enriched_sampling and source_criteria_directory is None:
        raise ValueError("Enriched sampling requires explicit --source-criteria-dir")
    source = Path(data_directory).resolve()
    destination = Path(output_directory_path).resolve() / run_id
    # Do not write inside, or replace, any normalized dataset directory.
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Pilot outputs must be separate from the prepared dataset")
    if destination.exists():
        raise FileExistsError("Pilot run ID already exists; use a new run ID")
    output_directory(destination.parent)
    destination.mkdir(exist_ok=False)
    clock = clock or time.perf_counter
    start = clock()
    telemetry = _Telemetry(clock, 2 * sample_size + sample_size // 2)
    records, sample, result, setup_error, fingerprint = (), (), None, None, None
    criteria_fingerprint = None
    try:
        # This loader opens only manifest metadata and development.jsonl.
        records = (load_prepared_partition(source, "development") if source_criteria_directory is None else
                   load_enriched_development(source, source_criteria_directory, enable_source_criteria=True))
        if source_criteria_directory is not None:
            criteria_fingerprint = hashlib.sha256((Path(source_criteria_directory) / "development.jsonl").read_bytes()).hexdigest()
        sample = (select_enriched_sample if criterion_enriched_sampling else select_sample)(records, sample_size, seed)
        telemetry.budget = 2 * sample_size + len({r.rubric.id for r in sample})
        telemetry.expected_references = {r.response.id: (r.ta_output, r.rubric) for r in sample}
        fingerprint = hashlib.sha256((source / "development.jsonl").read_bytes()).hexdigest()
        snapshots = {r.id: r.model_dump_json() for r in records}
        agents = build_agents(model=model, timeout=timeout, executable=executable)
        for role, agent, method_name in zip(("grader", "reflector", "refiner"), agents, ("grade", "reflect", "refine")):
            agent.provider = _TimedProvider(agent.provider, role, telemetry)
            setattr(agent, method_name, telemetry.wrap_agent(getattr(agent, method_name), role))
        result = OptimizationLoop(grader=agents[0], reflector=agents[1], refiner=agents[2], max_iterations=1).run(
            sample, AdaptationRules())
        if any(r.model_dump_json() != snapshots[r.id] for r in records):
            raise ValueError("Pilot input integrity violation")
    except Exception as error:
        setup_error = type(error).__name__
    paths = []
    if result is not None:
        for path in result.paths:
            history = path.history[0]
            paths.append(dict(rubric_id=path.rubric_id, stop_reason=path.stop_reason,
                refiner_returned_rules=history.proposed_candidate is not None,
                candidate_proposed=history.candidate_version is not None,
                candidate_changed=history.candidate_version is not None,
                candidate_version=history.candidate_version, candidate_validated_on_validation=False,
                candidate_selected_as_best=False,
                records=[dict(record_id=r.record_id, graded=r.grading is not None, discrepancy=r.discrepancy,
                              human_review_required=r.human_review_required) for r in history.records],
                failures=[dict(stage=f.stage, record_id=f.record_id, error_type=f.error_type) for f in history.failures],
                human_reviews=[dict(stage=f.stage, record_id=f.record_id, status="human_review") for f in history.human_reviews]))
    status = "failed" if setup_error else "completed"
    if not setup_error and any(p["failures"] or p["human_reviews"] for p in paths):
        status = "incomplete"
    report = PilotReport(run_id=run_id, status=status, development_sha256=fingerprint, seed=seed, model=model,
        timeout_seconds=timeout, sample_size=len(sample), full_development_size=len(records),
        max_provider_invocations=telemetry.budget, total_seconds=max(0, clock() - start),
        selected_records=tuple(dict(record_id=r.id, question_id=r.question.id, rubric_id=r.rubric.id,
            criterion_label_count=len(r.ta_output.criterion_scores),
            criterion_label_origins=dict(Counter(c.provenance.label_origin if c.provenance else "legacy_unspecified" for c in r.ta_output.criterion_scores)),
            evidence_availability=source_evidence_availability(r),
            provenance=r.provenance.model_dump(mode="json"), input_sha256=hashlib.sha256(r.model_dump_json().encode()).hexdigest()) for r in sample),
        invocations=tuple(telemetry.invocations), agent_outcomes=tuple(telemetry.outcomes), paths=tuple(paths),
        runtime_estimate=estimate_runtime(telemetry.invocations, telemetry.outcomes, result, records), setup_error_type=setup_error,
        source_criteria_enabled=source_criteria_directory is not None, source_criteria_sha256=criteria_fingerprint,
        sampling_mode="criterion_enriched" if criterion_enriched_sampling else "default",
        sampling_coverage=dict(selected_records=len(sample), rubric_paths=len({r.rubric.id for r in sample}),
            question_ids=sorted({r.question.id for r in sample}),
            label_origins=dict(Counter(r.provenance.label_origin for r in sample)),
            source_criterion_records=sum(source_evidence_availability(r)["eligible_for_enriched_sampling"] for r in sample)),
        criterion_label_summary=dict(records_with_criteria=sum(bool(r.ta_output.criterion_scores) for r in records),
            total_only_records=sum(not r.ta_output.criterion_scores for r in records),
            scores_by_origin=dict(Counter(c.provenance.label_origin if c.provenance else "legacy_unspecified"
                                        for r in records for c in r.ta_output.criterion_scores))),
        notes=("Development-only pilot; no validation or held-out record contents were read.",
               "ZIP labels are AI-derived supervision; disagreement does not establish grader error or human accuracy.",
               "Pilot candidates are diagnostic only, not validated, selected or exported for a final experiment.",
               "Elapsed time includes loading, setup and agent execution, excluding final report serialization/write.",
               "Provider success means transport returned; agent outcomes separately include JSON/schema/context validation and human review.",
               "AI source criterion labels, when explicitly enabled, are operational pseudo-ground truth, never human-verified. Remaining total-only records and unsupported findings retain all review guards.",
               "Enriched sampling, when enabled, selects by source-evidence availability, never discrepancies. Provider-prompt and agent-input checks verify exact reference values/provenance; genuine review still blocks refinement.",
               "Runtime extrapolation uses a small balanced-origin sample of selected scopes, not the full 80/20 population; unseen scopes, prompt length and aggregated Refiner input can change latency.",
               "No observed discrepancies is legitimate and does not guarantee no full-development discrepancies."))
    with (destination / "report.json").open("x", encoding="utf-8") as stream:
        stream.write(report.model_dump_json(indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Opt-in real development pilot; no validation or held-out access")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--data-dir", default="artifacts/real-mixed-v1")
    parser.add_argument("--output-dir", default="artifacts/real-pilot")
    parser.add_argument("--run-id", default="real-development-pilot-001")
    parser.add_argument("--sample-size", type=int, choices=(4, 6, 8), default=6)
    parser.add_argument("--seed", default=PILOT_SEED)
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--executable")
    parser.add_argument("--source-criteria-dir", help="Explicitly opt in to an enriched Development artifact, including AI pseudo-ground-truth criterion labels")
    parser.add_argument("--criterion-enriched-sampling", action="store_true", help="Select only source-criterion records; prefer multiple paired-origin rubric paths")
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("Explicit --live is required; nothing was loaded or executed")
    try:
        report = run_pilot(live=True, data_directory=args.data_dir, output_directory_path=args.output_dir,
            run_id=args.run_id, sample_size=args.sample_size, seed=args.seed, model=args.model,
            timeout=args.timeout, executable=args.executable, source_criteria_directory=args.source_criteria_dir,
            criterion_enriched_sampling=args.criterion_enriched_sampling)
    except Exception as error:
        print(f"Pilot not started ({type(error).__name__}); inputs and existing outputs unchanged.", file=sys.stderr)
        return 1
    print(f"Pilot {report.status}; {len(report.invocations)} provider invocations; {report.total_seconds:.2f}s.")
    print(f"Report: {Path(args.output_dir).resolve() / report.run_id / 'report.json'}")
    return 0 if report.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
