"""Freeze an existing real selection, then explicitly run one final test phase."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Literal

from pydantic import Field
from krugradeai.artifact_paths import output_directory
from krugradeai.codex_provider import CodexCLIProvider
from krugradeai.experiment import ExperimentAlreadyRunError
from krugradeai.experiment_models import (DatasetProvenance, ExperimentConfig, GuidelinesArtifact,
                                         HeldoutPathResult, RubricScope)
from krugradeai.grader import LLMGrader
from krugradeai.metrics import EvaluationMetrics
from krugradeai.models import AdaptationRule, AdaptationRules, Contract, Rubric, Text
from krugradeai.optimization_models import RecordHistory, RuleVersion, ValidationConfig
from krugradeai.providers import LLMProvider
from krugradeai.real_data import heldout_metrics_by_origin, load_prepared_partition
from krugradeai.validation import evaluate_labeled_records

DEFAULT_REPORT = "artifacts/real-optimization/real-optimization-full-001/report.json"
DEFAULT_FROZEN = "artifacts/real-final-evaluation/frozen-guidelines-full-001.json"


def sha(content):
    return hashlib.sha256(content).hexdigest()


def code_hashes():
    root = Path(__file__).parent
    return {name: sha((root/name).read_bytes()) for name in (
        "models.py", "grader.py", "prompts.py", "codex_provider.py", "providers.py", "execution.py",
        "validation.py", "metrics.py", "final_evaluation.py")}


class FrozenPath(Contract):
    rubric: Rubric
    baseline_rules: AdaptationRules
    selected_version: int = Field(ge=0, strict=True)
    selected_rules: AdaptationRules
    baseline_validation: EvaluationMetrics
    selected_validation: EvaluationMetrics
    selected_validation_comparison: EvaluationMetrics


class FrozenGuidelines(Contract):
    schema_version: Literal["1"] = "1"
    source_report: Text
    source_report_sha256: Text
    data_directory: Text
    manifest_sha256: Text
    heldout_sha256: Text
    configured_model: Text | None
    optimization_effective_model: Text | None = None
    model_evidence: Text
    model_verification: dict = Field(default_factory=dict)
    grading_code_sha256: dict[str, str]
    paths: tuple[FrozenPath, ...]
    validation_config: ValidationConfig = ValidationConfig()


def check_groups(manifest):
    """Inspect partition metadata only, never answer/label partitions."""
    seen, identities = {}, set()
    for row in manifest["lineage"]:
        partition = row["partition"]
        if partition not in ("development", "validation", "test"):
            continue
        if row["id"] in identities:
            raise ValueError("Duplicate prepared identity")
        identities.add(row["id"])
        tokens = [("connected", row["connected_group_id"]), ("group", row["group_id"])]
        for field in ("source_attempt_ids", "response_ids", "lineage"):
            tokens.extend((field, row["source"], value) for value in row.get(field, ()))
        for token in tokens:
            if token in seen and seen[token] != partition:
                raise ValueError("Prepared attempt/response lineage overlaps partitions")
            seen[token] = partition


def freeze_selection(report_path=DEFAULT_REPORT, data_directory="artifacts/real-mixed-v1", *, model_verification=None):
    """Validate selection from report/manifest only. Never loads held-out records."""
    source = Path(report_path).resolve()
    report_bytes = source.read_bytes()
    report = json.loads(report_bytes)
    root = Path(data_directory).resolve()
    manifest_bytes = (root/"manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if report.get("execution_error") is not None or report.get("held_out_accessed") is not False:
        raise ValueError("Selection requires a completed experiment without prior held-out access")
    if report.get("mode") != "full_development" or report.get("validation_label_origin") != "human_derived":
        raise ValueError("Selection must come from full Development and human-only Validation")
    hashes = {Path(k).name: v for k, v in report["input_sha256"].items()}
    if hashes.get("manifest.json") != sha(manifest_bytes):
        raise ValueError("Optimization and prepared split manifest differ")
    if hashes.get("validation.jsonl") != manifest["partition_sha256"]["validation"]:
        raise ValueError("Optimization Validation snapshot differs")
    if report["development_sha256"] != manifest["partition_sha256"]["development"]:
        raise ValueError("Optimization Development snapshot differs")
    check_groups(manifest)
    if any(r["partition"] == "validation" and r["label_origin"] != "human_derived" for r in manifest["lineage"]):
        raise ValueError("Frozen selection requires human-only Validation provenance")
    dev_ids = {r["id"] for r in manifest["lineage"] if r["partition"] == "development"}
    selected_ids = [r["record_id"] for r in report["selected_records"]]
    if len(selected_ids) != len(set(selected_ids)) or set(selected_ids) != dev_ids:
        raise ValueError("Optimization did not cover the full declared Development population")
    configured_model = report.get("model")
    verification = model_verification or dict(status="unknown", effective_model=None,
        verified_provider_calls=0, total_provider_calls=len(report.get("provider_calls", ())),
        reason="Source report does not retain call-attributed effective-model evidence")
    if verification.get("source_report_sha256", sha(report_bytes)) != sha(report_bytes):
        raise ValueError("Model-verification evidence belongs to another optimization")
    if verification.get("status") != "unknown" or verification.get("effective_model") is not None:
        raise ValueError("This runner cannot assert verified historical models without per-call evidence")
    rubrics = {r["id"]: Rubric.model_validate(r) for r in manifest["rubric_definitions"].values()}
    report_paths = {p["rubric_id"]: p for p in report["paths"]}
    if len(report_paths) != len(report["paths"]) or set(report_paths) != set(rubrics):
        raise ValueError("Selection must cover every immutable rubric exactly once")
    paths = []
    for rubric_id, path in sorted(report_paths.items()):
        rubric = rubrics[rubric_id]
        versions = tuple(RuleVersion.model_validate(v) for v in path["guideline_versions"])
        if ([v.version for v in versions] != list(range(len(versions))) or
                any(v.parent_version != (i-1 if i else None) for i,v in enumerate(versions))):
            raise ValueError("Guideline version chain is invalid")
        expected_ids = {r["id"] for r in manifest["lineage"] if r["partition"] == "validation"
                        and r["question_id"] in {q for q, rb in manifest["rubric_definitions"].items() if rb["id"] == rubric_id}}
        if not expected_ids:
            raise ValueError("Missing human Validation population for frozen scope")
        evaluations, fixed_ids = [], None
        for index, evaluation in enumerate(path["validation"]):
            metrics = EvaluationMetrics.model_validate(evaluation["comparison_metrics"])
            ids = tuple(evaluation["comparison_record_ids"])
            if (evaluation["version"] != index or len(ids) != len(set(ids)) or set(ids) != expected_ids
                    or (fixed_ids is not None and ids != fixed_ids)
                    or metrics.expected_count != len(ids)):
                raise ValueError("Validation population/version changed during selection")
            fixed_ids = ids
            complete = metrics.sample_count == metrics.expected_count and metrics.coverage == 1
            if type(evaluation["eligible"]) is not bool or evaluation["eligible"] != complete:
                raise ValueError("Invalid Validation selection coverage")
            evaluations.append((evaluation, metrics))
        if len(evaluations) != len(versions):
            raise ValueError("Every guideline version must have a Validation evaluation")
        eligible = [(e, m) for e, m in evaluations if e["eligible"] and m.mae is not None]
        if not eligible:
            raise ValueError("No valid best guidelines; final evaluation refused")
        # Existing MAE + earliest eligible version policy; never recompute labels.
        best, metric = min(eligible, key=lambda pair: (pair[1].mae, pair[0]["version"]))
        version = best["version"]
        rules = AdaptationRules.model_validate(path["selected_rules"])
        if type(path["selected_version"]) is not int or path["selected_version"] != version or rules != versions[version].rules:
            raise ValueError("Recorded selected rules/version contradict Validation selection")
        for candidate in (versions[0].rules, rules):
            if any(isinstance(r, AdaptationRule) and not set(r.criterion_ids) <= {c.id for c in rubric.criteria}
                   for r in candidate.rules):
                raise ValueError("Frozen rules reference an unknown rubric criterion")
        paths.append(FrozenPath(rubric=rubric, baseline_rules=versions[0].rules,
            selected_version=version, selected_rules=rules,
            baseline_validation=EvaluationMetrics.model_validate(evaluations[0][0]["metrics"]),
            selected_validation=EvaluationMetrics.model_validate(best["metrics"]),
            selected_validation_comparison=metric))
    return FrozenGuidelines(source_report=str(source), source_report_sha256=sha(report_bytes),
        data_directory=str(root), manifest_sha256=sha(manifest_bytes),
        heldout_sha256=manifest["partition_sha256"]["test"], configured_model=configured_model,
        optimization_effective_model=None, model_verification=verification,
        model_evidence="unknown_effective_optimization_model; explicit_evaluation_model_required",
        grading_code_sha256=code_hashes(), paths=tuple(paths))


def prepare(frozen_path=DEFAULT_FROZEN, **kwargs):
    snapshot = freeze_selection(**kwargs)
    path = Path(frozen_path).resolve()
    if path.exists():
        raise FileExistsError("Frozen guidelines already exist; nothing overwritten")
    output_directory(path.parent)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(snapshot.model_dump_json(indent=2)+"\n")
    return snapshot


def plan(snapshot):
    manifest = json.loads((Path(snapshot.data_directory)/"manifest.json").read_bytes())
    rows = [r for r in manifest["lineage"] if r["partition"] == "test"]
    scopes = {p.rubric.id: p for p in snapshot.paths}
    rubrics = manifest["rubric_definitions"]
    extra = sum(scopes[rubrics[r["question_id"]]["id"]].selected_version != 0 for r in rows)
    return dict(heldout_records=len(rows), label_origins=dict(Counter(r["label_origin"] for r in rows)),
        baseline_calls=len(rows), additional_selected_calls=extra, maximum_calls=len(rows)+extra,
        optimization_effective_model=snapshot.optimization_effective_model, explicit_evaluation_model_required=True,
        selected_versions={p.rubric.id: p.selected_version for p in snapshot.paths},
        heldout_contents_loaded=False, resume_supported=False)


class _FinalProvider(LLMProvider):
    def __init__(self, provider, maximum, journal):
        self.provider, self.maximum, self.journal = provider, maximum, journal
        self.calls = []

    def generate(self, prompt):
        if len(self.calls) >= self.maximum:
            raise RuntimeError("Final evaluation provider budget exhausted")
        started = time.perf_counter()
        item = dict(index=len(self.calls)+1, status="failure", error_type=None)
        self.calls.append(item)
        try:
            result = self.provider.generate(prompt)
            item["status"] = "success"
            return result
        except (Exception, KeyboardInterrupt) as error:
            item["error_type"] = type(error).__name__
            raise
        finally:
            item["seconds"] = time.perf_counter()-started
            with self.journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item)+"\n")


@dataclass(frozen=True)
class _CompletedGrades:
    records: tuple[RecordHistory, ...]


def evaluate_final(*, live=False, frozen_path=DEFAULT_FROZEN, output_root="artifacts/real-final-evaluation",
                   run_id="heldout-full-001", max_calls=29, timeout=120.0, model=None):
    if live is not True:
        raise ValueError("Final evaluation requires explicit --live")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Invalid final evaluation run ID")
    frozen_file = Path(frozen_path).resolve()
    frozen_bytes = frozen_file.read_bytes()
    snapshot = FrozenGuidelines.model_validate_json(frozen_bytes)
    expected = freeze_selection(snapshot.source_report, snapshot.data_directory, model_verification=snapshot.model_verification)
    if expected != snapshot:
        raise ValueError("Frozen rules, inputs or grading code changed; final evaluation refused")
    details = plan(snapshot)
    if model is None or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*", model):
        raise ValueError("Final evaluation requires an explicit --model; historical effective model is unknown")
    if snapshot.configured_model is not None and model != snapshot.configured_model:
        raise ValueError("Final evaluation cannot change a recorded optimization model setting")
    if type(max_calls) is not int or not details["maximum_calls"] <= max_calls <= 2*details["heldout_records"]:
        raise ValueError("Final evaluation budget does not reserve the fixed comparison population")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid final evaluation timeout")
    destination = Path(output_root).resolve()/run_id
    data = Path(snapshot.data_directory)
    if data == destination or data in destination.parents or destination in data.parents or destination == frozen_file.parent:
        raise ValueError("Final outputs must be separate from prepared data and frozen artifact directory")
    if destination.exists():
        raise FileExistsError("Final run exists; nothing executed or overwritten")
    # Fixed ledger location, independent of run ID and output directory.
    ledger = output_directory(data.parent/"final-heldout-ledger")
    receipt = ledger/(sha((snapshot.source_report_sha256+snapshot.heldout_sha256).encode())+".json")
    claim = dict(state="claimed", source_report_sha256=snapshot.source_report_sha256,
                 frozen_sha256=sha(frozen_bytes), heldout_sha256=snapshot.heldout_sha256,
                 evaluation_requested_model=model, optimization_effective_model=None)
    try:
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(claim, stream)
    except FileExistsError:
        raise ExperimentAlreadyRunError("This experiment's final evaluation attempt is already consumed") from None
    output_directory(destination)
    result, provider = None, None
    try:
        records = load_prepared_partition(data, "test")
        manifest = json.loads((data/"manifest.json").read_bytes())
        declared = {r["id"]: r for r in manifest["lineage"] if r["partition"] == "test"}
        scopes = {p.rubric.id: p for p in snapshot.paths}
        if len(records) != len(declared) or {r.id for r in records} != set(declared):
            raise ValueError("Held-out identities differ from frozen split")
        for record in records:
            if record.rubric.id not in scopes or record.rubric != scopes[record.rubric.id].rubric:
                raise ValueError("Held-out rubric differs from frozen immutable rubric")
            if record.provenance.label_origin != declared[record.id]["label_origin"] or record.question.id != declared[record.id]["question_id"]:
                raise ValueError("Held-out scope/provenance differs from manifest")
        provider = _FinalProvider(CodexCLIProvider(model=model, timeout_seconds=timeout),
                                  max_calls, destination/"provider_calls.jsonl")
        grader = LLMGrader(provider)
        paths, baseline_rows, selected_rows, summaries = [], [], [], []
        for selected in snapshot.paths:
            cohort = tuple(sorted((r for r in records if r.rubric.id == selected.rubric.id), key=lambda r: r.id))
            baseline = evaluate_labeled_records(grader, cohort, selected.baseline_rules, 0, snapshot.validation_config)
            final = baseline if selected.selected_version == 0 else evaluate_labeled_records(
                grader, cohort, selected.selected_rules, selected.selected_version, snapshot.validation_config)
            path = HeldoutPathResult(rubric_id=selected.rubric.id, selected_version=selected.selected_version,
                status="completed" if baseline.eligible and final.eligible else "incomplete", baseline=baseline, selected=final)
            paths.append(path)
            baseline_rows.extend(baseline.records); selected_rows.extend(final.records)
            summaries.append(dict(rubric_id=selected.rubric.id, selected_version=selected.selected_version,
                selected_rules=selected.selected_rules.model_dump(mode="json"), status=path.status,
                baseline={k:v.model_dump(mode="json") for k,v in heldout_metrics_by_origin(baseline, cohort).items()},
                selected={k:v.model_dump(mode="json") for k,v in heldout_metrics_by_origin(final, cohort).items()},
                failures=[f.model_dump(mode="json") for f in (*baseline.failures, *(() if final is baseline else final.failures))],
                human_review_count=len(baseline.human_reviews)+(0 if final is baseline else len(final.human_reviews))))
            artifact = GuidelinesArtifact(experiment_id=run_id, guideline_version=selected.selected_version,
                rubric_scope=RubricScope(rubric_id=selected.rubric.id, sha256=sha(selected.rubric.model_dump_json().encode())),
                rules=selected.selected_rules, validation_metrics=selected.selected_validation,
                validation_comparison_metrics=selected.selected_validation_comparison,
                heldout_baseline_metrics=baseline.metrics, heldout_selected_metrics=final.metrics, heldout_status=path.status,
                configuration=ExperimentConfig(max_iterations=1, validation=snapshot.validation_config,
                    initial_rules=selected.baseline_rules, runtime="codex_cli"),
                provenance=DatasetProvenance(kind="real", description="Mixed-source held-out labels; AI labels are not human ground truth",
                    sources=tuple(sorted({r.provenance.source for r in records}))))
            with (destination/(sha(selected.rubric.id.encode())[:12]+f"_g{selected.selected_version}.json")).open("x", encoding="utf-8") as stream:
                stream.write(artifact.model_dump_json(indent=2)+"\n")
        if frozen_file.read_bytes() != frozen_bytes or freeze_selection(snapshot.source_report, snapshot.data_directory, model_verification=snapshot.model_verification) != snapshot:
            raise ValueError("Protected selection or grading configuration changed during final evaluation")
        result = dict(status="completed" if all(p.status == "completed" for p in paths) else "incomplete",
            frozen_sha256=sha(frozen_bytes), source_report_sha256=snapshot.source_report_sha256,
            evaluation_model=model, evaluation_requested_model=model, optimization_effective_model=None,
            optimization_model_comparability="unknown", model_evidence=snapshot.model_evidence, plan=details,
            paths=summaries, provider_calls=provider.calls,
            baseline={k:v.model_dump(mode="json") for k,v in heldout_metrics_by_origin(_CompletedGrades(tuple(baseline_rows)), records).items()},
            selected={k:v.model_dump(mode="json") for k,v in heldout_metrics_by_origin(_CompletedGrades(tuple(selected_rows)), records).items()},
            selection_changed=False, interpretation="Mixed-source and AI-only metrics are operational agreement, not human grading accuracy")
        with (destination/"report.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
    except (Exception, KeyboardInterrupt) as error:
        claim.update(state="failed-consumed", error_type=type(error).__name__)
        receipt.write_text(json.dumps(claim), encoding="utf-8")
        with (destination/"failure.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(error_type=type(error).__name__, retry_allowed=False), stream)
        raise
    receipt.write_text(json.dumps({**claim, "state":result["status"]}), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--optimization-report", default=DEFAULT_REPORT)
    parser.add_argument("--data-directory", default="artifacts/real-mixed-v1")
    parser.add_argument("--frozen", default=DEFAULT_FROZEN)
    parser.add_argument("--model", help="Required for live evaluation; pinned for both baseline and selected passes")
    parser.add_argument("--model-verification", help="Optional safe historical audit JSON, used only when preparing")
    parser.add_argument("--run-id", default="heldout-full-001")
    parser.add_argument("--max-calls", type=int, default=29)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            evidence = json.loads(Path(args.model_verification).read_bytes()) if args.model_verification else None
            snapshot = prepare(args.frozen, report_path=args.optimization_report,
                               data_directory=args.data_directory, model_verification=evidence)
            print(json.dumps(plan(snapshot), indent=2))
        elif args.dry_run:
            if Path(args.frozen).exists():
                snapshot = FrozenGuidelines.model_validate_json(Path(args.frozen).read_bytes())
                if freeze_selection(snapshot.source_report, snapshot.data_directory,
                        model_verification=snapshot.model_verification) != snapshot:
                    raise ValueError("Frozen snapshot differs from original selection or code")
            else:
                snapshot = freeze_selection(args.optimization_report, args.data_directory)
            print(json.dumps(plan(snapshot), indent=2))
        else:
            result = evaluate_final(live=True, frozen_path=args.frozen, run_id=args.run_id,
                                    max_calls=args.max_calls, timeout=args.timeout, model=args.model)
            print(json.dumps(dict(status=result["status"], provider_calls=len(result["provider_calls"]))))
    except Exception as error:
        parser.exit(1, f"Final evaluation refused or incomplete ({type(error).__name__}); no automatic retry; private data withheld.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
