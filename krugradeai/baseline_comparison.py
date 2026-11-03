"""Compare a supplied historical prompt with frozen rules; no optimization calls."""
import argparse
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import re
import time

from krugradeai.agents import Grader
from krugradeai.artifact_paths import output_directory
from krugradeai.codex_provider import CodexCLIProvider
from krugradeai.final_evaluation import (FrozenGuidelines, DEFAULT_FROZEN, freeze_selection,
                                        sha, _CompletedGrades)
from krugradeai.grader import LLMGrader, HumanReviewRequired
from krugradeai.metrics import ScorePair, calculate_metrics
from krugradeai.models import AdaptationRules, GraderInput
from krugradeai.optimization_models import ValidationConfig
from krugradeai.prompts import GRADING_PROMPT
from krugradeai.providers import LLMProvider
from krugradeai.real_data import load_prepared_partition, heldout_metrics_by_origin
from krugradeai.validation import evaluate_labeled_records


class MissingOriginalPromptError(ValueError):
    pass


def read_original_prompt(path):
    """No discovery fallback: the caller must identify the actual old prompt."""
    if path is None:
        raise MissingOriginalPromptError("Supply the verified original system-prompt file; none is assumed")
    content = Path(path).read_bytes()
    text = content.decode("utf-8")  # Preserve BOM/newlines/spacing; never execute a prompt file.
    if not text.strip():
        raise MissingOriginalPromptError("Original prompt is empty")
    return text, sha(content)


class _OriginalPromptProvider(LLMProvider):
    """Reuse LLMGrader parsing/bounds; replace only its instruction prefix."""
    def __init__(self, provider, original_prompt):
        self.provider, self.original_prompt = provider, original_prompt

    def generate(self, request):
        prefix = GRADING_PROMPT + "\nINPUT_JSON\n"
        if not request.startswith(prefix):
            raise ValueError("Unexpected Grader serialization")
        payload = request[len(prefix):]
        parsed = GraderInput.model_validate_json(payload)
        if parsed.adaptation_rules != AdaptationRules():
            raise ValueError("Old-prompt baseline must not receive optimized adaptation rules")
        return self.provider.generate(self.original_prompt + "\nINPUT_JSON\n" + payload)


@dataclass
class _Calls:
    maximum: int
    journal: Path | None = None
    calls: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    context: dict = field(default_factory=dict)

    def emit(self, item):
        if self.journal is not None:
            with self.journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item)+"\n")


class _BudgetProvider(LLMProvider):
    def __init__(self, provider, state):
        self.provider, self.state = provider, state

    def generate(self, request):
        if len(self.state.calls) >= self.state.maximum:
            raise RuntimeError("Prompt comparison call budget exhausted; no retry")
        item = dict(kind="provider", index=len(self.state.calls)+1, **self.state.context,
                    status="failure", error_type=None)
        self.state.calls.append(item)
        started = time.perf_counter()
        try:
            response = self.provider.generate(request)
            item["status"] = "success"
            return response
        except (Exception, KeyboardInterrupt) as error:
            item["error_type"] = type(error).__name__
            raise
        finally:
            item["seconds"] = time.perf_counter()-started
            self.state.emit(item)


class _LoggedGrader(Grader):
    """Instrumentation of the existing Grader role, not another grading agent."""
    def __init__(self, grader, arm, state):
        self.grader, self.arm, self.state = grader, arm, state

    def grade(self, request):
        self.state.context = dict(configuration=self.arm, record_id=request.response.id, rubric_id=request.rubric.id)
        start, status, error_type = len(self.state.calls), "failure", None
        try:
            result = self.grader.grade(request)
            status = "scored"
            return result
        except HumanReviewRequired:
            status, error_type = "human_review", "HumanReviewRequired"
            raise
        except (Exception, KeyboardInterrupt) as error:
            error_type = type(error).__name__
            raise
        finally:
            item = dict(kind="outcome", **self.state.context, status=status, error_type=error_type,
                        provider_call_indices=[c["index"] for c in self.state.calls[start:]])
            self.state.outcomes.append(item)
            self.state.emit(item)


def metric_differences(old, optimized):
    """Positive means improvement; coverage and Exact Match increase, errors decrease."""
    result = {}
    with localcontext() as context:
        context.prec = 50
        for metric in ("mae", "normalized_mae", "exact_match", "coverage"):
            a, b = getattr(old, metric), getattr(optimized, metric)
            delta = None if a is None or b is None else a-b if metric in ("mae", "normalized_mae") else b-a
            result[metric] = str(delta) if delta is not None else None
        result["relative_mae_improvement"] = (str((old.mae-optimized.mae)/old.mae)
            if old.mae not in (None, 0) and optimized.mae is not None else None)
    return result


def compare_records(records, snapshot, old_prompt, provider_factory, *, maximum_calls, journal=None):
    """Injected providers, identical fixed records, existing evaluation/metric kernels."""
    records = tuple(records)
    if not isinstance(old_prompt,str) or not old_prompt.strip():
        raise MissingOriginalPromptError("A separate legacy prompt is required; Phase 1 G0 is not a fallback")
    if not records or len({r.id for r in records}) != len(records):
        raise ValueError("Comparison requires unique held-out records")
    if type(maximum_calls) is not int or maximum_calls < 2*len(records):
        raise ValueError("Reserve two grading calls per comparison record")
    if any(r.split != "test" or r.provenance is None or r.provenance.label_origin is None for r in records):
        raise ValueError("Comparison requires held-out records with explicit label provenance")
    if not any(r.provenance.label_origin=="human_derived" for r in records):
        raise ValueError("A human-reference comparison population is required")
    scopes = {p.rubric.id:p for p in snapshot.paths}
    if {r.rubric.id for r in records} != set(scopes) or any(r.rubric != scopes[r.rubric.id].rubric for r in records):
        raise ValueError("Comparison scopes must match the immutable frozen selection")
    protected = [r.model_dump_json() for r in records], snapshot.model_dump_json()
    state = _Calls(maximum_calls, journal)
    # The factory receives no arm-specific settings. Both instances use the same configuration.
    old_provider, optimized_provider = provider_factory(), provider_factory()
    agents = {
        "old_prompt": _LoggedGrader(LLMGrader(_OriginalPromptProvider(_BudgetProvider(old_provider, state), old_prompt)), "old_prompt", state),
        "optimized_guidelines": _LoggedGrader(LLMGrader(_BudgetProvider(optimized_provider, state)), "optimized_guidelines", state),
    }
    outcomes = {key:[] for key in agents}
    failures, reviews = {key:0 for key in agents}, {key:0 for key in agents}
    for rubric_id, selected in sorted(scopes.items()):
        cohort = tuple(sorted((r for r in records if r.rubric.id == rubric_id), key=lambda r:r.id))
        for arm, agent in agents.items():
            rules = AdaptationRules() if arm == "old_prompt" else selected.selected_rules
            evaluation = evaluate_labeled_records(agent, cohort, rules,
                0 if arm == "old_prompt" else selected.selected_version, ValidationConfig())
            outcomes[arm].extend(evaluation.records)
            failures[arm] += len(evaluation.failures); reviews[arm] += len(evaluation.human_reviews)
    if protected != ([r.model_dump_json() for r in records], snapshot.model_dump_json()):
        raise ValueError("Protected records or guidelines changed during comparison")
    metrics = {arm:heldout_metrics_by_origin(_CompletedGrades(tuple(rows)), records) for arm,rows in outcomes.items()}
    scored = {arm:{r.record_id:r for r in rows if r.grading is not None} for arm,rows in outcomes.items()}
    common = set(scored["old_prompt"]) & set(scored["optimized_guidelines"])
    paired = {}
    for origin, cohort in (("human_only",[r for r in records if r.provenance.label_origin=="human_derived"]),
                           ("ai_labeled_only",[r for r in records if r.provenance.label_origin=="ai_derived"]),
                           ("mixed_source",records)):
        paired[origin] = {arm:calculate_metrics([ScorePair(predicted=scored[arm][r.id].grading.grader_output.total_score,
            reference=r.ta_output.total_score, maximum=r.rubric.total_score) for r in cohort if r.id in common],
            expected_count=len(cohort)).model_dump(mode="json") for arm in agents}
    return dict(status="completed" if all(m["mixed_source"].coverage==1 for m in metrics.values()) else "incomplete",
        metrics={arm:{key:value.model_dump(mode="json") for key,value in groups.items()} for arm,groups in metrics.items()},
        improvement={key:metric_differences(metrics["old_prompt"][key],metrics["optimized_guidelines"][key]) for key in metrics["old_prompt"]},
        same_success_population=set(scored["old_prompt"])==set(scored["optimized_guidelines"]),
        paired_success_metrics=paired, failures=failures, human_reviews=reviews,
        provider_calls=state.calls, outcomes=state.outcomes,
        interpretation="Primary human-only comparison; AI/mixed agreement is not human grading accuracy. Positive deltas favor optimized. Incomplete or different success populations require paired-subset interpretation, not an all-record accuracy claim.")


def markdown_report(report):
    def number(value):
        return "N/A" if value is None else f"{Decimal(str(value)):.3f}"
    lines = ["# Old System Prompt vs Frozen Optimized Guidelines", "",
        f"Status: **{report['status']}**. Model setting: `{report['model']}`.", "",
        "The old prompt is supplied by the operator, preserved unchanged and identified by SHA-256. No historical prompt was invented.", ""]
    for cohort,label in (("human_only","Human reference scores (primary)"),("ai_labeled_only","AI-derived labels (supplementary)"),("mixed_source","Mixed-source labels (supplementary)")):
        lines += [f"## {label}", "", "| Metric | Old Prompt Baseline | Optimized Guidelines | Improvement |",
                  "| --- | --- | --- | --- |"]
        for key,name in (("mae","MAE"),("normalized_mae","Normalized MAE"),("exact_match","Exact Match"),("coverage","Coverage")):
            a,b=(report["metrics"][arm][cohort][key] for arm in ("old_prompt","optimized_guidelines"))
            lines.append(f"| {name} | {number(a)} | {number(b)} | {number(report['improvement'][cohort][key])} |")
        counts=[report["metrics"][arm][cohort] for arm in ("old_prompt","optimized_guidelines")]
        lines += [f"| Scored / expected | {counts[0]['sample_count']}/{counts[0]['expected_count']} | {counts[1]['sample_count']}/{counts[1]['expected_count']} | — |", ""]
    lines += ["Positive improvement means lower MAE/normalized MAE or higher Exact Match/coverage. Metrics use successful predictions; review/failure is never a zero score.", "",
        "If coverage differs, full-column differences are not a matched-success comparison. The JSON report includes paired-success metrics on the common records. Missing cases and small samples preclude an all-population accuracy claim.", "",
        "This is a post-hoc comparison on an already evaluated test partition, not a new independent confirmatory test. Frozen rules are never optimized or reselected from these outcomes.", "",
        "Changing both the base prompt and adaptation rules compares complete configurations; it does not isolate the causal effect of adaptation rules alone.", ""]
    return "\n".join(lines)


def run_comparison(*, live=False, old_prompt_path=None, frozen_path=DEFAULT_FROZEN,
                   model=None, timeout=120.0, executable=None, max_calls=48,
                   output_root="artifacts/old-prompt-comparison", run_id="old-prompt-comparison-001"):
    if live is not True:
        raise ValueError("Comparison requires explicit --live; no requests made")
    original_prompt, prompt_hash = read_original_prompt(old_prompt_path)
    if not isinstance(model,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*",model):
        raise ValueError("Use an explicit identical model setting for both configurations")
    if not math.isfinite(timeout) or timeout<=0 or type(max_calls) is not int or not 1<=max_calls<=512:
        raise ValueError("Invalid comparison timeout or budget")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}",run_id):
        raise ValueError("Use a unique safe comparison run ID")
    frozen_file=Path(frozen_path).resolve(); frozen_bytes=frozen_file.read_bytes()
    snapshot=FrozenGuidelines.model_validate_json(frozen_bytes)
    if freeze_selection(snapshot.source_report,snapshot.data_directory,model_verification=snapshot.model_verification)!=snapshot:
        raise ValueError("Frozen selection or original scoring implementation changed")
    data=Path(snapshot.data_directory); manifest=json.loads((data/"manifest.json").read_bytes())
    n=sum(r["partition"]=="test" for r in manifest["lineage"])
    if max_calls<2*n: raise ValueError("Budget must reserve both arms on the same fixed held-out population")
    destination=Path(output_root).resolve()/run_id
    reserved=Path(__file__).resolve().parent.parent/'data'/'raw'
    for source in (data,reserved):
        if destination==source or source in destination.parents or destination in source.parents:
            raise ValueError("Comparison outputs must be separate from all original input artifacts")
    for source_file in (Path(snapshot.source_report),frozen_file,Path(old_prompt_path).resolve()):
        if destination==source_file or destination in source_file.parents:
            raise ValueError("Comparison output cannot contain an existing input file")
    if destination.exists(): raise FileExistsError("Comparison exists; nothing overwritten or rerun")
    settings=dict(model=model,timeout_seconds=float(timeout),executable=executable)
    identity=sha((prompt_hash+sha(frozen_bytes)+snapshot.heldout_sha256+json.dumps(settings,sort_keys=True)).encode())
    ledger=output_directory(data.parent/"old-prompt-comparison-ledger")
    receipt=ledger/(identity+".json")
    with receipt.open("x",encoding="utf-8") as stream: json.dump(dict(state="claimed",identity=identity),stream)
    output_directory(destination)
    try:
        records=load_prepared_partition(data,"test")
        if {r.id for r in records}!={r['id'] for r in manifest['lineage'] if r['partition']=='test'}:
            raise ValueError("Held-out identities differ from frozen split")
        report=compare_records(records,snapshot,original_prompt,lambda:CodexCLIProvider(**settings),
            maximum_calls=max_calls,journal=destination/"events.jsonl")
        if frozen_file.read_bytes()!=frozen_bytes or read_original_prompt(old_prompt_path)[1]!=prompt_hash:
            raise ValueError("Frozen selection or original prompt changed during comparison")
        if freeze_selection(snapshot.source_report,snapshot.data_directory,model_verification=snapshot.model_verification)!=snapshot:
            raise ValueError("Original scoring conditions changed during comparison")
        report.update(model=model,timeout_seconds=timeout,old_prompt_sha256=prompt_hash,
            old_prompt_source=str(Path(old_prompt_path).resolve()),current_grading_prompt_sha256=sha(GRADING_PROMPT.encode()),
            frozen_sha256=sha(frozen_bytes),heldout_sha256=snapshot.heldout_sha256,
            historical_old_prompt_identity="operator_supplied_not_independently_authenticated",
            posthoc_test_reuse=True,selection_changed=False,
            comparison_identity=identity,settings=settings,
            selected_versions={p.rubric.id:p.selected_version for p in snapshot.paths})
        with (destination/"report.json").open("x",encoding="utf-8") as stream: json.dump(report,stream,indent=2)
        with (destination/"comparison.md").open("x",encoding="utf-8") as stream: stream.write(markdown_report(report))
    except (Exception,KeyboardInterrupt) as error:
        receipt.write_text(json.dumps(dict(state="failed-consumed",error_type=type(error).__name__)),encoding="utf-8")
        raise
    receipt.write_text(json.dumps(dict(state=report["status"],identity=identity)),encoding="utf-8")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",action="store_true")
    parser.add_argument("--old-prompt",required=True)
    parser.add_argument("--model",required=True)
    parser.add_argument("--frozen",default=DEFAULT_FROZEN)
    parser.add_argument("--timeout",type=float,default=120)
    parser.add_argument("--max-calls",type=int,default=48)
    parser.add_argument("--run-id",default="old-prompt-comparison-001")
    args=parser.parse_args(argv)
    try:
        report=run_comparison(live=args.live,old_prompt_path=args.old_prompt,model=args.model,frozen_path=args.frozen,
            timeout=args.timeout,max_calls=args.max_calls,run_id=args.run_id)
    except Exception as error:
        parser.exit(1,f"Prompt comparison refused/incomplete ({type(error).__name__}); no automatic retry, private data withheld.\n")
    print(json.dumps(dict(status=report["status"],provider_calls=len(report["provider_calls"]))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
