"""Exercise both independent pipeline entry points on a private full preparation.

All model responses are scripts. Metrics verify report plumbing, not accuracy.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from krugradeai.real_data import canonical, digest
from krugradeai.models import GraderInput
from krugradeai.optimization_models import DevelopmentRecord
from krugradeai.experiment_models import HeldoutRecord
from krugradeai.schema_mock import SchemaMockProvider
from krugradeai.prompts import GRADING_PROMPT
from krugradeai.run_optimization import run as optimize
from krugradeai.evaluation import run as evaluate


def run(prepared, destination, historical=None, original_untitled=None):
    prepared, out = Path(prepared).resolve(), Path(destination).resolve()
    if out.exists():
        raise FileExistsError('Use a fresh private offline directory')
    audit = json.loads((prepared/'audit.json').read_text(encoding='utf-8'))
    versions = None
    if original_untitled:
        import openpyxl
        original = Path(original_untitled).resolve()
        carrier = next(Path(p) for p in audit['input_sha256'] if Path(p).suffix=='.xlsx')
        a = openpyxl.load_workbook(original, read_only=True, data_only=True)
        b = openpyxl.load_workbook(carrier, read_only=True, data_only=True)
        try:
            equal = all(s in b.sheetnames and list(a[s].values)==list(b[s].values) for s in a.sheetnames)
            counts = {s.title:max(0,sum(any(v is not None for v in r) for r in s.values)-1) for s in a}
            versions = dict(original_file=str(original), original_sha256=digest(original.read_bytes()),
                            original_response_rows=sum(counts.values()), original_sheets=counts,
                            original_sheets_identical_in_carrier=equal)
            if not equal:
                raise ValueError('Original Untitled differs: inventory additional records before proceeding')
        finally:
            a.close()
            b.close()
    ledger = [json.loads(s) for s in (prepared/'master.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(ledger) == audit['total'] == sum(audit['assignments'].values())
    parts = {}
    for split, cls in [('development', DevelopmentRecord), ('test', HeldoutRecord)]:
        parts[split] = tuple(cls.model_validate_json(s) for s in (prepared/(split+'.jsonl')).read_text(encoding='utf-8').splitlines())
        assert {r.id for r in parts[split]} == {r['id'] for r in ledger if r['partition']==split}
    assert not {r.provenance.split_group for r in parts['development']} & {r.provenance.split_group for r in parts['test']}
    assert not (prepared/'validation.jsonl').read_bytes(), 'Human verification evidence requires explicit re-audit'
    for path, expected in audit['input_sha256'].items():
        assert digest(Path(path).read_bytes()) == expected
    out.mkdir(parents=True)
    def write(name, content):
        p = out/name
        p.write_text(canonical(content), encoding='utf-8')
        return p
    (out/'grader.txt').write_text(GRADING_PROMPT, encoding='utf-8')
    paths = len({r.rubric.id for r in parts['development']})
    budget = 2*len(parts['development'])+paths
    config = dict(development=str(prepared/'development.jsonl'), output_directory=str(out/'optimization'),
                  runtime='schema_mock', max_iterations=1, max_calls=budget)
    opt_path = write('optimization-mock.json', config)
    opt_plan = optimize(opt_path, dry_run=True)
    optimization = optimize(opt_path)
    assert all(p['stop_reason'] in ('human_review', 'no_discrepancies', 'no_change') for p in optimization['paths'])
    config.update(runtime='codex_cli', model='REPLACE_WITH_EXPLICIT_MODEL', output_directory=str(out/'future-live-optimization'))
    write('optimization-live-proposal.json', config)
    cohorts = defaultdict(list)
    for split in parts:
        for r in parts[split]:
            cohorts[(split, r.rubric.id)].append(r)
    evaluations = []
    for index, ((split, rubric), records) in enumerate(sorted(cohorts.items()), 1):
        rubric_path = write(f'rubric-{index}.json', records[0].rubric.model_dump(mode='json'))
        data = out/f'evaluation-{index}.jsonl'
        data.write_text(''.join(json.dumps(dict(r.model_dump(mode='json'), split='evaluation'))+'\n' for r in records), encoding='utf-8')
        provider = SchemaMockProvider('grader')
        responses = [json.loads(provider.generate('script\nINPUT_JSON\n'+GraderInput(question=r.question, rubric=r.rubric, response=r.response).model_dump_json())) for r in records]
        script = write(f'responses-{index}.json', responses)
        cfg = dict(dataset=str(data), rubric_file=str(rubric_path), prompt_file=str(out/'grader.txt'),
                   output_directory=str(out/f'evaluation-{index}'), runtime='mock', mock_responses=str(script))
        cp = write(f'evaluation-mock-{index}.json', cfg)
        plan = evaluate(cp, dry_run=True)
        result = evaluate(cp)
        assert result['status']=='completed'
        assert sum(m['expected_count'] for m in result['metrics_by_label_origin'].values())==len(records)
        evaluations.append(dict(split=split, rubric=rubric, records=len(records), provider_calls=result['provider_calls'], status=result['status'], plan=plan,
                                metrics_by_label_origin=result['metrics_by_label_origin']))
        if split=='test':
            cfg.pop('mock_responses')
            cfg.update(runtime='codex_cli', model='REPLACE_WITH_EXPLICIT_MODEL', output_directory=str(out/f'future-live-evaluation-{index}'))
            write(f'evaluation-live-proposal-{index}.json', cfg)
    historical_summary = None
    if historical:
        manifest = json.loads((Path(historical)/'manifest.json').read_text(encoding='utf-8'))
        historical_summary = dict(summary=manifest['summary'], policy=manifest['policy'],
            exclusion_reasons=dict(Counter(r['reason'] for r in manifest['eligibility_decisions'] if 'reason' in r)))
        qpath = Path(historical)/'quarantine.json'
        if qpath.exists():
            historical_summary['quarantine_reasons'] = dict(Counter(r['reason'] for r in json.loads(qpath.read_text(encoding='utf-8'))))
        write('historical-selection-evidence.json', historical_summary)
    report = dict(inventory=audit, source_versions=versions, optimization=optimization, optimization_plan=opt_plan, evaluations=evaluations,
                  evaluation_mock_calls=sum(r['provider_calls'] for r in evaluations),
                  proposed_one_iteration_optimization_bound=budget, proposed_single_heldout_evaluation_calls=len(parts['test']),
                  historical=historical_summary, live_calls=0,
                  interpretation='Scripted zeros verify software only. No human accuracy or prompt improvement claim. No human-only Validation is available; live selection is blocked.')
    write('offline-results.json', report)
    lines = ['# Private full-source dataset audit', '',
             'No live LLM calls were made. Original sources, historical results and frozen prompts were not changed.', '',
             f"ZIP: {audit['raw_zip']} source records. Untitled workbook: {audit['raw_untitled']} nonempty physical rows. Combined physical inventory: {audit['total']}.",
             'CSV/JSONL/pipeline-cases files in the ZIP are representations of the same 390 examples, not additional observations. Workbook review rows and alternate representations are retained individually in the ledger.', '',
             '| Source | Development | Validation | Test | Unscored | Quarantined |', '|---|---:|---:|---:|---:|---:|']
    for src in ('zip','untitled'):
        lines.append('| '+src+' | '+' | '.join(str(audit['by_source_partition'].get(src+'/'+s,0)) for s in ('development','validation','test','unscored','quarantined'))+' |')
    if versions:
        lines += ['', f"Original Untitled file: {versions['original_file']}; SHA-256 {versions['original_sha256']}. Its {versions['original_response_rows']} original response rows match the same sheets in the enriched carrier exactly. They are all included in the carrier inventory, not added a second time. The carrier adds the prepared and review sheets; counts distinguish physical representations from independent responses. No Untitled CSV was found in the searched Desktop, Downloads and Documents locations."]
    lines += ['', f"Supervised examples: {audit['supervised']}. Verified human evaluation examples: 0. Cross-source detected links: {audit['cross_source_links']}. No detected link is not proof of semantic independence.",
              f"Unscored inference with available question context: {audit['unscored_inference_ready']}; additional unscored records requiring question/context repair: {audit['unscored_requires_context_repair']}. These are disjoint subsets of the unscored column, not extra records.",
              f"Reconciliation: {' + '.join(str(v) for v in audit['assignments'].values())} = {audit['total']} physical rows.", '',
              'The 70/15/15 and source proportions are targets. Human-only Validation is empty because the review log does not identify a human reviewer. The available supervised population is assigned approximately 85/0/15 without discarding excess examples. The source ratio is constrained by intact components; no label or duplicate was invented.', '',
              'The master.jsonl ledger retains every raw row, source locator, source IDs, review metadata, original/augmented provenance, connected split_group, category, partition and exact reason. relationships.json records all detected ancestry, attempt, source-file and exact normalized answer links. See audit.json for question/provenance counts and source-documentation.json for the complete ZIP README, manifests and provenance.', '',
              'Quarantined alternate representations are linked to a preferred representation, not deleted. Missing-answer records cannot be used for inference. Other unscored rows retain their text for inference or manual repair; no supervised score is inferred. Q9/Q10 remain separate because their references are AI-generated.', '',
              'Historical 138: the preserved manifest documents 95 Development (76 ZIP + 19 workbook), 19 Validation (workbook), and 24 Test (5 ZIP + 19 workbook). The importer had 292 eligible ZIP and 57 assembled workbook parent examples. It allocated workbook groups approximately one third each, selected ZIP subsets to match source ratios, and reserved 211 eligible ZIP examples. The number 138 was the resulting selection, not the full inventory. Human parent assembly required complete disjoint criterion coverage; incomplete subparts were excluded. The historical human-derived designation relied on approved/corrected review joins and is not independent proof of human verification. Exact historical exclusion counts are in historical-selection-evidence.json.', '',
              f"Offline Agent Optimization Pipeline: {optimization['status']}; {len(parts['development'])} Development rows, {paths} rubric paths, {optimization['provider_calls']} scripted calls; no Validation or heldout input loaded.",
              f"Offline Independent Evaluation Pipeline: {len(evaluations)} scope/split runs, {report['evaluation_mock_calls']} scripted calls covering every supervised Development/Test example. AI, unverified and empty human cohorts are reported separately. Scripted metrics have no accuracy interpretation.", '',
              f'Proposed live optimization (one iteration): at most {budget} calls without Validation; if trustworthy Validation is later established, add 2 × its record count and recheck all groups/scopes. Independent heldout evaluation: {len(parts["test"])} calls per prompt, twice that for baseline/candidate comparison. No monetary estimate is available without a selected model/runtime.', '',
              'Commands (proposals only; replace model placeholders after human-label review):', '```powershell',
              f'python -m krugradeai.run_optimization --config "{out / "optimization-live-proposal.json"}" --dry-run',
              f'python -m krugradeai.run_optimization --config "{out / "optimization-live-proposal.json"}" --live',
              f'Get-ChildItem "{out}" -Filter evaluation-live-proposal-*.json | ForEach-Object {{ python -m krugradeai.run_evaluation --config $_.FullName --dry-run }}',
              '# After a separately authorized prompt and heldout protocol are locked, run each evaluation config with --live.', '```', '',
              'Independent evaluation proposal configs use the baseline instruction prefix. A selected candidate requires explicit exported rubric/rules inputs; the baseline configs do not automatically evaluate a candidate.', '',
              'Every private output is under Git-ignored artifacts/. No push, merge, frozen-prompt modification or historical-output rewrite was performed.']
    (out/'audit-report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--historical')
    parser.add_argument('--original-untitled')
    args = parser.parse_args()
    report = run(args.prepared, args.output_dir, args.historical, args.original_untitled)
    print(json.dumps(dict(status='completed', development=report['optimization']['records'], evaluation_calls=report['evaluation_mock_calls'], live_calls=0)))


if __name__=='__main__':
    main()
