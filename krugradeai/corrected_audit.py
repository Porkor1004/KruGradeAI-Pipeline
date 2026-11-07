"""Private corrected-provenance audit and mock-only testing of both PR pipelines."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from krugradeai.corrected_dataset import OWNER_POLICY, OWNER_STATEMENT
from krugradeai.real_data import canonical, digest
from krugradeai.optimization_models import DevelopmentRecord, ValidationRecord
from krugradeai.experiment_models import HeldoutRecord
from krugradeai.run_optimization import run as optimize
from krugradeai.evaluation import run as evaluate
from krugradeai.models import GraderInput
from krugradeai.schema_mock import SchemaMockProvider
from krugradeai.prompts import GRADING_PROMPT


def audit(prepared, output, historical):
    prepared,out,historical=Path(prepared).resolve(),Path(output).resolve(),Path(historical).resolve()
    if out.exists(): raise FileExistsError('Fresh offline output required')
    summary=json.loads((prepared/'audit.json').read_text(encoding='utf-8'))
    rows=[json.loads(s) for s in (prepared/'master.jsonl').read_text(encoding='utf-8').splitlines()]
    parts={}
    for s,cls in [('development',DevelopmentRecord),('validation',ValidationRecord),('test',HeldoutRecord)]:
        content=(prepared/(s+'.jsonl')).read_bytes()
        if digest(content)!=summary['partition_sha256'][s]: raise ValueError('Partition hash mismatch')
        parts[s]=[cls.model_validate_json(line) for line in content.decode().splitlines() if line.strip()]
        if {r.id for r in parts[s]}!={r['id'] for r in rows if r['partition']==s}: raise ValueError('Ledger mismatch')
    for p,h in summary['input_sha256'].items():
        if digest(Path(p).read_bytes())!=h: raise ValueError('Source hash mismatch')
    group_sets={s:{r.provenance.split_group for r in rs} for s,rs in parts.items()}
    for a,b in [('development','validation'),('development','test'),('validation','test')]:
        if group_sets[a]&group_sets[b]: raise ValueError('Group leakage')
    out.mkdir(parents=True)
    def write(name,data):
        p=out/name; p.write_text(canonical(data),encoding='utf-8'); return p
    def jsonl(name,records):
        p=out/name; p.write_text(''.join(r.model_dump_json()+'\n' for r in records),encoding='utf-8'); return p
    (out/'grader.txt').write_text(GRADING_PROMPT,encoding='utf-8')
    dev_scopes={r.rubric.id for r in parts['development']}; val_scopes={r.rubric.id for r in parts['validation']}
    if val_scopes-dev_scopes: raise ValueError('Validation scope has no Development path')
    covered=[r for r in parts['development'] if r.rubric.id in val_scopes]
    uncovered=[r for r in parts['development'] if r.rubric.id not in val_scopes]
    optimization=[]
    for name,dev,val in [('validated',covered,parts['validation']),('development-only',uncovered,[])]:
        if not dev: continue
        devp=jsonl(name+'-development.jsonl',dev)
        scope_count=len({r.rubric.id for r in dev})
        calls=2*len(dev)+scope_count+2*len(val)
        cfg=dict(development=str(devp),output_directory=str(out/(name+'-optimization')),runtime='schema_mock',max_iterations=1,max_calls=calls)
        if val: cfg['validation']=str(jsonl(name+'-validation.jsonl',val))
        cp=write(name+'-optimization-mock.json',cfg)
        plan=optimize(cp,dry_run=True); result=optimize(cp)
        if any(p['stop_reason']=='agent_failure' for p in result['paths']): raise ValueError('Optimization mock agent failure')
        optimization.append(dict(mode=name,records=len(dev),validation=len(val),plan=plan,result=result))
        cfg.update(runtime='codex_cli',output_directory=str(out/(name+'-future-live')))
        write(name+'-optimization-live-proposal.json',cfg)
    assert sum(r['records'] for r in optimization)==len(parts['development'])
    cohorts=defaultdict(list)
    for s,records in parts.items():
        for r in records: cohorts[(s,r.rubric.id)].append(r)
    evaluations=[]
    for i,((split,scope),records) in enumerate(sorted(cohorts.items()),1):
        rubric=write(f'rubric-{i}.json',records[0].rubric.model_dump(mode='json'))
        data=out/f'evaluation-{i}.jsonl'
        data.write_text(''.join(json.dumps(dict(r.model_dump(mode='json'),split='evaluation'))+'\n' for r in records),encoding='utf-8')
        provider=SchemaMockProvider('grader')
        responses=[json.loads(provider.generate('mock\nINPUT_JSON\n'+GraderInput(question=r.question,rubric=r.rubric,response=r.response).model_dump_json())) for r in records]
        script=write(f'responses-{i}.json',responses)
        cfg=dict(dataset=str(data),rubric_file=str(rubric),prompt_file=str(out/'grader.txt'),runtime='mock',mock_responses=str(script),output_directory=str(out/f'evaluation-{i}'))
        cp=write(f'evaluation-mock-{i}.json',cfg); plan=evaluate(cp,dry_run=True); result=evaluate(cp)
        if result['status']!='completed': raise ValueError('Evaluation mock incomplete')
        if sum(m['expected_count'] for m in result['metrics_by_score_and_ocr'].values())!=len(records): raise ValueError('Metric cohorts incomplete')
        evaluations.append(dict(split=split,scope=scope,records=len(records),provider_calls=result['provider_calls'],status=result['status'],metrics_by_score_and_ocr=result['metrics_by_score_and_ocr']))
        if split=='test':
            cfg.pop('mock_responses'); cfg.update(runtime='codex_cli',role='heldout_grader',output_directory=str(out/f'evaluation-future-live-{i}'))
            write(f'evaluation-live-proposal-{i}.json',cfg)
    # Overlap is declared without opening predictions or frozen heldout outcomes.
    old=json.loads((historical/'manifest.json').read_text(encoding='utf-8'))
    old_tokens={p:set() for p in ('development','validation','test','reserve')}
    for r in old['lineage']:
        old_tokens[r['partition']].update(r.get('response_ids',[]))
        old_tokens[r['partition']].update(r.get('lineage',[]))
        old_tokens[r['partition']].update(r.get('source_attempt_ids',[]))
    overlap={}
    for s,records in parts.items():
        overlap[s]={prior:sum(bool((set(r.provenance.source_ids)|{next(x['attempt'] for x in rows if x['id']==r.id)}) & tokens) for r in records) for prior,tokens in old_tokens.items()}
    previous=prepared.parent.parent/'full-dataset-audit-v4'
    if previous.exists():
        oldrows=[json.loads(s) for s in (previous/'master.jsonl').read_text(encoding='utf-8').splitlines()]
        previous_map={r['id']:r['partition'] for r in oldrows}
        transition=dict(Counter(previous_map.get(r['id'],'new')+' -> '+r['partition'] for r in rows))
    else: transition={}
    supervised=[r for r in rows if r['category']=='supervised']
    result=dict(inventory=summary,owner_policy=OWNER_POLICY,owner_statement=OWNER_STATEMENT,
                optimization=optimization,evaluations=evaluations,evaluation_mock_calls=sum(r['provider_calls'] for r in evaluations),
                optimization_mock_calls=sum(r['result']['provider_calls'] for r in optimization),
                optimization_live_bound=sum(r['plan']['reserved_maximum_calls'] for r in optimization),test_calls_per_prompt=len(parts['test']),
                development_only_scopes=sorted(dev_scopes-val_scopes),historical_input_overlap=overlap,previous_split_transition=transition,
                canonical_provenance={k:dict(Counter(r[k] for r in supervised)) for k in ('answer_origin','score_origin','ocr_review_origin','rubric_origin')},
                group_leakage=False,live_calls=0,interpretation='Scripted zeros are software tests, not accuracy measurements. Historical overlap prevents a fresh-unseen-test claim.')
    write('offline-results.json',result)
    report=['# Corrected provenance: private dataset design for review','',OWNER_STATEMENT,'',
            'Final preparation directory: '+str(prepared), 'Offline evidence directory: '+str(out),'',
            'This supersedes the prior blanket unverified/AI score interpretation. Source files, old experiments, frozen prompts and heldout results were not modified. No live LLM calls, push or merge occurred.','',
            f"Complete inventory: {summary['source_rows']['zip']} ZIP examples + {summary['source_rows']['untitled']} nonempty workbook rows = {summary['total']}. Workbook rows include original sheets, review entries and prepared alternate representations; they are not all distinct examples.",'',
            '| Source | Development | Validation | Test | Unscored | Quarantined |','|---|---:|---:|---:|---:|---:|']
    for source in ('zip','untitled'):
        report.append('| '+source+' | '+' | '.join(str(summary['by_source_partition'].get(source+'/'+s,0)) for s in ('development','validation','test','unscored','quarantined'))+' |')
    report+=['',f"Eligible distinct supervised examples: {summary['supervised']}. Human-reviewed OCR + original TA score, passing row/rubric integrity gates: {summary['benchmark_eligible']}.",
             f"Reconciliation: {' + '.join(str(summary['assignments'].get(s,0)) for s in ('development','validation','test','unscored','quarantined'))} = {summary['total']}.",'',
             '| Split | Count | Eligible population | ZIP | Untitled |','|---|---:|---:|---:|---:|']
    for s in ('development','validation','test'):
        n=len(parts[s]); z=summary['by_source_partition'].get('zip/'+s,0); u=n-z
        report.append(f'| {s} | {n} | {100*n/summary["supervised"]:.2f}% | {z} ({100*z/n:.2f}%) | {u} ({100*u/n:.2f}%) |')
    report+=['','Provenance counts below distinguish physical source rows (including aliases) from distinct supervised examples. They must not be summed across different provenance dimensions.','',
             '| Dimension | Physical rows | Canonical supervised examples |','|---|---|---|']
    for k,counts in summary['provenance_counts'].items(): report.append('| '+k+' | '+json.dumps(counts)+' | '+json.dumps(result['canonical_provenance'][k])+' |')
    report+=['',f"Score-missing physical rows: {summary['score_missing']}. All other unusable rows and exact reasons are listed below and per-row in prepared/master.jsonl.",'',
             'Original TA authorship is based on the owner clarification plus actual row score fields. ZIP source-reviewed targets must match teacher_score_before and training_score. Explicit AI-generated target labels stay AI-generated even when numerically equal to an earlier TA score. Untitled expected_score is joined to approved/corrected non-stale review w when present; the owner identifies those reviewed values as TA score readings/corrections. For approved unchanged reviews only, an available original_score is retained when w is blank; corrected reviews with a missing w do not fall back. Original_score is retained, including before/after differences; its absence does not imply no TA score. Blank scores are never supplied from this policy.','',
             'Row-level benchmark eligibility requires original student work, original TA score, human-reviewed OCR, teacher rubric and successful integrity checks. ZIP source benchmark_eligible=false is retained and ZIP-original-TA metrics are reported separately by AI OCR review. AI-derived answers/labels never acquire a human benchmark designation. Q9/Q10 AI rubric examples remain separate and unscored.','',
             'Grouping uses response ancestry, attempts, source-file identity and substantive exact text across both sources. Common short answers such as True/False are logged as match candidates, not evidence that unrelated students are the same response. Parent/subpart criterion overlaps within a known original attempt are represented once, with redundant rows quarantined and linked. No scores are added or imputed.','',
             f"Group leakage: none across all three supervised splits. Detected cross-source hard/substantive-exact links: {len(summary['cross_source_edges'])}. Absence of such links is not proof against unknown semantic or scan relationships. See prepared/relationships.json and exact-text-candidates.json.",'',
             'Previous split: 428 / 0 / 75, with 306 unscored and 267 quarantined. The corrected split above uses owner-supported TA/human OCR provenance, admits suitable review-only rows, and removes overlapping parent/subpart counts. The complete row transition matrix is in offline-results.json. The old files remain intact.','',
             'This is a new experiment version, not a new unseen population. Historical input overlap is recorded in offline-results.json using response ancestry and attempts only. No previous prediction/result was used for selection. Overlapping historical Test work may already have been exposed; new results must not be represented as pristine first-use heldout evidence.','',
             'Remaining per-scope limitation: the following Development scopes have no compatible independent Validation scope; they run Development-only and cannot claim Validation-selected G1. Other Development paths use the fixed Validation partition. No teacher rubric was changed to manufacture compatibility:',*['- '+s for s in result['development_only_scopes']],'',
             'Exclusion and assignment reasons (physical rows):',*['- '+reason+': '+str(n) for reason,n in sorted(summary['reasons'].items())],'',
             f"Offline optimization: {len(optimization)} scope-compatible batches cover all {len(parts['development'])} Development records and {len(parts['validation'])} Validation records; {result['optimization_mock_calls']} scripted calls. Scripted zero discrepancies correctly stop unsupported refinement for review. Baseline Validation selection is exercised; G1 refinement/selection remains covered by synthetic regression tests, not claimed from these mocks.",
             f"Independent evaluation: {len(evaluations)} scope/split runs cover all {result['evaluation_mock_calls']} supervised records using scripted responses. Score-origin/OCR-review metrics are separate; mock metrics have no accuracy meaning.",'',
             f"Proposed one-iteration optimization upper bound: {result['optimization_live_bound']} calls = 2 × Development + rubric paths + 2 × Validation. Test evaluation: {len(parts['test'])} calls per prompt, or {2*len(parts['test'])} for a paired G0/candidate comparison. No live calls are authorized by these configurations.",'',
             'Proposed commands after dataset-design review (model placeholders must be replaced; no execution now):','```powershell',
             f'Get-ChildItem "{out}" -Filter *-optimization-live-proposal.json | ForEach-Object {{ python -m krugradeai.run_optimization --config $_.FullName --dry-run }}',
             '# After separate approval, use --live for approved optimization configs.',
             '# Lock exported candidate rules and heldout protocol before adapting the baseline evaluation proposals to that candidate.',
             '```','',
             'All datasets, ledger rows, reports, model-script responses and configs are private under Git-ignored artifacts/. Review this dataset design before live execution.']
    report=[line.replace('prepared/master.jsonl',prepared.name+'/master.jsonl').replace('prepared/relationships.json',prepared.name+'/relationships.json') for line in report]
    discovery=out.parent/'source-discovery.json'
    if discovery.exists():
        versions=json.loads(discovery.read_text(encoding='utf-8'))
        report+=['',f"Source discovery: the original Untitled workbook has {versions['original_total']} response rows in five sheets, verified identical to the same sheets in the enriched local carrier. They are counted once in the carrier inventory. Other located carrier copies are documented by SHA-256 in source-discovery.json; no Untitled CSV was found in the searched Downloads, Documents and local raw directories."]
    report+=['',f"Historical Test overlap: {overlap['development']['test']} new Development and {overlap['validation']['test']} new Validation records share known response/attempt evidence with historical Test. New Test overlaps {overlap['test']['development']} historical Development and {overlap['test']['validation']} historical Validation records, and {overlap['test']['test']} historical Test records by checked keys. These overlapping inputs require explicit review before any live experiment."]
    (out.parent/'audit-report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--historical',required=True)
    a=p.parse_args();r=audit(a.prepared,a.output_dir,a.historical)
    print(json.dumps(dict(status='offline_checks_passed',evaluation_calls=r['evaluation_mock_calls'],live_calls=0)))


if __name__=='__main__': main()
