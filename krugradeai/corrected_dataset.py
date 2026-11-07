"""Owner-clarified provenance and exhaustive, private, group-aware preparation.

Score authorship and OCR review are independent. Every physical source row is
retained; only disjoint compatible representations enter supervised partitions.
No provider imports, live calls, edits to sources or historical artifacts.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
import random
import re
import zipfile

from krugradeai.real_data import canonical, digest, _definition, _decimal, _answer, _marked, FLAGS
from krugradeai.full_dataset import normalize
from krugradeai.models import Question, Rubric, StudentResponse, TAReference
from krugradeai.optimization_models import DevelopmentRecord, ValidationRecord, RecordProvenance
from krugradeai.experiment_models import HeldoutRecord

OWNER_POLICY = 'dataset-owner-clarification-2026-11-07'
OWNER_STATEMENT = ('Original examinations and original scores were TA graded. ZIP OCR was AI reviewed; '
                   'Untitled OCR was human reviewed. Augmented/synthetic ZIP targets retain their explicit AI label source. '
                   'Blank reviewer identities do not negate this owner evidence. Reviewed w scores are human-reviewed '
                   'readings/corrections of TA scores under this declaration; original_score may be an absent pre-review OCR field. '
                   'This does not authenticate source scans or every transcription independently.')
PREFIX = 'com-eng-math-ii-2026-1-p'


def zip_origins(item, prov):
    kind = item['generation_kind']
    answer = 'fully_synthetic' if kind=='synthetic_full_response' else 'ai_augmented' if kind in ('inferred_spans','assumed_transcription') else 'original_student' if kind in ('source_reviewed','source_answer_ai_score') else 'unknown'
    target = item.get('target',{}).get('score')
    origin = 'missing' if target is None else 'ai_generated' if item.get('label_source')=='ai_generated' else 'unknown'
    if target is not None and item.get('label_source')=='teacher_derived' and answer=='original_student':
        try:
            if _decimal(target)==_decimal(prov.get('teacher_score_before'))==_decimal(prov.get('training_score')):
                origin='original_ta'
        except ValueError:
            pass
    return answer, origin


def reviewed_score(row):
    """An approved unchanged score may live only in original_score, never invent w."""
    if row.get('w') is not None: return row['w'], 'w'
    if row.get('review_status')=='approved' and row.get('original_score') is not None:
        return row['original_score'], 'original_score'
    return None, 'w'


def to_record(r, split):
    q, rubric = r['definition']
    cls = {'development':DevelopmentRecord,'validation':ValidationRecord,'test':HeldoutRecord}[split]
    fields = ('answer_origin','score_origin','ocr_review_origin','rubric_origin','source_dataset','source_response_id',
              'response_id','benchmark_eligible','benchmark_evidence','score_evidence')
    return cls(id=r['id'],question=q,rubric=rubric,response=StudentResponse(id=r['id'],question_id=q.id,text=_answer(r['text'])),
               ta_output=TAReference(total_score=r['score']),provenance=RecordProvenance(source=r['source_dataset'],item_id=r['id'],
               label_origin='human_derived' if r['score_origin']=='original_ta' else 'ai_derived',
               split_group=r['split_group'],source_ids=tuple(r['source_ids']),generation_kind=r['generation_kind'],reference_origin=r['rubric_origin'],
               **{k:r[k] for k in fields}))


def definition_for(row, definitions):
    scope=row.get('question_id') or ''
    parent=scope.split('.')[0]
    if parent not in definitions:
        raise ValueError('missing_teacher_rubric_or_unknown_question_scope')
    q,rubric=definitions[parent]
    declared={c['criterion_id']:(c['description']+'\npartial_credit: '+c['partial_credit'],_decimal(c['points'])) for c in json.loads(row['rubric_criteria'])}
    expected={c.id:(c.description,c.max_score) for c in rubric.criteria}
    if not declared or any(expected.get(k)!=v for k,v in declared.items()):
        raise ValueError('teacher_criterion_subset_mismatch')
    if set(declared)!=set(expected):
        q=Question(id=scope,text=q.text+'\nScore only source scope '+scope+' using the supplied criterion subset.',reference_solution=q.reference_solution)
        rubric=Rubric(id=scope+'@teacher-subset',criteria=tuple(c for c in rubric.criteria if c.id in declared),total_score=sum(v[1] for v in declared.values()))
    if _decimal(row['rubric_max_score'])!=rubric.total_score:
        raise ValueError('rubric_maximum_mismatch')
    return q,rubric


def link(rows):
    """Hard identities/attempts/files; substantive exact text only is conservative.

    Common Boolean/short answers from distinct students are duplicate candidates,
    not evidence of identical student work. They must not collapse all attempts.
    """
    parent={r['id']:r['id'] for r in rows}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    owners,edges,texts={},[],defaultdict(list)
    for r in rows:
        keys=[('response',str(s)) for s in r['source_ids'] if s]
        if r.get('attempt'): keys.append(('attempt',str(r['attempt'])))
        if r.get('source_file'): keys.append(('file',normalize(str(r['source_file']).replace('\\','/'))))
        text=normalize(r.get('text') or '')
        if text:
            scope=r['question_id']
            texts[(scope,text)].append(r['id'])
            if len(text)>=80: keys.append(('substantive_exact_text',scope+':'+digest(text.encode())))
        for key in keys:
            if key in owners:
                other=owners[key]
                a,b=root(r['id']),root(other)
                parent[max(a,b)]=min(a,b)
                edges.append(dict(left=other,right=r['id'],relation=key[0]))
            else: owners[key]=r['id']
    for r in rows: r['split_group']=root(r['id'])
    candidates=[dict(question_id=k[0],text_sha256=digest(k[1].encode()),record_ids=ids,
                     decision='grouped_substantive_exact_match' if len(k[1])>=80 else 'common_short_answer_not_identity_evidence') for k,ids in texts.items() if len(ids)>1]
    return edges,candidates


def remove_representations(rows):
    """Select nonoverlapping criterion scopes per known attempt and question.

    Prefer a full compatible scope, then largest disjoint subsets. Preserve every
    redundant row and its selected representative; never add their scores.
    """
    buckets=defaultdict(list)
    for r in rows:
        if r['category']=='supervised':
            # Only deduplicate within hard original attempt/file relationships.
            identity=r.get('source_file') or r.get('attempt') or r['response_id']
            buckets[(r['source_dataset'],identity,r['question_id'].split('.')[0])].append(r)
    for rs in buckets.values():
        selected=[]
        for r in sorted(rs,key=lambda r:(-len(r['definition'][1].criteria),r.get('representation_priority',0),r['id'])):
            ids={c.id for c in r['definition'][1].criteria}
            overlap=[s for s in selected if ids & {c.id for c in s['definition'][1].criteria}]
            if overlap:
                r.update(category='quarantined',reason='overlapping_original_response_criterion_representation',canonical_ids=[s['id'] for s in overlap])
            else: selected.append(r)


def split_rows(rows, seed=731107):
    groups=defaultdict(list)
    for r in rows:
        if r['category']=='supervised': groups[r['split_group']].append(r)
    if not groups: raise ValueError('No supervised records')
    allowed_val={g for g,rs in groups.items() if all(r['benchmark_eligible'] and r['source_dataset']=='untitled' for r in rs)}
    # Score independent randomized search: target sizes, source balance and scope
    # coverage. No target score, model prediction or past outcome enters search.
    all_scopes={r['definition'][1].id for rs in groups.values() for r in rs}
    counts={g:Counter(r['source_dataset'] for r in rs) for g,rs in groups.items()}
    scopes={g:{r['definition'][1].id for r in rs} for g,rs in groups.items()}
    total=sum(map(len,groups.values()))
    def objective(a):
        n=Counter(); src={s:Counter() for s in ('development','validation','test')}; cover={s:set() for s in src}
        for g,s in a.items():
            n[s]+=len(groups[g]); src[s].update(counts[g]); cover[s].update(scopes[g])
        missing_val=len(cover['development']-cover['validation'])
        missing_dev=len((cover['validation']|cover['test'])-cover['development'])
        return (10000*missing_dev+1000*missing_val+
                sum(abs(n[s]-ratio*total) for s,ratio in [('development',.7),('validation',.15),('test',.15)])+
                .35*(abs(src['development']['zip']-.8*n['development'])+abs(src['test']['untitled']-.8*n['test'])))
    rng=random.Random(seed); best=None
    keys=sorted(groups)
    # Multistart coordinate search can preserve mandatory scope coverage while
    # varying whole attempts. Deterministic seed and counts recorded in manifest.
    for trial in range(100):
        assignment={g:('validation' if g in allowed_val and rng.random()<.3 else 'test' if rng.random()<.2 else 'development') for g in keys}
        for _ in range(8):
            improved=False; order=keys[:]; rng.shuffle(order)
            for g in order:
                old=assignment[g]; current=objective(assignment)
                for target in ('development','validation','test'):
                    if target=='validation' and g not in allowed_val: continue
                    assignment[g]=target; value=objective(assignment)
                    if value<current:
                        old,current,improved=target,value,True
                assignment[g]=old
            if not improved: break
        value=objective(assignment)
        if best is None or value<best[0]: best=value,assignment.copy()
    for r in rows:
        r['partition']=best[1][r['split_group']] if r['category']=='supervised' else r['category']
    return dict(seed=seed,search_trials=100,objective=best[0],validation_eligible_components=len(allowed_val),rubric_scopes=len(all_scopes))


def prepare(raw_dir, output):
    import openpyxl
    raw,out=Path(raw_dir).resolve(),Path(output).resolve()
    if out.exists(): raise FileExistsError('Fresh experiment directory required')
    archive=raw/'training-groups-signal320.zip'
    books=[p for p in raw.glob('*.xlsx') if 'untitled' in p.name.casefold()]
    if len(books)!=1: raise ValueError('Expected one complete Untitled carrier')
    book=books[0]
    hashes={str(p):digest(p.read_bytes()) for p in (archive,book)}
    rows,definitions,documents=[],{},{}
    with zipfile.ZipFile(archive) as z:
        prefix=next(n[:-9] for n in z.namelist() if n.endswith('/README.md'))
        # Complete documentation is read before processing any example.
        for n in z.namelist():
            if any(k in n.lower() for k in ('readme','manifest','provenance')): documents[n]=z.read(n).decode('utf-8')
        manifest=json.loads(documents[prefix+'manifest.json'])
        for n,h in manifest['files_sha256'].items():
            if digest(z.read(prefix+n))!=h: raise ValueError('ZIP integrity mismatch')
        loaded=[]
        for group in ('01_source_reviewed','02_ai_augmented'):
            ps=json.loads(documents[prefix+group+'/provenance.json'])
            prov={p['response_id']:p for p in ps}
            items=[json.loads(s) for s in z.read(prefix+group+'/training.jsonl').decode('utf-8').splitlines() if s.strip()]
            if len(items)!=manifest['groups'][group]['rows'] or len(prov)!=len(ps): raise ValueError('ZIP inventory/provenance identity mismatch')
            for n,item in enumerate(items,1):
                p=prov[item['response_id']]; inp=item['input']
                if item['reference_origin']=='teacher_supplied':
                    definition=_definition(inp)
                    if inp['question_id'] in definitions and definitions[inp['question_id']]!=definition: raise ValueError('Conflicting teacher scope')
                    definitions[inp['question_id']]=definition
                loaded.append((group,n,item,p))
        for group,n,item,p in loaded:
            inp=item['input']; answer,score_origin=zip_origins(item,p)
            r=base(f'zip:{group}:{n}','zip',f'{archive.name}!{group}/training.jsonl:{n}',item,item['response_id'],inp['question_id'],inp.get('answer_text'))
            r.update(provenance=p,source_response_id=item['source_response_id'],source_ids=sorted(set([item['response_id'],item['source_response_id'],*p['source_response_ids']])),
                     attempt=item['split_group'],source_file=p['source_file'],answer_origin=answer,score_origin=score_origin,ocr_review_origin='ai',
                     generation_kind=item['generation_kind'],rubric_origin='teacher_supplied' if item['reference_origin']=='teacher_supplied' else 'ai_generated',
                     score=item.get('target',{}).get('score'),score_evidence=[r['locator']+'#target.score',r['locator']+'#label_source','provenance.teacher_score_before','provenance.training_score',OWNER_POLICY],
                     benchmark_evidence=['ZIP benchmark_eligible=false retained; AI OCR is not human-reviewed OCR.'],source_benchmark_eligible=p.get('benchmark_eligible'))
            try:
                if item['split_group']!=p['source_attempt_id'] or item['source_response_id']!=p['source_response_id'] or inp['question_id']!=p['question_id']: raise ValueError('inconsistent_source_lineage')
                if any(item.get(k)!=p.get(k) for k in ('label_source','generation_kind','reference_origin')) or _decimal(r['score'])!=_decimal(p.get('training_score')): raise ValueError('inconsistent_source_target_provenance')
                if r['rubric_origin']!='teacher_supplied': raise ValueError('Q9_Q10_AI_generated_rubric_separate_unscored')
                r['definition']=definitions[r['question_id']]
                validate(r)
            except (ValueError,KeyError,TypeError) as e: r['reason']=str(e)
            rows.append(r)
    if len(rows)!=manifest['total_rows']: raise ValueError('ZIP total mismatch')
    w=openpyxl.load_workbook(book,read_only=True,data_only=True); sheets={}
    try:
        for s in w:
            it=iter(s.values); header=next(it,())
            sheets[s.title]=[(n,{str(k):v for k,v in zip(header,vals) if k is not None}) for n,vals in enumerate(it,2) if any(v is not None for v in vals)]
    finally: w.close()
    reviews={d['response_id']:(n,d) for n,d in sheets['krugrade-human-review success31']}
    if len(reviews)!=len(sheets['krugrade-human-review success31']): raise ValueError('Duplicate review identities')
    scope_definitions=definitions.copy()
    for _,d in sheets['usable-319']:
        try:
            definition=definition_for(d,definitions)
            if d['question_id'] in scope_definitions and scope_definitions[d['question_id']]!=definition: raise ValueError('Conflicting scope')
            scope_definitions[d['question_id']]=definition
        except (ValueError,KeyError,TypeError): pass
    seen={}
    for sheet in sorted(sheets,key=lambda s:(s!='usable-319',s)):
        for n,d in sheets[sheet]:
            rid=d.get('response_id'); review=reviews.get(rid); evidence=review[1] if review else None
            is_review=sheet=='krugrade-human-review success31'
            qid=d.get('question_id') or d.get('corrected_question_id') or d.get('original_question_id') or ''
            if is_review and not qid:
                label=str(d.get('corrected_question_label') or d.get('question_label') or '')
                match=re.fullmatch(r'(\d+)(?:\.(\d+))?[.)]*',label)
                if match: qid=PREFIX+match[1]+('.'+match[2] if match[2] and match[2]!='0' else '')
            text=(d['corrected_answer'] if d.get('corrected_answer') is not None else d.get('original_answer')) if is_review else d.get('answer_text')
            r=base(f'untitled:{sheet}:{n}','untitled',f'{book.name}!{sheet}:{n}',d,rid,qid,text)
            score,score_field=reviewed_score(d) if is_review else (d.get('expected_score'),'expected_score')
            r.update(provenance=evidence,source_response_id=rid,source_ids=[rid] if rid else [],attempt=d.get('attempt_id'),
                     source_file=evidence.get('source_pdf') if evidence else None,answer_origin='original_student',
                     score_origin='original_ta' if score is not None else 'missing',ocr_review_origin='human',score=score,
                     score_evidence=[r['locator']+'#'+score_field,OWNER_POLICY],
                     representation_priority=0 if sheet=='usable-319' else 2 if is_review else 1,
                     benchmark_evidence=[OWNER_POLICY,'Original student answer + available TA score + human-reviewed OCR; row integrity gates apply.'])
            if qid in scope_definitions: r['rubric_origin']='teacher_supplied'
            if review:
                r['score_evidence'] += [f'{book.name}!krugrade-human-review success31:{review[0]}#'+reviewed_score(evidence)[1], 'review.original_score is retained; approved unchanged originals are usable when w is absent']
            try:
                if rid in FLAGS: raise ValueError(FLAGS[rid])
                if rid in seen:
                    r.update(category='quarantined',reason='alternate_sheet_representation',canonical_ids=[seen[rid]])
                else:
                    if evidence and (evidence['review_status'] not in ('approved','corrected') or evidence.get('stale') is not False): raise ValueError('deferred_excluded_or_stale_review')
                    if is_review:
                        if qid not in scope_definitions: raise ValueError('review_only_missing_teacher_scope')
                        r['definition']=scope_definitions[qid]
                    else: r['definition']=definition_for(d,definitions)
                    r['rubric_origin']='teacher_supplied'
                    if evidence:
                        if _decimal(score)!=_decimal(reviewed_score(evidence)[0]): raise ValueError('review_score_mismatch')
                        original=evidence['corrected_answer'] if evidence.get('corrected_answer') is not None else evidence.get('original_answer')
                        if normalize(original)!=normalize(text): raise ValueError('review_answer_mismatch')
                        if evidence.get('verified_max_score') is not None and _decimal(evidence['verified_max_score'])!=r['definition'][1].total_score: raise ValueError('review_maximum_mismatch')
                    validate(r)
            except (ValueError,KeyError,TypeError) as e: r['reason']=str(e)
            if rid: seen.setdefault(rid,r['id'])
            rows.append(r)
    for r in rows:
        if not r.get('text') and r.get('text') is not False: r.update(category='quarantined',reason='missing_answer_text; '+r['reason'])
    edges,candidates=link(rows)
    remove_representations(rows)
    split_policy=split_rows(rows)
    verify(rows)
    out.mkdir(parents=True)
    for split in ('development','validation','test'):
        (out/(split+'.jsonl')).write_text(''.join(to_record(r,split).model_dump_json()+'\n' for r in rows if r['partition']==split),encoding='utf-8')
    (out/'master.jsonl').write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='definition'},ensure_ascii=False,default=str)+'\n' for r in rows),encoding='utf-8')
    for name,data in [('source-documentation.json',documents),('relationships.json',edges),('exact-text-candidates.json',candidates),('owner-provenance-policy.json',dict(id=OWNER_POLICY,statement=OWNER_STATEMENT))]:
        (out/name).write_text(canonical(data),encoding='utf-8')
    summary=dict(experiment_version='full-dataset-corrected-provenance-v1',owner_policy=OWNER_POLICY,input_sha256=hashes,total=len(rows),
                 source_rows=dict(Counter(r['source_dataset'] for r in rows)),workbook_sheets={s:len(rs) for s,rs in sheets.items()},
                 assignments=dict(Counter(r['partition'] for r in rows)),by_source_partition=dict(Counter(r['source_dataset']+'/'+r['partition'] for r in rows)),
                 provenance_counts={k:dict(Counter(r[k] for r in rows)) for k in ('answer_origin','score_origin','ocr_review_origin','rubric_origin')},
                 split_provenance={s:{k:dict(Counter(r[k] for r in rows if r['partition']==s)) for k in ('source_dataset','answer_origin','score_origin','ocr_review_origin')} for s in ('development','validation','test')},
                 by_question_partition=dict(Counter(r['question_id']+'/'+r['partition'] for r in rows)),
                 reasons=dict(Counter(r['reason'] for r in rows)),score_missing=sum(r['score_origin']=='missing' for r in rows),
                 supervised=sum(r['category']=='supervised' for r in rows),benchmark_eligible=sum(r['category']=='supervised' and r['benchmark_eligible'] for r in rows),
                 cross_source_edges=[e for e in edges if e['left'].startswith('zip:')!=e['right'].startswith('zip:')],split_policy=split_policy,
                 partition_sha256={s:digest((out/(s+'.jsonl')).read_bytes()) for s in ('development','validation','test')},
                 reconciliation='Every physical row is assigned exactly once; aliases and overlapping criterion representations remain in quarantine.',
                 limitations=['Owner evidence establishes authorship/review process, not independent reinspection of source scans.',
                              'ZIP original TA labels retain AI OCR and source benchmark_eligible=false.',
                              'Common short answers are not duplicate identities; unknown semantic cross-source relations remain possible.'])
    if len(rows)!=manifest['total_rows']+sum(len(rs) for rs in sheets.values()): raise ValueError('Source-row reconciliation mismatch')
    for p,h in hashes.items():
        if digest(Path(p).read_bytes())!=h: raise ValueError('Source changed')
    (out/'audit.json').write_text(canonical(summary),encoding='utf-8')
    return summary


def base(id,source,locator,raw,rid,qid,text):
    return dict(id=id,source_dataset=source,locator=locator,raw=raw,response_id=str(rid) if rid else id,source_response_id=None,
                source_ids=[],question_id=qid,text=text,generation_kind='original_student_response',category='unscored',reason='not_yet_eligible',
                answer_origin='unknown',score_origin='unknown',ocr_review_origin='unknown',rubric_origin='unknown',
                benchmark_eligible=False,benchmark_evidence=[],score_evidence=[],split_group=id)


def validate(r):
    _answer(r['text'])
    if _marked(str(r['text'])): raise ValueError('unresolved_transcription_marker')
    if r['score_origin'] not in ('original_ta','ai_generated'): raise ValueError('missing_or_unknown_score_origin')
    r['score']=str(_decimal(r['score']))
    to_record(r,'development')
    r.update(category='supervised',reason='eligible_teacher_rubric_and_traceable_score')
    r['benchmark_eligible']=bool(r['source_dataset']=='untitled' and r['answer_origin']=='original_student' and r['score_origin']=='original_ta' and r['ocr_review_origin']=='human')


def verify(rows):
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Ledger identity duplication')
    owners={}
    ids={}
    for r in rows:
        if r['category']!='supervised': continue
        s=r['partition']
        if s=='validation' and not (r['source_dataset']=='untitled' and r['benchmark_eligible']): raise ValueError('Validation provenance violation')
        for key in [r['split_group'],*['response:'+i for i in r['source_ids']]]:
            if owners.setdefault(key,s)!=s: raise ValueError('Cross-partition group/ancestry leakage')
        if ids.setdefault(r['id'],s)!=s: raise ValueError('Record duplicated across splits')
    if sum(Counter(r['partition'] for r in rows).values())!=len(rows): raise ValueError('Reconciliation failure')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--raw-dir',required=True); p.add_argument('--output-dir',required=True)
    args=p.parse_args(); summary=prepare(args.raw_dir,args.output_dir)
    print(canonical({k:summary[k] for k in ('total','source_rows','assignments','by_source_partition','supervised')}))


if __name__=='__main__': main()
