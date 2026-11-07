"""Private, exhaustive source-row ledger. No provider calls; never edits sources.

Physical workbook rows are counted separately from unique responses. Review and
alternate-sheet representations remain in the ledger, linked to their response.
Unknown review authorship is never promoted to verified human ground truth.
"""
from collections import Counter, defaultdict
from decimal import Decimal
import argparse
import json
from pathlib import Path
import zipfile
import unicodedata

from krugradeai.real_data import canonical, digest, _definition, _decimal, _answer, _marked, FLAGS
from krugradeai.models import StudentResponse, TAReference
from krugradeai.optimization_models import DevelopmentRecord, RecordProvenance
from krugradeai.experiment_models import HeldoutRecord


def normalize(text):
    return ''.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def connect(rows):
    """Use identities, attempts, ancestry and scoped exact text, across sources."""
    parents = {r['id']: r['id'] for r in rows}
    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    owners, edges = {}, []
    for r in rows:
        keys = ['response:' + str(x) for x in r['source_ids'] if x]
        keys += ['attempt:' + str(r['attempt'])] if r.get('attempt') else []
        if r.get('source_file'):
            keys += ['file:' + normalize(str(r['source_file']).replace('\\', '/'))]
        if r.get('text') and r.get('question_id'):
            keys += ['text:' + r['question_id'] + ':' + digest(normalize(r['text']).encode())]
        for key in keys:
            if key in owners:
                other = owners[key]
                a, b = root(r['id']), root(other)
                parents[max(a, b)] = min(a, b)
                edges.append(dict(left=other, right=r['id'], relation=key.split(':')[0]))
            else:
                owners[key] = r['id']
    for r in rows:
        r['split_group'] = root(r['id'])
    return edges


def assign(rows):
    """No reserve/cap: every supervised row gets a whole-component partition.

    Human certification is a separate evidence gate. With no certified rows,
    validation stays empty; available data go to Development and Test. Search
    uses only component counts/source, never scores or observed model quality.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[r['split_group']].append(r)
    usable = {g: [r for r in rs if r['category'] == 'supervised'] for g, rs in groups.items()}
    usable = {g: rs for g, rs in usable.items() if rs}
    total = sum(map(len, usable.values()))
    # Greedy deterministic best improvements to target heldout 15%, 80% Untitled.
    selected = set()
    def objective(gs):
        rs = [r for g in gs for r in usable[g]]
        n = len(rs)
        untitled = sum(r['source'] == 'untitled' for r in rs)
        return abs(n - .15 * total) + .35 * abs(untitled - .8 * n)
    while usable:
        options = [(objective(selected ^ {g}), g) for g in usable]
        value, best = min(options)
        if value >= objective(selected):
            break
        selected ^= {best}
    for r in rows:
        if r['category'] == 'supervised':
            r['partition'] = 'test' if r['split_group'] in selected else 'development'
            r['reason'] = 'valid_label_and_teacher_rubric; whole_connected_component; label_origin=' + r['label_origin']
        else:
            r['partition'] = r['category']


def prepare(raw_directory, destination):
    import openpyxl
    raw, out = Path(raw_directory).resolve(), Path(destination).resolve()
    if out.exists():
        raise FileExistsError('Use a fresh private output directory')
    archives = list(raw.glob('training-groups-signal320.zip'))
    books = [p for p in raw.glob('*.xlsx') if 'untitled' in p.name.casefold()]
    if len(archives) != 1 or len(books) != 1:
        raise ValueError('Expected exactly one named ZIP and one Untitled workbook')
    inputs = {str(p): digest(p.read_bytes()) for p in archives + books}
    rows, documents, definitions = [], {}, {}
    with zipfile.ZipFile(archives[0]) as z:
        prefix = next(n[:-len('README.md')] for n in z.namelist() if n.endswith('/README.md'))
        # Read COMPLETE documentation and provenance before reading training rows.
        for name in z.namelist():
            if any(k in name.lower() for k in ('readme', 'manifest', 'provenance')):
                documents[name] = z.read(name).decode('utf-8')
        manifest = json.loads(documents[prefix + 'manifest.json'])
        for name, expected in manifest['files_sha256'].items():
            if digest(z.read(prefix + name)) != expected:
                raise ValueError('ZIP member integrity mismatch')
        for group in ('01_source_reviewed', '02_ai_augmented'):
            prov = {p['response_id']: p for p in json.loads(documents[prefix+group+'/provenance.json'])}
            records = [json.loads(s) for s in z.read(prefix+group+'/training.jsonl').decode().splitlines() if s.strip()]
            if len(records) != manifest['groups'][group]['rows']:
                raise ValueError('ZIP record count mismatch')
            for n, item in enumerate(records, 1):
                inp, p = item['input'], prov[item['response_id']]
                r = dict(id=f'zip:{group}:{n}', source='zip', locator=f'{archives[0].name}!{group}/training.jsonl:{n}',
                         raw=item, provenance=p, source_ids=sorted(set([item['response_id'], item['source_response_id'], *p['source_response_ids']])),
                         attempt=item['split_group'], source_file=p.get('source_file'), question_id=inp['question_id'], text=inp.get('answer_text'),
                         label_origin='ai_derived', generation_kind=item['generation_kind'], reference_origin=item['reference_origin'])
                r.update(category='unscored', reason='missing_teacher_rubric_Q9_Q10')
                if item['reference_origin'] == 'teacher_supplied':
                    try:
                        if item['split_group'] != p['source_attempt_id'] or item['source_response_id'] != p['source_response_id'] or inp['question_id'] != p['question_id']:
                            raise ValueError('inconsistent_source_lineage')
                        definition = _definition(inp)
                        if inp['question_id'] in definitions and definitions[inp['question_id']] != definition:
                            raise ValueError('conflicting_teacher_definition')
                        definitions[inp['question_id']] = definition
                        r['definition'] = definition
                        r['score'] = str(_decimal(item['target']['score']))
                        _validate(r)
                    except (ValueError, TypeError, KeyError) as e:
                        r.update(category='unscored', reason=str(e))
                rows.append(r)
        if len(rows) != manifest['total_rows']:
            raise ValueError('ZIP total mismatch')
    w = openpyxl.load_workbook(books[0], read_only=True, data_only=True)
    sheets = {}
    try:
        for s in w:
            it = iter(s.values)
            header = next(it, ())
            sheets[s.title] = [(n, {str(k): v for k, v in zip(header, values) if k is not None})
                               for n, values in enumerate(it, 2) if any(v is not None for v in values)]
    finally:
        w.close()
    review = {d.get('response_id'): d for _, d in sheets.get('krugrade-human-review success31', [])}
    # Prefer the prepared sheet as a representation, never discard other rows.
    seen = {}
    for sheet in sorted(sheets, key=lambda s: (s != 'usable-319', s)):
        for n, d in sheets[sheet]:
            rid = d.get('response_id')
            is_review = sheet == 'krugrade-human-review success31'
            r = dict(id=f'untitled:{sheet}:{n}', source='untitled', locator=f'{books[0].name}!{sheet}:{n}',
                     raw=d, provenance=review.get(rid), source_ids=[str(rid)] if rid else [],
                     attempt=d.get('attempt_id'), source_file=(review.get(rid) or {}).get('source_pdf'),
                     question_id=d.get('question_id') or d.get('corrected_question_id') or d.get('original_question_id') or '',
                     text=d.get('answer_text') if not is_review else (d.get('corrected_answer') or d.get('original_answer')),
                     label_origin='unverified', generation_kind='review_log' if is_review else 'source_response',
                     reference_origin='unconfirmed', category='unscored', reason='unverified_label_or_missing_rubric_context')
            if rid and rid in seen:
                r.update(category='quarantined', reason='alternate_representation_not_additional_training_example', canonical_id=seen[rid])
            elif not is_review:
                try:
                    if rid in FLAGS:
                        raise ValueError(FLAGS[rid])
                    parent = r['question_id'].split('.')[0]
                    q, rubric = definitions[parent]
                    criteria = json.loads(d['rubric_criteria'])
                    declared = {c['criterion_id']: (c['description']+'\npartial_credit: '+c['partial_credit'], _decimal(c['points'])) for c in criteria}
                    expected = {c.id: (c.description, c.max_score) for c in rubric.criteria}
                    if not declared or any(expected.get(k) != v for k, v in declared.items()):
                        raise ValueError('teacher_criterion_subset_mismatch')
                    from krugradeai.models import Question, Rubric
                    if set(declared) != set(expected):
                        q = Question(id=r['question_id'], text=q.text+'\nScore only source scope '+r['question_id']+' using the supplied criterion subset.', reference_solution=q.reference_solution)
                        rubric = Rubric(id=r['question_id']+'@teacher-subset', criteria=tuple(c for c in rubric.criteria if c.id in declared), total_score=sum(v[1] for v in declared.values()))
                    if _decimal(d['rubric_max_score']) != rubric.total_score:
                        raise ValueError('rubric_maximum_mismatch')
                    r.update(definition=(q, rubric), score=str(_decimal(d['expected_score'])), reference_origin='teacher_supplied_criterion_match')
                    evidence = review.get(rid)
                    if evidence and (evidence['review_status'] not in ('approved', 'corrected') or evidence['stale'] is not False):
                        raise ValueError('deferred_excluded_or_stale_review')
                    if evidence and _decimal(evidence['w']) != _decimal(r['score']):
                        raise ValueError('review_score_mismatch')
                    if evidence:
                        answer = evidence.get('corrected_answer') or evidence.get('original_answer')
                        if normalize(answer) != normalize(r['text']):
                            raise ValueError('review_answer_mismatch')
                    _validate(r)
                except (ValueError, TypeError, KeyError) as e:
                    r.update(category='unscored', reason=str(e))
            if rid:
                seen.setdefault(rid, r['id'])
            rows.append(r)
    for r in rows:
        if r['category'] == 'unscored' and (not r.get('text') or not str(r['text']).strip()):
            r.update(category='quarantined', reason='missing_answer_text; '+r['reason'])
        r['inference_ready'] = bool(r['category']=='unscored' and r.get('text') and r.get('question_id') and
                                    (r.get('definition') or r['raw'].get('question_text') or r['raw'].get('input',{}).get('question_text')))
    edges = connect(rows)
    assign(rows)
    out.mkdir(parents=True)
    for split in ('development', 'validation', 'test'):
        selected = [record(r, split) for r in rows if r['partition'] == split]
        (out/(split+'.jsonl')).write_text(''.join(r.model_dump_json()+'\n' for r in selected), encoding='utf-8')
    serial = [{k: v for k, v in r.items() if k != 'definition'} for r in rows]
    (out/'master.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, default=str)+'\n' for r in serial), encoding='utf-8')
    (out/'source-documentation.json').write_text(canonical(documents), encoding='utf-8')
    (out/'relationships.json').write_text(canonical(edges), encoding='utf-8')
    cross = [e for e in edges if e['left'].startswith('zip:') != e['right'].startswith('zip:')]
    summary = dict(input_sha256=inputs, raw_zip=sum(r['source']=='zip' for r in rows),
                   raw_untitled=len(rows)-manifest['total_rows'], total=len(rows),
                   workbook_sheets={s:len(rs) for s,rs in sheets.items()},
                   unique_untitled_response_ids=len(seen), cross_source_links=len(cross),
                   cross_source_links_by_kind=dict(Counter(e['relation'] for e in cross)),
                   supervised=sum(r['category']=='supervised' for r in rows), human_ground_truth=0,
                   unscored_inference_ready=sum(r['inference_ready'] for r in rows),
                   unscored_requires_context_repair=sum(r['category']=='unscored' and not r['inference_ready'] for r in rows),
                   assignments=dict(Counter(r['partition'] for r in rows)),
                   by_source_partition=dict(Counter(r['source']+'/'+r['partition'] for r in rows)),
                   by_question_provenance_partition=dict(Counter(r['question_id']+'/'+r['label_origin']+'/'+r['partition'] for r in rows)),
                   reasons=dict(Counter(r['reason'] for r in rows)),
                   limitations=['No independently authenticated human reviewer: Validation empty; live selection blocked.',
                                'Unverified labels are supervision only, never human ground truth.',
                                'Q9/Q10 AI reference records remain unscored.',
                                'Exact normalized text and declared ancestry checked; unknown semantic relationships cannot be ruled out.',
                   'Physical rows include alternate representations, not independent observations.'])
    assert sum(summary['assignments'].values()) == summary['total']
    for p, h in inputs.items():
        if digest(Path(p).read_bytes()) != h:
            raise ValueError('Source changed')
    (out/'audit.json').write_text(canonical(summary), encoding='utf-8')
    return summary


def _validate(r):
    text = _answer(r['text'])
    if _marked(text):
        raise ValueError('unresolved_transcription_marker')
    record(r, 'development')  # typed score/rubric/schema bounds validation
    r.update(category='supervised', reason='schema_valid_supervision')


def record(r, split):
    q, rubric = r['definition']
    cls = HeldoutRecord if split == 'test' else DevelopmentRecord
    return cls(id=r['id'], question=q, rubric=rubric,
               response=StudentResponse(id=r['id'], question_id=q.id, text=_answer(r['text'])),
               ta_output=TAReference(total_score=r['score']),
               provenance=RecordProvenance(source=r['source'], item_id=r['id'], label_origin=r['label_origin'],
                   split_group=r.get('split_group'), source_ids=tuple(r['source_ids']),
                   generation_kind=r['generation_kind'], reference_origin=r['reference_origin']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(canonical(prepare(args.raw_dir, args.output_dir)))


if __name__ == '__main__':
    main()
