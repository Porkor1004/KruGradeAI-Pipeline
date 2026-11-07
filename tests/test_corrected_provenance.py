"""Synthetic tests: score authorship, OCR review, representations and all splits."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from krugradeai.corrected_dataset import zip_origins, base, link, split_rows, verify, prepare, remove_representations, reviewed_score
from krugradeai.optimization_models import RecordProvenance, ValidationRecord, ValidationConfig
from krugradeai.synthetic import make_synthetic_dataset
from krugradeai.validation import prepare_validation


class CorrectedProvenanceTests(unittest.TestCase):
    def test_approved_original_score_retained_when_w_is_blank(self):
        self.assertEqual(reviewed_score(dict(review_status='approved',w=None,original_score=2)),(2,'original_score'))
        self.assertEqual(reviewed_score(dict(review_status='corrected',w=None,original_score=2)),(None,'w'))
        self.assertEqual(reviewed_score(dict(review_status='corrected',w=0,original_score=2)),(0,'w'))
    def test_ai_ocr_does_not_imply_ai_score(self):
        item=dict(generation_kind='source_reviewed',label_source='teacher_derived',target={'score':'2'})
        self.assertEqual(zip_origins(item,dict(teacher_score_before=2,training_score='2')),('original_student','original_ta'))

    def test_equal_numeric_ai_target_remains_ai_generated(self):
        item=dict(generation_kind='inferred_spans',label_source='ai_generated',target={'score':'2'})
        self.assertEqual(zip_origins(item,dict(teacher_score_before=2,training_score=2)),('ai_augmented','ai_generated'))

    def test_synthetic_never_inherits_original_ta_by_numeric_agreement(self):
        item=dict(generation_kind='synthetic_full_response',label_source='teacher_derived',target={'score':'2'})
        self.assertEqual(zip_origins(item,dict(teacher_score_before=2,training_score=2)),('fully_synthetic','unknown'))

    def test_missing_and_mismatched_original_scores_not_invented(self):
        item=dict(generation_kind='source_reviewed',label_source='teacher_derived',target={'score':None})
        self.assertEqual(zip_origins(item,{}),('original_student','missing'))
        item['target']['score']='2'
        self.assertEqual(zip_origins(item,dict(teacher_score_before=1,training_score=2)),('original_student','unknown'))

    def test_model_rejects_conflated_label_origin_and_benchmark(self):
        with self.assertRaises(ValueError): RecordProvenance(source='x',item_id='a',label_origin='human_derived',score_origin='ai_generated')
        with self.assertRaises(ValueError): RecordProvenance(source='x',item_id='a',source_dataset='zip',benchmark_eligible=True,score_origin='ai_generated')

    def test_common_boolean_answers_do_not_collapse_students(self):
        rows=[base('a','untitled','a',{},'a','q','True'),base('b','untitled','b',{},'b','q','True')]
        for r in rows: r['source_ids']=[r['response_id']]
        edges,candidates=link(rows)
        self.assertFalse(edges)
        self.assertNotEqual(rows[0]['split_group'],rows[1]['split_group'])
        self.assertEqual(candidates[0]['decision'],'common_short_answer_not_identity_evidence')

    def test_cross_source_original_ancestry_and_variants_stay_together(self):
        rows=[base('a','zip','a',{},'a','q','Long text '*20),base('b','untitled','b',{},'b','q','different')]
        rows[0]['source_ids']=['original-response'];rows[1]['source_ids']=['original-response']
        edges,_=link(rows)
        self.assertEqual(rows[0]['split_group'],rows[1]['split_group'])
        self.assertEqual(edges[0]['relation'],'response')
        for r in rows: r.update(category='supervised',definition=(make_synthetic_dataset().development[0].question,make_synthetic_dataset().development[0].rubric),benchmark_eligible=r['source_dataset']=='untitled')
        split_rows(rows)
        self.assertEqual(rows[0]['partition'],rows[1]['partition'])
        self.assertNotEqual(rows[0]['partition'],'validation')
        verify(rows)

    def test_overlapping_parent_subparts_are_counted_once(self):
        model=make_synthetic_dataset().development[0]
        rows=[base('a','untitled','a',{},'a','q','answer'),base('b','untitled','b',{},'b','q','same answer')]
        for r in rows:r.update(category='supervised',attempt='same',definition=(model.question,model.rubric))
        remove_representations(rows)
        self.assertEqual(sum(r['category']=='supervised' for r in rows),1)
        self.assertEqual(rows[1]['canonical_ids'],['a'])

    def test_generic_validation_rejects_explicit_ai_ocr(self):
        data=make_synthetic_dataset()
        r=data.validation[0]
        p=RecordProvenance(source='zip',item_id=r.id,source_dataset='zip',answer_origin='original_student',score_origin='original_ta',ocr_review_origin='ai',rubric_origin='teacher_supplied',benchmark_eligible=False)
        with self.assertRaisesRegex(ValueError,'human-reviewed'):
            prepare_validation(data.development,(r.model_copy(update={'provenance':p}),),ValidationConfig())

    def test_evaluation_separates_ta_labels_by_ocr_and_from_ai_labels(self):
        from krugradeai.evaluation import run
        from krugradeai.schema_mock import SchemaMockProvider
        from krugradeai.models import GraderInput
        template=make_synthetic_dataset().development[0]
        records=[]
        for i,(score,ocr) in enumerate([('original_ta','human'),('original_ta','ai'),('ai_generated','ai')]):
            rid='cohort-'+str(i)
            p=RecordProvenance(source='test',item_id=rid,label_origin='human_derived' if score=='original_ta' else 'ai_derived',
                               score_origin=score,ocr_review_origin=ocr)
            records.append(template.model_copy(update={'id':rid,'response':template.response.model_copy(update={'id':rid}),'provenance':p}))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'dataset.jsonl').write_text(''.join(json.dumps(dict(r.model_dump(mode='json'),split='evaluation'))+'\n' for r in records))
            (root/'rubric.json').write_text(template.rubric.model_dump_json())
            (root/'prompt.txt').write_text('Offline mock test')
            provider=SchemaMockProvider('grader')
            responses=[json.loads(provider.generate('mock\nINPUT_JSON\n'+GraderInput(question=r.question,rubric=r.rubric,response=r.response).model_dump_json())) for r in records]
            (root/'responses.json').write_text(json.dumps(responses))
            cfg=dict(dataset='dataset.jsonl',rubric_file='rubric.json',prompt_file='prompt.txt',runtime='mock',mock_responses='responses.json',output_directory='result')
            (root/'config.json').write_text(json.dumps(cfg))
            result=run(root/'config.json')
            self.assertEqual(set(result['metrics_by_score_and_ocr']),{'original_ta/human','original_ta/ai','ai_generated/ai'})
            self.assertTrue(all(m['expected_count']==1 for m in result['metrics_by_score_and_ocr'].values()))

    def test_complete_loader_owner_basis_reconciles_and_preserves_sources(self):
        from test_real_data import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir();fixture(raw)
            with zipfile.ZipFile(raw/'synthetic.zip') as z:
                members={n:z.read(n) for n in z.namelist() if n!='synthetic/manifest.json'}
            groups={}
            for group in ('01_source_reviewed','02_ai_augmented'):
                items=[json.loads(s) for s in members['synthetic/'+group+'/training.jsonl'].splitlines()]
                ps=json.loads(members['synthetic/'+group+'/provenance.json'])
                for p,item in zip(ps,items):
                    p.update(label_source=item['label_source'],generation_kind=item['generation_kind'],training_score=item['target']['score'],teacher_score_before=item['target']['score'])
                members['synthetic/'+group+'/provenance.json']=json.dumps(ps).encode()
                groups[group]=dict(rows=len(items))
            members['synthetic/README.md']=b'Complete synthetic instructions: original TA scores, AI OCR; generated scores explicit.'
            manifest=dict(total_rows=sum(v['rows'] for v in groups.values()),groups=groups,files_sha256={n.removeprefix('synthetic/'):hashlib.sha256(v).hexdigest() for n,v in members.items()})
            with zipfile.ZipFile(raw/'training-groups-signal320.zip','w') as z:
                for n,v in members.items():z.writestr(n,v)
                z.writestr('synthetic/manifest.json',json.dumps(manifest))
            (raw/'synthetic.xlsx').rename(raw/'Untitled synthetic.xlsx')
            before={p.name:p.read_bytes() for p in raw.iterdir()}
            summary=prepare(raw,root/'prepared')
            self.assertEqual(summary['total'],sum(summary['assignments'].values()))
            self.assertEqual(summary['source_rows']['zip'],64)
            self.assertGreater(summary['assignments'].get('validation',0),0)
            self.assertGreater(summary['benchmark_eligible'],0)
            self.assertEqual(before,{p.name:p.read_bytes() for p in raw.iterdir()})
            rs=[json.loads(s) for s in (root/'prepared/master.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['ocr_review_origin']=='human' and r['score_origin']=='original_ta' for r in rs if r['partition']=='validation'))
            with self.assertRaises(FileExistsError): prepare(raw,root/'prepared')


if __name__=='__main__':unittest.main()
