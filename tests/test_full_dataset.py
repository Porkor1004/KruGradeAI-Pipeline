"""Synthetic checks for exhaustive inventory, conservative groups and offline paths."""
import json
from pathlib import Path
import tempfile
import unittest
from krugradeai.full_dataset import connect, assign
from krugradeai.optimization_models import RecordProvenance, ValidationConfig
from krugradeai.synthetic import make_synthetic_dataset
from krugradeai.validation import prepare_validation
from krugradeai.run_optimization import run
from krugradeai.evaluation import run as evaluate
from krugradeai.schema_mock import SchemaMockProvider
from krugradeai.models import GraderInput
from krugradeai.full_dataset import prepare


class FullDatasetTests(unittest.TestCase):
    def test_full_loader_reconciles_all_representations_and_preserves_inputs(self):
        import zipfile
        import hashlib
        from test_real_data import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            raw=root/'raw'
            raw.mkdir()
            fixture(raw)
            with zipfile.ZipFile(raw/'synthetic.zip') as z:
                members={n:z.read(n) for n in z.namelist() if not n.endswith('manifest.json')}
            members['synthetic/README.md']=b'Complete synthetic documentation. No human verification.'
            groups={g:dict(rows=len(members['synthetic/'+g+'/training.jsonl'].splitlines())) for g in ('01_source_reviewed','02_ai_augmented')}
            manifest=dict(groups=groups,total_rows=sum(g['rows'] for g in groups.values()),files_sha256={n.removeprefix('synthetic/'):hashlib.sha256(v).hexdigest() for n,v in members.items()})
            with zipfile.ZipFile(raw/'training-groups-signal320.zip','w') as z:
                for n,v in members.items(): z.writestr(n,v)
                z.writestr('synthetic/manifest.json',json.dumps(manifest))
            (raw/'synthetic.xlsx').rename(raw/'Untitled synthetic.xlsx')
            before={p.name:p.read_bytes() for p in raw.iterdir()}
            first=prepare(raw,root/'prepared-a')
            second=prepare(raw,root/'prepared-b')
            self.assertEqual(first,second)
            self.assertEqual(first['raw_zip'],64)
            self.assertEqual(first['total'],sum(first['assignments'].values()))
            self.assertGreater(first['raw_untitled'],0)
            self.assertEqual(first['human_ground_truth'],0)
            self.assertEqual(before,{p.name:p.read_bytes() for p in raw.iterdir()})
            self.assertEqual((root/'prepared-a/master.jsonl').read_bytes(),(root/'prepared-b/master.jsonl').read_bytes())
            with self.assertRaises(FileExistsError): prepare(raw,root/'prepared-a')

    def row(self, id, source='zip', **overrides):
        return dict(id=id, source=source, source_ids=[id], attempt=id,
                    text=id, question_id='q', category='supervised', label_origin='ai_derived', **overrides)

    def test_cross_source_ancestry_and_exact_text_are_transitive(self):
        rows = [self.row('a'), self.row('b', 'untitled'), self.row('c', 'untitled')]
        rows[1]['source_ids'] += ['a']
        rows[2]['text'] = ' B '
        edges = connect(rows)
        self.assertEqual(len({r['split_group'] for r in rows}), 1)
        self.assertEqual({e['relation'] for e in edges}, {'response', 'text'})

    def test_every_eligible_record_assigned_without_historical_cap(self):
        rows = [self.row(str(i), 'zip' if i<250 else 'untitled') for i in range(400)]
        connect(rows)
        assign(rows)
        self.assertEqual(len(rows), 400)
        self.assertTrue(all(r['partition'] in ('development', 'test') for r in rows))
        self.assertNotIn('validation', {r['partition'] for r in rows})
        self.assertEqual(sum(r['partition']=='test' for r in rows),60)

    def test_unscored_and_quarantine_remain_accounted(self):
        rows = [self.row('a'), self.row('b'), self.row('c')]
        rows[1].update(category='unscored', reason='missing_teacher_rubric')
        rows[2].update(category='quarantined', reason='missing_answer')
        connect(rows)
        assign(rows)
        self.assertEqual(rows[1]['partition'], 'unscored')
        self.assertEqual(rows[2]['partition'], 'quarantined')
        self.assertEqual(len(rows),3)

    def test_group_guard_rejects_distinct_ids_in_same_component(self):
        data = make_synthetic_dataset()
        def grouped(r):
            return r.model_copy(update={'provenance': RecordProvenance(source='s',item_id=r.id,split_group='same')})
        with self.assertRaisesRegex(ValueError,'split groups'):
            prepare_validation(tuple(grouped(r) for r in data.development), tuple(grouped(r) for r in data.validation), ValidationConfig())

    def test_schema_mock_never_accepts_labels_in_grader_input(self):
        with self.assertRaisesRegex(ValueError,'leaked'):
            SchemaMockProvider('grader').generate('mock\nINPUT_JSON\n'+json.dumps({'ta_output':{}}))

    def test_both_runners_accept_unverified_origin_and_mock_only(self):
        data=make_synthetic_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[r.model_copy(update={'provenance':RecordProvenance(source='private-test',item_id=r.id,label_origin='unverified',split_group=r.id)}) for r in data.development]
            (root/'dev.jsonl').write_text(''.join(r.model_dump_json()+'\n' for r in rows))
            cfg=dict(development='dev.jsonl',output_directory='opt',runtime='schema_mock',max_iterations=1,max_calls=5)
            (root/'opt.json').write_text(json.dumps(cfg))
            result=run(root/'opt.json')
            self.assertEqual(result['records'],2)
            self.assertEqual(result['paths'][0]['stop_reason'],'human_review')
            (root/'rubric.json').write_text(rows[0].rubric.model_dump_json())
            (root/'prompt.txt').write_text('Offline script only')
            (root/'eval.jsonl').write_text(''.join(json.dumps(dict(r.model_dump(mode='json'),split='evaluation'))+'\n' for r in rows))
            provider=SchemaMockProvider('grader')
            responses=[json.loads(provider.generate('mock\nINPUT_JSON\n'+GraderInput(question=r.question,rubric=r.rubric,response=r.response).model_dump_json())) for r in rows]
            (root/'responses.json').write_text(json.dumps(responses))
            cfg=dict(dataset='eval.jsonl',rubric_file='rubric.json',prompt_file='prompt.txt',output_directory='eval',runtime='mock',mock_responses='responses.json')
            (root/'eval.json').write_text(json.dumps(cfg))
            result=evaluate(root/'eval.json')
            self.assertEqual(result['metrics_by_label_origin']['human_derived']['expected_count'],0)
            self.assertEqual(result['metrics_by_label_origin']['unverified']['expected_count'],2)
            self.assertEqual(result['provider_calls'],2)


if __name__=='__main__':
    unittest.main()
