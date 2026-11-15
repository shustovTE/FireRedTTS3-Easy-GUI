"""Real Parquet expansion: historical holdouts, safe cleanup and shard resume."""
from collections import Counter
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from fireredtts3_local.training_data import dataset_fingerprint, file_digest, write_manifest
from fireredtts3_local.yo_expansion import (REPO, REVISION, checked_rows, choose_shards,
    plan_expansion, remove_old, run_expansion, write_json)
from fireredtts3_local.yo_dataset import import_yo


class YoExpansionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.root/'raw'
        (self.source/'data').mkdir(parents=True)
        self.assets=self.root/'remote'; (self.assets/'data').mkdir(parents=True)
        self.serial=0
        def anchor(label):
            rows=[]
            for index in range(2):
                path=self.root/f'{label}-{index}.wav'
                sf.write(path,np.sin(np.arange(48000)*(.02+self.serial*.001))*.1,24000)
                self.serial+=1
                rows.append(dict(audio=str(path),audio_sha256=file_digest(path),text='Старый пример.',
                    speaker_id='yo:'+label,language='Russian',utt_id=f'yo:{label}-{index}:0',
                    segment_source_id=f'yo:{label}-{index}',source_id='YO-CPT-ru:'+label))
            path=write_manifest(self.root/f'{label}.jsonl',rows)
            loaded=checked_rows(path)
            return dict(path=str(path),sha256=file_digest(path),fingerprint=dataset_fingerprint(loaded))
        self.anchors={'train':[anchor('oldtrain')],'validation':[anchor('oldval')],'protected':[]}
        self.old=self.shard('data/ru000-000.parquet',[self.row('old-used','oldtrain')])
        shutil.copyfile(self.assets/self.old['path'],self.source/self.old['path'])
        self.new=[
            self.shard('data/ru001-000.parquet',[self.row('a1','newa',12),self.row('oldtrain-0','oldtrain')]),
            self.shard('data/ru001-001.parquet',[self.row('a2','newa',12),self.row('b1','newb')]),
            self.shard('data/ru001-002.parquet',[self.row('b2','newb'),self.row('a3','newa',12),
                self.row('bad-mos','newb',mos=3.),self.row('unfinished','newb',text='Привет, мир…')]),
            self.shard('data/ru001-003.parquet',[self.row('v1','oldval'),self.row('v2','oldval'),
                self.row('t1','oldtrain'),self.row('t2','oldtrain'),self.row('alone','singleton')]),
        ]
        self.plan=dict(schema=1,repo=REPO,revision=REVISION,source_dir=str(self.source),seed=1234,
            selected=self.new,old_parquet=[self.old],anchors=self.anchors,report={},
            filters=dict(segmentation='complete-utterance',min_mos=4.,max_hours=0.,max_speaker_minutes=.5))
        self.plan_path=self.root/'plan.json';write_json(self.plan_path,self.plan)

    def row(self,utt,speaker,duration=2.,mos=4.5,text='Привет, мир!'):
        self.serial+=1
        signal=np.sin(np.arange(round(duration*24000))*(.025+self.serial*.001))*.1
        buf=io.BytesIO();sf.write(buf,signal,24000,format='WAV')
        return dict(utt_id=utt,local_spk_id=speaker,global_spk_id=None,lang='ru',mos_score=mos,
            duration=duration,text='привет мир',text_denorm=text,
            text_alignment=f'привет(0.01,{duration/2:.3f}) мир({duration/2:.3f},{duration-.01:.3f})',
            audio={'bytes':buf.getvalue(),'path':utt+'.wav'})

    def shard(self,name,rows):
        path=self.assets/name
        pq.write_table(pa.Table.from_pylist(rows),path)
        return dict(path=name,size=path.stat().st_size,sha256=file_digest(path))

    def download(self,source,names,repo,revision,**kwargs):
        self.assertEqual((repo,revision),(REPO,REVISION))
        self.assertTrue(kwargs['authenticated'])
        for name in names:
            path=Path(source)/name;path.parent.mkdir(parents=True,exist_ok=True)
            if name.endswith('.parquet'):
                shutil.copyfile(self.assets/name,path)
            elif name=='shard_map.tsv':
                path.write_text('shard\thours\n'+''.join(f'{Path(r["path"]).stem}\t1.0\n' for r in [self.old]+self.new))
            else:path.write_text('fixture metadata')

    def execute(self,output,**kwargs):
        with patch('fireredtts3_local.yo_expansion.download_files',side_effect=self.download):
            return run_expansion(self.plan_path,self.root/output,log=lambda _:None,**kwargs)

    def test_stratified_selection_is_reproducible_new_and_half_of_all(self):
        catalog=[dict(path=f'data/ru{part:03d}-{index:03d}.parquet',size=1,sha256='a'*64)
                 for part,count in [(0,10),(1,8),(2,2)] for index in range(count)]
        excluded={r['path'] for r in catalog[:2]}
        chosen=choose_shards(catalog,excluded,.5,1234)
        self.assertEqual(chosen,choose_shards(list(reversed(catalog)),excluded,.5,1234))
        self.assertEqual(len(chosen),10)
        self.assertFalse(excluded & {r['path'] for r in chosen})
        self.assertEqual(set(Counter(Path(r['path']).stem.split('-')[0] for r in chosen)),{'ru000','ru001','ru002'})
        with self.assertRaisesRegex(ValueError,'only 18 remain'):
            choose_shards(catalog,excluded,1)

    def test_plan_requires_auth_uses_pinned_inventory_and_protects_manifests(self):
        catalog=[self.old]+self.new
        info=SimpleNamespace(sha=REVISION,siblings=[SimpleNamespace(rfilename=r['path'],size=r['size'],
             lfs=SimpleNamespace(sha256=r['sha256'])) for r in catalog])
        with patch('huggingface_hub.get_token',return_value=None),self.assertRaisesRegex(ValueError,'auth login'):
            plan_expansion(self.source,self.root/'unauthenticated')
        with patch('huggingface_hub.get_token',return_value='fixture-token'),patch('huggingface_hub.HfApi') as api, \
             patch('fireredtts3_local.yo_expansion.download_files',side_effect=self.download):
            api.return_value.dataset_info.return_value=info
            report=plan_expansion(self.source,self.root/'planned',train_anchors=[self.anchors['train'][0]['path']],
                validation_anchors=[self.anchors['validation'][0]['path']],protected_manifests=[])
            api.assert_called_once_with(token=True)
            api.return_value.dataset_info.assert_called_once_with(REPO,revision=REVISION,files_metadata=True)
        plan=json.loads((self.root/'planned'/'plan.json').read_text())
        self.assertNotIn('fixture-token',json.dumps(plan))
        self.assertEqual(report['selected_new_shards'],3)
        self.assertEqual(report['old_parquet_files'],1)
        self.assertNotIn(self.old['path'],{r['path'] for r in plan['selected']})
        self.assertTrue((self.source/self.old['path']).is_file())

    def test_old_cleanup_checks_all_files_and_wavs_and_is_idempotent(self):
        old=self.source/self.old['path']
        historical=checked_rows(self.anchors['train'][0]['path'])[0]['audio']
        sf.write(historical,np.zeros(48000),24000)
        with self.assertRaisesRegex(ValueError,'recorded WAV checksum'): remove_old(self.plan)
        self.assertTrue(old.exists())

    def test_old_cleanup_rejects_corrupt_parquet_or_overlap_before_delete(self):
        original=(self.source/self.old['path']).read_bytes()
        (self.source/self.old['path']).write_bytes(b'corrupt parquet')
        with self.assertRaisesRegex(ValueError,'Parquet size/checksum'): remove_old(self.plan)
        (self.source/self.old['path']).write_bytes(original)
        changed=dict(self.plan,selected=self.new+[self.old])
        with self.assertRaisesRegex(ValueError,'overlap'): remove_old(changed)
        extra=self.new[-1]
        shutil.copyfile(self.assets/extra['path'],self.source/extra['path'])
        (self.source/extra['path']).write_bytes(b'corrupt second candidate')
        changed=dict(self.plan,selected=self.new[:-1],old_parquet=[self.old,extra])
        with self.assertRaisesRegex(ValueError,'Parquet size/checksum'): remove_old(changed)
        self.assertTrue((self.source/self.old['path']).exists())
        report=remove_old(self.plan)
        self.assertEqual(report['deleted_files'],1)
        self.assertFalse((self.source/self.old['path']).exists())
        self.assertTrue(Path(checked_rows(self.anchors['train'][0]['path'])[0]['audio']).exists())
        self.assertEqual(remove_old(self.plan)['deleted_files'],0)

    def test_full_import_keeps_cross_shard_pairs_budgets_and_old_holdouts(self):
        report=self.execute('full',delete_old=True)
        train=checked_rows(self.root/'full'/'train.jsonl')
        validation=checked_rows(self.root/'full'/'validation.jsonl')
        rows=train+validation
        self.assertEqual(sum(r['speaker_id']=='yo:newa' for r in rows),2)
        self.assertEqual(sum(r['speaker_id']=='yo:newb' for r in rows),2)
        self.assertTrue(any(r['duration']>10 for r in rows))
        self.assertTrue(all(r in train for r in rows if r['speaker_id']=='yo:oldtrain'))
        self.assertTrue(all(r in validation for r in rows if r['speaker_id']=='yo:oldval'))
        self.assertEqual(report['rejections']['previously_used_utterance'],1)
        self.assertEqual(report['rejections']['speaker_budget'],1)
        self.assertEqual(report['rejections']['language_or_quality'],1)
        self.assertEqual(report['rejections']['incomplete_sentence_ending'],1)
        self.assertEqual(report['unused_extracted_audio'],1)
        self.assertEqual(len(list((self.root/'full'/'parts').rglob('*.wav'))),len(rows))
        with patch('fireredtts3_local.yo_expansion.download_files',side_effect=AssertionError('completed run')):
            self.assertEqual(run_expansion(self.plan_path,self.root/'full',log=lambda _:None),report)

    def test_resume_after_interrupted_shard_matches_uninterrupted_selection(self):
        calls=0
        def interrupted(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2: raise KeyboardInterrupt()
            return import_yo(*args,**kwargs)
        with patch('fireredtts3_local.yo_expansion.import_yo',side_effect=interrupted),self.assertRaises(KeyboardInterrupt):
            self.execute('paused')
        self.assertEqual(json.loads((self.root/'paused'/'expansion_state.json').read_text())['status'],'interrupted')
        with patch('fireredtts3_local.yo_expansion.import_yo',wraps=import_yo) as importer:
            self.execute('paused')
        self.assertEqual(importer.call_count,3)
        self.execute('full')
        for split in ['train','validation']:
            def signature(folder):
                return [(r['utt_id'],r['audio_sha256'],r['text']) for r in checked_rows(self.root/folder/f'{split}.jsonl')]
            self.assertEqual(signature('paused'),signature('full'))
        changed=dict(self.plan,seed=999);write_json(self.plan_path,changed)
        with self.assertRaisesRegex(ValueError,'different plan'):
            self.execute('paused')


if __name__ == '__main__':
    unittest.main()
