"""Original WAV preparation and bounded deletion of obsolete YO data."""
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from fireredtts3_local.dialogs_only import audit_prepared, find_original_plan, prepare
from fireredtts3_local.cleanup_yo import candidates, check_protected, cleanup
from fireredtts3_local.training_data import file_digest, load_manifest
import test_dialogs_dataset as fixtures


class DialogsOnlyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DialogsTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.plan, _ = self.fixture.plan()
        plan = json.loads(self.plan.read_text())
        # Small genuine WAV fixture within the production upper budgets.
        plan['report'].update(max_seconds=10., train_minutes_per_speaker=30.,
                              validation_minutes_per_speaker=10/3)
        self.plan.write_text(json.dumps(plan))
        source_recipe = Path(__file__).resolve().parents[1]/'recipes/russian_dialogs_base_96gb.json'
        config = json.loads(source_recipe.read_text())
        config['output_dir'] = str(self.root/'training/outputs/new-dialogs')
        config['options']['cache_dir'] = str(self.root/'.runtime/training-cache-dialogs-base-08')
        self.recipe = self.root/'template.json'
        self.recipe.write_text(json.dumps(config))
        self.output = self.root/'training/datasets/dialogs-base-08'
        (self.root/'fireredtts3_local').mkdir()

    def prepared(self):
        return prepare(self.plan, self.output, self.recipe, log=lambda _: None)

    def test_reimports_full_original_audio_without_yo_and_preserves_thirds(self):
        report = self.prepared()
        checked, config, rows = audit_prepared(self.output)
        self.assertEqual(report, checked)
        self.assertIsNone(config['initial_adapter'])
        self.assertIsNone(config['resume_checkpoint'])
        self.assertEqual(config['options']['replay_manifest'], '')
        self.assertEqual(config['options']['validation_scope'], 'recording')
        self.assertTrue(config['options']['validation_all_targets'])
        for row in rows:
            # Fixture source is already 24 kHz PCM16: preserve every sample,
            # including audible ends (no ASR/VAD/alignment trim is run).
            self.assertEqual(Path(row['audio']).read_bytes(),
                             (self.fixture.source/row['audio_path']).read_bytes())
            self.assertNotIn('dialogs_tail_review', row)
        for split in ('train', 'validation'):
            for stats in report['datasets'][split]['actors'].values():
                self.assertEqual(set(stats['slots'].values()), {stats['clips']//3})
        with self.assertRaisesRegex(ValueError, 'must be new'):
            self.prepared()
        Path(rows[0]['audio']).write_bytes(Path(rows[1]['audio']).read_bytes())
        with self.assertRaises(ValueError):
            audit_prepared(self.output)

    def test_lookup_uses_original_import_hash_not_latest_reviewed_manifest(self):
        folder = self.root/'training/plans/dialogs-neutral-v2'
        folder.mkdir(parents=True)
        (folder/'plan.json').write_bytes(self.plan.read_bytes())
        previous = self.root/'training/datasets/dialogs-neutral'
        previous.mkdir(parents=True)
        (previous/'import_report.json').write_text(json.dumps(dict(plan_sha256=file_digest(self.plan))))
        cwd = Path.cwd()
        try:
            os.chdir(self.root)
            self.assertEqual(find_original_plan().resolve(), (folder/'plan.json').resolve())
        finally:
            os.chdir(cwd)

    def test_cleanup_dry_run_then_delete_preserves_dialogs_models_and_other_hub_data(self):
        self.prepared()
        yo = self.root/'training/raw/yo'
        yo.mkdir(parents=True)
        (yo/'part.parquet').write_bytes(b'old YO shard')
        cache = self.root/'.runtime/training-cache'
        cache.mkdir(parents=True)
        (cache/'cached.pt').write_bytes(b'old mixed features')
        hub = self.root/'hub'
        hub_yo = hub/'datasets--NCSpeech--YO-CPT-ru'
        hub_yo.mkdir(parents=True)
        (hub_yo/'blob').write_bytes(b'YO cache')
        protected = self.root/'models/base.safetensors'
        protected.parent.mkdir()
        protected.write_bytes(b'original model')
        other = hub/'models--CKermit--FireRedTTS3-Base'
        other.mkdir()
        (other/'blob').write_bytes(b'other model')
        # An interior link is removed, never followed by recursive deletion.
        (yo/'model-link').symlink_to(protected)
        report = cleanup(self.root, self.output, hubs=[hub])
        self.assertFalse(report['apply'])
        self.assertTrue((yo/'part.parquet').exists())
        with patch('fireredtts3_local.cleanup_yo.assert_no_direct_workers'):
            applied = cleanup(self.root, self.output, apply=True, hubs=[hub])
        self.assertTrue(applied['complete'])
        self.assertFalse(yo.exists())
        self.assertFalse(cache.exists())
        self.assertFalse(hub_yo.exists())
        self.assertEqual(protected.read_bytes(), b'original model')
        self.assertTrue((other/'blob').exists())
        audit_prepared(self.output)

    def test_cleanup_rejects_symlink_roots_overlap_and_active_job(self):
        self.prepared()
        yo = self.root/'training/raw/yo'
        yo.parent.mkdir(parents=True)
        yo.symlink_to(self.fixture.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'linked'):
            candidates(self.root, [])
        yo.unlink()
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            check_protected([self.output.parent], [self.output])
        lock = self.root/'.runtime/training-job-locks/job.lock'
        lock.parent.mkdir(parents=True)
        with lock.open('w') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, 'active job'):
                cleanup(self.root, self.output, apply=True, hubs=[])
        audit_prepared(self.output)


if __name__ == '__main__':
    unittest.main()
