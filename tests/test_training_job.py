"""Real subprocess checks; no GPU or model download required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from fireredtts3_local.training_job import launch, read_status, run_commands


@unittest.skipUnless(os.name == 'posix', 'Linux/POSIX job runner')
class BackgroundJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_detached_worker_survives_launcher_exit_and_reports_real_failure(self):
        recipe = self.root/'recipe.json'
        recipe.write_text(json.dumps({'manifest_path': str(self.root/'missing.jsonl'),
                                      'output_dir': str(self.root/'output')}))
        job = self.root/'job'
        result = subprocess.run([sys.executable, '-m', 'fireredtts3_local.training_job',
                                 'start', str(recipe), '--job-dir', str(job)],
                                capture_output=True, text=True, timeout=15, check=True)
        started = json.loads(result.stdout)
        pid = started['pid']
        self.assertEqual(os.getsid(pid), pid)  # independent of launching terminal/session
        # The output lock is held across launcher exit and the child process.
        with self.assertRaisesRegex(ValueError, 'already owns'):
            launch(recipe, self.root/'duplicate')
        deadline = time.monotonic()+30
        while time.monotonic() < deadline:
            status = read_status(job)
            if not status['active']:
                break
            time.sleep(.05)
        self.assertFalse(status['active'])
        self.assertEqual(status['status'], 'failed')
        self.assertNotEqual(status['exit_code'], 0)
        self.assertIn('missing.jsonl', (job/'job.log').read_text())
        self.assertEqual(json.loads((job/'recipe.json').read_text()), json.loads(recipe.read_text()))

    def test_precompute_failure_does_not_start_training(self):
        marker = self.root/'trained'
        stages = [('precompute', [sys.executable, '-c', 'raise SystemExit(7)']),
                  ('train', [sys.executable, '-c', f'open({str(marker)!r}, "w").close()'])]
        result = run_commands(self.root, {}, stages, None)
        self.assertEqual(result, 7)
        self.assertFalse(marker.exists())
        status = json.loads((self.root/'status.json').read_text())
        self.assertEqual((status['status'], status['phase'], status['exit_code']), ('failed', 'precompute', 7))

    def test_stop_request_signals_child_and_waits_for_save(self):
        ready, saved = self.root/'ready', self.root/'saved'
        script = ('import signal\n'
                  'def stop(*args):\n'
                  f'    open({str(saved)!r}, "w").write("checkpoint")\n'
                  '    raise SystemExit(0)\n'
                  'signal.signal(signal.SIGTERM, stop)\n'
                  f'open({str(ready)!r}, "w").close()\n'
                  'signal.pause()\n')
        def request():
            deadline = time.monotonic()+5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            (self.root/'STOP').touch()
        thread = threading.Thread(target=request)
        thread.start()
        try:
            result = run_commands(self.root, {}, [('train', [sys.executable, '-c', script])], None)
        finally:
            thread.join(timeout=6)
        self.assertEqual(result, 0)
        self.assertEqual(saved.read_text(), 'checkpoint')
        self.assertEqual(json.loads((self.root/'status.json').read_text())['status'], 'stopped')
