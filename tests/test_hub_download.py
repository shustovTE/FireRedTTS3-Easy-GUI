"""Rate limiting, content-verified reuse and recovery without live Hub requests."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from huggingface_hub import constants
from huggingface_hub._local_folder import write_download_metadata

from fireredtts3_local.dialogs_dataset import REPO, REVISION
from fireredtts3_local.hub_download import download_files, retry_delay, verified_local


class HubDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.states = []

    def receipt(self, name, payload=b'fixture audio', revision=REVISION, git=False):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        digest = (hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
                  if git else hashlib.sha256(payload).hexdigest())
        write_download_metadata(self.root, name, revision, digest)
        return str(path)

    def run_download(self, names, **kwargs):
        return download_files(self.root, names, REPO, REVISION, interval=0,
                              log=lambda _:None, progress=self.states.append, **kwargs)

    def error(self, status, retry_after=None):
        response=httpx.Response(status,request=httpx.Request('GET','https://huggingface.co/example'),
                                headers={'Retry-After':retry_after} if retry_after else {})
        return httpx.HTTPStatusError('fixture HTTP error',request=response.request,response=response)

    def test_verified_lfs_and_git_files_reused_with_zero_requests(self):
        self.receipt('wavs/a.wav')
        self.receipt('train.csv',b'header|value\n',git=True)
        self.assertTrue(verified_local(self.root,'wavs/a.wav',REVISION))
        self.assertTrue(verified_local(self.root,'train.csv',REVISION))
        old=constants.HF_HUB_DISABLE_XET
        with patch('huggingface_hub.hf_hub_download',side_effect=AssertionError('no network')) as fetch:
            result=self.run_download(['wavs/a.wav','train.csv','wavs/a.wav'])
        fetch.assert_not_called()
        self.assertEqual((result['reused'],result['downloaded'],result['total']),(2,0,2))
        self.assertTrue(result['complete'])
        self.assertEqual(constants.HF_HUB_DISABLE_XET,old)

    def test_429_waits_before_retry_and_next_file_using_http(self):
        events=[]
        first=True
        old=constants.HF_HUB_DISABLE_XET
        def fetch(repo,name,**options):
            nonlocal first
            self.assertTrue(constants.HF_HUB_DISABLE_XET)
            self.assertEqual((repo,options['revision'],options['repo_type']),(REPO,REVISION,'dataset'))
            events.append(name)
            if first:
                first=False
                raise self.error(429,'125')
            return self.receipt(name)
        def wait(seconds):
            if seconds: events.append(seconds)
        with patch('huggingface_hub.hf_hub_download',side_effect=fetch), \
             patch('fireredtts3_local.hub_download.wait_seconds',side_effect=wait):
            result=self.run_download(['wavs/a.wav','wavs/b.wav'])
        self.assertEqual(events,['wavs/a.wav',125.,'wavs/a.wav','wavs/b.wav'])
        self.assertEqual((result['downloaded'],result['retries']),(2,1))
        self.assertTrue(any(s['status']=='waiting' and s['retry_wait_seconds']==125 for s in self.states))
        self.assertEqual(constants.HF_HUB_DISABLE_XET,old)

    def test_failed_run_resumes_only_missing_file_and_keeps_partial_transfer(self):
        def fetch(repo,name,**options):
            if name=='wavs/a.wav': return self.receipt(name)
            self.assertFalse(options['force_download'])
            raise httpx.ConnectError('interrupted transfer')
        with patch('huggingface_hub.hf_hub_download',side_effect=fetch), self.assertRaises(httpx.ConnectError):
            self.run_download(['wavs/a.wav','wavs/b.wav'],max_retries=0)
        self.assertEqual(self.states[-1]['status'],'failed')
        self.assertEqual(self.states[-1]['processed'],1)
        with patch('huggingface_hub.hf_hub_download',side_effect=lambda repo,name,**opts:self.receipt(name)) as fetch:
            result=self.run_download(['wavs/a.wav','wavs/b.wav'])
        self.assertEqual(fetch.call_count,1)
        self.assertEqual(fetch.call_args.args[1],'wavs/b.wav')
        self.assertFalse(fetch.call_args.kwargs['force_download'])
        self.assertEqual((result['reused'],result['downloaded']),(1,1))

    def test_interval_spaces_new_requests_and_wrapped_hub_errors_back_off(self):
        with patch('huggingface_hub.hf_hub_download',side_effect=lambda repo,name,**opts:self.receipt(name)), \
             patch('fireredtts3_local.hub_download.time.monotonic',side_effect=[100.,100.5,102.]), \
             patch('fireredtts3_local.hub_download.wait_seconds') as wait:
            download_files(self.root,['wavs/a.wav','wavs/b.wav'],REPO,REVISION,log=lambda _:None)
        wait.assert_called_once_with(1.5)
        for code, expected in [(429,125.),(403,None),(404,None)]:
            wrapped=ValueError('Force download failed due to the above error.')
            wrapped.__cause__=self.error(code,'125')
            self.assertEqual(retry_delay(wrapped,0),expected)
        from huggingface_hub.errors import LocalEntryNotFoundError
        wrapped=LocalEntryNotFoundError('HEAD failure')
        wrapped.__cause__=httpx.ConnectError('network')
        self.assertEqual(retry_delay(wrapped,0),10.)

    def test_corrupt_or_wrong_revision_cache_is_replaced(self):
        for name,revision in [('wavs/corrupt.wav',REVISION),('wavs/old.wav','0'*40)]:
            self.receipt(name,revision=revision)
        (self.root/'wavs/corrupt.wav').write_bytes(b'changed content')
        for name in ('wavs/corrupt.wav','wavs/old.wav'):
            self.assertFalse(verified_local(self.root,name,REVISION))
        def fetch(repo,name,**options):
            self.assertTrue(options['force_download'])
            return self.receipt(name)
        with patch('huggingface_hub.hf_hub_download',side_effect=fetch):
            result=self.run_download(['wavs/corrupt.wav','wavs/old.wav'])
        self.assertEqual(result['downloaded'],2)

    def test_bad_resumed_xet_content_is_retried_once_from_zero(self):
        forced=[]
        def fetch(repo,name,**options):
            forced.append(options['force_download'])
            result=self.receipt(name)
            if len(forced)==1:
                (self.root/name).write_bytes(b'old noncontiguous Xet reconstruction')
            return result
        with patch('huggingface_hub.hf_hub_download',side_effect=fetch), \
             patch('fireredtts3_local.hub_download.wait_seconds'):
            result=self.run_download(['wavs/a.wav'])
        self.assertEqual(forced,[False,True])
        self.assertEqual((result['downloaded'],result['retries']),(1,1))
        self.assertTrue(result['complete'])

    def test_permanent_failures_and_invalid_download_are_not_success(self):
        for code in (401,403,404):
            with self.subTest(status=code), \
                 patch('huggingface_hub.hf_hub_download',side_effect=self.error(code)) as fetch, \
                 patch('fireredtts3_local.hub_download.wait_seconds') as wait, self.assertRaises(httpx.HTTPStatusError):
                self.run_download(['wavs/a.wav'])
            self.assertEqual(fetch.call_count,1)
            wait.assert_not_called()
            self.assertFalse(self.states[-1]['complete'])
        def bad(repo,name,**options):
            self.receipt(name)
            (self.root/name).write_bytes(b'corrupt download')
        with patch('huggingface_hub.hf_hub_download',side_effect=bad), self.assertRaisesRegex(ValueError,'content verification'):
            self.run_download(['wavs/a.wav'])
        self.assertEqual(self.states[-1]['processed'],0)

    def test_retry_budget_retry_after_date_and_permanent_xet_status(self):
        with patch('huggingface_hub.hf_hub_download',side_effect=self.error(429)) as fetch, \
             patch('fireredtts3_local.hub_download.wait_seconds') as wait, self.assertRaises(httpx.HTTPStatusError):
            self.run_download(['wavs/a.wav'],max_retries=2)
        self.assertEqual(fetch.call_count,3)
        self.assertEqual([c.args[0] for c in wait.call_args_list if c.args[0]>0],[60.,120.])
        self.assertEqual(self.states[-1]['retries'],2)
        now=datetime(2026,11,15,tzinfo=timezone.utc)
        error=self.error(429,format_datetime(now+timedelta(seconds=700),usegmt=True))
        self.assertEqual(retry_delay(error,0,now),700.)
        self.assertEqual(retry_delay(httpx.ReadTimeout('timeout'),0),10.)
        self.assertIsNone(retry_delay(OSError('disk full'),0))
        self.assertEqual(retry_delay(ConnectionError('HTTP status client error (429 Too Many Requests)'),0),60.)
        self.assertIsNone(retry_delay(ConnectionError('HTTP status client error (403 Forbidden)'),0))

    def test_invalid_options_and_paths_fail_before_requests(self):
        with patch('huggingface_hub.hf_hub_download') as fetch:
            for name in ('../escape','/tmp/escape','wavs/../escape','wavs\\escape'):
                with self.subTest(name=name), self.assertRaises(ValueError): self.run_download([name])
            for retry in (-1,1.5,True):
                with self.subTest(retry=retry), self.assertRaises(ValueError): self.run_download([],max_retries=retry)
        fetch.assert_not_called()
