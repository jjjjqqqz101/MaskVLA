"""Video HTTP/API contracts using isolated byte fixtures, without robot/models."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from http.server import ThreadingHTTPServer

from task_agent.web import Console,Handler


class QuietHandler(Handler):
    def log_message(self,*args):
        pass


class TaskVideoHTTPContracts(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.console=Console(self.root/'runs')
        self.server=ThreadingHTTPServer(('127.0.0.1',0),QuietHandler)
        self.server.console=self.console
        self.thread=threading.Thread(target=self.server.serve_forever,
                                     kwargs={'poll_interval':.01},daemon=True)
        self.thread.start()
        self.base='http://127.0.0.1:'+str(self.server.server_port)
        # These are transport-only bytes, not a purported decodable recording.
        # Real recorded MP4 decode validation is performed separately.
        self.bytes=b'\x00\x00\x00\x18ftypmp42'+bytes(range(256))*8

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
        self.temporary.cleanup()

    def fixture(self,name,status='completed',content=None):
        run_id='task_agent_video_'+name
        directory=self.console.runs_root/run_id;directory.mkdir()
        (directory/'summary.json').write_text(json.dumps(dict(
            status=status,success=status=='completed',instruction='transport fixture')))
        if content is not None:(directory/'execution.mp4').write_bytes(content)
        return run_id,directory

    def request(self,path,headers=None):
        with urlopen(Request(self.base+path,headers=headers or {}),timeout=3) as response:
            return response.status,dict(response.headers),response.read()

    def snapshot(self,run_id):
        status,_,body=self.request('/api/status?run_id='+run_id)
        self.assertEqual(status,200)
        return json.loads(body)['run']

    def test_terminal_outcomes_retain_playback_and_explicit_download(self):
        for outcome in ('completed','failed','retry_budget_exhausted','stopped','error','aborted','timeout'):
            with self.subTest(outcome=outcome):
                run_id,directory=self.fixture(outcome,status=outcome,content=self.bytes)
                run=self.snapshot(run_id)
                video='/artifacts/'+run_id+'/execution.mp4'
                self.assertTrue(run['finished']);self.assertFalse(run['busy'])
                self.assertEqual(run['artifacts']['video'],video)
                self.assertEqual(run['artifacts']['video_download'],video+'?download=1')
                status,headers,body=self.request(video)
                self.assertEqual(status,200);self.assertEqual(headers['Content-Type'],'video/mp4')
                self.assertEqual(body,self.bytes)
                self.assertFalse(headers.get('Content-Disposition','').lower().startswith('attachment'))
                status,headers,body=self.request(run['artifacts']['video_download'])
                self.assertEqual(status,200);self.assertEqual(body,self.bytes)
                self.assertEqual(headers['Content-Disposition'],f'attachment; filename="{run_id}.mp4"')
                self.assertEqual((directory/'execution.mp4').read_bytes(),self.bytes)

    def test_active_video_is_not_advertised_until_finished(self):
        run_id,directory=self.fixture('active',status='executing',content=self.bytes)
        self.console.active_run_id=run_id
        self.console.process=SimpleNamespace(poll=lambda:None)
        run=self.snapshot(run_id)
        self.assertTrue(run['busy']);self.assertFalse(run['finished'])
        self.assertNotIn('video',run['artifacts']);self.assertNotIn('video_download',run['artifacts'])
        self.assertEqual((directory/'execution.mp4').read_bytes(),self.bytes)

    def test_missing_or_empty_video_is_not_advertised_ready(self):
        for name,content in (('missing',None),('empty',b'')):
            with self.subTest(name=name):
                run_id,directory=self.fixture(name,status='stopped',content=content)
                run=self.snapshot(run_id)
                self.assertTrue(run['finished'])
                self.assertNotIn('video',run['artifacts']);self.assertNotIn('video_download',run['artifacts'])
                if content is not None:
                    self.assertTrue((directory/'execution.mp4').is_file())
                    self.assertEqual((directory/'execution.mp4').stat().st_size,0)

    def test_playback_and_download_support_identical_byte_ranges(self):
        run_id,_=self.fixture('ranges',content=self.bytes)
        video='/artifacts/'+run_id+'/execution.mp4';size=len(self.bytes)
        ranges=[('bytes=0-31',0,31),('bytes=128-255',128,255),
                ('bytes=-24',size-24,size-1),(f'bytes={size-24}-',size-24,size-1)]
        for suffix in ('','?download=1'):
            for value,start,end in ranges:
                with self.subTest(download=bool(suffix),range=value):
                    status,headers,body=self.request(video+suffix,{'Range':value})
                    self.assertEqual(status,206)
                    self.assertEqual(headers['Content-Range'],f'bytes {start}-{end}/{size}')
                    self.assertEqual(headers['Content-Length'],str(end-start+1))
                    self.assertEqual(headers['Accept-Ranges'],'bytes')
                    self.assertEqual(headers['Content-Type'],'video/mp4')
                    self.assertEqual(body,self.bytes[start:end+1])
                    if suffix:self.assertEqual(headers['Content-Disposition'],f'attachment; filename="{run_id}.mp4"')

    def test_download_does_not_bypass_symlink_file_protection(self):
        run_id,directory=self.fixture('symlink',status='stopped')
        private=self.root/'private.mp4';private.write_bytes(self.bytes)
        (directory/'execution.mp4').symlink_to(private)
        run=self.snapshot(run_id)
        self.assertNotIn('video',run['artifacts']);self.assertNotIn('video_download',run['artifacts'])
        for suffix in ('','?download=1'):
            with self.subTest(download=bool(suffix)):
                with self.assertRaises(HTTPError) as response:
                    self.request('/artifacts/'+run_id+'/execution.mp4'+suffix)
                self.assertEqual(response.exception.code,400)


if __name__=='__main__':unittest.main()
