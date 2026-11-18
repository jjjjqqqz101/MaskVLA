"""Console and HTTP contracts, without robot simulation or external API calls."""
import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from agentic_act.web import Console, Handler, ThreadingHTTPServer


def payload():
    return dict(instruction='先把 cube1 放入 A，再把 cube2 放入 B。',
                steps=[dict(object_id='cube1', region_id='zone_A'),
                       dict(object_id='cube2', region_id='zone_B')],
                seed=2026092500, max_attempts=3,
                inject_first_failure=False, plan_only=False)


class FakeProcess:
    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.returncode = None

    def poll(self):
        return self.returncode


class ConsoleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.console = Console(self.root / 'runs', self.root / 'act')

    def tearDown(self):
        self.temporary.cleanup()

    def test_invalid_requests_never_launch_a_process(self):
        cases = []
        for key, value in [('seed', True), ('seed', -1), ('seed', 2**31),
                           ('max_attempts', True), ('max_attempts', 0), ('max_attempts', 6),
                           ('instruction', ''), ('steps', []),
                           ('inject_first_failure', 'false'), ('plan_only', 1)]:
            case = payload(); case[key] = value; cases.append(case)
        case = payload(); case['steps'][1]['object_id'] = 'cube1'; cases.append(case)
        case = payload(); case['steps'][1]['region_id'] = 'zone_A'; cases.append(case)
        case = payload(); case['steps'][0]['skill'] = 'anything'; cases.append(case)
        case = payload(); case['steps'][0]['object_id'] = 'unknown'; cases.append(case)
        case = payload(); case['checkpoint'] = '/unrequested/model.pt'; cases.append(case)
        for case in cases:
            with self.subTest(case=case), mock.patch('agentic_act.web.subprocess.Popen') as launch:
                with self.assertRaises(ValueError):
                    self.console.start(case)
                launch.assert_not_called()
        self.assertEqual(list(self.console.runs_root.iterdir()), [])

    def test_single_active_job_and_cooperative_stop(self):
        with mock.patch('agentic_act.web.subprocess.Popen', side_effect=FakeProcess) as launch:
            row = self.console.start(payload())
            self.assertTrue(self.console.busy())
            with self.assertRaises(RuntimeError):
                self.console.start(payload())
            self.assertEqual(launch.call_count, 1)
        process = self.console.process
        command = process.command
        self.assertIn('--web-output', command)
        self.assertIn('--stop-file', command)
        self.assertEqual(command[command.index('--order')+1:command.index('--seed')],
                         ['cube1:zone_A', 'cube2:zone_B'])
        directory = self.console.directory(row['id'])
        bootstrap = json.loads((directory/'web_request.json').read_text())
        self.assertEqual(bootstrap['task']['steps'][0]['object_id'], 'cube1')
        self.assertTrue(self.console.stop()['requested'])
        self.assertTrue((directory/'stop.request').is_file())
        self.assertEqual(self.console.status()['run']['status'], 'stopping')
        self.assertIsNone(process.returncode)

    def test_progress_then_terminal_summary_and_history(self):
        with mock.patch('agentic_act.web.subprocess.Popen', side_effect=FakeProcess):
            row = self.console.start(payload())
        directory = self.console.directory(row['id'])
        (directory/'summary.json').write_text(json.dumps(dict(status='judging', completed_steps=0)))
        (directory/'status.json').write_text(json.dumps(dict(status='executing', completed_steps=1,
              current_step=1, total_steps=2, attempts={'0': 1, '1': 1})))
        (directory/'backend_status.json').write_text(json.dumps(dict(step=300, phase='grasp_lift')))
        active = self.console.snapshot(row['id'])
        self.assertEqual(active['status'], 'executing')
        self.assertEqual(active['completed_steps'], 1)
        self.assertEqual(active['backend']['step'], 300)
        self.assertFalse(active['finished'])
        (directory/'summary.json').write_text(json.dumps(dict(status='completed', success=True,
              completed_steps=2, final_verification={'success':True})))
        self.console.process.returncode = 0
        terminal = self.console.snapshot(row['id'])
        self.assertEqual(terminal['status'], 'completed')
        self.assertTrue(terminal['success'])
        self.assertTrue(terminal['finished'])
        self.assertEqual(self.console.history()[0]['instruction'], payload()['instruction'])
        self.assertEqual(self.console.history()[0]['completed_steps'], 2)

    def test_crashed_worker_is_not_reported_as_running_or_successful(self):
        with mock.patch('agentic_act.web.subprocess.Popen', side_effect=FakeProcess):
            row = self.console.start(payload())
        self.console.process.returncode = 1
        snap = self.console.snapshot(row['id'])
        self.assertEqual(snap['status'], 'error')
        self.assertFalse(snap['success'])
        self.assertTrue(snap['finished'])
        self.assertTrue(snap['error'])

    def test_run_directory_traversal_and_external_symlinks_rejected(self):
        outside = self.root/'outside'; outside.mkdir()
        (self.console.runs_root/'agentic_act_link').symlink_to(outside, target_is_directory=True)
        for value in ['../outside', 'agentic_act_../outside', 'agentic_act_link']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.console.directory(value)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.console = Console(self.root/'runs', self.root/'act')
        self.directory = self.console.runs_root/'agentic_act_http_test'
        self.directory.mkdir()
        (self.directory/'execution.mp4').write_bytes(b'0123456789')
        (self.directory/'summary.json').write_text(json.dumps(dict(status='completed',success=True,
              task={'instruction':'fixture','steps':[]},completed_steps=2)))
        self.server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        self.server.console = self.console
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
        self.temporary.cleanup()

    def request(self, method, path, data=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=3)
        body = None if data is None else json.dumps(data).encode()
        request_headers = {'Content-Type':'application/json'}
        request_headers.update(headers or {})
        connection.request(method,path,body=body,headers=request_headers)
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_missing_csrf_and_untrusted_host_cannot_start_jobs(self):
        with mock.patch('agentic_act.web.subprocess.Popen') as launch:
            status, _, _ = self.request('POST','/api/start',payload())
            self.assertEqual(status,403)
            status, _, _ = self.request('POST','/api/start',payload(),
                {'Host':'untrusted.example','X-CSRF-Token':self.console.token})
            self.assertEqual(status,403)
            launch.assert_not_called()
        status, _, body = self.request('GET','/api/config')
        self.assertEqual(status,200)
        self.assertEqual(json.loads(body)['csrf_token'],self.console.token)
        self.assertNotIn('api_key',json.loads(body))

    def test_video_byte_ranges_and_invalid_ranges(self):
        route='/artifacts/agentic_act_http_test/execution.mp4'
        status, headers, body=self.request('GET',route,headers={'Range':'bytes=2-5'})
        self.assertEqual(status,206);self.assertEqual(body,b'2345')
        self.assertEqual(headers['Content-Range'],'bytes 2-5/10')
        self.assertEqual(headers['Content-Length'],'4')
        status, _, body=self.request('GET',route,headers={'Range':'bytes=-3'})
        self.assertEqual(status,206);self.assertEqual(body,b'789')
        status, _, _=self.request('GET',route,headers={'Range':'bytes=90-100'})
        self.assertEqual(status,416)

    def test_authorized_start_double_submission_and_stop_over_http(self):
        headers={'X-CSRF-Token':self.console.token}
        with mock.patch('agentic_act.web.subprocess.Popen', side_effect=FakeProcess) as launch:
            status, _, body=self.request('POST','/api/start',payload(),headers)
            self.assertEqual(status,202)
            run_id=json.loads(body)['id']
            status, _, _=self.request('POST','/api/start',payload(),headers)
            self.assertEqual(status,409)
            self.assertEqual(launch.call_count,1)
            status, _, body=self.request('POST','/api/stop',{},headers)
            self.assertEqual(status,200)
            self.assertTrue(json.loads(body)['requested'])
        self.assertTrue((self.console.directory(run_id)/'stop.request').is_file())

    def test_artifact_paths_cannot_disclose_private_files(self):
        secret=self.root/'private.json';secret.write_bytes(b'not-for-browser')
        (self.directory/'secret.png').symlink_to(secret)
        for path in ['/artifacts/agentic_act_http_test/secret.png',
                     '/artifacts/agentic_act_http_test/%2e%2e%2fprivate.json',
                     '/api/artifact?run=agentic_act_http_test&name=../private.json']:
            with self.subTest(path=path):
                status, _, body=self.request('GET',path)
                self.assertEqual(status,400)
                self.assertNotIn(b'not-for-browser',body)

    def test_frame_symlink_cannot_disclose_private_file(self):
        secret=self.root/'private.json';secret.write_bytes(b'not-for-browser')
        (self.directory/'live.png').symlink_to(secret)
        status, _, body=self.request('GET','/api/frame?run=agentic_act_http_test')
        self.assertIn(status,(400,404))
        self.assertNotIn(b'not-for-browser',body)

    def test_partial_event_line_does_not_break_status(self):
        (self.directory/'events.jsonl').write_text('{"event":"step_completed"}\n{"event":')
        status, _, body=self.request('GET','/api/run?id=agentic_act_http_test')
        self.assertEqual(status,200)
        self.assertEqual(json.loads(body)['events'],[{'event':'step_completed'}])


if __name__=='__main__':
    unittest.main()
