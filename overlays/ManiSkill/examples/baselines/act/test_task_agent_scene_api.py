"""Scene-before-prompt HTTP protocol tests with inert child-process doubles."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from task_agent.web import Console, Handler


class FakeProcess:
    def __init__(self): self.returncode = None
    def poll(self): return self.returncode


class ProcessFactory:
    def __init__(self):
        self.calls = []
        self.processes = []

    def __call__(self, command, **options):
        process = FakeProcess()
        self.calls.append((command, options))
        self.processes.append(process)
        return process


class QuietHandler(Handler):
    def log_message(self, *args): pass


class SceneAPIContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.factory = ProcessFactory()
        self.console = Console(Path(self.temp.name)/'runs', act_root=Path(self.temp.name),
                               process_factory=self.factory)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        self.server.console = self.console
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.01), daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:%d' % self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def get(self, route):
        with urlopen(self.url+route, timeout=3) as response:
            return json.load(response)

    def post(self, route, payload, token=True, origin=None):
        headers = {'Content-Type': 'application/json'}
        if token: headers['X-CSRF-Token'] = self.console.token
        if origin is not None: headers['Origin'] = origin
        request = Request(self.url+route, data=json.dumps(payload, allow_nan=True).encode(), headers=headers)
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def prepare(self, seed=17):
        code, response = self.post('/api/scene', dict(seed=seed))
        self.assertEqual(code, 202, response)
        scene = self.get('/api/status')['scene']
        self.assertIsInstance(scene, dict)
        self.assertTrue(scene['id'])
        return scene

    def folder(self, scene):
        return self.console.directory(scene['id'])

    def ready(self, scene, manifest_changes=None):
        folder = self.folder(scene)
        manifest = dict(scene_id=scene['id'], seed=scene['seed'], reset_count=1)
        manifest.update(manifest_changes or {})
        (folder/'scene.json').write_text(json.dumps(manifest))
        (folder/'status.json').write_text(json.dumps(dict(scene_id=scene['id'], seed=scene['seed'],
            status='scene_ready', phase='scene_ready', step_generation=0)))
        (folder/'live.png').write_bytes(b'prepared-camera-image')
        return self.get('/api/status')['scene']

    def task(self, scene, instruction='收拾球类', **extra):
        return dict(scene_id=scene['id'], instruction=instruction, **extra)

    def assert_conflict(self, route, payload):
        code, response = self.post(route, payload)
        self.assertEqual(code, 409, response)

    def test_preparation_creates_one_prompt_free_worker_and_publishes_frozen_seed(self):
        scene = self.prepare(17)
        self.assertEqual(scene['seed'], 17)
        self.assertFalse(scene['ready'])
        self.assertFalse(scene['submitted'])
        self.assertFalse(scene['has_frame'])
        command, options = self.factory.calls[0]
        self.assertEqual(command[command.index('-m')+1], 'task_agent.scene_worker')
        self.assertEqual(int(command[command.index('--seed')+1]), 17)
        self.assertEqual(Path(command[command.index('--output')+1]), self.folder(scene))
        self.assertNotIn('--instruction', command)
        self.assertNotIn('shell', options)
        folder = self.folder(scene)
        self.assertFalse((folder/'task_request.json').exists())
        self.assertFalse((folder/'plan.json').exists())
        self.assertFalse((folder/'vlm_calls.jsonl').exists())
        request = json.loads((folder/'web_request.json').read_text())
        self.assertEqual(request['mode'], 'scene_first')
        self.assertEqual(request['seed'], 17)

    def test_null_seed_becomes_a_concrete_bounded_seed_shared_by_status_and_worker(self):
        with patch('task_agent.web.secrets.randbelow', return_value=123456) as random_seed:
            scene = self.prepare(None)
        random_seed.assert_called_once_with(2**31)
        self.assertEqual(scene['seed'], 123456)
        self.assertIs(type(scene['seed']), int)
        self.assertTrue(0 <= scene['seed'] <= 2**31-1)
        command = self.factory.calls[0][0]
        self.assertEqual(int(command[command.index('--seed')+1]), scene['seed'])
        self.assertEqual(json.loads((self.folder(scene)/'web_request.json').read_text())['seed'], scene['seed'])

    def test_prepare_seed_boundaries_and_strict_payload_validation(self):
        for value in (-1, 2**31, True, False, 1.5, '17', float('nan'), float('inf'), [], {}):
            with self.subTest(seed=value):
                code, response = self.post('/api/scene', dict(seed=value))
                self.assertEqual(code, 400, response)
        for extra in ({'instruction': 'should not plan yet'}, {'policy': 'pi05'}, {'scene_id': 'old'}):
            with self.subTest(extra=extra):
                code, response = self.post('/api/scene', dict(seed=17, **extra))
                self.assertEqual(code, 400, response)
        self.assertEqual(self.factory.calls, [])
        for seed in (0, 2**31-1):
            scene = self.prepare(seed)
            self.assertEqual(scene['seed'], seed)
            self.factory.processes[-1].returncode = 0

    def test_prepared_worker_blocks_another_scene_and_legacy_start(self):
        scene = self.prepare()
        for ready in (False, True):
            if ready: self.ready(scene)
            self.assert_conflict('/api/scene', dict(seed=29))
            self.assert_conflict('/api/start', dict(instruction='球类'))
        self.assertEqual(len(self.factory.calls), 1)

    def test_ready_requires_matching_manifest_not_just_a_status_label(self):
        scene = self.prepare()
        folder = self.folder(scene)
        (folder/'status.json').write_text(json.dumps(dict(status='scene_ready', phase='scene_ready',
                                                        scene_id=scene['id'], seed=scene['seed'])))
        self.assertFalse(self.get('/api/status')['scene']['ready'])
        for changes in ({'scene_id': 'task_agent_other'}, {'seed': 29}):
            with self.subTest(changes=changes):
                self.assertFalse(self.ready(scene, changes)['ready'])
                self.assert_conflict('/api/scene/task', self.task(scene))
        ready = self.ready(scene)
        self.assertTrue(ready['ready'])
        self.assertTrue(ready['has_frame'])
        self.assertFalse(ready['submitted'])

    def test_task_is_atomic_and_consumes_same_scene_without_a_second_process_or_seed_override(self):
        scene = self.prepare(29)
        self.ready(scene)
        folder = self.folder(scene)
        config = dict(enabled=True, start_seconds=2.5, duration_seconds=.5)
        original_manifest = (folder/'scene.json').read_bytes()
        real_replace = Path.replace
        with patch.object(Path, 'replace', autospec=True, side_effect=real_replace) as replace:
            code, response = self.post('/api/scene/task', self.task(scene, '  先收拾球类  ', policy='act',
                max_retries=1, max_subtasks=3, monitor_hz=1, mask_dropout=config))
        self.assertEqual(code, 202, response)
        self.assertEqual(len(self.factory.calls), 1)
        request = json.loads((folder/'task_request.json').read_text())
        self.assertEqual(request['scene_id'], scene['id'])
        self.assertEqual(request['seed'], 29)
        self.assertEqual(request['instruction'], '先收拾球类')
        self.assertEqual(request['policy'], 'act')
        self.assertEqual(request['max_retries'], 1)
        self.assertEqual(request['max_subtasks'], 3)
        self.assertEqual(request['monitor_hz'], 1)
        self.assertEqual(request['mask_dropout'], config)
        self.assertEqual((folder/'scene.json').read_bytes(), original_manifest)
        self.assertFalse((folder/'stop.request').exists())
        self.assertTrue(any(Path(c.args[1]) == folder/'task_request.json' for c in replace.call_args_list))
        status = self.get('/api/status')
        self.assertEqual(status['active_run_id'], scene['id'])
        self.assertTrue(status['scene']['submitted'])
        self.assertFalse(status['scene']['ready'])

    def test_early_wrong_and_duplicate_submissions_do_not_create_or_overwrite_task(self):
        scene = self.prepare()
        path = self.folder(scene)/'task_request.json'
        self.assert_conflict('/api/scene/task', self.task(scene))
        self.assertFalse(path.exists())
        self.ready(scene)
        self.assert_conflict('/api/scene/task', dict(scene_id='task_agent_wrong', instruction='球类'))
        self.assertFalse(path.exists())
        code, response = self.post('/api/scene/task', self.task(scene, 'first instruction'))
        self.assertEqual(code, 202, response)
        original = path.read_bytes()
        self.assert_conflict('/api/scene/task', self.task(scene, 'second instruction'))
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(len(self.factory.calls), 1)

    def test_two_concurrent_submissions_have_exactly_one_winner(self):
        scene = self.prepare()
        self.ready(scene)
        barrier = threading.Barrier(2)
        def submit(instruction):
            barrier.wait(timeout=2)
            return self.post('/api/scene/task', self.task(scene, instruction))
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(submit, ['first instruction', 'second instruction']))
        self.assertEqual(sorted(code for code, _ in responses), [202, 409])
        request = json.loads((self.folder(scene)/'task_request.json').read_text())
        winner = ['first instruction', 'second instruction'][next(i for i, r in enumerate(responses) if r[0] == 202)]
        self.assertEqual(request['instruction'], winner)
        self.assertEqual(len(self.factory.calls), 1)

    def test_task_settings_reject_unknown_seed_override_and_invalid_bounds(self):
        scene = self.prepare()
        self.ready(scene)
        invalid = [{'seed': 99}, {'scene_id': None}, {'instruction': ''}, {'instruction': ' '*3},
            {'instruction': 'x'*2001}, {'instruction': None}, {'policy': 'shell'},
            {'max_retries': True}, {'max_retries': -1}, {'max_retries': 4},
            {'max_subtasks': 0}, {'max_subtasks': 5}, {'monitor_hz': True},
            {'monitor_hz': 2}, {'monitor_hz': float('nan')}, {'object_id': 'cube1'},
            {'output': '/tmp/another-run'}, {'stop_file': '/tmp/stop'},
            {'mask_dropout_control': '/tmp/control'},
            {'mask_dropout': {'enabled': 1}}, {'mask_dropout': {'duration_seconds': float('inf')}}]
        for changes in invalid:
            with self.subTest(changes=changes):
                payload = self.task(scene)
                payload.update(changes)
                code, response = self.post('/api/scene/task', payload)
                self.assertIn(code, (400, 409), response)
                self.assertFalse((self.folder(scene)/'task_request.json').exists())
        self.assertEqual(len(self.factory.calls), 1)

    def test_durable_stop_request_prevents_ready_and_task_submission(self):
        scene = self.prepare()
        self.ready(scene)
        # A stop from another local controller must be honored even before
        # worker status or the web process's in-memory flag catches up.
        (self.folder(scene)/'stop.request').touch()
        self.assertFalse(self.console.stop_requested)
        self.assertFalse(self.get('/api/status')['scene']['ready'])
        self.assert_conflict('/api/scene/task', self.task(scene))
        self.assertFalse((self.folder(scene)/'task_request.json').exists())

    def test_preview_rejects_mask_interruption_and_malformed_close_payloads(self):
        scene = self.prepare()
        self.ready(scene)
        self.assert_conflict('/api/mask_dropout', dict(run_id=scene['id'], action='restore'))
        self.assertFalse((self.folder(scene)/'mask_dropout_control.json').exists())
        for payload in ({}, {'scene_id': None}, {'scene_id': scene['id'], 'seed': 17}):
            code, response = self.post('/api/scene/close', payload)
            self.assertIn(code, (400, 409), response)
        self.assertFalse((self.folder(scene)/'stop.request').exists())

    def test_close_is_only_for_current_unsubmitted_scene_and_waits_for_process_exit(self):
        scene = self.prepare()
        self.assert_conflict('/api/scene/close', dict(scene_id='task_agent_wrong'))
        self.assertFalse((self.folder(scene)/'stop.request').exists())
        code, response = self.post('/api/scene/close', dict(scene_id=scene['id']))
        self.assertIn(code, (200, 202), response)
        self.assertTrue((self.folder(scene)/'stop.request').exists())
        self.assertIsNone(self.factory.processes[0].returncode)
        self.ready(scene)  # A late ready status cannot undo an earlier close.
        self.assertFalse(self.get('/api/status')['scene']['ready'])
        self.assert_conflict('/api/scene/task', self.task(scene))
        self.assert_conflict('/api/scene', dict(seed=29))
        self.factory.processes[0].returncode = 0
        new_scene = self.prepare(29)
        self.assertNotEqual(new_scene['id'], scene['id'])

    def test_ready_scene_can_close_but_submitted_or_executing_scene_cannot(self):
        scene = self.prepare()
        self.ready(scene)
        code, response = self.post('/api/scene/task', self.task(scene))
        self.assertEqual(code, 202, response)
        self.assert_conflict('/api/scene/close', dict(scene_id=scene['id']))
        self.assertFalse((self.folder(scene)/'stop.request').exists())
        (self.folder(scene)/'status.json').write_text(json.dumps(dict(status='executing', phase='grasp',
            scene_id=scene['id'], seed=scene['seed'])))
        self.assert_conflict('/api/scene/close', dict(scene_id=scene['id']))

    def test_close_and_submit_race_cannot_both_win(self):
        scene = self.prepare()
        self.ready(scene)
        barrier = threading.Barrier(2)
        def request(action):
            barrier.wait(timeout=2)
            return self.post('/api/scene/'+action,
                             self.task(scene) if action == 'task' else dict(scene_id=scene['id']))
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(request, ['task', 'close']))
        successful = [code for code, _ in responses if code in (200, 202)]
        self.assertEqual(len(successful), 1, responses)
        self.assertIn(409, [code for code, _ in responses])
        folder = self.folder(scene)
        self.assertNotEqual((folder/'task_request.json').exists(), (folder/'stop.request').exists())
        self.assertEqual(len(self.factory.calls), 1)

    def test_dead_or_historical_worker_cannot_consume_task_or_close_new_scene(self):
        old = self.prepare()
        self.ready(old)
        self.factory.processes[0].returncode = 0
        self.assertFalse(self.get('/api/status')['scene']['ready'])
        self.assert_conflict('/api/scene/task', self.task(old))
        current = self.prepare(29)
        self.ready(current)
        self.assert_conflict('/api/scene/task', self.task(old))
        self.assert_conflict('/api/scene/close', dict(scene_id=old['id']))
        self.assertFalse((self.folder(current)/'stop.request').exists())
        self.assertFalse((self.folder(old)/'task_request.json').exists())

    def test_all_scene_mutations_share_csrf_and_origin_guards(self):
        for token, origin in ((False, None), (True, 'https://unrelated.example')):
            code, response = self.post('/api/scene', dict(seed=17), token=token, origin=origin)
            self.assertEqual(code, 403, response)
        self.assertEqual(self.factory.calls, [])
        scene = self.prepare()
        self.ready(scene)
        for route, payload in (('/api/scene/task', self.task(scene)),
                               ('/api/scene/close', dict(scene_id=scene['id']))):
            for token, origin in ((False, None), (True, 'https://unrelated.example')):
                code, response = self.post(route, payload, token=token, origin=origin)
                self.assertEqual(code, 403, response)
        self.assertFalse((self.folder(scene)/'task_request.json').exists())
        self.assertFalse((self.folder(scene)/'stop.request').exists())

    def test_stopped_preview_record_and_video_remain_in_history(self):
        scene = self.prepare()
        self.ready(scene)
        self.post('/api/scene/close', dict(scene_id=scene['id']))
        folder = self.folder(scene)
        (folder/'execution.mp4').write_bytes(b'0123456789')
        (folder/'summary.json').write_text(json.dumps(dict(status='stopped', success=False,
            phase='stopped', scene_id=scene['id'], seed=scene['seed'], instruction='')))
        self.factory.processes[0].returncode = 0
        run = self.get('/api/status')['run']
        self.assertEqual(run['status'], 'stopped')
        self.assertIn('video', run['artifacts'])
        self.assertIn(scene['id'], [r['id'] for r in self.get('/api/runs')['runs']])
        request = Request(self.url+run['artifacts']['video'], headers={'Range': 'bytes=2-5'})
        with urlopen(request, timeout=3) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b'2345')

    def test_legacy_start_still_uses_runner_without_a_prepared_scene(self):
        code, response = self.post('/api/start', dict(instruction='球类', seed=17))
        self.assertEqual(code, 202, response)
        command = self.factory.calls[0][0]
        self.assertEqual(command[command.index('-m')+1], 'task_agent.runner')
        self.assertEqual(command[command.index('--instruction')+1], '球类')
        self.assertIsNone(self.get('/api/status')['scene'])


if __name__ == '__main__':
    unittest.main()
