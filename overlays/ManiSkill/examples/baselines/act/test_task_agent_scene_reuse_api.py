"""Explicit initial-layout comparison contracts with inert workers and HTTP."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from task_agent.web import Console, Handler


HASHES = dict(base_rgb='0'*64, wrist_rgb='1'*64, overview_rgb='a'*64)


class FakeProcess:
    def __init__(self): self.returncode = None
    def poll(self): return self.returncode


class ProcessFactory:
    def __init__(self): self.calls = []; self.processes = []
    def __call__(self, command, **options):
        process = FakeProcess()
        self.calls.append((command, options)); self.processes.append(process)
        return process


class QuietHandler(Handler):
    def log_message(self, *args): pass


class InitialLayoutComparisonContracts(unittest.TestCase):
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
        self.source_counter = 0

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
        self.temp.cleanup()

    def get(self, route):
        with urlopen(self.url+route, timeout=3) as response: return json.load(response)

    def post(self, route, payload, token=True, origin=None):
        headers = {'Content-Type': 'application/json'}
        if token: headers['X-CSRF-Token'] = self.console.token
        if origin is not None: headers['Origin'] = origin
        request = Request(self.url+route, data=json.dumps(payload, allow_nan=True).encode(), headers=headers)
        try:
            with urlopen(request, timeout=3) as response: return response.status, json.load(response)
        except HTTPError as error: return error.code, json.load(error)

    def source(self, request_changes=None, manifest_changes=None):
        self.source_counter += 1
        scene_id = 'task_agent_history_%d' % self.source_counter
        folder = self.console.runs_root/scene_id
        folder.mkdir()
        request = dict(mode='scene_first', scene_id=scene_id, seed=17, instruction='旧任务不要继承',
            policy='act', mask_dropout=dict(enabled=True, start_seconds=99, duration_seconds=20))
        request.update(request_changes or {})
        manifest = dict(scene_id=scene_id, seed=17, reset_count=1, frame_id=5, rgb_sha256=dict(HASHES))
        manifest.update(manifest_changes or {})
        (folder/'web_request.json').write_text(json.dumps(request))
        (folder/'scene.json').write_text(json.dumps(manifest))
        (folder/'summary.json').write_text(json.dumps(dict(status='completed', success=True,
            completed_steps=1, instruction=request['instruction'], seed=17, policy='act')))
        (folder/'events.jsonl').write_text('{"event":"old_scene_finished"}\n')
        return scene_id, folder

    def clone(self, source_id):
        code, response = self.post('/api/scene', dict(mode='current', source_scene_id=source_id))
        self.assertEqual(code, 202, response)
        return self.get('/api/status')['scene']

    def ready(self, scene, source_id, verified=None):
        folder = self.console.directory(scene['id'])
        manifest = dict(scene_id=scene['id'], seed=scene['seed'], reset_count=1,
                        frame_id=5, rgb_sha256=dict(HASHES), comparison_source_scene_id=source_id)
        if verified is not None: manifest['comparison_initial_layout_verified'] = verified
        (folder/'scene.json').write_text(json.dumps(manifest))
        (folder/'status.json').write_text(json.dumps(dict(status='scene_ready', phase='scene_ready',
            scene_id=scene['id'], seed=scene['seed'])))
        (folder/'live.png').write_bytes(b'new-worker-frame')
        return self.get('/api/status')['scene']

    def test_explicit_history_source_creates_new_record_and_worker_without_changing_source(self):
        source_id, source = self.source()
        original = {p.name: p.read_bytes() for p in source.iterdir()}
        scene = self.clone(source_id)
        self.assertNotEqual(scene['id'], source_id)
        self.assertEqual(scene['seed'], 17)
        self.assertEqual(scene['generation_mode'], 'current')
        self.assertEqual(scene['comparison_source_scene_id'], source_id)
        self.assertFalse(scene['comparison_initial_layout_verified'])
        self.assertFalse(scene['ready'])
        folder = self.console.directory(scene['id'])
        reference = json.loads((folder/'scene_reference.json').read_text())
        self.assertEqual(reference, dict(source_scene_id=source_id, seed=17, rgb_sha256=HASHES))
        command, options = self.factory.calls[0]
        self.assertEqual(command[command.index('-m')+1], 'task_agent.scene_worker')
        self.assertEqual(int(command[command.index('--seed')+1]), 17)
        self.assertEqual(Path(command[command.index('--output')+1]), folder)
        self.assertEqual({x for x in command if x.startswith('--')},
                         {'--scene-id', '--seed', '--output', '--stop-file', '--task-file'})
        self.assertNotIn('--instruction', command)
        self.assertNotIn('shell', options)
        request = json.loads((folder/'web_request.json').read_text())
        self.assertEqual(request['mode'], 'scene_first')
        self.assertEqual(request['generation_mode'], 'current')
        self.assertEqual(request['comparison_source_scene_id'], source_id)
        self.assertEqual(request['instruction'], '')
        self.assertNotEqual(request.get('policy'), 'act')
        self.assertFalse(request['mask_dropout']['enabled'])
        self.assertFalse((folder/'task_request.json').exists())
        self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, original)

    def test_snapshot_and_history_advertise_only_valid_initial_scene_sources(self):
        source_id, _ = self.source()
        legacy_id, _ = self.source(request_changes=dict(mode='legacy_direct'))
        self.assertTrue(self.console.snapshot(source_id)['initial_scene_available'])
        self.assertFalse(self.console.snapshot(legacy_id)['initial_scene_available'])
        rows = {r['id']: r for r in self.get('/api/runs')['runs']}
        self.assertTrue(rows[source_id]['initial_scene_available'])
        self.assertFalse(rows[legacy_id]['initial_scene_available'])
        scene = self.clone(source_id)
        row = next(r for r in self.get('/api/runs')['runs'] if r['id'] == scene['id'])
        self.assertEqual(row['generation_mode'], 'current')
        self.assertEqual(row['comparison_source_scene_id'], source_id)
        self.assertFalse(row['initial_scene_available'])

    def test_current_mode_requires_explicit_source_and_rejects_seed_or_other_settings(self):
        source_id, _ = self.source()
        invalid = [dict(mode='current'), dict(mode='current', source_scene_id=None),
            dict(mode='new', source_scene_id=source_id), dict(mode='random'),
            dict(mode='current', source_scene_id=source_id, seed=17),
            dict(mode='current', source_scene_id=source_id, seed=None),
            dict(mode='current', source_scene_id=source_id, instruction='do not inherit a prompt'),
            dict(mode='current', source_scene_id=source_id, policy='act'),
            dict(source_scene_id=source_id)]
        for payload in invalid:
            with self.subTest(payload=payload):
                code, response = self.post('/api/scene', payload)
                self.assertEqual(code, 400, response)
        self.assertEqual(self.factory.calls, [])

    def test_legacy_or_missing_source_cannot_be_used_even_if_directory_exists(self):
        legacy_id, _ = self.source(request_changes=dict(mode='legacy_direct'))
        for source_id in (legacy_id, 'task_agent_not_found'):
            code, response = self.post('/api/scene', dict(mode='current', source_scene_id=source_id))
            self.assertEqual(code, 400, response)
        self.assertEqual(self.factory.calls, [])

    def test_invalid_source_manifest_seed_reset_or_hashes_are_rejected(self):
        invalid = [dict(scene_id='task_agent_other'), dict(seed=29), dict(seed=True),
            dict(seed='17'), dict(reset_count=0), dict(reset_count=2), dict(reset_count=True),
            dict(rgb_sha256=None), dict(rgb_sha256={}),
            dict(rgb_sha256=dict(base_rgb='0'*64, wrist_rgb='1'*64)),
            dict(rgb_sha256=dict(HASHES, base_rgb='0'*63)),
            dict(rgb_sha256=dict(HASHES, wrist_rgb='z'*64)),
            dict(rgb_sha256=dict(HASHES, overview_rgb=None))]
        for manifest in invalid:
            with self.subTest(manifest=manifest):
                source_id, _ = self.source(manifest_changes=manifest)
                self.assertFalse(self.console.snapshot(source_id)['initial_scene_available'])
                code, response = self.post('/api/scene', dict(mode='current', source_scene_id=source_id))
                self.assertEqual(code, 400, response)
        source_id, _ = self.source(request_changes=dict(seed=19))
        code, response = self.post('/api/scene', dict(mode='current', source_scene_id=source_id))
        self.assertEqual(code, 400, response)
        source_id, _ = self.source(request_changes=dict(scene_id='task_agent_wrong_request_id'))
        code, response = self.post('/api/scene', dict(mode='current', source_scene_id=source_id))
        self.assertEqual(code, 400, response)
        self.assertEqual(self.factory.calls, [])

    def test_source_paths_and_symlink_records_are_rejected(self):
        source_id, source = self.source()
        for bad in ('../'+source_id, str(source), 'task_agent_../private', 'task_agent_%2e%2e'):
            code, response = self.post('/api/scene', dict(mode='current', source_scene_id=bad))
            self.assertEqual(code, 400, response)
        directory_link = self.console.runs_root/'task_agent_source_link'
        directory_link.symlink_to(source, target_is_directory=True)
        code, response = self.post('/api/scene', dict(mode='current', source_scene_id=directory_link.name))
        self.assertEqual(code, 400, response)
        for filename in ('scene.json', 'web_request.json'):
            source_id, source = self.source()
            external = Path(self.temp.name)/('external_'+filename)
            external.write_bytes((source/filename).read_bytes())
            (source/filename).unlink()
            (source/filename).symlink_to(external)
            code, response = self.post('/api/scene', dict(mode='current', source_scene_id=source_id))
            self.assertEqual(code, 400, response)
        self.assertEqual(self.factory.calls, [])

    def test_any_busy_worker_blocks_clone_until_closed_and_exited(self):
        source_id, _ = self.source()
        code, response = self.post('/api/scene', dict(seed=29))
        self.assertEqual(code, 202, response)
        prepared = self.get('/api/status')['scene']
        payload = dict(mode='current', source_scene_id=source_id)
        code, response = self.post('/api/scene', payload)
        self.assertEqual(code, 409, response)
        code, response = self.post('/api/scene/close', dict(scene_id=prepared['id']))
        self.assertEqual(code, 202, response)
        code, response = self.post('/api/scene', payload)
        self.assertEqual(code, 409, response)
        self.assertEqual(len(self.factory.calls), 1)
        self.factory.processes[0].returncode = 0
        scene = self.clone(source_id)
        self.assertEqual(scene['seed'], 17)
        self.assertEqual(len(self.factory.calls), 2)

    def test_executing_legacy_task_also_blocks_current_mode(self):
        source_id, _ = self.source()
        code, response = self.post('/api/start', dict(instruction='running task'))
        self.assertEqual(code, 202, response)
        code, response = self.post('/api/scene', dict(mode='current', source_scene_id=source_id))
        self.assertEqual(code, 409, response)
        self.assertEqual(len(self.factory.calls), 1)

    def test_new_worker_must_verify_reference_before_ready_or_task_submission(self):
        source_id, _ = self.source(manifest_changes=dict(comparison_initial_layout_verified=True,
                                                        comparison_source_scene_id='task_agent_older'))
        scene = self.clone(source_id)
        self.assertFalse(scene['comparison_initial_layout_verified'])
        for verified, reported_source in ((None, source_id), (False, source_id), (1, source_id),
                                           (True, 'task_agent_wrong_source')):
            with self.subTest(verified=verified, reported_source=reported_source):
                current = self.ready(scene, reported_source, verified)
                self.assertFalse(current['ready'])
                self.assertFalse(current['comparison_initial_layout_verified'])
                code, response = self.post('/api/scene/task', dict(scene_id=scene['id'], instruction='new task'))
                self.assertEqual(code, 409, response)
                self.assertFalse((self.console.directory(scene['id'])/'task_request.json').exists())
        ready = self.ready(scene, source_id, True)
        self.assertTrue(ready['ready'])
        self.assertTrue(ready['comparison_initial_layout_verified'])
        code, response = self.post('/api/scene/task', dict(scene_id=scene['id'], instruction='新任务'))
        self.assertEqual(code, 202, response)
        request = json.loads((self.console.directory(scene['id'])/'task_request.json').read_text())
        self.assertEqual(request['instruction'], '新任务')
        self.assertEqual(request['policy'], 'pi05')
        self.assertFalse(request['mask_dropout']['enabled'])
        self.assertEqual(request['seed'], 17)
        self.assertEqual(len(self.factory.calls), 1)

    def test_verified_flag_cannot_belong_to_a_different_new_scene_or_seed(self):
        source_id, _ = self.source()
        scene = self.clone(source_id)
        for changes in ({'scene_id': 'task_agent_another_new_scene'}, {'seed': 29},
                        {'rgb_sha256': dict(HASHES, base_rgb='f'*64)}):
            with self.subTest(changes=changes):
                self.ready(scene, source_id, True)
                path = self.console.directory(scene['id'])/'scene.json'
                manifest = json.loads(path.read_text())
                manifest.update(changes)
                path.write_text(json.dumps(manifest))
                current = self.get('/api/status')['scene']
                self.assertFalse(current['ready'])
                self.assertFalse(current['comparison_initial_layout_verified'])
                self.assertFalse(self.console.snapshot(scene['id'])['comparison_initial_layout_verified'])

    def test_clone_requires_same_csrf_and_origin_guards_without_mutating_source(self):
        source_id, source = self.source()
        original = {p.name: p.read_bytes() for p in source.iterdir()}
        for token, origin in ((False, None), (True, 'https://unrelated.example')):
            code, response = self.post('/api/scene', dict(mode='current', source_scene_id=source_id),
                                       token=token, origin=origin)
            self.assertEqual(code, 403, response)
        self.assertEqual(self.factory.calls, [])
        self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, original)

    def test_only_verified_comparison_layouts_can_chain_even_when_task_itself_failed(self):
        source_id, _ = self.source()
        unverified = self.clone(source_id)
        self.ready(unverified, source_id, False)
        self.factory.processes[-1].returncode = 0
        self.assertFalse(self.console.snapshot(unverified['id'])['initial_scene_available'])
        code, response = self.post('/api/scene', dict(mode='current', source_scene_id=unverified['id']))
        self.assertEqual(code, 400, response)
        self.assertEqual(len(self.factory.calls), 1)

        verified = self.clone(source_id)
        self.ready(verified, source_id, True)
        folder = self.console.directory(verified['id'])
        # A failed manipulation does not invalidate an already verified
        # initial layout: compare initial conditions, not task outcome.
        (folder/'summary.json').write_text(json.dumps(dict(status='retry_budget_exhausted',
            success=False, completed_steps=0, seed=17)))
        self.factory.processes[-1].returncode = 0
        self.assertTrue(self.console.snapshot(verified['id'])['initial_scene_available'])
        original = {p.name: p.read_bytes() for p in folder.iterdir()}
        chained = self.clone(verified['id'])
        reference = json.loads((self.console.directory(chained['id'])/'scene_reference.json').read_text())
        self.assertEqual(reference, dict(source_scene_id=verified['id'], seed=17, rgb_sha256=HASHES))
        self.assertEqual({p.name: p.read_bytes() for p in folder.iterdir()}, original)
        self.assertEqual(json.loads((folder/'scene_reference.json').read_text())['source_scene_id'], source_id)
        self.assertEqual(len(self.factory.calls), 3)

    def test_legacy_seed_api_remains_compatible_and_creates_no_reference_file(self):
        for seed in (None, 17):
            with self.subTest(seed=seed):
                code, response = self.post('/api/scene', dict(seed=seed))
                self.assertEqual(code, 202, response)
                scene = self.get('/api/status')['scene']
                folder = self.console.directory(scene['id'])
                self.assertFalse((folder/'scene_reference.json').exists())
                self.assertFalse(scene['comparison_initial_layout_verified'])
                self.assertIsNone(scene['comparison_source_scene_id'])
                if seed is not None: self.assertEqual(scene['seed'], seed)
                self.factory.processes[-1].returncode = 0


if __name__ == '__main__':
    unittest.main()
