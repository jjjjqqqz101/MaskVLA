"""Triple-ACT policy routing and provenance contracts; CPU doubles only."""
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from task_agent.policies import POLICY_IDS, policy_metadata, policy_options, policy_record
from task_agent.scene_objects import normalize_scene_spec, resolve_scene_spec
from task_agent.scene_worker import SceneWorker, rgb_hashes, validate_task_request
from task_agent.web import Console, DEFAULTS, Handler


class ProcessFactory:
    def __init__(self): self.calls = []; self.processes = []
    def __call__(self, command, **options):
        process = SimpleNamespace(returncode=None)
        process.poll = lambda: process.returncode
        self.calls.append((command, options)); self.processes.append(process)
        return process


def task(scene_id, policy='act_triple', seed=17):
    return dict(scene_id=scene_id, instruction='收拾球类', policy=policy, seed=seed,
                max_retries=1, max_subtasks=4, monitor_hz=1, mask_dropout=None)


class PolicyMetadataContracts(unittest.TestCase):
    def test_distinct_checkpoint_families_have_stable_public_metadata(self):
        expected = {'act': (100000, 2, 28, 8, 100),
                    'act_triple': (160000, 3, 28, 25, 50),
                    'pi05': (49999, 3, 8, 25, 50)}
        for policy, values in expected.items():
            row = policy_metadata(policy)
            self.assertEqual((row['checkpoint_step'], len(row['rgb_streams']), row['state_dim'],
                              row['training_object_count'], row['training_episodes_per_object']), values)
            self.assertEqual(row['rgb_streams'][-1], 'base_masked_rgb')
        self.assertEqual(DEFAULTS['policy'], 'pi05')
        self.assertEqual({row['policy'] for row in policy_options()}, set(POLICY_IDS))
        serialized = json.dumps(policy_options())
        self.assertNotIn('/home/', serialized); self.assertNotIn('checkpoint_path', serialized)
        changed = policy_metadata('act_triple'); changed['rgb_streams'].clear()
        self.assertEqual(len(policy_metadata('act_triple')['rgb_streams']), 3)

    def test_historical_act_cannot_inherit_triple_checkpoint_metadata(self):
        result = policy_record('act', policy_metadata('act_triple'))
        self.assertEqual(result['model']['checkpoint_step'], 100000)
        self.assertEqual(len(result['model']['rgb_streams']), 2)
        saved = policy_metadata('act_triple')
        saved.update(checkpoint_path='/home/private/checkpoint.pt', access_token='private')
        public = policy_record('act_triple', saved)
        self.assertNotIn('checkpoint_path', public['model'])
        self.assertNotIn('access_token', public['model'])

    def test_worker_task_schema_accepts_new_id_and_keeps_scene_immutable(self):
        for policy in POLICY_IDS:
            self.assertEqual(validate_task_request(task('scene', policy), 'scene', 17)['policy'], policy)
        for change in ({'seed': 18}, {'policy': 'other'}, {'policy': []},
                       {'scene_spec': normalize_scene_spec()}, {'objects': ['025_mug']}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_task_request(dict(task('scene'), **change), 'scene', 17)


class PolicyWebContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.factory = ProcessFactory()
        self.console = Console(Path(self.temp.name)/'runs', process_factory=self.factory)

    def ready_scene(self):
        response = self.console.prepare_scene({'seed': 17})
        directory = self.console.directory(response['scene_id'])
        (directory/'scene.json').write_text(json.dumps(dict(scene_id=directory.name, seed=17, reset_count=1)))
        (directory/'status.json').write_text(json.dumps(dict(status='scene_ready', phase='scene_ready')))
        return directory

    def test_original_direct_api_routes_all_policy_ids_and_persists_model(self):
        for policy in POLICY_IDS:
            response = self.console.start(dict(instruction='收拾球类', policy=policy))
            command = self.factory.calls[-1][0]
            self.assertEqual(command[command.index('--policy')+1], policy)
            self.assertEqual(response['run']['model'], policy_metadata(policy))
            directory = self.console.directory(response['run_id'])
            request = json.loads((directory/'web_request.json').read_text())
            self.assertEqual(request['model'], policy_metadata(policy))
            self.factory.processes[-1].returncode = 0

    def test_scene_submission_keeps_exact_worker_schema_and_frozen_inventory(self):
        directory = self.ready_scene()
        original_spec = (directory/'scene_spec.json').read_bytes()
        payload = dict(scene_id=directory.name, instruction='收拾球类', policy='act_triple')
        for change in ({'seed': 18}, {'objects': []}, {'scene_spec': normalize_scene_spec()},
                       {'model': policy_metadata('act')}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.console.submit_scene_task(dict(payload, **change))
        self.assertFalse((directory/'task_request.json').exists())
        response = self.console.submit_scene_task(payload)
        self.assertEqual(response['run']['model']['checkpoint_step'], 160000)
        self.assertEqual((directory/'scene_spec.json').read_bytes(), original_spec)
        self.assertEqual(len(self.factory.calls), 1)
        request = json.loads((directory/'task_request.json').read_text())
        self.assertEqual(set(request), set(task(directory.name)))
        self.assertEqual(validate_task_request(request, directory.name, 17)['policy'], 'act_triple')
        saved = json.loads((directory/'web_request.json').read_text())
        self.assertEqual(saved['model'], policy_metadata('act_triple'))

    def test_legacy_history_and_new_runs_keep_their_own_policy_model(self):
        for policy in POLICY_IDS:
            directory = self.console.runs_root/('task_agent_saved_'+policy); directory.mkdir()
            # Historical records predate public model snapshots.
            request = dict(policy=policy, instruction='收拾球类', created_at='2026-11-05T00:00:00+08:00')
            (directory/'web_request.json').write_text(json.dumps(request))
            (directory/'summary.json').write_text(json.dumps(dict(status='completed', success=True)))
        rows = {row['policy']: row for row in self.console.history()}
        self.assertEqual(set(rows), set(POLICY_IDS))
        for policy in POLICY_IDS:
            self.assertEqual(rows[policy]['model'], policy_metadata(policy))
            self.assertEqual(rows[policy]['policy_label'], policy_metadata(policy)['policy_label'])
        self.assertIn('双路', rows['act']['policy_label'])

    def test_config_and_catalog_publish_model_options_without_private_paths(self):
        class QuietHandler(Handler):
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler); server.console = self.console
        worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
        worker.start()
        try:
            root = 'http://127.0.0.1:%d' % server.server_port
            with urlopen(root+'/api/config', timeout=3) as response: config = json.load(response)
            self.assertEqual(config['defaults']['policy'], 'pi05')
            self.assertEqual(config['policies'], list(POLICY_IDS))
            self.assertEqual(config['policy_options'], policy_options())
            self.assertNotIn('/home/', json.dumps(config))
            with urlopen(root+'/api/object_catalog', timeout=3) as response: catalog = json.load(response)
            self.assertEqual(sum(row['act_trained'] for row in catalog['objects']), 8)
            self.assertEqual(sum(row['act_triple_trained'] for row in catalog['objects']), 25)
            self.assertTrue(all(row['act_triple_trained'] == row['pi05_trained'] for row in catalog['objects']))
        finally:
            server.shutdown(); server.server_close(); worker.join(timeout=2)


class SyntheticRGB:
    def __init__(self, value): self.value = value
    def tobytes(self): return bytes([self.value])*16


class SceneBindingContracts(unittest.TestCase):
    def run_scene(self, matching=True):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        output = Path(temp.name); scene_id = 'task_agent_triple_scene'
        stop = output/'stop.request'; task_file = output/'task_request.json'
        spec = resolve_scene_spec(dict(mode='manual', object_ids=['025_mug', '055_baseball']))
        (output/'scene_spec.json').write_text(json.dumps(spec))
        rgb = {name: SyntheticRGB(index+1) for index, name in enumerate(('base_rgb', 'wrist_rgb', 'overview_rgb'))}
        hashes = rgb_hashes(rgb)
        if not matching: hashes['wrist_rgb'] = '0'*64
        (output/'scene_reference.json').write_text(json.dumps(dict(source_scene_id='task_agent_source',
                                                                  seed=17, rgb_sha256=hashes)))
        backend = SimpleNamespace(reset_count=1, selected_object_ids=['025_mug', '055_baseball'], binds=[])
        backend.refresh_observation = lambda: dict(frame_id=9, **rgb)
        def bind(**kwargs): backend.binds.append(kwargs); return backend
        backend.bind_task = bind; backend.close = lambda: None
        backend_calls = []
        def backend_factory(**kwargs): backend_calls.append(kwargs); return backend
        def wait(seconds): task_file.write_text(json.dumps(task(scene_id)))
        def runner_factory(**kwargs):
            def run():
                kwargs['backend_factory'](policy=kwargs['policy'], seed=kwargs['seed'], output=kwargs['output'])
                return dict(status='completed', success=True, policy=kwargs['policy'])
            return SimpleNamespace(run=run)
        result = SceneWorker(17, output, stop, task_file, scene_id, backend_factory=backend_factory,
                             runner_factory=runner_factory, sleep=wait).run()
        return output, backend, backend_calls, spec, result

    def test_triple_task_binds_once_without_reset_frame_or_rgb_changes(self):
        output, backend, calls, spec, result = self.run_scene()
        self.assertEqual(result['status'], 'completed'); self.assertEqual(backend.reset_count, 1)
        self.assertEqual(len(calls), 1); self.assertTrue(calls[0]['defer_policy_loading'])
        self.assertEqual(calls[0]['policy'], 'pi05')
        self.assertEqual(len(backend.binds), 1); self.assertEqual(backend.binds[0]['policy'], 'act_triple')
        summary = json.loads((output/'summary.json').read_text())
        self.assertEqual(summary['model'], policy_metadata('act_triple'))
        self.assertEqual(summary['scene_spec'], spec)
        self.assertTrue(summary['comparison_initial_layout_verified'])
        reused = json.loads((output/'scene_reuse.json').read_text())
        self.assertTrue(reused['rgb_unchanged'])
        self.assertEqual(reused['preview_frame_id'], reused['after_bind_frame_id'])

    def test_rgb_mismatch_prevents_triple_task_or_policy_binding(self):
        output, backend, calls, spec, result = self.run_scene(matching=False)
        self.assertEqual(result['status'], 'error'); self.assertFalse(backend.binds)
        self.assertEqual(backend.reset_count, 1)
        comparison = json.loads((output/'scene_comparison.json').read_text())
        self.assertFalse(comparison['verified']); self.assertEqual(comparison['reason'], 'initial_rgb_mismatch')


class RunnerPolicyContracts(unittest.TestCase):
    def test_new_policy_reaches_runner_and_final_summary_with_metadata(self):
        from task_agent.runner import TaskRunner
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            backend = SimpleNamespace(zone={'capacity': 4}, snapshot=lambda: {}, close=lambda: None)
            def backend_factory(**kwargs): calls.append(kwargs); return backend
            vlm = SimpleNamespace(plan=lambda snapshot: dict(steps=[], summary='请明确任务',
                selection_rule='球类', needs_clarification=True))
            result = TaskRunner('收拾球类', 'act_triple', 17, directory,
                                backend_factory=backend_factory, vlm=vlm).run()
            self.assertEqual(calls[0]['policy'], 'act_triple')
            self.assertEqual(result['model'], policy_metadata('act_triple'))
            self.assertEqual(json.loads((Path(directory)/'summary.json').read_text())['model'], result['model'])

    def test_original_runner_cli_accepts_new_policy_id(self):
        from task_agent import runner
        result = SimpleNamespace(run=lambda: dict(status='completed'))
        with patch.object(runner, 'TaskRunner', return_value=result) as factory, \
             patch('sys.argv', ['task_agent.runner', '--instruction', '收拾球类',
                                '--policy', 'act_triple', '--output', '/tmp/triple-policy-cli']), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 0)
        self.assertEqual(factory.call_args.kwargs['policy'], 'act_triple')


if __name__ == '__main__': unittest.main()
