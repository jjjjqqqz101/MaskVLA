"""Fake-process HTTP contracts for mask-loss controls; no robot/model calls."""
import io
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from task_agent.web import Console, Handler


OFF = dict(enabled=False, start_seconds=2, duration_seconds=1)


class FakeProcess:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode


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
    def log_message(self, *args):
        pass


class MaskDropoutAPIContracts(unittest.TestCase):
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
        if token:
            headers['X-CSRF-Token'] = self.console.token
        if origin is not None:
            headers['Origin'] = origin
        # allow_nan deliberately exercises the server's finite-number guard.
        request = Request(self.url+route, data=json.dumps(payload, allow_nan=True).encode(), headers=headers)
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def start(self, config=None):
        payload = dict(instruction='收拾球类')
        if config is not None:
            payload['mask_dropout'] = config
        code, result = self.post('/api/start', payload)
        self.assertEqual(code, 202, result)
        return result

    def progress(self, run_id, phase='grasp', generation=3, **extra):
        row = dict(status='executing', phase=phase, step_generation=generation)
        row.update(extra)
        (self.console.directory(run_id)/'status.json').write_text(json.dumps(row))

    def command_path(self, run_id):
        return self.console.directory(run_id)/'mask_dropout_control.json'

    def reject_command(self, payload, allowed_codes=(400, 409)):
        code, result = self.post('/api/mask_dropout', payload)
        self.assertIn(code, allowed_codes, result)
        self.assertNotEqual(result.get('ok'), True)

    def test_start_defaults_and_automatic_config_reach_literal_child_argv(self):
        for config in (None, {}, dict(enabled=True, start_seconds=0, duration_seconds=.1),
                       dict(enabled=True, start_seconds=120, duration_seconds=30)):
            with self.subTest(config=config):
                result = self.start(config)
                expected = dict(OFF, **(config or {}))
                self.assertEqual(result['run']['mask_dropout'], expected)
                self.assertEqual(result['run']['step_generation'], 0)
                command, options = self.factory.calls[-1]
                passed = json.loads(command[command.index('--mask-dropout-config')+1])
                self.assertEqual(passed, expected)
                control = Path(command[command.index('--mask-dropout-control')+1])
                self.assertEqual(control, self.command_path(result['run_id']))
                self.assertTrue(control.is_absolute())
                self.assertNotIn('shell', options)
                saved = json.loads((control.parent/'web_request.json').read_text())
                self.assertEqual(saved['mask_dropout'], expected)
                self.factory.processes[-1].returncode = 0

    def test_automatic_config_rejects_unknown_nonfinite_boolean_and_out_of_range_values(self):
        invalid = [[], 'enabled', {'enabled': 1}, {'enabled': 'true'}, {'enabled': None},
                   {'unknown': 1}]
        invalid += [{'start_seconds': x} for x in (True, False, None, '2', -.01, 120.01,
                                                  float('nan'), float('inf'), -float('inf'))]
        invalid += [{'duration_seconds': x} for x in (True, False, None, '1', 0, .09, 30.01,
                                                     float('nan'), float('inf'))]
        for config in invalid:
            with self.subTest(config=config):
                code, result = self.post('/api/start', dict(instruction='球类', mask_dropout=config))
                self.assertEqual(code, 400, result)
        self.assertEqual(self.factory.calls, [])
        self.assertFalse(self.console.busy())

    def test_trigger_is_atomic_queued_and_does_not_claim_robot_already_applied_it(self):
        result = self.start()  # Manual loss also works when automatic loss is off.
        run_id = result['run_id']
        state = dict(active=False, source='none', remaining_seconds=0)
        self.progress(run_id, mask_dropout_state=state)
        path = self.command_path(run_id)
        real_replace = Path.replace
        # pathlib on Python3.9 caches its os.replace accessor at import time;
        # observe the public atomic Path.replace operation instead.
        with patch.object(Path, 'replace', autospec=True, side_effect=real_replace) as replace:
            code, queued = self.post('/api/mask_dropout', dict(run_id=run_id, action='trigger', duration_seconds=1.5),
                                     origin=self.url)
        self.assertEqual(code, 202, queued)
        self.assertTrue(queued.get('queued'))
        self.assertFalse(queued.get('applied', False))
        self.assertFalse(queued.get('active', False))
        command = json.loads(path.read_text())
        self.assertEqual(command['action'], 'trigger')
        self.assertEqual(command['duration_seconds'], 1.5)
        self.assertEqual(command['step_generation'], 3)
        self.assertEqual(command['command_id'], queued['command_id'])
        self.assertIsInstance(command['command_id'], str)
        self.assertTrue(command['command_id'])
        self.assertIs(type(command['step_generation']), int)
        self.assertIsNot(type(command['issued_at']), bool)
        self.assertTrue(math.isfinite(command['issued_at']))
        atomic_replacements = [call for call in replace.call_args_list if Path(call.args[1]) == path]
        self.assertTrue(atomic_replacements, 'Control file must be committed with an atomic replacement')
        self.assertNotEqual(Path(atomic_replacements[-1].args[0]), path)
        self.assertEqual(self.console.snapshot(run_id)['mask_dropout_state'], state)

    def test_restore_is_allowed_in_any_busy_phase_without_duration_even_before_first_grasp(self):
        result = self.start(dict(enabled=True, start_seconds=2, duration_seconds=1))
        run_id = result['run_id']
        command_ids = []
        for phase, generation in (('queued', 0), ('transport', 2), ('verify', 2), ('recovery', 2)):
            with self.subTest(phase=phase):
                self.progress(run_id, phase, generation)
                code, queued = self.post('/api/mask_dropout', dict(run_id=run_id, action='restore'))
                self.assertEqual(code, 202, queued)
                self.assertTrue(queued.get('queued'))
                command = json.loads(self.command_path(run_id).read_text())
                self.assertEqual(command['action'], 'restore')
                self.assertNotIn('duration_seconds', command)
                self.assertEqual(command['step_generation'], generation)
                command_ids.append(command['command_id'])
        self.assertEqual(len(set(command_ids)), len(command_ids))

    def test_trigger_rejects_every_non_grasp_phase_and_invalid_generation(self):
        run_id = self.start()['run_id']
        payload = dict(run_id=run_id, action='trigger', duration_seconds=1)
        for phase in ('queued', 'idle', 'planning', 'transport', 'release', 'verify', 'recovery', 'goal_verify'):
            with self.subTest(phase=phase):
                self.progress(run_id, phase=phase)
                self.reject_command(payload)
        for generation in (0, -1, True, False, 1.5, '2', None):
            with self.subTest(generation=generation):
                self.progress(run_id, generation=generation)
                self.reject_command(payload)
        self.assertFalse(self.command_path(run_id).exists())

    def test_invalid_manual_payload_cannot_overwrite_a_valid_pending_command(self):
        run_id = self.start()['run_id']
        self.progress(run_id)
        good = dict(run_id=run_id, action='trigger', duration_seconds=1)
        code, queued = self.post('/api/mask_dropout', good)
        self.assertEqual(code, 202, queued)
        original = self.command_path(run_id).read_bytes()
        invalid = [dict(good, action='unknown'), dict(good, object_id='cube1'),
                   dict(run_id=run_id, action='trigger'),
                   dict(run_id=run_id, action='restore', duration_seconds=1),
                   dict(action='trigger', duration_seconds=1),
                   dict(good, run_id=None)]
        invalid += [dict(good, duration_seconds=x) for x in (True, False, None, '1', 0, .09, 30.01,
                                                           float('nan'), float('inf'), -float('inf'))]
        for payload in invalid:
            with self.subTest(payload=payload):
                self.reject_command(payload)
                self.assertEqual(self.command_path(run_id).read_bytes(), original)

    def test_manual_duration_bounds_are_accepted(self):
        run_id = self.start()['run_id']
        self.progress(run_id)
        for duration in (.1, 30):
            with self.subTest(duration=duration):
                code, response = self.post('/api/mask_dropout', dict(run_id=run_id, action='trigger', duration_seconds=duration))
                self.assertEqual(code, 202, response)
                self.assertEqual(json.loads(self.command_path(run_id).read_text())['duration_seconds'], duration)

    def test_history_and_finished_runs_cannot_receive_manual_commands(self):
        self.reject_command(dict(run_id='task_agent_missing', action='restore'))
        old = self.start()['run_id']
        self.factory.processes[-1].returncode = 0
        current = self.start()['run_id']
        self.progress(current)
        for action in ('trigger', 'restore'):
            payload = dict(run_id=old, action=action)
            if action == 'trigger':
                payload['duration_seconds'] = 1
            self.reject_command(payload)
        self.assertFalse(self.command_path(old).exists())
        self.assertFalse(self.command_path(current).exists())
        self.factory.processes[-1].returncode = 0
        self.reject_command(dict(run_id=current, action='restore'))
        self.assertFalse(self.command_path(current).exists())

    def test_mask_control_requires_same_csrf_and_origin_guards_as_task_start(self):
        run_id = self.start()['run_id']
        self.progress(run_id)
        for token, origin in ((False, None), (True, 'https://unrelated.example')):
            with self.subTest(token=token, origin=origin):
                code, response = self.post('/api/mask_dropout', dict(run_id=run_id, action='trigger', duration_seconds=1),
                                           token=token, origin=origin)
                self.assertEqual(code, 403, response)
        self.assertFalse(self.command_path(run_id).exists())

    def test_state_is_passed_through_status_snapshot_history_and_video_still_works(self):
        config = dict(enabled=True, start_seconds=2.5, duration_seconds=1.25)
        run_id = self.start(config)['run_id']
        state = dict(active=True, source='manual', remaining_seconds=.75, command_id='applied_elsewhere')
        self.progress(run_id, mask_dropout=config, mask_dropout_state=state)
        for entry in (self.console.snapshot(run_id), self.get('/api/status')['run'], self.get('/api/runs')['runs'][0]):
            self.assertEqual(entry['mask_dropout'], config)
            self.assertEqual(entry['mask_dropout_state'], state)
            self.assertEqual(entry['step_generation'], 3)
        folder = self.console.directory(run_id)
        (folder/'execution.mp4').write_bytes(b'0123456789')
        self.factory.processes[-1].returncode = 0
        (folder/'summary.json').write_text(json.dumps(dict(status='stopped', success=False,
            mask_dropout=config, mask_dropout_state=dict(active=False, source='none'))))
        entry = self.get('/api/status')['run']
        self.assertIn('video', entry['artifacts'])
        request = Request(self.url+entry['artifacts']['video'], headers={'Range': 'bytes=2-5'})
        with urlopen(request, timeout=3) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b'2345')

    def test_legacy_record_without_dropout_fields_defaults_off(self):
        folder = self.console.runs_root/'task_agent_legacy'
        folder.mkdir()
        (folder/'summary.json').write_text(json.dumps(dict(status='stopped', success=False, instruction='旧任务')))
        for entry in (self.console.snapshot(folder.name), self.console.status(folder.name)['run'], self.console.history()[0]):
            self.assertEqual(entry['mask_dropout'], OFF)
            self.assertIn('mask_dropout_state', entry)
            self.assertFalse((entry['mask_dropout_state'] or {}).get('active', False))
            self.assertEqual(entry['step_generation'], 0)


class MaskDropoutRunnerCLIContracts(unittest.TestCase):
    def test_start_argv_is_understood_by_runner_without_starting_a_backend(self):
        from task_agent import runner
        with tempfile.TemporaryDirectory() as folder:
            factory = ProcessFactory()
            console = Console(Path(folder)/'runs', process_factory=factory)
            config = dict(enabled=True, start_seconds=2.5, duration_seconds=1.25)
            result = console.start(dict(instruction='球类', mask_dropout=config))
            command = factory.calls[0][0]
            received = {}

            class FakeRunner:
                def __init__(self, **options):
                    received.update(options)

                def run(self):
                    return dict(status='completed')

            module_index = command.index('task_agent.runner')
            with patch.object(runner, 'TaskRunner', FakeRunner), patch('sys.argv', ['task_agent.runner']+command[module_index+1:]), patch('sys.stdout', io.StringIO()):
                self.assertEqual(runner.main(), 0)
            parsed = received['mask_dropout']
            parsed = json.loads(parsed) if isinstance(parsed, str) else parsed
            self.assertEqual(parsed, config)
            self.assertEqual(Path(received['mask_dropout_control']), console.directory(result['run_id'])/'mask_dropout_control.json')


if __name__ == '__main__':
    unittest.main()
