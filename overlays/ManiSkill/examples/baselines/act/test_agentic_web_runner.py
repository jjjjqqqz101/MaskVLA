"""Multi-target progress and cooperative cancellation without simulator/API access."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import agentic_act.__main__ as runner
from agentic_act.contracts import ContractError, Step, StopRequested, TaskSpec, validate_plan


class Image:
    def save(self, path):
        Path(path).write_bytes(b'frame')


class WebRunnerTests(unittest.TestCase):
    def exercise(self, temporary, count=3, stop_before=False, stop_after_plan=False):
        root = Path(temporary)
        stop = root / 'stop'
        if stop_before:
            stop.touch()
        steps = [Step('cube'+str(i+1), 'zone_'+chr(65+i)) for i in range(count)]
        state = {'gripper_empty': True, 'objects': {
            'cube'+str(i+1): dict(placed_region=None, held=False, stable=True) for i in range(3)}}
        observed_statuses = []
        executions = []
        created_sessions = []
        output = root / 'run'

        def status():
            observed_statuses.append(json.loads((output/'status.json').read_text())['status'])

        class Session:
            def __init__(self, **kwargs):
                self.closed = False
                created_sessions.append(self)

            def observe(self):
                return copy.deepcopy(state)

            def image(self):
                return Image()

            def execute(self, object_id, region_id, **kwargs):
                status()
                executions.append(object_id)
                state['objects'][object_id].update(placed_region=region_id)
                return {'success': True}

            def close(self):
                self.closed = True

        class Planner:
            def __init__(self, client):
                pass

            def plan(self, *args):
                status()
                if stop_after_plan:
                    stop.touch()
                return {'steps': [step.as_dict() for step in steps]}

            def next_action(self, task, plan, observation, completed, *args):
                status()
                return dict(decision='complete' if completed == len(task.steps) else 'execute',
                            step_index=completed, reason='Use current verified progress.')

            def judge(self, *args):
                status()
                return dict(status='success', reason='Test placement visible.')

        backend = ModuleType('agentic_act.backend')
        backend.RobotSession = Session
        args = SimpleNamespace(order=[s.object_id+':'+s.region_id for s in steps],
            instruction='按照指定顺序放置目标', output=output, config=None, checkpoint=None,
            seed=123, device='cpu', inject_first_failure=False, plan_only=False,
            max_attempts=2, stop_file=stop)
        with mock.patch.dict(sys.modules, {'agentic_act.backend': backend}), \
             mock.patch.object(runner, 'QwenClient', return_value=SimpleNamespace(calls=0, redact=str)), \
             mock.patch.object(runner, 'QwenPlanner', Planner), mock.patch('builtins.print'):
            result = runner.run(args)
        return result, executions, observed_statuses, created_sessions, json.loads((output/'status.json').read_text())

    def test_three_targets_complete_in_order_with_live_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, stages, sessions, final = self.exercise(temporary)
        self.assertTrue(result['success'])
        self.assertEqual(calls, ['cube1', 'cube2', 'cube3'])
        self.assertEqual(result['completed_steps'], 3)
        self.assertEqual(final['total_steps'], 3)
        self.assertTrue(final['finished'])
        self.assertEqual(final['status'], 'completed')
        self.assertIn('planning', stages)
        self.assertIn('executing', stages)
        self.assertIn('judging', stages)
        self.assertTrue(sessions[0].closed)

    def test_single_target_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, _, _, _ = self.exercise(temporary, count=1)
        self.assertTrue(result['success'])
        self.assertEqual(calls, ['cube1'])

    def test_cancel_before_setup_issues_no_actions_or_api_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, _, sessions, final = self.exercise(temporary, stop_before=True)
        self.assertEqual(result['status'], 'stopped')
        self.assertFalse(result['success'])
        self.assertEqual(calls, [])
        self.assertEqual(sessions, [])
        self.assertTrue(final['finished'])

    def test_cancel_during_api_call_is_checked_before_motion(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, _, sessions, final = self.exercise(temporary, stop_after_plan=True)
        self.assertEqual(result['status'], 'stopped')
        self.assertFalse(final['success'])
        self.assertEqual(calls, [])
        self.assertTrue(sessions[0].closed)

    def test_three_step_contract_rejects_duplicate_region_and_wrong_order(self):
        task = TaskSpec('three', tuple(Step('cube'+str(i+1), 'zone_'+chr(65+i)) for i in range(3)))
        self.assertEqual(len(validate_plan({'steps': [s.as_dict() for s in task.steps]}, task)), 3)
        with self.assertRaises(ContractError):
            validate_plan({'steps': [s.as_dict() for s in reversed(task.steps)]}, task)
        with self.assertRaises(ContractError):
            TaskSpec('bad', (Step('cube1','zone_A'), Step('cube2','zone_A')))


def backend_classes():
    """Load the actual cancellation control paths without importing GPU libraries."""
    source = Path(runner.__file__).with_name('backend.py').read_text()
    tree = ast.parse(source)
    recorded = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_RecordedEnv')
    session = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RobotSession')
    session.body = [n for n in session.body if isinstance(n, ast.FunctionDef) and n.name in ('execute','_check_stop')]

    class Wrapper:
        def __init__(self, env):
            self.env = env

    namespace = {'gym': SimpleNamespace(Wrapper=Wrapper), 'StopRequested': StopRequested, 'Path': Path}
    module = ast.Module(body=[recorded, session], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<actual-backend-cancellation>', 'exec'), namespace)
    return namespace['_RecordedEnv'], namespace['RobotSession']


class BackendCancellationTests(unittest.TestCase):
    def test_step_cancels_before_next_physical_action(self):
        Recorded, _ = backend_classes()
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary)/'stop'
            calls = []
            def check():
                if marker.exists():
                    raise StopRequested('stop')
            def step(action):
                calls.append(action)
                marker.touch()
                return 'state'
            owner = SimpleNamespace(_check_stop=check, _after_step=lambda *args: None)
            env = Recorded(SimpleNamespace(step=step), owner)
            with self.assertRaises(StopRequested):
                env.step('first')
            with self.assertRaises(StopRequested):
                env.step('second')
            self.assertEqual(calls, ['first'])

    def fake_session(self, cancel_in_planner=False):
        _, Session = backend_classes()
        robot = Session()
        robot.stop_file = None
        robot.steps = 0
        robot.zones = {'zone_A': {}}
        state = {'gripper_empty': True, 'objects': {'cube1': {'held': False}}}
        robot.observe = lambda: state
        robot._event = lambda *args, **kwargs: None
        calls = []
        def pick(index):
            calls.append('act')
            if not cancel_in_planner:
                raise StopRequested('cancel during ACT')
            return True, 'held', 1
        robot._act_pick = pick
        def planner():
            calls.append('planner')
            raise StopRequested('cancel during planning')
        robot._planner = planner
        return robot, calls

    def test_cancel_during_act_skips_recovery_actions(self):
        robot, calls = self.fake_session()
        with self.assertRaises(StopRequested):
            robot.execute('cube1', 'zone_A')
        self.assertEqual(calls, ['act'])

    def test_cancel_during_planning_skips_recovery_actions(self):
        robot, calls = self.fake_session(cancel_in_planner=True)
        with self.assertRaises(StopRequested):
            robot.execute('cube1', 'zone_A')
        self.assertEqual(calls, ['act', 'planner'])


if __name__ == '__main__':
    unittest.main()
