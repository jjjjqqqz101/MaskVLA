"""Runner behavior tests with deterministic fake vision and robot backends.

These tests exercise orchestration and acceptance decisions, not grasp quality.
They never import a simulator, read credentials or issue an API request.
"""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import agentic_act.__main__ as runner


class FakeImage:
    def save(self, path):
        Path(path).write_bytes(b'fake-test-image')


class Scenario:
    def __init__(self, visual_statuses=None, decisions=None, max_attempts=2):
        self.sessions = []
        self.executions = []
        self.visual_statuses = list(visual_statuses or [])
        self.decisions = list(decisions or [])
        self.max_attempts = max_attempts
        self.state = {'gripper_empty': True, 'objects': {
            'cube1': {'placed_region': None, 'held': False, 'stable': True},
            'cube2': {'placed_region': None, 'held': False, 'stable': True},
            'cube3': {'placed_region': None, 'held': False, 'stable': True},
        }}

    def make_client(self, config, output):
        return SimpleNamespace(calls=0, redact=lambda value: str(value))

    def make_session(self, **kwargs):
        scenario = self

        class Session:
            closed = False

            def observe(self):
                return copy.deepcopy(scenario.state)

            def image(self):
                return FakeImage()

            def execute(self, object_id, region_id, attempt, inject_failure=False):
                scenario.executions.append(dict(object_id=object_id,
                    region_id=region_id, attempt=attempt, inject_failure=inject_failure))
                if inject_failure:
                    return {'success': False, 'reason': 'injected_no_motion_timeout',
                            'injected': True}
                scenario.state['objects'][object_id].update(
                    placed_region=region_id, held=False, stable=True)
                return {'success': True, 'reason': 'placement_verified'}

            def close(self):
                self.closed = True

        result = Session()
        self.sessions.append(result)
        return result

    def make_planner(self, client):
        scenario = self

        class Planner:
            def plan(self, instruction, observation, image, repair):
                client.calls += 1
                return {'steps': [
                    {'skill': 'pick_place', 'object_id': 'cube1', 'region_id': 'zone_A'},
                    {'skill': 'pick_place', 'object_id': 'cube2', 'region_id': 'zone_B'},
                ], 'reason': 'Preserve the user instruction order.'}

            def next_action(self, task, plan, observation, completed, attempts,
                            last_result, image, repair):
                client.calls += 1
                if scenario.decisions:
                    return scenario.decisions.pop(0)
                if completed == 2:
                    decision = 'complete'
                else:
                    decision = 'retry' if attempts.get(completed, 0) else 'execute'
                return {'decision': decision, 'step_index': completed,
                        'reason': 'Use the next unfinished step after observed feedback.'}

            def judge(self, step, before, after):
                client.calls += 1
                if scenario.visual_statuses:
                    status = scenario.visual_statuses.pop(0)
                else:
                    obj = scenario.state['objects'][step.object_id]
                    status = ('success' if obj['placed_region'] == step.region_id and
                              obj['stable'] and not obj['held'] else 'failure')
                return {'status': status, 'reason': 'Deterministic test judgment.'}

        return Planner()

    def run(self, directory, inject=False):
        backend = ModuleType('agentic_act.backend')
        backend.RobotSession = self.make_session
        args = SimpleNamespace(order=['cube1:zone_A', 'cube2:zone_B'],
            instruction='先放 cube1 到 zone_A，然后放 cube2 到 zone_B',
            output=Path(directory) / 'run', config=None, checkpoint=None,
            seed=123, device='cpu', inject_first_failure=inject,
            plan_only=False, max_attempts=self.max_attempts)
        with mock.patch.dict(sys.modules, {'agentic_act.backend': backend}), \
             mock.patch.object(runner, 'QwenClient', side_effect=self.make_client), \
             mock.patch.object(runner, 'QwenPlanner', side_effect=self.make_planner), \
             mock.patch('builtins.print'):
            summary = runner.run(args)
        events = [json.loads(line) for line in (args.output / 'events.jsonl').read_text().splitlines()]
        saved = json.loads((args.output / 'summary.json').read_text())
        return summary, events, saved, args.output


class RunnerTests(unittest.TestCase):
    def test_ordered_two_step_success_keeps_one_scene(self):
        scenario = Scenario()
        with tempfile.TemporaryDirectory() as directory:
            summary, events, saved, output = scenario.run(directory)
            self.assertTrue(summary['success'])
            self.assertEqual(summary['status'], 'completed')
            self.assertTrue(summary['final_verification']['success'])
            self.assertTrue(saved['success'])
            self.assertEqual(summary['completed_steps'], 2)
            self.assertEqual([(x['object_id'], x['region_id']) for x in scenario.executions],
                             [('cube1', 'zone_A'), ('cube2', 'zone_B')])
            self.assertEqual([x['step_index'] for x in events if x['event'] == 'step_completed'], [0, 1])
            self.assertEqual(len(scenario.sessions), 1)
            self.assertTrue(scenario.sessions[0].closed)
            self.assertTrue((output / 'report.html').is_file())

    def test_injected_failure_requires_explicit_retry(self):
        scenario = Scenario()
        with tempfile.TemporaryDirectory() as directory:
            summary, events, _, _ = scenario.run(directory, inject=True)
        self.assertTrue(summary['success'])
        self.assertEqual([x['decision'] for x in events if x['event'] == 'decision'],
                         ['execute', 'retry', 'execute', 'complete'])
        self.assertEqual([x['attempt'] for x in scenario.executions], [1, 2, 1])
        self.assertEqual([x['inject_failure'] for x in scenario.executions], [True, False, False])
        self.assertEqual(summary['attempts'], {0: 2, 1: 1})
        self.assertEqual(len(scenario.sessions), 1)

    def test_visual_disagreement_does_not_become_success(self):
        scenario = Scenario(visual_statuses=['failure', 'uncertain'])
        with tempfile.TemporaryDirectory() as directory:
            summary, events, saved, _ = scenario.run(directory)
        self.assertFalse(summary['success'])
        self.assertFalse(saved['success'])
        self.assertEqual(summary['status'], 'visual_verification_disagreement')
        self.assertEqual(summary['completed_steps'], 0)
        self.assertEqual(len(scenario.executions), 1)
        self.assertTrue(scenario.state['objects']['cube1']['placed_region'] == 'zone_A')
        self.assertTrue(any(x['event'] == 'visual_recheck' for x in events))
        self.assertFalse(any(x['event'] == 'task_completed' for x in events))

    def test_early_vlm_completion_is_rejected_before_motion(self):
        early = {'decision': 'complete', 'step_index': 0, 'reason': 'Incorrect early completion.'}
        scenario = Scenario(decisions=[copy.deepcopy(early) for _ in range(3)])
        with tempfile.TemporaryDirectory() as directory:
            summary, events, saved, _ = scenario.run(directory)
        self.assertFalse(summary['success'])
        self.assertFalse(saved['success'])
        self.assertEqual(summary['status'], 'error')
        self.assertEqual(len(scenario.executions), 0)
        self.assertEqual(sum(x['event'] == 'decision_rejected' for x in events), 3)
        self.assertTrue(scenario.sessions[0].closed)

    def test_retry_budget_stops_without_second_skill(self):
        scenario = Scenario(max_attempts=1)
        with tempfile.TemporaryDirectory() as directory:
            summary, events, _, _ = scenario.run(directory, inject=True)
        self.assertFalse(summary['success'])
        self.assertEqual(summary['status'], 'retry_budget_exhausted')
        self.assertEqual(len(scenario.executions), 1)
        self.assertEqual(summary['completed_steps'], 0)
        self.assertTrue(any(x['event'] == 'retry_budget_exhausted' for x in events))


if __name__ == '__main__':
    unittest.main()
