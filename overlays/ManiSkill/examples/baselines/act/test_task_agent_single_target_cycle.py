"""Sequential target identity and post-subtask audit; no models or physics."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from task_agent.runner import TaskRunner
from task_agent.contracts import validate_target


class IdleMonitor:
    def __init__(self, *args, **kwargs): pass
    def start(self): pass
    def stop(self, **kwargs): pass
    def stats(self): return {'inflight_count': 0}


class SingleTargetCycleTests(unittest.TestCase):
    def run_task(self, retry_first=False, complete_after_first=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        calls = []
        state = dict(frame_id=0, captured_at=0., phase='idle', target_text='',
                     step_generation=0, base_rgb=np.zeros((32, 32, 3), np.uint8))
        runner = TaskRunner('依次收拾红色和蓝色方块', 'act', 17, directory.name)
        targets = []
        def context(phase, target, generation):
            calls.append(('context', phase, target))
            state.update(phase=phase, target_text=target, step_generation=generation)
        def target(snapshot, plan, completed, hint=None):
            name = 'red cube' if not completed else 'blue cube'
            calls.append(('bbox', name));targets.append((name, hint))
            return dict(decision='pick', description=name, reason='Visible single cube', bbox_xyxy=[2, 2, 8, 8])
        def grasp(selected, should_stop=None, validate_mask=None):
            calls.append(('grasp', selected['description']))
            if retry_first and sum(c[0] == 'grasp' for c in calls) == 1:
                return {'reason': 'mask_target_mismatch', 'evidence': 'A second object was included'}
            return {'reason': 'grasped'}
        def hold(seconds, should_stop=None):
            state['frame_id'] += 1;state['captured_at'] += 1
            return {'reason': 'hold_completed'}
        def finish(snapshot, completed):
            calls.append(('goal_check', len(completed)))
            done = complete_after_first or len(completed) == 2
            return dict(complete=done, reason='Scene audit', remaining_description='' if done else 'blue cube')
        backend = SimpleNamespace(zone={'capacity': 4}, snapshot=lambda: dict(state),
            set_context=context, grasp=grasp, hold=hold, close=lambda: calls.append(('close',)),
            transport_place=lambda *args: {'reason': 'placement_motion_completed'},
            recover_open_home=lambda *args: {'reason': 'recovery_motion_completed'})
        runner.backend_factory = lambda **kwargs: backend
        runner.vlm = SimpleNamespace(plan=lambda snap: dict(summary='Two cubes', selection_rule='cubes',
                needs_clarification=False, steps=[]), target=target, finish=finish, judge=lambda *args: None)
        runner._verify_place = lambda: 'verified'
        with patch('task_agent.runner.VisualSupervisor', IdleMonitor):
            status = runner.run()
        return status, calls, targets, Path(directory.name)

    def test_each_completed_subtask_is_audited_before_next_single_bbox(self):
        status, calls, targets, directory = self.run_task()
        self.assertEqual(status['status'], 'completed')
        work = [c for c in calls if c[0] in ('bbox', 'grasp', 'goal_check')]
        self.assertEqual(work, [('bbox', 'red cube'), ('grasp', 'red cube'), ('goal_check', 1),
            ('bbox', 'blue cube'), ('grasp', 'blue cube'), ('goal_check', 2), ('goal_check', 2)])
        self.assertEqual(targets[1][1], 'blue cube')
        events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        self.assertEqual([e['completed_steps'] for e in events if e['event']=='post_subtask_goal_check'], [1, 2])

    def test_retry_relocalizes_same_identity_without_premature_global_success(self):
        status, calls, targets, _ = self.run_task(retry_first=True)
        self.assertEqual(status['status'], 'completed')
        self.assertEqual([t[0] for t in targets], ['red cube', 'red cube', 'blue cube'])
        self.assertEqual(targets[1][1]['description'], 'red cube')
        first_audit = next(i for i, c in enumerate(calls) if c[0] == 'goal_check')
        self.assertEqual(sum(c[0] == 'grasp' for c in calls[:first_audit]), 2)

    def test_global_completion_stops_before_a_second_bbox(self):
        status, _, targets, _ = self.run_task(complete_after_first=True)
        self.assertEqual(status['status'], 'completed')
        self.assertEqual(len(targets), 1)

    def test_array_of_boxes_is_rejected_at_grounding_boundary(self):
        with self.assertRaises(ValueError):
            validate_target(dict(decision='pick', description='cubes', reason='Two objects',
                bbox_xyxy=[[2, 2, 8, 8], [12, 12, 18, 18]], coordinate_space='pixels'), 32, 32)

    def test_retiring_target_clears_public_bbox_before_next_grounding(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        runner = TaskRunner('收拾方块', 'act', 17, directory.name)
        runner.backend = SimpleNamespace(set_context=lambda *args: None)
        for phase in ('idle', 'goal_verify'):
            runner.update(active_bbox=[2, 2, 8, 8])
            runner._context(phase, '')
            self.assertIsNone(runner.status['active_bbox'])
        runner.update(active_bbox=[2, 2, 8, 8])
        runner._context('grasp', 'red cube')
        self.assertEqual(runner.status['active_bbox'], [2, 2, 8, 8])


if __name__ == '__main__':
    unittest.main()
