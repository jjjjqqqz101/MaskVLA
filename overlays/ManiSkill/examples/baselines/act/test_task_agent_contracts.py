"""CPU-only boundaries for visual task votes, bbox transfer and final audit."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from perception_mask.client import PerceptionMaskClient, encode_rgb
from perception_mask.server import Application
from task_agent.contracts import (StopRequested, validate_judgement,
                                  validate_mask_check, validate_target)
from task_agent.runner import AbortRequested, TaskRunner, VisualVotes
from task_agent.vlm import TaskVLM


class VisualVoteContracts(unittest.TestCase):
    def setUp(self):
        self.votes = VisualVotes(max_age=4)
        self.snapshot = dict(step_generation=3, phase='verify', target_text='red ball')
        self.clock = patch('task_agent.runner.time.monotonic', return_value=100.)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def row(self, frame, at, **changes):
        row = dict(self.snapshot, accepted=True, frame_id=frame, captured_at=at,
                   assessment='placed', decision='continue', confidence=.6)
        row.update(changes)
        return row

    def test_two_distinct_spaced_fresh_votes_confirm(self):
        self.votes.add(self.row(11, 98.8))
        self.assertFalse(self.votes.confirmed(self.snapshot, 'placed'))
        self.votes.add(self.row(12, 99.6))
        self.assertTrue(self.votes.confirmed(self.snapshot, 'placed'))

    def test_six_second_window_accepts_observed_vlm_delay_but_still_expires_old_votes(self):
        delayed = [self.row(11, 95.6), self.row(12, 96.6)]
        for row in delayed:
            self.votes.add(row)
        self.assertFalse(self.votes.confirmed(self.snapshot, 'placed'))
        votes = VisualVotes(max_age=6)
        for row in delayed:
            votes.add(row)
        self.assertTrue(votes.confirmed(self.snapshot, 'placed'))
        votes.clear()
        votes.add(self.row(13, 93.9))
        votes.add(self.row(14, 94.9))
        self.assertFalse(votes.confirmed(self.snapshot, 'placed'))

    def test_duplicates_close_frames_low_confidence_and_rejected_votes_do_not_confirm(self):
        for changes in ({'frame_id': 11}, {'captured_at': 98.9},
                        {'confidence': .59}, {'accepted': False},
                        {'decision': 'retry'}, {'decision': 'abort'}):
            with self.subTest(changes=changes):
                votes = VisualVotes()
                votes.add(self.row(11, 98.8))
                votes.add(self.row(12, 99.6, **changes))
                self.assertFalse(votes.confirmed(self.snapshot, 'placed'))

    def test_stale_future_and_other_contexts_cannot_confirm_or_abort(self):
        for changes in ({'captured_at': 95.}, {'captured_at': 101.},
                        {'step_generation': 2}, {'phase': 'transport'},
                        {'target_text': 'blue ball'}):
            with self.subTest(changes=changes):
                votes = VisualVotes()
                votes.add(self.row(11, 98.8))
                votes.add(self.row(12, 99.6, **changes))
                self.assertFalse(votes.confirmed(self.snapshot, 'placed'))
                votes.add(self.row(13, 99.8, **dict(changes, decision='abort')))
                self.assertFalse(votes.abort(self.snapshot))

    def test_an_uncertain_recent_vote_breaks_consecutive_confirmation(self):
        self.votes.add(self.row(11, 98.))
        self.votes.add(self.row(12, 98.6, assessment='uncertain'))
        self.votes.add(self.row(13, 99.5))
        self.assertFalse(self.votes.confirmed(self.snapshot, 'placed'))
        self.votes.add(self.row(14, 100.))
        self.assertTrue(self.votes.confirmed(self.snapshot, 'placed'))

    def test_retry_requires_two_confident_current_opinions(self):
        self.votes.add(self.row(11, 98.8, decision='retry', assessment='failed'))
        self.assertFalse(self.votes.retry(self.snapshot))
        self.votes.add(self.row(12, 99.6, decision='retry', assessment='failed', confidence=.59))
        self.assertFalse(self.votes.retry(self.snapshot))
        self.votes.add(self.row(13, 100., decision='retry', assessment='failed'))
        self.assertFalse(self.votes.retry(self.snapshot))
        self.votes.add(self.row(14, 100., decision='retry', assessment='failed'))
        self.assertTrue(self.votes.retry(self.snapshot))

    def test_empty_gripper_after_release_does_not_retry_or_confirm_success(self):
        for motion_phase in ('release', 'retreat_raise', 'retreat_home', 'retreat_joint_home', 'settle'):
            with self.subTest(motion_phase=motion_phase):
                snapshot = dict(self.snapshot, phase='transport', motion_phase=motion_phase)
                votes = VisualVotes()
                for frame, at in ((11, 98.8), (12, 99.6)):
                    votes.add(self.row(frame, at, phase='transport', assessment='not_grasped',
                                       decision='retry', confidence=.95))
                self.assertFalse(votes.retry(snapshot))
                self.assertFalse(votes.confirmed(snapshot, 'placed'))
                self.assertFalse(votes.confirmed(snapshot, 'grasped_lifted'))

    def test_not_grasped_before_release_still_requests_retry(self):
        for phase, motion_phase in (('grasp', 'grasp_lift'), ('transport', 'transport_raise'),
                                    ('transport', 'transport_to_green_zone'),
                                    ('transport', 'deposit_lower'), ('transport', 'deposit_settle_closed')):
            with self.subTest(phase=phase, motion_phase=motion_phase):
                snapshot = dict(self.snapshot, phase=phase, motion_phase=motion_phase)
                votes = VisualVotes()
                for frame, at in ((11, 98.8), (12, 99.6)):
                    votes.add(self.row(frame, at, phase=phase, assessment='not_grasped',
                                       decision='retry', confidence=.95))
                self.assertTrue(votes.retry(snapshot))

    def test_explicit_placement_failure_after_release_still_requires_confident_retry(self):
        for motion_phase in ('release', 'retreat_raise', 'retreat_joint_home', 'settle'):
            with self.subTest(motion_phase=motion_phase):
                snapshot = dict(self.snapshot, phase='transport', motion_phase=motion_phase)
                votes = VisualVotes()
                # An explicit failure is actionable even if the model's decision
                # field did not separately repeat "retry".
                votes.add(self.row(11, 98.8, phase='transport', assessment='failed',
                                   decision='continue', confidence=.95))
                votes.add(self.row(12, 99.6, phase='transport', assessment='failed',
                                   decision='continue', confidence=.95))
                self.assertTrue(votes.retry(snapshot))
                votes.add(self.row(13, 100., phase='transport', assessment='failed',
                                   decision='retry', confidence=.59))
                self.assertFalse(votes.retry(snapshot))


class TargetContracts(unittest.TestCase):
    def target(self, box, **changes):
        row = dict(decision='pick', reason='Visible red ball outside collection area',
                   description='red ball', bbox_xyxy=box)
        row.update(changes)
        return row

    def test_normalized_box_maps_to_original_dimensions_not_upscaled_vlm_input(self):
        result = validate_target(self.target([100, 200, 500, 600]), 200, 100)
        self.assertEqual(result['bbox_xyxy'], [20., 20., 100., 60.])
        result = validate_target(self.target([20, 20, 100, 60], coordinate_space='pixels'), 200, 100)
        self.assertEqual(result['bbox_xyxy'], [20., 20., 100., 60.])

    def test_invalid_box_or_model_confidence_cannot_become_a_success(self):
        for box in ([500, 200, 100, 600], [-1, 200, 500, 600],
                    [100, 200, 1001, 600], [100, 200, float('nan'), 600]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_target(self.target(box), 200, 100)
        for confidence in (True, float('nan'), float('inf'), -.1, 1.1):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                validate_judgement(dict(assessment='placed', decision='complete',
                                        confidence=confidence, reason='Looks placed'))


class FakeLocator:
    def __init__(self):
        self.calls = []

    def locate(self, image, description):
        self.calls.append(description)
        return dict(found=True, bbox_xyxy=[10, 10, 18, 18], coordinate_space='pixels')


class FakeSegmenter:
    def __init__(self):
        self.boxes = []

    def segment(self, image, box):
        self.boxes.append(list(box))
        mask = np.zeros(image.shape[:2], dtype=bool)
        x0, y0, x1, y1 = map(int, box)
        mask[y0:y1, x0:x1] = True
        return mask, .99


class FakeTracker:
    def __init__(self):
        self.mask = None
        self.closed = False

    def initialize(self, image, mask): self.mask = mask.copy()
    def predict(self, image): return self.mask.copy(), .99
    def accept(self): pass
    def reject(self): pass
    def close(self): self.closed = True


class UpstreamBoxContracts(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        self.locator = FakeLocator()
        self.segmenter = FakeSegmenter()
        self.trackers = []

        def tracker_factory():
            tracker = FakeTracker()
            self.trackers.append(tracker)
            return tracker

        self.app = Application(self.locator, self.segmenter, tracker_factory)
        self.addCleanup(lambda: [s.close() for s in self.app.sessions.values()])

    def payload(self):
        return dict(image_png=encode_rgb(self.rgb), target_text='red ball',
                    bbox_xyxy=[2, 3, 8, 9], request_id='task_vlm_000007')

    def test_client_server_use_task_box_once_and_force_reacquire_grounds_again(self):
        client = PerceptionMaskClient()
        # Exercise the actual new client endpoint and real service contract,
        # replacing only transport and learned perception adapters.
        client._request = lambda path, data=None, timeout=None: self.app.request(path, data)
        first = client.start_from_vlm(self.rgb, 'red ball', [2, 3, 8, 9],
                                      request_id='task_vlm_000007')
        self.assertEqual(self.locator.calls, [])
        self.assertEqual(self.segmenter.boxes, [[2., 3., 8., 9.]])
        self.assertTrue(first.mask[3:9, 2:8].all())
        event = next(e for e in first.metadata['events'] if e['type'] == 'vlm_bbox')
        self.assertEqual(event['origin'], 'upstream_task_vlm')
        self.assertEqual(event['request_id'], 'task_vlm_000007')
        self.assertEqual(client.step(self.rgb, 1).metadata['generation'], 1)
        self.assertEqual(self.locator.calls, [])
        reacquired = client.step(self.rgb, 2, force_reacquire=True)
        self.assertEqual(self.locator.calls, ['red ball'])
        self.assertEqual(reacquired.metadata['generation'], 2)
        self.assertTrue(self.trackers[0].closed)
        client.close()
        self.assertEqual(self.app.sessions, {})

    def test_service_rejects_privileged_fields_at_both_boundaries(self):
        for key in ('gt_mask', 'segmentation', 'target_idx', 'object_id', 'target_pose',
                    'world_position', 'is_grasping'):
            with self.subTest(field=key, boundary='request'), self.assertRaises(ValueError):
                self.app.request('/v1/start_from_vlm', dict(self.payload(), **{key: 1}))
            with self.subTest(field=key, boundary='options'), self.assertRaises(ValueError):
                self.app.request('/v1/start_from_vlm', dict(self.payload(), options={key: 1}))
        self.assertEqual(self.locator.calls, [])
        self.assertEqual(self.app.sessions, {})
        self.assertEqual(self.trackers, [])


class MaskPreflightContracts(unittest.TestCase):
    def make_runner(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        runner = TaskRunner('收拾球类', 'pi05', 17, directory.name, max_retries=1)
        state = dict(frame_id=0, captured_at=0., phase='idle', target_text='', step_generation=0,
                     base_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
                     mask=np.ones((32, 32), dtype=bool))
        targets = []
        grasps = []
        accepted_skills = []
        recoveries = []

        def set_context(phase, target, generation):
            state.update(phase=phase, target_text=target, step_generation=generation)

        def grasp(target, should_stop=None, validate_mask=None):
            grasps.append(dict(target))
            self.assertTrue(callable(validate_mask))
            validation = validate_mask(dict(state))
            if not validation['valid']:
                return dict(reason='mask_target_mismatch', steps=0, evidence=validation['reason'])
            accepted_skills.append(dict(target))
            return dict(reason='abort', steps=0)

        def recover(should_stop=None):
            recoveries.append(True)
            return dict(reason='recovery_motion_completed')

        def target(snapshot, plan, completed, pending_hint=None):
            targets.append(pending_hint)
            return dict(decision='pick', description='red ball', reason='Visible ball',
                        bbox_xyxy=[2, 3, 8, 9] if len(targets) == 1 else [10, 12, 20, 22])

        backend = SimpleNamespace(snapshot=lambda: dict(state), set_context=set_context,
            grasp=grasp, recover_open_home=recover, zone={'capacity': 4}, closed=False)
        def close(): backend.closed = True
        backend.close = close
        runner.backend_factory = lambda **unused: backend
        runner.backend = backend
        runner.vlm = SimpleNamespace(
            plan=lambda snapshot: dict(summary='Collect balls', selection_rule='visible spheres',
                                       needs_clarification=False, steps=[]),
            target=target, judge=lambda *args: dict(assessment='uncertain', decision='continue',
                                                   confidence=.1, reason='No physical claim'))
        return runner, targets, grasps, accepted_skills, recoveries

    def test_retry_hint_keeps_failed_box_and_visual_evidence_before_corrected_grasp(self):
        runner, hints, grasps, accepted, recoveries = self.make_runner()
        checks = []
        evidence = 'Segmented pixels show wooden table, not the requested red ball'
        def check_mask(snapshot):
            checks.append(snapshot['step_generation'])
            return dict(valid=len(checks) > 1, reason=evidence if len(checks) == 1 else 'Visible red ball silhouette')
        runner.vlm.check_mask = check_mask
        result = runner.run()
        self.assertEqual(result['status'], 'aborted')
        self.assertEqual(len(hints), 2)
        self.assertEqual(hints[1]['description'], 'red ball')
        self.assertEqual(hints[1]['previous_bbox_xyxy'], [2, 3, 8, 9])
        self.assertEqual(hints[1]['failure_reason'], 'mask_target_mismatch')
        self.assertEqual(hints[1]['evidence'], evidence)
        self.assertEqual([g['bbox_xyxy'] for g in grasps], [[2, 3, 8, 9], [10, 12, 20, 22]])
        self.assertEqual([g['bbox_xyxy'] for g in accepted], [[10, 12, 20, 22]])
        self.assertEqual(checks, [1, 2])
        self.assertEqual(len(recoveries), 1)
        self.assertTrue(runner.backend.closed)

    def test_stop_during_preflight_cannot_start_skill_or_retry(self):
        runner, hints, grasps, accepted, recoveries = self.make_runner()
        def stop_while_checking(snapshot):
            runner.stop.set()
            return dict(valid=True, reason='Visual check completed after user stop')
        runner.vlm.check_mask = stop_while_checking
        result = runner.run()
        self.assertEqual(result['status'], 'stopped')
        self.assertFalse(result['success'])
        self.assertEqual(len(grasps), 1)
        self.assertEqual(len(hints), 1)
        self.assertEqual(accepted, [])
        self.assertEqual(recoveries, [])
        self.assertTrue(runner.backend.closed)

    def test_recovery_callback_ignores_repeated_retry_but_preserves_abort_and_stop(self):
        for situation, expected_status in (('retry', 'needs_input'), ('abort', 'aborted'), ('stop', 'stopped')):
            with self.subTest(situation=situation):
                runner, hints, grasps, accepted, _ = self.make_runner()
                runner.vlm.check_mask = lambda snapshot: dict(valid=False, reason='Wrong target mask')
                callback_results = []

                def recover(should_stop=None):
                    snapshot = runner.backend.snapshot()
                    now = time.monotonic()
                    for frame, age in ((11, .8), (12, .1)):
                        runner.votes.add(dict(snapshot, accepted=True, frame_id=frame,
                            captured_at=now-age, assessment='failed', decision='retry', confidence=.95))
                    if situation == 'abort':
                        runner.votes.add(dict(snapshot, accepted=True, frame_id=13, captured_at=now,
                            assessment='failed', decision='abort', confidence=.95))
                    elif situation == 'stop':
                        runner.stop.set()
                    decision = should_stop()
                    callback_results.append(decision)
                    # Finish this isolated run after a normal recovery, before
                    # another learned skill can begin.
                    runner.vlm.target = lambda *args: dict(decision='uncertain', reason='End isolated test')
                    return dict(reason=decision or 'recovery_motion_completed')

                runner.backend.recover_open_home = recover
                result = runner.run()
                self.assertEqual(result['status'], expected_status)
                self.assertFalse(result['success'])
                self.assertTrue(runner.backend.closed)
                self.assertEqual(len(grasps), 1)
                self.assertEqual(len(hints), 1)
                self.assertEqual(accepted, [])
                self.assertEqual(callback_results, [None] if situation == 'retry' else
                                 ['abort'] if situation == 'abort' else [])

    def test_existing_stop_prevents_a_new_preflight_model_call(self):
        runner, *_ = self.make_runner()
        def forbidden(snapshot):
            raise AssertionError('Stopped task must not start a new mask judgement')
        runner.vlm.check_mask = forbidden
        runner.stop.set()
        with self.assertRaises(StopRequested):
            runner._validate_mask(runner.backend.snapshot())

    def test_empty_mask_rejected_without_vlm_and_valid_flag_is_strict_boolean(self):
        vlm = TaskVLM.__new__(TaskVLM)
        def forbidden(*args):
            raise AssertionError('Empty mask needs no external model call')
        vlm._request = forbidden
        result = vlm.check_mask(dict(base_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
                                     mask=np.zeros((32, 32), dtype=bool)))
        self.assertFalse(result['valid'])
        for invalid in ('false', 'true', 0, 1, None):
            with self.subTest(valid=invalid), self.assertRaises(ValueError):
                validate_mask_check(dict(valid=invalid, reason='Background pixels'))
        self.assertFalse(validate_mask_check(dict(valid=False, reason='Background pixels'))['valid'])


class FinalAuditContracts(unittest.TestCase):
    def runner(self, hold_reason='hold_completed', finish_effect=None, hold_effect=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        runner = TaskRunner('收拾球类', 'pi05', 17, directory.name)
        calls = []
        audited_snapshots = []
        holds = []

        def finish(snapshot, completed):
            calls.append(snapshot['frame_id'])
            audited_snapshots.append(dict(snapshot))
            if finish_effect:
                finish_effect(runner, len(calls))
            return dict(complete=True, reason='All balls visibly released inside collection area',
                        remaining_description='')

        state = dict(frame_id=0, captured_at=100., phase='idle', target_text='', step_generation=0)

        def set_context(phase, target, generation):
            state.update(phase=phase, target_text=target, step_generation=generation)

        def hold(seconds, should_stop=None):
            state['frame_id'] += 1
            state['captured_at'] += 1
            holds.append(dict(seconds=seconds, frame_id=state['frame_id'], captured_at=state['captured_at']))
            if hold_effect:
                hold_effect(runner, len(holds))
            reason = hold_reason[len(holds)-1] if isinstance(hold_reason, (list, tuple)) else hold_reason
            return dict(reason=reason)

        runner.backend = SimpleNamespace(snapshot=lambda: dict(state),
            set_context=set_context, hold=hold, holds=holds, audited_snapshots=audited_snapshots)
        runner.vlm = SimpleNamespace(finish=finish)
        return runner, calls

    def test_final_completion_requires_two_different_scene_observations(self):
        runner, calls = self.runner()
        self.assertTrue(runner._goal_audit()['complete'])
        self.assertEqual(calls, [1, 2])
        self.assertEqual([h['seconds'] for h in runner.backend.holds], [.1, 1.1])
        self.assertEqual([s['captured_at'] for s in runner.backend.audited_snapshots], [101., 102.])
        self.assertEqual([(s['frame_id'], s['captured_at']) for s in runner.backend.audited_snapshots],
                         [(h['frame_id'], h['captured_at']) for h in runner.backend.holds])

    def test_live_abort_cannot_be_overridden_by_second_completion_call(self):
        for abort_at in (1, 2):
            with self.subTest(abort_at=abort_at):
                runner, calls = self.runner(['abort'] if abort_at == 1 else ['hold_completed', 'abort'])
                with self.assertRaises(AbortRequested):
                    runner._goal_audit()
                self.assertEqual(calls, [] if abort_at == 1 else [1])

    def test_live_retry_blocks_terminal_completion(self):
        for retry_at in (1, 2):
            with self.subTest(retry_at=retry_at):
                runner, calls = self.runner(['retry'] if retry_at == 1 else ['hold_completed', 'retry'])
                self.assertFalse(runner._goal_audit()['complete'])
                self.assertEqual(calls, [] if retry_at == 1 else [1])

    def test_stop_before_or_during_either_hold_prevents_next_final_call(self):
        runner, calls = self.runner()
        runner.stop.set()
        with self.assertRaises(StopRequested):
            runner._goal_audit()
        self.assertEqual(calls, [])
        self.assertEqual(runner.backend.holds, [])
        for stop_at in (1, 2):
            with self.subTest(stop_at=stop_at):
                def stop_during_hold(runner, number):
                    if number == stop_at:
                        runner.stop.set()
                runner, calls = self.runner(hold_effect=stop_during_hold)
                with self.assertRaises(StopRequested):
                    runner._goal_audit()
                self.assertEqual(calls, [] if stop_at == 1 else [1])

    def test_stop_during_either_blocking_final_call_wins_over_success(self):
        for stop_at in (1, 2):
            with self.subTest(stop_at=stop_at):
                def stop_on_call(runner, number):
                    if number == stop_at:
                        runner.stop.set()
                runner, calls = self.runner(finish_effect=stop_on_call)
                with self.assertRaises(StopRequested):
                    runner._goal_audit()
                self.assertEqual(len(calls), stop_at)

    def test_run_stop_and_vlm_error_close_resources_and_persist_truthful_terminal_state(self):
        for failure, expected in (('stop', 'stopped'), ('error', 'error'), ('abort', 'aborted')):
            with self.subTest(failure=failure):
                def interrupt(runner, number):
                    if number == 2:
                        if failure == 'stop':
                            runner.stop.set()
                        elif failure == 'error':
                            raise RuntimeError('simulated VLM failure')

                runner, _ = self.runner('abort' if failure == 'abort' else 'hold_completed', interrupt)
                backend = runner.backend
                backend.zone = {'capacity': 4}
                backend.closed = False
                def close(): backend.closed = True
                backend.close = close
                runner.backend_factory = lambda **unused: backend
                runner.vlm.plan = lambda snapshot: dict(summary='Collect visible balls',
                    selection_rule='visible spheres', needs_clarification=False, steps=[])
                runner.vlm.target = lambda *args: dict(decision='complete', reason='No remaining ball')
                runner.vlm.judge = lambda *args: dict(assessment='uncertain', decision='continue',
                                                       confidence=.1, reason='No physical claim')
                result = runner.run()
                self.assertEqual(result['status'], expected)
                self.assertFalse(result['success'])
                self.assertTrue(backend.closed)
                self.assertFalse(runner.monitor.stats()['running'])
                self.assertEqual(runner.votes.max_age, 6)
                self.assertEqual(runner.monitor.max_age, runner.votes.max_age)
                self.assertEqual(result['visual_max_age_seconds'], runner.votes.max_age)
                summary = json.loads((Path(runner.output)/'summary.json').read_text())
                self.assertEqual(summary['status'], expected)
                self.assertFalse(summary['success'])


if __name__ == '__main__':
    unittest.main()
