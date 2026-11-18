"""VLM observation boundaries: exercise image assembly without network or models."""
import unittest

import numpy as np

from task_agent.vlm import TaskVLM


class VLMObservationContracts(unittest.TestCase):
    def snapshot(self, **values):
        shape = (12, 16, 3)
        snapshot = dict(base_rgb=np.full(shape, [120, 40, 20], dtype=np.uint8),
                    overview_rgb=np.full(shape, [15, 90, 30], dtype=np.uint8),
                    wrist_rgb=np.full(shape, [5, 10, 200], dtype=np.uint8),
                    mask=np.zeros(shape[:2], dtype=bool), frame_id=20, captured_at=3.,
                    phase='grasp', target_text='左侧红球')
        snapshot.update(values)
        return snapshot

    def client(self, response):
        client = TaskVLM.__new__(TaskVLM)
        client.instruction = '先收拾红球，再收拾蓝球'
        calls = []
        def request(purpose, system, prompt, snapshots, images):
            calls.append(dict(purpose=purpose, system=system, prompt=prompt,
                              snapshots=snapshots, images=images))
            return response, 'observation_test_request'
        client._request = request
        return client, calls

    def test_empty_policy_mask_keeps_raw_evidence_without_local_failure_override(self):
        snapshot = self.snapshot()
        originals = {name: snapshot[name].copy() for name in ('base_rgb', 'overview_rgb', 'wrist_rgb', 'mask')}
        snapshot['mask_dropout_state'] = dict(active=True, source='manual')
        response = dict(assessment='uncertain', decision='continue', confidence=.4,
                        reason='目标被夹爪遮挡，继续观察原始画面')
        client, calls = self.client(response)
        result = client.judge(snapshot)
        self.assertEqual(result, response)
        self.assertEqual(len(calls), 1, 'An empty effective mask must still be visually judged')
        call = calls[0]
        self.assertEqual(call['prompt']['image_order'], ['current overview view', 'current base view',
                         'current wrist view', 'current single-target masked base RGB'])
        for image, color in zip(call['images'][:3], ([15, 90, 30], [120, 40, 20], [5, 10, 200])):
            np.testing.assert_array_equal(np.array(image)[0, 0], color)
        self.assertFalse(np.array(call['images'][-1]).any())
        self.assertIn('empty or partly occluded mask alone is not proof of grasp failure', call['system'])
        self.assertNotIn('mask_dropout_state', call['prompt'])
        for name, original in originals.items():
            np.testing.assert_array_equal(snapshot[name], original)

    def test_previous_scene_is_first_and_current_mask_is_last(self):
        snapshot = self.snapshot()
        snapshot['mask'][3:7, 4:9] = True
        previous = self.snapshot()
        previous['overview_rgb'][:] = [90, 70, 10]
        previous['frame_id'] = 10
        client, calls = self.client(dict(assessment='not_grasped', decision='continue',
                                        confidence=.8, reason='尚未抓起'))
        client.judge(snapshot, previous)
        call = calls[0]
        self.assertEqual(call['prompt']['image_order'][0], 'earlier overview view')
        self.assertEqual(call['prompt']['image_order'][-1], 'current single-target masked base RGB')
        self.assertEqual([s['frame_id'] for s in call['snapshots']], [10, 20])
        np.testing.assert_array_equal(np.array(call['images'][0])[0, 0], [90, 70, 10])
        self.assertFalse(np.array(call['images'][-1])[0, 0].any())
        self.assertTrue(np.array(call['images'][-1]).any())

    def test_verification_keeps_current_target_separate_from_remaining_task(self):
        snapshot = self.snapshot()
        snapshot.update(phase='verify', motion_phase='settle')
        response = dict(assessment='placed', decision='continue', confidence=.9,
                        reason='当前红球已释放在绿色区，蓝球仍待收拾')
        client, calls = self.client(response)
        self.assertEqual(client.judge(snapshot), response)
        call = calls[0]
        self.assertEqual(call['prompt']['current_target'], '左侧红球')
        self.assertEqual(call['prompt']['phase'], 'verify')
        self.assertEqual(call['prompt']['motion_phase'], 'settle')
        self.assertIn('even if other task objects remain outside', call['system'])
        self.assertIn('An empty gripper after release is EXPECTED', call['system'])

    def test_full_task_audit_uses_raw_scene_even_after_empty_target_mask(self):
        snapshot = self.snapshot()
        snapshot.update(phase='goal_verify', target_text='')
        response = dict(complete=False, reason='蓝球还在绿色区外', remaining_description='蓝球')
        client, calls = self.client(response)
        self.assertEqual(client.finish(snapshot, [{'description': '红球'}]), response)
        call = calls[0]
        self.assertEqual(call['purpose'], 'task_completion')
        self.assertEqual(len(call['images']), 2)
        for image in call['images']:
            self.assertTrue(np.array(image).any())
        self.assertEqual(call['prompt']['previous_visual_completions'], [{'description': '红球'}])

    def test_preflight_rejects_multiobject_mask_with_raw_crop_mask_order(self):
        snapshot = self.snapshot()
        snapshot['mask'][3:7, 4:9] = True
        response = dict(valid=False, reason='分割还包含旁边的第二个物体')
        client, calls = self.client(response)
        self.assertEqual(client.check_mask(snapshot), response)
        call = calls[0]
        self.assertEqual(call['purpose'], 'mask_target_confirmation')
        self.assertEqual(len(call['images']), 3)
        np.testing.assert_array_equal(np.array(call['images'][0])[0, 0], [120, 40, 20])
        self.assertFalse(np.array(call['images'][2])[0, 0].any())
        self.assertTrue(np.array(call['images'][1]).any())
        self.assertIn('exactly ONE requested physical object', call['system'])
        self.assertIn('even another object of the same category, return valid=false', call['system'])


if __name__ == '__main__':
    unittest.main()
