"""ACT RGB-only perception boundary tests; no simulator, network or checkpoint.

Run in the ACT environment: python -m unittest test_act_perception_integration -v
Selected definitions are loaded without starting the HTTP lab or a simulator.
"""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from urllib.parse import urlparse

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T

from train_dual_rgb import (DualRGBObservationWrapper, dual_rgb,
                            policy_state as dual_policy_state)
from train_single_rgb import policy_state, single_rgb


def load_boundary():
    source = Path(__file__).with_name('target_test_web.py')
    names = {'array', 'perception_config', 'TrackingObservationWrapper', 'hold_joint_action'}
    tree = ast.parse(source.read_text(), filename=str(source))
    selected = ast.Module(body=[node for node in tree.body if getattr(node, 'name', None) in names], type_ignores=[])
    scope = dict(np=np, torch=torch, gym=gym, T=T, threading=threading,
                 urlparse=urlparse, dual_rgb=dual_rgb, dual_policy_state=dual_policy_state,
                 policy_state=policy_state, single_rgb=single_rgb)
    exec(compile(selected, str(source), 'exec'), scope)
    return scope


BOUNDARY = load_boundary()
TrackingObservationWrapper = BOUNDARY['TrackingObservationWrapper']
perception_config = BOUNDARY['perception_config']
hold_joint_action = BOUNDARY['hold_joint_action']


class FakeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({})
        self.cubes = [SimpleNamespace(per_scene_id=torch.tensor([7], dtype=torch.int32)),
                      SimpleNamespace(per_scene_id=torch.tensor([9], dtype=torch.int32))]
        self.target_idx = torch.tensor([0])
        seg = torch.full((1, 224, 224, 1), 9, dtype=torch.int32)
        seg[:, 90:130, 90:130] = 7
        qpos = torch.zeros((1, 9), dtype=torch.float32)
        qpos[:, 7:] = .03
        self._init_raw_obs = dict(agent=dict(qpos=qpos, qvel=torch.zeros_like(qpos)),
            extra=dict(tcp_pose=torch.zeros((1, 7)), goal_pos=torch.zeros((1, 3))),
            sensor_data=dict(base_camera=dict(rgb=torch.full((1, 224, 224, 3), 128, dtype=torch.uint8),
                                              segmentation=seg)))

    def update_obs_space(self, observation):
        self.observation_space = gym.spaces.Dict({
            k: gym.spaces.Box(0, 255, tuple(v.shape[1:]), dtype=np.uint8)
            if k == 'rgb' else gym.spaces.Box(-np.inf, np.inf, tuple(v.shape[1:]), dtype=np.float32)
            for k, v in observation.items()})


class FakeClient:
    def __init__(self, mask, generation=1):
        self.mask = mask
        self.generation = generation
        self.calls = []
        self.closed = 0

    def _result(self):
        return SimpleNamespace(mask=self.mask, metadata=dict(state='tracking', quality=.9,
            generation=self.generation, bbox_xyxy=[0, 0, 40, 40], latency_ms=3.0))

    def start(self, rgb, target_text, options):
        self.calls.append(('start', rgb.copy(), target_text, dict(options)))
        return self._result()

    def step(self, rgb, frame_index, force_reacquire=False):
        self.calls.append(('step', rgb.copy(), frame_index, force_reacquire))
        if force_reacquire:
            self.generation += 1
        return self._result()

    def close(self):
        self.closed += 1


class PerceptionBoundaryTests(unittest.TestCase):
    def test_oracle_default_exactly_matches_existing_wrapper(self):
        baseline = DualRGBObservationWrapper(FakeEnv())
        candidate = TrackingObservationWrapper(FakeEnv())
        expected = baseline.observation(baseline.base_env._init_raw_obs)
        actual = candidate.observation(candidate.base_env._init_raw_obs)
        self.assertTrue(torch.equal(expected['rgb'], actual['rgb']))
        self.assertTrue(torch.equal(expected['state'], actual['state']))
        self.assertEqual(candidate.mask_source, 'oracle')

    def test_learned_mask_replaces_gt_without_leaking_labels(self):
        env = FakeEnv()
        wrapper = TrackingObservationWrapper(env)
        wrong_mask = np.zeros((224, 224), dtype=bool)
        wrong_mask[:40, :40] = True
        client = FakeClient(wrong_mask)
        original = env._init_raw_obs['sensor_data']['base_camera']['rgb']
        options = dict(loss_mode='observe', intervention=dict(kind='none'))
        result = wrapper.start_perception(env._init_raw_obs, 'left red cube', options, client)
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        self.assertEqual(call[0], 'start')
        np.testing.assert_array_equal(call[1], original[0].numpy())
        self.assertEqual(call[2:], ('left red cube', options))
        self.assertTrue(torch.equal(result['rgb'][:, 0], original.permute(0, 3, 1, 2)))
        self.assertTrue(result['rgb'][0, 1, :, :40, :40].any())
        self.assertFalse(result['rgb'][0, 1, :, 90:130, 90:130].any())
        self.assertEqual(wrapper.last_metrics['gt_iou'], 0.0)
        self.assertEqual(wrapper.last_metrics['mask_source'], 'perception')

    def test_empty_perception_mask_stays_empty_and_never_uses_gt(self):
        env = FakeEnv()
        wrapper = TrackingObservationWrapper(env)
        client = FakeClient(np.zeros((224, 224), dtype=bool))
        result = wrapper.start_perception(env._init_raw_obs, 'banana', {}, client)
        self.assertFalse(result['rgb'][:, 1].any())
        self.assertTrue(result['rgb'][:, 0].any())
        self.assertEqual(wrapper.last_metrics['policy_mask_pixels'], 0)
        self.assertGreater(wrapper.last_metrics['gt_mask_pixels'], 0)
        self.assertEqual(wrapper.last_metrics['gt_iou'], 0.0)

    def test_explicit_session_lifetime_and_reacquire_flag(self):
        env, signal = FakeEnv(), threading.Event()
        wrapper = TrackingObservationWrapper(env, reacquire=signal)
        client = FakeClient(np.ones((224, 224), dtype=bool))
        self.assertEqual(client.calls, [])
        wrapper.start_perception(env._init_raw_obs, 'banana', {}, client)
        signal.set()
        wrapper.observation(env._init_raw_obs)
        self.assertEqual([c[0] for c in client.calls], ['start', 'step'])
        self.assertEqual(client.calls[-1][2:], (1, True))
        self.assertFalse(signal.is_set())
        self.assertEqual(wrapper.last_metrics['generation'], 2)
        wrapper.close_perception()
        self.assertEqual(client.closed, 1)
        with self.assertRaisesRegex(RuntimeError, '已关闭'):
            wrapper.observation(env._init_raw_obs)
        wrapper.use_oracle()
        self.assertTrue(wrapper.observation(env._init_raw_obs)['rgb'][:, 1].any())

    def test_bad_mask_and_service_error_are_not_replaced_with_gt(self):
        for mask in (np.ones((10, 10), dtype=bool), np.ones((224, 224), dtype=np.uint8)):
            env = FakeEnv()
            wrapper = TrackingObservationWrapper(env)
            with self.assertRaisesRegex(ValueError, '未回退'):
                wrapper.start_perception(env._init_raw_obs, 'mug', {}, FakeClient(mask))
            self.assertEqual(wrapper.mask_source, 'perception')
        class Broken(FakeClient):
            def start(self, *args):
                raise RuntimeError('service unavailable')
        env = FakeEnv()
        wrapper = TrackingObservationWrapper(env)
        with self.assertRaisesRegex(RuntimeError, 'service unavailable'):
            wrapper.start_perception(env._init_raw_obs, 'mug', {}, Broken(None))
        self.assertEqual(wrapper.mask_source, 'perception')

    def test_masked_only_mode_keeps_single_image_contract(self):
        env = FakeEnv()
        wrapper = TrackingObservationWrapper(env, input_mode='masked_only')
        result = wrapper.start_perception(env._init_raw_obs, 'mug', {},
            FakeClient(np.zeros((224, 224), dtype=bool)))
        self.assertEqual(tuple(result['rgb'].shape), (1, 1, 3, 224, 224))
        self.assertFalse(result['rgb'].any())
        self.assertEqual(tuple(result['state'].shape), (1, 28))

    def test_pause_action_holds_joint_targets_and_current_gripper(self):
        state = torch.zeros((1, 28))
        state[0, :7] = torch.arange(7) / 10
        state[0, 7:9] = .03
        action = hold_joint_action(dict(state=state))
        np.testing.assert_allclose(action[0, :7], state[0, :7])
        self.assertAlmostEqual(float(action[0, 7]), .5, places=5)
        self.assertEqual(action.shape, (1, 8))

    def test_config_requires_visible_description_and_bounded_intervention(self):
        with self.assertRaises(ValueError):
            perception_config(dict(mask_source='perception'))
        with self.assertRaises(ValueError):
            perception_config(dict(intervention=dict(duration=0)))
        with self.assertRaises(ValueError):
            perception_config(dict(max_pause_frames=0))
        with self.assertRaises(ValueError):
            perception_config(dict(endpoint='http://example.com:18830'))
        with self.assertRaises(ValueError):
            perception_config(dict(endpoint=None))
        valid = perception_config(dict(mask_source='perception', target_text=' left red cube ',
            intervention=dict(kind='dropout', start=20, duration=10)))
        self.assertEqual(valid['target_text'], 'left red cube')
        self.assertEqual(valid['intervention']['dx'], 20)
        self.assertEqual(perception_config()['mask_source'], 'oracle')


if __name__ == '__main__':
    unittest.main()
