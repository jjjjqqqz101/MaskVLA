"""Integration checks for stored/offline and live/online dual RGB observations."""
import argparse
import copy
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_dual_rgb import (
    Args, Agent, DualRGBObservationWrapper, SmallDemoDataset_ACTPolicy,
    STATE_FIELDS, STREAM_ORDER, policy_state,
)
from act.make_env import make_eval_envs


def state_at(group, frame):
    return {key: state_at(value, frame) if isinstance(value, h5py.Group)
            else torch.as_tensor(value[frame:frame + 1]) for key, value in group.items()}


def run(args):
    torch.set_num_threads(4)
    torch.zeros(1, device="cuda")  # Exercise CPU physics + CUDA sensor output.
    dataset = SmallDemoDataset_ACTPolicy(args.demo_path, 30, 10, control_mode="pd_joint_pos")
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert batch['observations']['rgb'].shape == (4, 2, 3, 224, 224)
    assert batch['observations']['state'].shape == (4, 28)
    assert batch['actions'].shape == (4, 30, 8)
    env = DualRGBObservationWrapper(gym.make('TargetPickCube-v1', obs_mode='rgb+segmentation',
                                           control_mode='pd_joint_pos', sim_backend='physx_cpu'))
    checks = 0
    with h5py.File(args.demo_path) as source:
        for episode in range(10):
            env.reset(seed=episode)
            group = source[f'traj_{episode}']
            for frame in [0, len(group['actions']) // 2, len(group['actions'])]:
                env.unwrapped.set_state_dict(state_at(group['env_states'], frame))
                raw = env.unwrapped.get_obs()
                live = env.observation(raw)
                offline = dataset.trajectories['observations'][episode]
                # Render round-off can affect isolated RGB pixels; mask construction
                # itself is checked against the serialized original + masked pairs.
                state_error = (live['state'][0].cpu() - offline['state'][frame]).abs().max().item()
                assert state_error < 1e-6, state_error
                assert live['rgb'].shape == (1, 2, 3, 224, 224)
                modified = copy.deepcopy(raw)
                for section in ['agent', 'extra']:
                    for field in ['target_idx', 'target_seg_id', 'cube_seg_ids', 'target_pose', 'cube_poses']:
                        modified[section][field] = torch.full((1, 99), 99999.)
                modified['alignment'] = {'target_idx': torch.tensor([99])}
                assert torch.equal(policy_state(modified), policy_state(raw))
                # Independent stream ordering check against the exact stored arrays.
                stored_rgb = torch.from_numpy(group['obs/sensor_data/base_camera/rgb'][frame]).permute(2, 0, 1)
                stored_masked = torch.from_numpy(group['derived/base_camera/masked_rgb'][frame]).permute(2, 0, 1)
                assert torch.equal(offline['rgb'][frame, 0], dataset.transforms(stored_rgb))
                assert torch.equal(offline['rgb'][frame, 1], dataset.transforms(stored_masked))
                checks += 1
        # A counterfactual target switch must affect only stream 1.
        env.unwrapped.set_state_dict(state_at(source['traj_0/env_states'], 0))
        raw = env.unwrapped.get_obs()
        before = env.observation(raw)
        env.unwrapped.target_idx[:] = (env.unwrapped.target_idx + 1) % 3
        after = env.observation(raw)
        assert torch.equal(before['state'], after['state'])
        assert torch.equal(before['rgb'][:, 0], after['rgb'][:, 0])
        assert not torch.equal(before['rgb'][:, 1], after['rgb'][:, 1])
    env.close()
    result = dict(passed=True, demos=10, sample_rgb_shape=[2, 3, 224, 224],
                  batch_rgb_shape=[4, 2, 3, 224, 224], state_dim=28,
                  stream_order=STREAM_ORDER, state_fields=STATE_FIELDS,
                  state_and_stream_checks=checks, injected_metadata_ignored=True,
                  target_switch_only_changes_masked_stream=True)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
        config = Args(**checkpoint['args'])
        envs = make_eval_envs('TargetPickCube-v1', 1, 'physx_cpu',
                             dict(obs_mode='rgb+segmentation', control_mode='pd_joint_pos'),
                             None, wrappers=[DualRGBObservationWrapper])
        agent = Agent(envs, config).cuda().eval()
        agent.load_state_dict(checkpoint['ema_agent'])
        calls = []
        hook = agent.model.backbones[0].register_forward_pre_hook(
            lambda module, inputs: calls.append(list(inputs[0].shape)))
        with torch.no_grad():
            prediction = agent.get_action({k: v[:1].cuda() for k, v in batch['observations'].items()})
        hook.remove()
        assert calls == [[1, 3, 224, 224], [1, 3, 224, 224]], calls
        assert prediction.shape == (1, 30, 8) and torch.isfinite(prediction).all()
        result.update(shared_backbone_calls=calls, action_shape=list(prediction.shape), checkpoint_reloaded=True)
        features = []
        def retain_stream_features(module, inputs, outputs):
            feature = outputs[0][0]
            feature.retain_grad()
            features.append(feature)
        hook = agent.model.backbones[0].register_forward_hook(retain_stream_features)
        loss = agent.compute_loss({k: v[:1].cuda() for k, v in batch['observations'].items()},
                                  batch['actions'][:1].cuda())['loss']
        loss.backward()
        hook.remove()
        assert len(features) == 2
        norms = [float(feature.grad.norm()) for feature in features]
        assert all(torch.isfinite(feature.grad).all() for feature in features)
        assert all(value > 0 for value in norms)
        result['backward_gradient_norm_per_stream'] = norms
        envs.close()
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-path', default=Args().demo_path)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
