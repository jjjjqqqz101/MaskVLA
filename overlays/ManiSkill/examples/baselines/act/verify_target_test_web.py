"""Audit a completed three-target comparison and replay each first ACT inference."""
import argparse
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from train_dual_rgb import Agent, Args, STATE_FIELDS, STREAM_ORDER
from target_test_web import file_digest


def verify(path):
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    report = json.loads(path.read_text())
    assert report['complete'] and report['same_initial_rgb'] and report['same_initial_state']
    assert report['different_masks'] and len(report['trials']) == 3
    ckpt = torch.load(report['checkpoint'], map_location='cpu', weights_only=False)
    args = Args(**{f.name: ckpt['args'][f.name] for f in fields(Args) if f.name in ckpt['args']})
    adapter = SimpleNamespace(single_observation_space=gym.spaces.Dict({
        'state': gym.spaces.Box(-np.inf, np.inf, (28,), dtype=np.float32),
        'rgb': gym.spaces.Box(0, 255, (2, 3, 224, 224), dtype=np.uint8)}),
        single_action_space=gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32))
    device = torch.device('cuda')
    model = Agent(adapter, args).to(device).eval()
    model.load_state_dict(ckpt['ema_agent'])
    stats = {k: v.to(device) for k, v in ckpt['norm_stats'].items()}
    checksum = file_digest(Path(report['checkpoint']))
    inputs, checks = [], []
    for target, trial in enumerate(report['trials'], 1):
        folder = path.parent / 'trials' / trial
        result = json.loads((folder / 'result.json').read_text())
        assert result['checkpoint_sha256'] == checksum
        assert result['target'] == target and result['target_idx'] == target-1
        assert tuple(result['stream_order']) == STREAM_ORDER
        assert tuple(tuple(x) for x in result['state_fields']) == STATE_FIELDS
        data = np.load(folder / 'initial_policy_input.npz')
        rgb, state = data['rgb'], data['state']
        assert rgb.shape == (1, 2, 3, 224, 224) and rgb.dtype == np.uint8
        assert state.shape == (1, 28)
        mask = rgb[0, 1].any(axis=0)
        assert mask.any() and np.array_equal(rgb[0, 1], np.where(mask[None], rgb[0, 0], 0))
        frame = np.asarray(Image.open(folder / 'initial.png'))
        for stream in (0, 1):
            expected = np.repeat(np.repeat(rgb[0, stream].transpose(1, 2, 0), 2, axis=0), 2, axis=1)
            assert np.array_equal(frame[144:592, (stream+1)*448:(stream+2)*448], expected)
        if inputs:
            assert np.array_equal(inputs[0][0][:, 0], rgb[:, 0])
            assert np.array_equal(inputs[0][1], state)
            for old_rgb, _ in inputs:
                assert not (mask & old_rgb[0, 1].any(axis=0)).any()
        inputs.append((rgb, state))
        actions = np.load(folder / 'actions.npy')
        assert actions.shape == (500, 8) and np.isfinite(actions).all()
        with torch.inference_mode():
            prediction = model.get_action(dict(rgb=torch.as_tensor(rgb, device=device),
                state=(torch.as_tensor(state, device=device)-stats['state_mean'])/stats['state_std']))
            action = prediction[:, 0] * stats['action_std'] + stats['action_mean']
        difference = float(np.abs(action.cpu().numpy()[0]-actions[0]).max())
        assert difference < 2e-5, difference
        traces = json.loads((folder / 'step_metrics.json').read_text())
        assert len(traces) == 500
        assert result['wrong_object_grasped_ever'] == any(t['wrong_object_grasped'] for t in traces)
        assert result['correct_pick'] == (any(t['is_target_grasped'] and t['target_lifted'] for t in traces)
                                           and not any(t['wrong_object_grasped'] for t in traces))
        with imageio.get_reader(folder / 'video.mp4') as reader:
            frames = reader.count_frames()
            assert frames == 501 and reader.get_data(0).shape == (640, 1344, 3)
        checks.append(dict(target=target, video_frames=frames, masked_pixels=int(mask.sum()),
                           first_action_max_error=difference, correct_pick=result['correct_pick']))
    verified = dict(passed=True, comparison=str(path), checkpoint_sha256=checksum,
                    same_initial_rgb_and_state=True, disjoint_target_masks=True,
                    policy_pixels_match_video_panels=True, ema_first_action_reproduced=True,
                    per_step_metrics_match_summary=True, checks=checks)
    destination = path.with_name(path.stem+'_verification.json')
    destination.write_text(json.dumps(verified, indent=2))
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', type=Path)
    verify(parser.parse_args().comparison)
