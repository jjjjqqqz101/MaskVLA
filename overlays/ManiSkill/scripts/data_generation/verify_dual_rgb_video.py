"""Check numbered video panels against policy pixels and real grasp contacts."""
import json
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path('examples/baselines/act').resolve()))
from train_dual_rgb import Args, Agent, DualRGBObservationWrapper
from act.dual_rgb_video import DualRGBVideoWrapper
from act.make_env import make_eval_envs
from mani_skill.examples.motionplanning.panda.solutions.target_pick_cube import solve
from mani_skill.utils.wrappers import RecordEpisode


torch.set_num_threads(2)
out = Path('outputs/dual_rgb_eval_video_preview')
out.mkdir(exist_ok=False)
kwargs = dict(obs_mode='rgb+segmentation', control_mode='pd_joint_pos', sim_backend='physx_cpu',
              sensor_configs=dict(width=224, height=224, shader_pack='minimal'),
              render_mode='rgb_array', max_episode_steps=500)
base_env = gym.make('TargetPickCube-v1', **kwargs)
policy_env = DualRGBObservationWrapper(base_env)
video_env = DualRGBVideoWrapper(policy_env)
recorder = RecordEpisode(video_env, output_dir=out, save_trajectory=False,
                        save_video=True, info_on_video=False, video_fps=20,
                        source_type='motionplanning', source_desc='Expert-only visualization verification; not ACT performance')
result = solve(recorder, seed=10000)
assert not isinstance(result, int) and bool(result[-1]['full_success'].item())
observation = result[0]
before = {key: value.clone() for key, value in observation.items()}
frame = video_env.render()[0]
assert frame.shape == (640, 1344, 3)
for stream in range(2):
    source = before['rgb'][0, stream].cpu().numpy().transpose(1, 2, 0)
    expected = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)
    actual = frame[144:592, (stream + 1) * 448:(stream + 2) * 448]
    assert np.array_equal(actual, expected)
assert all(torch.equal(before[key], observation[key]) for key in before)
metadata = video_env.last_frame_metadata[0]
target = int(base_env.unwrapped.target_idx.item())
assert metadata['holding'] == [target] and not metadata['wrong_object_grasped']
Image.fromarray(frame).save(out / 'target_held.png')
recorder.flush_video(name='expert_video_format_preview')

# Same actual grasp, counterfactual target: verify the wrong-object display path.
base_env.unwrapped.target_idx[:] = (target + 1) % 3
info = base_env.unwrapped.get_info()
counterfactual = policy_env.observation(base_env.unwrapped.get_obs(info))
video_env._remember(counterfactual, info)
wrong_frame = video_env.render()[0]
assert video_env.last_frame_metadata[0]['holding'] == [target]
assert video_env.last_frame_metadata[0]['wrong_object_grasped']
Image.fromarray(wrong_frame).save(out / 'wrong_object_display_test.png')
base_env.unwrapped.target_idx[:] = target
recorder.close()

# Exercise exactly the recording construction used by train_dual_rgb.py.
envs = make_eval_envs('TargetPickCube-v1', 1, 'physx_cpu', kwargs, None,
                     video_dir=str(out / 'act_forward_check'),
                     wrappers=[DualRGBObservationWrapper, DualRGBVideoWrapper], video_info=False)
obs, _ = envs.reset(seed=23)
assert obs['rgb'].shape == (1, 2, 3, 224, 224) and obs['state'].shape == (1, 28)
agent = Agent(envs, Args()).cuda().eval()
with torch.no_grad():
    prediction = agent.get_action({key: torch.as_tensor(value, device='cuda') for key, value in obs.items()})
assert prediction.shape == (1, 30, 8) and torch.isfinite(prediction).all()
# One neutral control step tests CPU wrapper, recorder, and initial/step capture.
envs.step(np.zeros((1, 8), dtype=np.float32))
envs.close()
with imageio.get_reader(out / 'expert_video_format_preview.mp4') as reader:
    frames = reader.count_frames()
    assert reader.get_data(0).shape == (640, 1344, 3)
    Image.fromarray(reader.get_data(0)).save(out / 'initial_frame.png')
assert frames == video_env.frame_number + 1
report = dict(passed=True, output_shape=[640, 1344, 3], policy_rgb_shape=[1, 2, 3, 224, 224],
              video_frames=frames, successful_expert_target_idx=target, holding_check=True,
              wrong_grasp_display_check=True, policy_pixels_exact_before_video_compression=True,
              observations_unchanged=True, act_forward_passed=True,
              caveat='Preview uses expert actions; the ACT forward compatibility check uses an untrained ACT head.')
(out / 'verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
