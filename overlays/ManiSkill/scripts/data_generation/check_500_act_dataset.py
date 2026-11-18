"""Load all 500 real demos and verify one ACT forward/backward batch."""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path('examples/baselines/act').resolve()))
from train_dual_rgb import Args, Agent, DualRGBObservationWrapper, SmallDemoDataset_ACTPolicy, STATE_FIELDS
from act.make_env import make_eval_envs

torch.set_num_threads(4)
root = Path('demos/TargetPickCube-v1/motionplanning/target_pick_500_mask/dataset')
path = root / 'trajectory.rgb+segmentation.h5'
dataset = SmallDemoDataset_ACTPolicy(str(path), 30, 500, control_mode='pd_joint_pos')
assert dataset.num_traj == 500
batch = next(iter(DataLoader(dataset, batch_size=4)))
assert batch['observations']['rgb'].shape == (4, 2, 3, 224, 224)
assert batch['observations']['state'].shape == (4, 28)
assert batch['actions'].shape == (4, 30, 8)
assert all(torch.isfinite(value).all() for value in dataset.norm_stats.values())
info = json.loads(path.with_suffix('.json').read_text())
kwargs = dict(info['env_info']['env_kwargs'])
kwargs['max_episode_steps'] = info['env_info']['max_episode_steps']
envs = make_eval_envs('TargetPickCube-v1', 1, 'physx_cpu', kwargs, None, wrappers=[DualRGBObservationWrapper])
observation, _ = envs.reset(seed=912345)
assert observation['rgb'].shape == (1, 2, 3, 224, 224)
base = envs.envs[0].unwrapped
camera = base._sensors['base_camera'].get_params()
frozen = json.loads((root / 'frozen_scene.json').read_text())
for key, value in frozen['camera'].items():
    assert np.array_equal(camera[key][0].cpu().numpy(), np.array(value, dtype=np.float32))
agent = Agent(envs, Args()).cuda()
losses = agent.compute_loss({key: value.cuda() for key, value in batch['observations'].items()}, batch['actions'].cuda())
assert all(torch.isfinite(value).all() for value in losses.values())
losses['loss'].backward()
norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), float('inf'), error_if_nonfinite=True)
assert norm.item() > 0
envs.close()
report = dict(demos_loaded=dataset.num_traj, action_samples=len(dataset),
              batch_rgb_shape=[4, 2, 3, 224, 224], state_shape=[4, 28], action_shape=[4, 30, 8],
              state_fields=STATE_FIELDS, normalization_finite=True,
              evaluation_camera_matches_dataset=True, forward_backward_passed=True,
              loss=float(losses['loss'].detach()), gradient_norm=float(norm),
              peak_host_memory_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
              optimizer_steps=0, passed=True)
(root / 'act_dataset_check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
