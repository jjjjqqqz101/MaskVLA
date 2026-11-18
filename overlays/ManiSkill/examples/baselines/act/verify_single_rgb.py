"""Read-only data/forward/evaluation checks. Never backpropagates or updates weights."""
import argparse
import ast
import copy
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from train_single_rgb import Args, Agent, SmallDemoDataset_ACTPolicy, SingleRGBObservationWrapper, STATE_FIELDS, STREAM_ORDER, policy_state
from act.single_rgb_video import SingleRGBVideoWrapper
from act.make_env import make_eval_envs
from evaluate_dual_rgb import evaluate


def run(cli):
    torch.set_num_threads(4)
    torch.manual_seed(1)
    out = cli.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    trees = [ast.parse((here / f'train_{mode}_rgb.py').read_text()) for mode in ('single', 'dual')]
    def node(tree, name):
        return next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    for name in ('Agent', 'policy_state', 'kl_divergence'):
        assert ast.dump(node(trees[0], name)) == ast.dump(node(trees[1], name)), name
    datasets = [node(t, 'SmallDemoDataset_ACTPolicy') for t in trees]
    for name in ('__getitem__', 'get_norm_stats'):
        bodies = [next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name) for c in datasets]
        assert ast.dump(bodies[0]) == ast.dump(bodies[1]), name

    # Instrument every actual H5 dataset read, not just the resulting tensor shape.
    accessed = set()
    original_getitem = h5py.Dataset.__getitem__
    def read(dataset, key):
        accessed.add(dataset.name.split('/', 2)[-1])
        return original_getitem(dataset, key)
    h5py.Dataset.__getitem__ = read
    try:
        dataset = SmallDemoDataset_ACTPolicy(str(cli.demo_path), 30, 450)
    finally:
        h5py.Dataset.__getitem__ = original_getitem
    allowed = {'actions', 'obs/sensor_data/base_camera/rgb', 'obs/agent/qpos', 'obs/agent/qvel',
               'obs/extra/tcp_pose', 'obs/extra/goal_pos'}
    assert accessed == allowed, accessed
    assert dataset.num_traj == 450 and len(dataset) == 74785
    checkpoint = torch.load(cli.reference_checkpoint, map_location='cpu', weights_only=False)
    stat_errors = {}
    for name in ('state_mean', 'state_std', 'action_mean', 'action_std'):
        stat_errors[name] = (dataset.norm_stats[name] - checkpoint['norm_stats'][name]).abs().max().item()
        assert torch.equal(dataset.norm_stats[name], checkpoint['norm_stats'][name]), stat_errors
    print('ALL_450_DEMOS_AND_NORMALIZATION_MATCH', flush=True)

    batch = next(iter(DataLoader(dataset, batch_size=256, shuffle=False)))
    assert batch['observations']['rgb'].shape == (256, 1, 3, 224, 224)
    assert batch['observations']['state'].shape == (256, 28)
    assert batch['actions'].shape == (256, 30, 8)
    with h5py.File(cli.demo_path, 'r') as source:
        for i in (0, 224, 449):
            group = source[f'traj_{i}']
            for t in (0, len(group['actions'])//2, len(group['actions'])):
                stored = torch.from_numpy(group['obs/sensor_data/base_camera/rgb'][t]).permute(2, 0, 1)
                assert torch.equal(dataset.trajectories['observations'][i]['rgb'][t, 0], dataset.transforms(stored))

    meta = json.loads(cli.demo_path.with_suffix('.json').read_text())
    cfg = meta['env_info']['env_kwargs']
    kwargs = dict(obs_mode='rgb', control_mode='pd_joint_pos', sim_backend='physx_cpu',
                  render_mode='rgb_array', max_episode_steps=2)
    for key in ('sensor_configs', 'robot_uids', 'robot_init_qpos_noise'):
        if key in cfg:
            kwargs[key] = cfg[key]
    env = gym.make('TargetPickCube-v1', **kwargs)
    policy = SingleRGBObservationWrapper(env)
    video = SingleRGBVideoWrapper(policy)
    live, info = video.reset(seed=2026092400)
    assert set(live) == {'rgb', 'state'}
    assert live['rgb'].shape == (1, 1, 3, 224, 224)
    base = env.unwrapped
    raw = base.get_obs(base.get_info())
    assert set(raw['sensor_data']['base_camera']) == {'rgb'}
    before = {k: v.clone() for k, v in live.items()}
    # Changing the scored target must not change ANY policy input.
    for target in range(3):
        base.target_idx[:] = target
        raw = base.get_obs(base.get_info())
        changed = policy.observation(raw)
        assert all(torch.equal(before[k], changed[k]) for k in before)
        modified = copy.deepcopy(raw)
        for key in ('target_idx', 'target_pose', 'cube_poses', 'target_seg_id', 'cube_seg_ids'):
            modified['extra'][key] = torch.full((1, 99), 99999.)
        assert torch.equal(policy_state(modified), policy_state(raw))
        video._remember(changed, base.get_info())
        frame = video.render()[0]
        expected = np.repeat(np.repeat(changed['rgb'][0, 0].cpu().numpy().transpose(1, 2, 0), 2, axis=0), 2, axis=1)
        assert frame.shape == (640, 896, 3)
        assert np.array_equal(frame[144:592, 448:896], expected)
    Image.fromarray(frame).save(out / 'single_rgb_eval_preview.png')
    video.close()

    envs = make_eval_envs('TargetPickCube-v1', 1, 'physx_cpu', kwargs, None,
                         video_dir=str(out / 'two_step_eval_video'),
                         wrappers=[SingleRGBObservationWrapper, SingleRGBVideoWrapper], video_info=False)
    agent = Agent(envs, Args()).cuda().eval()
    # This is an untrained ACT head, used only to verify the plumbing.
    weights_before = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
    calls = []
    hook = agent.model.backbones[0].register_forward_pre_hook(lambda module, inputs: calls.append(list(inputs[0].shape)))
    with torch.no_grad():
        prediction = agent.get_action({k: v.cuda() for k, v in batch['observations'].items()})
    hook.remove()
    assert calls == [[256, 3, 224, 224]], calls
    assert prediction.shape == (256, 30, 8) and torch.isfinite(prediction).all()
    with torch.no_grad():
        losses = agent.compute_loss({k: v.cuda() for k, v in batch['observations'].items()}, batch['actions'].cuda())
    assert all(torch.isfinite(v).all() for v in losses.values())
    eval_kwargs = dict(stats={k:v.cuda() for k,v in dataset.norm_stats.items()}, num_queries=30,
                       temporal_agg=True, max_timesteps=2, device=torch.device('cuda'), sim_backend='physx_cpu')
    metrics = evaluate(2, agent, envs, eval_kwargs)
    envs.close()
    assert len(metrics['success_at_end']) == 2
    assert all(torch.equal(weights_before[k], v.cpu()) for k, v in agent.state_dict().items())
    assert all(p.grad is None for p in agent.parameters())
    videos = sorted((out / 'two_step_eval_video').glob('*.mp4'))
    assert len(videos) == 2
    for path in videos:
        with imageio.get_reader(path) as reader:
            assert reader.count_frames() == 3
            assert reader.get_data(0).shape == (640, 896, 3)
    result = dict(passed=True, training_started=False, backward_calls=0, optimizer_steps=0,
                  parameter_updates=0, demos=dataset.num_traj, samples=len(dataset),
                  actual_h5_read_paths=sorted(accessed), state_fields=STATE_FIELDS, stream_order=STREAM_ORDER,
                  batch_rgb_shape=[256,1,3,224,224], batch_state_shape=[256,28], action_shape=[256,30,8],
                  normalization_matches_dual_checkpoint=True, normalization_max_errors=stat_errors,
                  shared_architecture_and_loss_unchanged=True, backbone_forward_calls=calls,
                  target_switch_changes_no_policy_input=True, privileged_metadata_ignored=True,
                  policy_pixels_match_video=True, evaluation_episodes=2, evaluation_steps_per_episode=2,
                  model_weights_unchanged=True, finite_losses={k:float(v) for k,v in losses.items()},
                  caveat='Forward and two-step evaluation use an untrained ACT head; no learning or performance test.')
    (out / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-path', type=Path, default=Path(Args().demo_path))
    parser.add_argument('--reference-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
