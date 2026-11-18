"""Evaluate single- and dual-RGB checkpoints on identical seeded layouts and targets.

Extends evaluate_rgb_ablation.py to a third input mode, the target+gripper
masked stream of train_dual_rgb1.py. The observation adapter is chosen from each
checkpoint's own stream_order, so a model is always evaluated on the stream 1 it
was trained on; an unrecognised layout is refused rather than silently evaluated
through the wrong adapter.

No training data images are loaded. No optimizer or gradient updates are used.
Note: for both dual modes the evaluation video reuses DualRGBVideoWrapper, whose
third panel is captioned "Masked RGB" regardless of what the mask contains. The
pixels shown are the policy's actual stream 1; only the caption is generic.
"""
import argparse
import hashlib
import json
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from train_single_rgb import Args, Agent, SingleRGBObservationWrapper, STATE_FIELDS
from train_dual_rgb import DualRGBObservationWrapper
from train_dual_rgb1 import DualRGBObservationWrapper as GripperDualRGBObservationWrapper
from act.single_rgb_video import SingleRGBVideoWrapper
from act.dual_rgb_video import DualRGBVideoWrapper

HERE = Path(__file__).resolve().parent

# stream_order -> (label, observation adapter, video adapter). Exhaustive by
# design: the adapter rebuilds stream 1 online, so picking the wrong one would
# feed the model a different second stream than it was trained on and still run.
MODES = {
    ('original_rgb',):
        ('single_rgb', SingleRGBObservationWrapper, SingleRGBVideoWrapper),
    ('original_rgb', 'masked_rgb'):
        ('dual_rgb_target', DualRGBObservationWrapper, DualRGBVideoWrapper),
    ('original_rgb', 'masked_target_and_gripper_rgb'):
        ('dual_rgb_target_gripper', GripperDualRGBObservationWrapper, DualRGBVideoWrapper),
}


def config(checkpoint):
    streams = tuple(checkpoint['stream_order'])
    if streams not in MODES:
        raise ValueError(f'Unsupported checkpoint streams: {streams}; known: {sorted(MODES)}')
    if tuple(tuple(x) for x in checkpoint['state_fields']) != STATE_FIELDS:
        raise ValueError('State whitelist mismatch')
    args = Args(**{f.name: checkpoint['args'][f.name] for f in fields(Args) if f.name in checkpoint['args']})
    if args.env_id != 'TargetPickCube-v1' or args.control_mode != 'pd_joint_pos' or args.include_depth:
        raise ValueError('Expected TargetPickCube-v1, pd_joint_pos and RGB only')
    return args, streams


def make_env(checkpoint, horizon):
    args, streams = config(checkpoint)
    meta_path = (HERE / args.demo_path).resolve().with_suffix('.json')
    meta = json.loads(meta_path.read_text())
    cfg = meta['env_info']['env_kwargs']
    # Both methods use the same rendering configuration. The single-RGB adapter
    # discards segmentation and never reads target_idx or a target mask.
    kwargs = dict(obs_mode='rgb+segmentation', control_mode='pd_joint_pos',
                  reward_mode='sparse', sim_backend='physx_cpu', render_mode='rgb_array',
                  reconfiguration_freq=1, max_episode_steps=horizon)
    for key in ('sensor_configs', 'robot_uids', 'robot_init_qpos_noise'):
        if key in cfg:
            kwargs[key] = cfg[key]
    env = gym.make('TargetPickCube-v1', **kwargs)
    _, policy_type, video_type = MODES[streams]
    policy = policy_type(env)
    return video_type(policy), policy, meta


def load_policy(checkpoint, device):
    args, streams = config(checkpoint)
    adapter = SimpleNamespace(single_observation_space=gym.spaces.Dict({
        'state': gym.spaces.Box(-np.inf, np.inf, (28,), dtype=np.float32),
        'rgb': gym.spaces.Box(0, 255, (len(streams), 3, 224, 224), dtype=np.uint8)}),
        single_action_space=gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32))
    agent = Agent(adapter, args).to(device).eval()
    agent.load_state_dict(checkpoint['ema_agent'], strict=True)
    stats = {k: checkpoint['norm_stats'][k].to(device) for k in
             ('state_mean', 'state_std', 'action_mean', 'action_std')}
    for k, value in stats.items():
        if not torch.isfinite(value).all() or ('std' in k and not (value > 0).all()):
            raise ValueError(f'Invalid normalization: {k}')
    return agent, args, stats


def sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def rollout(video, policy, agent, args, stats, seed, target, horizon, fingerprint, video_path=None):
    video.reset(seed=seed)
    base = video.unwrapped
    base.target_idx[:] = target - 1
    info = base.get_info()
    obs = policy.observation(base.get_obs(info))
    video._remember(obs, info)
    initial = (sha(obs['rgb'][:, 0]), sha(obs['state']),
               sha(torch.stack([c.pose.raw_pose for c in base.cubes], dim=1)))
    if seed in fingerprint and initial != fingerprint[seed]:
        raise RuntimeError(f'Initial RGB/state/cube pose mismatch at seed {seed}; comparison stopped')
    fingerprint[seed] = initial
    device = next(agent.parameters()).device
    history = []
    q = args.num_queries
    held_ever = set()
    any_pick = target_pick = wrong_ever = success_once = False
    writer = None
    if video_path:
        writer = imageio.get_writer(video_path, fps=base.control_freq, codec='libx264', quality=7,
                                    ffmpeg_params=['-movflags', '+faststart'])
        writer.append_data(video.render()[0])
    try:
        for step in range(horizon):
            inputs = {k:v.clone().to(device) for k,v in obs.items()}
            inputs['state'] = (inputs['state']-stats['state_mean'])/stats['state_std']
            if args.temporal_agg or step % q == 0:
                prediction = agent.get_action(inputs)
                if prediction.shape != (1, q, 8) or not torch.isfinite(prediction).all():
                    raise ValueError('Invalid ACT predictions')
            if args.temporal_agg:
                history.append((step, prediction))
                history = [(t,p) for t,p in history if step-t < q]
                current = torch.stack([p[:, step-t] for t,p in history], dim=1)
                w = torch.exp(-.01*torch.arange(len(history), device=device))
                raw_action = (current*(w/w.sum())[None,:,None]).sum(dim=1)
            else:
                raw_action = prediction[:, step % q]
            action = raw_action*stats['action_std']+stats['action_mean']
            if not torch.isfinite(action).all():
                raise ValueError('Invalid denormalized actions')
            obs, _, terminated, truncated, info = video.step(action.cpu().numpy())
            grasped = torch.stack([base.agent.is_grasping(c) for c in base.cubes], dim=1)
            lifted = torch.stack([c.pose.p[:,2] > base.cube_half_size+base.lift_height for c in base.cubes], dim=1)
            held_ever.update((torch.nonzero(grasped[0]).flatten().cpu().numpy()+1).tolist())
            any_pick |= bool((grasped & lifted).any().item())
            target_pick |= bool(info['is_target_grasped'].item() and info['target_lifted'].item())
            wrong_ever |= bool(info['wrong_object_grasped'].item())
            success_once |= bool(info['success'].item())
            if writer:
                writer.append_data(video.render()[0])
            if bool(truncated.item()):
                break
    finally:
        if writer:
            writer.close()
    return dict(seed=seed, target=target, steps=step+1, grasped_cube_ids=sorted(held_ever),
                any_object_picked=any_pick, target_grasped_and_lifted=target_pick,
                correct_pick=target_pick and not wrong_ever, wrong_object_grasped_ever=wrong_ever,
                success_once=success_once, success_at_end=bool(info['success'].item()),
                initial_rgb_sha256=initial[0], initial_state_sha256=initial[1], cube_poses_sha256=initial[2],
                video=str(video_path) if video_path else None)


def main(cli):
    if cli.num_scenes < 1 or cli.seed_start < 0 or cli.seed_start+cli.num_scenes >= 2**31:
        raise ValueError('Invalid scene count or seed range')
    if cli.max_steps is not None and not 1 <= cli.max_steps <= 500:
        raise ValueError('max-steps must be in [1, 500]')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = cli.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fingerprints, summaries = {}, []
    for index, path in enumerate(cli.checkpoints):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        args, streams = config(checkpoint)
        horizon = cli.max_steps or args.max_episode_steps
        if horizon is None:
            raise ValueError('Checkpoint must specify max_episode_steps')
        if index and horizon != summaries[0]['horizon']:
            raise ValueError('Both models must use the same evaluation horizon')
        torch.manual_seed(1)
        video, policy, meta = make_env(checkpoint, horizon)
        seen = {e.get('episode_seed') for e in meta['episodes']}
        seeds = list(range(cli.seed_start, cli.seed_start+cli.num_scenes))
        if any(s in seen for s in seeds):
            video.close()
            raise ValueError('Evaluation seeds overlap the checkpoint training episodes')
        mode, policy_type, _ = MODES[streams]
        folder = out / f'{index:02d}_{mode}_{path.stem}'
        folder.mkdir()
        agent, args, stats = load_policy(checkpoint, device)
        records = []
        try:
            with (folder / 'episodes.jsonl').open('w', buffering=1) as log:
                for seed in seeds:
                    for target in (1,2,3):
                        movie = folder / f'seed{seed}_cube{target}.mp4' if cli.capture_video else None
                        result = rollout(video, policy, agent, args, stats, seed, target, horizon, fingerprints, movie)
                        records.append(result)
                        log.write(json.dumps(result)+'\n')
                        print(f'{mode} seed={seed} target={target} pick={result["correct_pick"]} wrong={result["wrong_object_grasped_ever"]}', flush=True)
        finally:
            video.close()
        names = ('any_object_picked','target_grasped_and_lifted','correct_pick','wrong_object_grasped_ever','success_once','success_at_end')
        summary = dict(checkpoint=str(path.resolve()), stream_order=streams, mode=mode,
                       observation_wrapper=f'{policy_type.__module__}.{policy_type.__qualname__}',
                       weights='ema_agent',
                       num_scenes=cli.num_scenes, episodes=len(records), horizon=horizon,
                       seed_start=cli.seed_start, shared_initial_inputs_verified=True,
                       rates={k:float(np.mean([r[k] for r in records])) for k in names},
                       all_three_targets_correct_rate=float(np.mean([all(r['correct_pick'] for r in records[i:i+3])
                                                                      for i in range(0,len(records),3)])))
        summaries.append(summary)
        (folder/'summary.json').write_text(json.dumps(summary, indent=2))
        (out/'summary.json').write_text(json.dumps(summaries, indent=2))
        del agent, stats, checkpoint
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    parser.add_argument('--num-scenes', type=int, default=50)
    parser.add_argument('--seed-start', type=int, default=2026092400)
    parser.add_argument('--max-steps', type=int, help='Optional short structural check; omit for full evaluation')
    parser.add_argument('--capture-video', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE.parents[2]/'outputs'/f'rgb_ablation_{datetime.now():%Y%m%d_%H%M%S}')
    main(parser.parse_args())
