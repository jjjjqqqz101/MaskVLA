"""Seeded C/D ACT rollouts with explicit target-selection and leakage checks."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .data import STATE_FIELDS, encode_observation, extract_state
from .model import ACTPolicy, ModelConfig


ENV_ID = 'TargetPickCube-v1'
VIDEO_FPS = 20
METRICS = ('any_object_picked', 'target_grasped_and_lifted', 'correct_pick',
           'wrong_object_grasped_ever', 'success_once', 'success_at_end')
STAT_WIDTHS = {'state_mean': 28, 'state_std': 28, 'action_mean': 8, 'action_std': 8}


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _fingerprint(value):
    array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _validate_seeds(seeds):
    values = list(seeds)
    if not values or any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
                         or not 0 <= int(seed) < 2**31 for seed in values):
        raise ValueError('seeds must be nonempty integers in [0, 2**31)')
    values = [int(seed) for seed in values]
    if len(set(values)) != len(values):
        raise ValueError('Evaluation seeds must be unique')
    return values


def _normalization(stats, device):
    if not isinstance(stats, dict):
        raise ValueError('norm_stats must be a dictionary')
    result = {}
    for key, width in STAT_WIDTHS.items():
        if key not in stats:
            raise ValueError(f'Missing normalization statistic: {key}')
        value = torch.as_tensor(stats[key], dtype=torch.float32, device=device)
        if value.shape != (1, width) or not torch.isfinite(value).all():
            raise ValueError(f'{key} must have finite shape [1,{width}]')
        if key.endswith('_std') and not (value > 0).all():
            raise ValueError(f'{key} must be strictly positive')
        result[key] = value
    return result


def _environment_kwargs(env_kwargs, max_steps):
    if not isinstance(env_kwargs, dict):
        raise ValueError('env_kwargs must be a dictionary')
    if env_kwargs.get('control_mode', 'pd_joint_pos') != 'pd_joint_pos':
        raise ValueError('C/D evaluation requires pd_joint_pos actions')
    kwargs = dict(env_kwargs)
    kwargs.update(obs_mode='rgb+segmentation', control_mode='pd_joint_pos',
                  reward_mode='sparse', sim_backend='physx_cpu', num_envs=1,
                  render_mode='rgb_array', reconfiguration_freq=1,
                  max_episode_steps=max_steps)
    return kwargs


def _make_env(kwargs):
    import gymnasium as gym
    import mani_skill.envs  # Register ManiSkill tasks, including TargetPickCube.

    return gym.make(ENV_ID, **kwargs)


def _encode(raw, base, target, condition):
    camera = raw['sensor_data']['base_camera']
    rgb = torch.as_tensor(camera['rgb'])
    segmentation = torch.as_tensor(camera['segmentation'])
    if segmentation.ndim == 4 and segmentation.shape[-1] == 1:
        segmentation = segmentation[..., 0]
    if segmentation.ndim != 3 or tuple(segmentation.shape) != tuple(rgb.shape[:3]):
        raise ValueError('Expected one visible actor-ID segmentation channel matching RGB')
    actor_id = base.cubes[target - 1].per_scene_id.to(segmentation.device).reshape(-1, 1, 1)
    mask = segmentation == actor_id
    state = extract_state(raw)
    return encode_observation(rgb, mask, state, condition), mask


def _condition_snapshot(observation, mask, condition):
    result = {'visible_mask_pixels': int(mask.sum().item()),
              'visible_mask_sha256': _fingerprint(mask)}
    if condition == 'C':
        result.update(target_xy=observation['target_xy'][0].detach().cpu().tolist(),
                      target_valid=bool(observation['target_valid'][0].item()))
    else:
        result['binary_input_sha256'] = _fingerprint(observation['rgb'][:, 1])
    return result


def _video_frame(observation, condition, seed, target, step, held, success):
    image = observation['rgb'][0, 0].detach().cpu().permute(1, 2, 0).numpy()
    left = Image.fromarray(image).resize((448, 448), Image.Resampling.NEAREST)
    try:
        title_font = ImageFont.truetype('DejaVuSans.ttf', 18)
        detail_font = ImageFont.truetype('DejaVuSans.ttf', 15)
    except OSError:
        title_font = detail_font = ImageFont.load_default()
    if condition == 'C':
        right = left.copy()
        valid = bool(observation['target_valid'][0].item())
        xy = observation['target_xy'][0].detach().cpu().tolist()
        if valid:
            x, y = ((float(value) + 1) * 447 / 2 for value in xy)
            point = ImageDraw.Draw(right)
            point.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(40, 255, 100), width=3)
            point.line((x - 11, y, x + 11, y), fill=(40, 255, 100), width=2)
            point.line((x, y - 11, x, y + 11), fill=(40, 255, 100), width=2)
        title = 'Target point'
        detail = f'target_xy=({xy[0]:+.3f}, {xy[1]:+.3f}), valid={valid}'
    else:
        binary = observation['rgb'][0, 1].detach().cpu().permute(1, 2, 0).numpy()
        right = Image.fromarray(binary).resize((448, 448), Image.Resampling.NEAREST)
        title = 'Binary mask input'
        detail = f'Visible target pixels: {int((binary[..., 0] > 0).sum())}'
    canvas = Image.new('RGB', (896, 536), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 7), 'Original RGB input', font=title_font, fill='#202729')
    draw.text((458, 7), title, font=title_font, fill='#202729')
    canvas.paste(left, (0, 36))
    canvas.paste(right, (448, 36))
    draw.text((10, 492), f'{condition} / seed {seed} / target cube{target} / step {step}',
              font=detail_font, fill='#202729')
    draw.text((458, 492), detail, font=detail_font, fill='#202729')
    draw.text((10, 514), f'Held cubes: {held} / success: {success}', font=detail_font, fill='#202729')
    return np.asarray(canvas)


def _episode(env, policy, config, stats, seed, target, max_steps,
             temporal_agg, device, fingerprints, video_path):
    env.reset(seed=seed)
    base = env.unwrapped
    base.target_idx[:] = target - 1
    info = base.get_info()
    raw = base.get_obs(info)
    observation, mask = _encode(raw, base, target, config.condition)
    initial = (_fingerprint(raw['sensor_data']['base_camera']['rgb']),
               _fingerprint(observation['state']),
               _fingerprint(torch.stack([cube.pose.raw_pose for cube in base.cubes], dim=1)))
    if seed in fingerprints and fingerprints[seed] != initial:
        raise RuntimeError(f'Seed {seed}: initial RGB/state/cube poses differ between targets')
    fingerprints[seed] = initial
    initial_condition = _condition_snapshot(observation, mask, config.condition)
    history, held_ever, picked_ever = [], set(), set()
    held = []
    first_lifted, first_lift_step, first_lifted_ids = None, None, []
    target_pick = wrong_ever = any_pick = False
    success_once = bool(torch.as_tensor(info['success']).item())
    terminated_seen = False
    truncated_at_end = False
    invisible_steps = 0
    writer = None
    if video_path is not None:
        import imageio.v2 as imageio

        writer = imageio.get_writer(video_path, fps=VIDEO_FPS, codec='libx264',
                                    quality=7, macro_block_size=1,
                                    ffmpeg_params=['-movflags', '+faststart'])
    started = time.monotonic()
    try:
        if writer is not None:
            writer.append_data(_video_frame(observation, config.condition, seed, target, 0,
                                            held, success_once))
        for step in range(max_steps):
            invisible_steps += int(not bool(mask.any().item()))
            inputs = {key: value.to(device) for key, value in observation.items()}
            inputs['state'] = (inputs['state'] - stats['state_mean']) / stats['state_std']
            if temporal_agg or step % config.num_queries == 0:
                prediction = policy.get_action(inputs)
                if prediction.shape != (1, config.num_queries, 8) or not torch.isfinite(prediction).all():
                    raise ValueError('Expected finite normalized ACT predictions [1,num_queries,8]')
            if temporal_agg:
                history.append((step, prediction))
                history = [(time_step, chunk) for time_step, chunk in history
                           if step - time_step < config.num_queries]
                actions = torch.stack([chunk[:, step - time_step] for time_step, chunk in history], dim=1)
                # Match the existing ACT convention: older predictions receive larger weights.
                weights = torch.exp(-0.01 * torch.arange(len(history), device=device))
                normalized = (actions * (weights / weights.sum())[None, :, None]).sum(dim=1)
            else:
                normalized = prediction[:, step % config.num_queries]
            action = normalized * stats['action_std'] + stats['action_mean']
            if not torch.isfinite(action).all():
                raise ValueError('Nonfinite denormalized pd_joint_pos action')
            raw, _, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            grasped = torch.stack([base.agent.is_grasping(cube) for cube in base.cubes], dim=1)[0]
            lifted = torch.stack([cube.pose.p[:, 2] > base.cube_half_size + base.lift_height
                                  for cube in base.cubes], dim=1)[0]
            picked = grasped & lifted
            held = (torch.nonzero(grasped).flatten().cpu() + 1).tolist()
            picked_ids = (torch.nonzero(picked).flatten().cpu() + 1).tolist()
            held_ever.update(held)
            picked_ever.update(picked_ids)
            if first_lifted is None and picked_ids:
                first_lifted_ids = picked_ids
                first_lifted, first_lift_step = min(picked_ids), step + 1
            any_pick |= bool(picked.any().item())
            target_pick |= bool(picked[target - 1].item())
            wrong_ever |= any(cube_id != target for cube_id in held)
            success = bool(torch.as_tensor(info['success']).item())
            success_once |= success
            terminated_seen |= bool(torch.as_tensor(terminated).any().item())
            truncated_at_end = bool(torch.as_tensor(truncated).any().item())
            observation, mask = _encode(raw, base, target, config.condition)
            if writer is not None:
                writer.append_data(_video_frame(observation, config.condition, seed, target,
                                                step + 1, held, success))
            if truncated_at_end:
                break
    finally:
        if writer is not None:
            writer.close()
    return dict(seed=seed, target=target, condition=config.condition, steps=step + 1,
                elapsed_seconds=time.monotonic() - started,
                any_object_picked=any_pick, target_grasped_and_lifted=target_pick,
                correct_pick=target_pick and not wrong_ever,
                wrong_object_grasped_ever=wrong_ever, success_once=success_once,
                success_at_end=bool(torch.as_tensor(info['success']).item()),
                first_lifted_cube=first_lifted, first_lifted_cube_ids=first_lifted_ids,
                first_lift_step=first_lift_step, held_cube_ids=held,
                held_cube_ids_ever=sorted(held_ever), lifted_cube_ids_ever=sorted(picked_ever),
                target_invisible_input_steps=invisible_steps,
                terminated_seen=terminated_seen, truncated_at_end=truncated_at_end,
                initial_rgb_sha256=initial[0], initial_state_sha256=initial[1],
                cube_poses_sha256=initial[2], initial_condition=initial_condition,
                video=str(video_path) if video_path is not None else None)


@torch.no_grad()
def run_rollouts(policy, config, norm_stats, env_kwargs, seeds, max_steps,
                 temporal_agg=True, video_dir=None):
    """Evaluate all three target assignments on each identical seeded scene.

    A pick requires simultaneous grasp and lift above the environment threshold.
    The first such step determines first_lifted_cube; simultaneous ties also
    appear in first_lifted_cube_ids. Evaluation continues after task termination
    until the configured horizon or a time-limit truncation. Seed leakage is
    checked by the checkpoint CLI; training callers provide their own schedule.
    """
    if not isinstance(config, ModelConfig):
        raise TypeError('config must be the new C/D ModelConfig')
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')
    seeds = _validate_seeds(seeds)
    parameter = next(policy.parameters(), None)
    device = parameter.device if parameter is not None else torch.device('cpu')
    stats = _normalization(norm_stats, device)
    kwargs = _environment_kwargs(env_kwargs, max_steps)
    if video_dir is not None:
        video_dir = Path(video_dir).resolve()
        video_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            for target in (1, 2, 3):
                path = video_dir / f'{config.condition}_seed{seed}_cube{target}.mp4'
                if path.exists():
                    raise FileExistsError(f'Refusing to overwrite video: {path}')
    training = policy.training
    env = None
    episodes, fingerprints = [], {}
    started = time.monotonic()
    policy.eval()
    try:
        env = _make_env(kwargs)
        for seed in seeds:
            for target in (1, 2, 3):
                path = video_dir / f'{config.condition}_seed{seed}_cube{target}.mp4' if video_dir else None
                episodes.append(_episode(env, policy, config, stats, seed, target, max_steps,
                                         temporal_agg, device, fingerprints, path))
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            policy.train(training)
    rates = {metric: float(np.mean([episode[metric] for episode in episodes])) for metric in METRICS}
    rates['all_three_targets_correct'] = float(np.mean([
        all(episode['correct_pick'] for episode in episodes[index:index + 3])
        for index in range(0, len(episodes), 3)]))
    return dict(experiment='target_ablation_cd', condition=config.condition,
                episodes=episodes, rates=rates, num_episodes=len(episodes),
                shared_initial_inputs_verified=True, elapsed_seconds=time.monotonic() - started,
                parameters=_json_value(dict(model_config=asdict(config), seeds=seeds,
                                            targets=[1, 2, 3], max_steps=max_steps,
                                            temporal_agg=bool(temporal_agg), temporal_weight_decay=0.01,
                                            temporal_weight_order='oldest_to_newest',
                                            environment_id=ENV_ID, env_kwargs=kwargs,
                                            input_env_kwargs=env_kwargs, video_fps=VIDEO_FPS,
                                            video_dir=video_dir, state_fields=STATE_FIELDS,
                                            termination='max_steps_or_truncation',
                                            first_lift='first_simultaneous_grasp_and_lift')))


def _load_checkpoint(path, device, weights):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get('format_version') != 1:
        raise ValueError('Expected a version-1 target_ablation_cd checkpoint')
    if checkpoint.get('experiment') != 'target_ablation_cd':
        raise ValueError('Checkpoint is not from the isolated target_ablation_cd experiment')
    required = ('model_config', 'condition', 'norm_stats', 'agent', 'ema_agent', 'env_kwargs',
                'state_fields', 'iteration', 'train_episode_seeds', 'validation_episode_seeds',
                'selection_eval_seeds', 'stream_order')
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f'Missing C/D checkpoint fields: {missing}')
    if not isinstance(checkpoint['model_config'], dict):
        raise ValueError('Checkpoint model_config must be a dictionary')
    config = ModelConfig(**checkpoint['model_config'])
    if checkpoint['condition'] not in ('C', 'D') or checkpoint['condition'] != config.condition:
        raise ValueError('Checkpoint condition and ModelConfig disagree')
    if tuple(tuple(item) for item in checkpoint['state_fields']) != STATE_FIELDS:
        raise ValueError('Checkpoint state whitelist differs from the 28-dimensional C/D state')
    expected_streams = ['original_rgb'] if config.condition == 'C' else ['original_rgb', 'binary_mask_rgb']
    if list(checkpoint['stream_order']) != expected_streams:
        raise ValueError('Checkpoint stream order does not match the C/D input contract')
    _normalization(checkpoint['norm_stats'], device)
    _environment_kwargs(checkpoint['env_kwargs'], 1)
    policy = ACTPolicy(config).to(device)
    policy.load_state_dict(checkpoint[weights], strict=True)
    return checkpoint, policy, config


def _seed_audit(checkpoint, seeds, allow_selection_seeds):
    audit = {}
    for key in ('train_episode_seeds', 'validation_episode_seeds', 'selection_eval_seeds'):
        stored = checkpoint[key]
        if not isinstance(stored, (list, tuple)) or any(isinstance(seed, bool)
                or not isinstance(seed, (int, np.integer)) for seed in stored):
            raise ValueError(f'Checkpoint {key} must be an integer list')
        overlap = sorted(set(seeds).intersection(int(seed) for seed in stored))
        audit[key] = overlap
        if overlap and (key != 'selection_eval_seeds' or not allow_selection_seeds):
            raise ValueError(f'Evaluation seeds overlap {key}: {overlap}')
    audit['allow_selection_seeds'] = bool(allow_selection_seeds)
    return audit


def main(cli):
    seeds = _validate_seeds(cli.seeds)
    if cli.max_steps < 1:
        raise ValueError('max-steps must be positive')
    torch.set_num_threads(cli.torch_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(cli.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    checkpoint_path = cli.checkpoint.resolve()
    checkpoint, policy, config = _load_checkpoint(checkpoint_path, device, cli.weights)
    audit = _seed_audit(checkpoint, seeds, cli.allow_selection_seeds)
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = run_rollouts(policy, config, checkpoint['norm_stats'], checkpoint['env_kwargs'],
                          seeds, cli.max_steps, temporal_agg=cli.temporal_agg,
                          video_dir=output / 'videos' if cli.capture_video else None)
    result.update(created_at=datetime.now(timezone.utc).isoformat(), seed_audit=audit,
                  checkpoint=dict(path=str(checkpoint_path), sha256=_file_sha256(checkpoint_path),
                                  weights=cli.weights, iteration=_json_value(checkpoint['iteration'])))
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (output / 'episodes.jsonl').open('x') as stream:
        for episode in result['episodes']:
            stream.write(json.dumps(episode) + '\n')
    print(json.dumps(dict(output=str(output), condition=config.condition,
                          num_episodes=result['num_episodes'], rates=result['rates']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026110000])
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture-video', action='store_true')
    parser.add_argument('--temporal-agg', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--allow-selection-seeds', action='store_true')
    parser.add_argument('--weights', choices=('ema_agent', 'agent'), default='ema_agent')
    parser.add_argument('--device', help='Torch device; defaults to CUDA when available')
    parser.add_argument('--torch-threads', type=int, default=4)
    main(parser.parse_args())
