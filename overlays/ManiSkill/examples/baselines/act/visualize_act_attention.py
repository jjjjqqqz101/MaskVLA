#!/usr/bin/env python3
"""Record ACT inputs and attention maps on policy rollouts or H5 demonstrations.

Run with --help for options. The current policy's decoder selection is preserved.
Attention maps describe information routing, not causal explanations of actions.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torchvision import transforms as T

from act.attention_maps import AttentionRecorder
from evaluate_rgb_ablation import config, load_policy, make_env
from train_dual_rgb import dual_rgb
from train_single_rgb import STATE_FIELDS, single_rgb


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_CHECKPOINTS = [
    HERE / 'runs/TargetPickCube-v1__train_dual_rgb__1__001/checkpoints/best_eval_success_once.pt',
    HERE / 'runs/TargetPickCube-v1__train_single_rgb__1__1790243190/checkpoints/best_eval_success_once.pt',
]
COLOR_MAP = colormaps['inferno']
RESIZE = T.Resize((224, 224), antialias=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def font(size):
    return ImageFont.truetype('DejaVuSans.ttf', size)


def outline(rgb, mask):
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    outer = np.asarray(mask_image.filter(ImageFilter.MaxFilter(3))) > 0
    inner = np.asarray(mask_image.filter(ImageFilter.MinFilter(3))) > 0
    result = rgb.copy()
    result[outer & ~inner] = (75, 255, 150)
    return result


def heat_overlay(rgb, weights, smooth):
    maximum = float(np.max(weights))
    scaled = weights / maximum if maximum > 0 else np.zeros_like(weights)
    resample = Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST
    intensity = np.asarray(Image.fromarray(scaled.astype(np.float32)).resize(
        (rgb.shape[1], rgb.shape[0]), resample))
    intensity = np.clip(intensity, 0, 1)
    colors = COLOR_MAP(intensity)[..., :3] * 255
    alpha = (0.7 * intensity)[..., None]
    return np.round(rgb * (1 - alpha) + colors * alpha).astype(np.uint8), maximum


def render_frame(obs, mask, maps, step, target, source, status, cli):
    rgb = obs['rgb'][0, 0].cpu().numpy().transpose(1, 2, 0)
    streams = obs['rgb'].shape[1]
    dual = streams == 2
    original = outline(rgb, mask)
    second = (obs['rgb'][0, 1].cpu().numpy().transpose(1, 2, 0) if dual else
              np.repeat(mask[..., None], 3, axis=2).astype(np.uint8) * 255)
    action_image, action_max = heat_overlay(rgb, maps['action_original'], cli.smooth)
    relation_valid = bool(maps['mask_relation_valid']) if dual else True
    relation_map = maps['mask_to_original'] if dual else maps['decoder_original']
    relation_image, relation_max = heat_overlay(rgb, relation_map, cli.smooth)
    if not relation_valid:
        relation_image = rgb.copy()
    panels = [original, second, outline(action_image, mask), outline(relation_image, mask)]
    titles = ['Original RGB + target outline',
              'Masked RGB: policy input' if dual else 'Reference mask: display only',
              'Action -> RGB: attention rollout',
              'Mask target -> RGB: attention' if dual else 'Action -> RGB: direct attention']
    notes = ['Green outline = visible target',
             'Original RGB is also a policy input' if dual else 'Mask is not passed to this policy',
             'Decoder 1, through all encoder layers',
             f'Encoder {int(maps["selected_encoder_layer"]) + 1}, target query cells' if dual
             else 'Decoder 1 -> mixed encoder memory']
    canvas = Image.new('RGB', (1536, 576), '#191b1d')
    draw = ImageDraw.Draw(canvas)
    mode = 'ACT + MASK' if dual else 'ACT RGB ONLY'
    query = f'mean of {maps["query_count"]} queries' if cli.query_index == -1 else f'action query {cli.query_index}'
    draw.text((12, 12), f'{mode} | {source} | frame {step:04d} | target cube{target} | {query}',
              font=font(21), fill='white')
    for index, (panel, title, note) in enumerate(zip(panels, titles, notes)):
        x = index * 384
        draw.text((x + 8, 53), title, font=font(17), fill='white')
        draw.text((x + 8, 78), note, font=font(13), fill='#c6cdcf')
        canvas.paste(Image.fromarray(panel).resize((384, 384), Image.Resampling.NEAREST), (x, 104))
    draw.text((10, 500), status, font=font(15), fill='#c6cdcf')
    visible = int(mask.sum())
    draw.text((394, 500), f'Target: {visible} visible pixels', font=font(15), fill='#c6cdcf')
    for index, maximum in ((2, action_max), (3, relation_max)):
        x = index * 384 + 8
        ramp = COLOR_MAP(np.linspace(0, 1, 352))[None, :, :3]
        canvas.paste(Image.fromarray((np.repeat(ramp, 9, axis=0) * 255).astype(np.uint8)), (x, 499))
        text = f'0 to {maximum:.5f} weight / feature cell'
        if index == 3 and not relation_valid:
            text = 'Unavailable: target mask is empty'
        draw.text((x, 514), text, font=font(13), fill='#c6cdcf')
    draw.text((12, 548),
              f'{maps["feature_height"]}x{maps["feature_width"]} spatial grid | RGB attention mass: {float(maps["action_mass_original"]):.3f}'
              ' | Current prediction attention; not a causal attribution of the executed action.',
              font=font(16), fill='#c6cdcf')
    return np.asarray(canvas)


def predict(agent, obs, stats):
    device = next(agent.parameters()).device
    inputs = {key: value.clone().to(device) for key, value in obs.items()}
    inputs['state'] = (inputs['state'] - stats['state_mean']) / stats['state_std']
    with torch.no_grad():
        return agent.get_action(inputs)


def live_mask(base, info):
    camera = base.get_obs(info)['sensor_data']['base_camera']
    target_id = base.cubes[int(base.target_idx.item())].per_scene_id.to(camera['segmentation'].device)
    mask = (camera['segmentation'] == target_id.reshape(1, 1, 1, 1))[0, ..., 0]
    mask = mask.cpu().numpy().astype(np.uint8) * 255
    return np.asarray(Image.fromarray(mask).resize((224, 224), Image.Resampling.NEAREST)) > 0


def dataset_obs(trajectory, step, streams):
    original = torch.from_numpy(trajectory['obs/sensor_data/base_camera/rgb'][step:step + 1])
    if len(streams) == 2:
        masked = torch.from_numpy(trajectory['derived/base_camera/masked_rgb'][step:step + 1])
        rgb = dual_rgb(original, masked, RESIZE)
    else:
        rgb = single_rgb(original, RESIZE)
    state = torch.cat([torch.from_numpy(trajectory[f'obs/{group}/{key}'][step:step + 1]).float()
                       for group, key in STATE_FIELDS], dim=-1)
    mask = trajectory['derived/base_camera/target_mask'][step, ..., 0]
    mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (224, 224), Image.Resampling.NEAREST)) > 0
    return dict(rgb=rgb, state=state), mask


class Exporter:
    def __init__(self, folder, metadata, cli, horizon, fps):
        self.folder, self.metadata, self.cli, self.horizon = folder, metadata, cli, horizon
        self.saved_maps, self.records, self.last_frame = [], [], None
        self.viser_frames = []
        self.snapshots = {0, (horizon // 2 // cli.stride) * cli.stride, horizon}
        self.writer = imageio.get_writer(folder / 'attention.mp4', fps=fps, codec='libx264',
                                         quality=8, ffmpeg_params=['-movflags', '+faststart', '-threads', '2'])

    def capture(self, obs, mask, recorder, prediction, step, target, source, status):
        maps = recorder.maps(obs['rgb'].shape[1], mask, query_index=self.cli.query_index,
                             encoder_layer=self.cli.encoder_layer)
        maps['query_count'] = prediction.shape[1]
        self.last_frame = render_frame(obs, mask, maps, step, target, source, status, self.cli)
        self.saved_maps.append({key: value for key, value in maps.items()
                                if isinstance(value, np.ndarray) and value.shape == recorder.feature_shape})
        scalars = {key: value.item() if isinstance(value, np.generic) else value
                   for key, value in maps.items() if np.isscalar(value)}
        self.records.append(dict(step=step, target=target, target_pixels=int(mask.sum()), **scalars))
        # Store clean inputs and raw attention for interactive query/layer selection.
        self.viser_frames.append(dict(
            rgb=obs['rgb'][0].detach().cpu().numpy().astype(np.uint8, copy=True),
            target_mask=np.array(mask, dtype=np.bool_, copy=True),
            state=obs['state'][0].detach().float().cpu().numpy().copy(),
            predicted_normalized_actions=prediction[0].detach().float().cpu().numpy().copy(),
            encoder_attention=np.stack(recorder.encoder_attention)[:, 0].astype(np.float32, copy=True),
            decoder_attention=np.array(recorder.decoder_attention[0], dtype=np.float32, copy=True),
        ))
        if step in self.snapshots:
            Image.fromarray(self.last_frame).save(self.folder / f'frame_{step:04d}.png')
            np.savez_compressed(self.folder / f'frame_{step:04d}_attention.npz',
                                rgb=obs['rgb'].cpu().numpy(), state=obs['state'].cpu().numpy(),
                                target_mask=mask, predicted_normalized_actions=prediction.cpu().numpy(),
                                encoder_attention=np.stack(recorder.encoder_attention),
                                decoder_attention=recorder.decoder_attention,
                                **{key: value for key, value in maps.items() if isinstance(value, np.ndarray)})
        if step == 0:
            Image.fromarray(obs['rgb'][0, 0].cpu().numpy().transpose(1, 2, 0)).save(self.folder / 'original.png')
            Image.fromarray(mask.astype(np.uint8) * 255).save(self.folder / 'target_mask.png')
            if obs['rgb'].shape[1] == 2:
                Image.fromarray(obs['rgb'][0, 1].cpu().numpy().transpose(1, 2, 0)).save(self.folder / 'masked_rgb.png')

    def write(self):
        self.writer.append_data(self.last_frame)

    def close(self):
        self.writer.close()
        if self.saved_maps:
            if len(self.viser_frames) != len(self.saved_maps) or len(self.records) != len(self.saved_maps):
                raise RuntimeError('Attention maps, Viser inputs, and frame records are not aligned')
            steps = np.array([record['step'] for record in self.records], dtype=np.int64)
            arrays = {key: np.stack([frame[key] for frame in self.saved_maps]) for key in self.saved_maps[0]}
            np.savez_compressed(self.folder / 'attention_maps.npz',
                                steps=steps, **arrays)
            inputs = {key: np.stack([frame[key] for frame in self.viser_frames])
                      for key in self.viser_frames[0]}
            np.savez_compressed(self.folder / 'viser_frames.npz', steps=steps, **inputs)
            self.metadata.update(viser_frames_file='viser_frames.npz', viser_frames_version=1,
                                 captured_frames=len(steps),
                                 feature_shape=list(self.saved_maps[0]['action_original'].shape))
        (self.folder / 'frames.json').write_text(json.dumps(self.records, indent=2))
        (self.folder / 'metadata.json').write_text(json.dumps(self.metadata, indent=2))


def metadata_for(checkpoint_path, checkpoint_hash, streams, cli):
    return dict(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=checkpoint_hash,
                weights='ema_agent', stream_order=list(streams),
                source='h5_demonstration' if cli.dataset else 'closed_loop_policy_rollout',
                seed=None if cli.dataset else cli.seed,
                dataset=str(cli.dataset.resolve()) if cli.dataset else None,
                trajectory=cli.trajectory if cli.dataset else None,
                decoder_layer=0, encoder_relation_layer=cli.encoder_layer,
                action_query=cli.query_index, capture_stride=cli.stride,
                action_attention_method='mean-head decoder-0 cross attention times residual encoder rollout',
                relation_method='target-occupancy-weighted masked-query to original-key encoder attention',
                heat_scale='Each panel: zero to its displayed maximum raw weight; no min subtraction.',
                limitations=['Attention routing is not a causal attribution.',
                             'Native spatial resolution is 7x7; ResNet receptive fields overlap.',
                             'Current forward attention does not explain the full temporal action aggregate.',
                             'The existing decoder-0 selection is preserved.'])


def run_dataset(agent, checkpoint, stats, streams, folder, metadata, cli):
    with h5py.File(cli.dataset, 'r') as data:
        if cli.trajectory not in data:
            raise ValueError(f'Missing trajectory: {cli.trajectory}')
        trajectory = data[cli.trajectory]
        horizon = min(cli.max_steps, len(trajectory['actions']))
        target = int(trajectory['env_states/target_idx'][0].item()) + 1
        metadata.update(target=target, steps=horizon, fps=20, closed_loop=False)
        exporter = Exporter(folder, metadata, cli, horizon, 20)
        try:
            first_obs, _ = dataset_obs(trajectory, 0, streams)
            reference = predict(agent, first_obs, stats)
            with AttentionRecorder(agent.model) as recorder:
                for step in range(horizon + 1):
                    if step % cli.stride == 0 or step == horizon:
                        obs, mask = dataset_obs(trajectory, step, streams)
                        recorder.clear()
                        prediction = predict(agent, obs, stats)
                        if step == 0:
                            torch.testing.assert_close(prediction, reference, rtol=1e-5, atol=1e-6)
                            metadata['attention_hooks_preserve_predictions'] = True
                        exporter.capture(obs, mask, recorder, prediction, step, target,
                                         f'H5 demonstration {cli.trajectory}', 'Recorded expert observations')
                    exporter.write()
            metadata['completed'] = True
        finally:
            exporter.close()


def run_rollout(agent, checkpoint, stats, streams, folder, metadata, cli, fingerprints):
    args, _ = config(checkpoint)
    video, policy, _ = make_env(checkpoint, cli.max_steps)
    exporter = None
    try:
        video.reset(seed=cli.seed)
        base = video.unwrapped
        base.target_idx[:] = cli.target - 1
        info = base.get_info()
        obs = policy.observation(base.get_obs(info))
        fingerprint = tuple(hashlib.sha256(obs[key].cpu().numpy().tobytes()).hexdigest()
                            for key in ('state',)) + (hashlib.sha256(obs['rgb'][:, 0].cpu().numpy().tobytes()).hexdigest(),)
        if fingerprints and fingerprint != fingerprints[0]:
            raise RuntimeError('Initial original RGB/state differ across models')
        fingerprints.append(fingerprint)
        metadata.update(target=cli.target, steps=cli.max_steps, fps=base.control_freq,
                        closed_loop=True, initial_state_sha256=fingerprint[0], initial_rgb_sha256=fingerprint[1],
                        temporal_aggregation=args.temporal_agg)
        exporter = Exporter(folder, metadata, cli, cli.max_steps, base.control_freq)
        reference = predict(agent, obs, stats)
        history, actions = [], []
        success_once = target_pick = False
        with AttentionRecorder(agent.model) as recorder:
            for step in range(cli.max_steps + 1):
                recorder.clear()
                prediction = predict(agent, obs, stats)
                if not torch.isfinite(prediction).all():
                    raise ValueError('Nonfinite ACT prediction')
                if step == 0:
                    torch.testing.assert_close(prediction, reference, rtol=1e-5, atol=1e-6)
                    metadata['attention_hooks_preserve_predictions'] = True
                if step % cli.stride == 0 or step == cli.max_steps:
                    mask = live_mask(base, info)
                    status = f'Held: {int(info["is_target_grasped"].item())} | success: {int(info["success"].item())}'
                    exporter.capture(obs, mask, recorder, prediction, step, cli.target,
                                     f'policy rollout / seed {cli.seed}', status)
                exporter.write()
                if step == cli.max_steps:
                    break
                if args.temporal_agg:
                    history.append((step, prediction))
                    history = [(t, p) for t, p in history if step - t < args.num_queries]
                    current = torch.stack([p[:, step - t] for t, p in history], dim=1)
                    weights = torch.exp(-0.01 * torch.arange(len(history), device=prediction.device))
                    action = (current * (weights / weights.sum())[None, :, None]).sum(dim=1)
                else:
                    if step % args.num_queries == 0:
                        chunk = prediction
                    action = chunk[:, step % args.num_queries]
                action = action * stats['action_std'] + stats['action_mean']
                actions.append(action.cpu().numpy()[0])
                obs, _, _, _, info = video.step(action.cpu().numpy())
                success_once |= bool(info['success'].item())
                target_pick |= bool(info['is_target_grasped'].item() and info['target_lifted'].item())
        np.save(folder / 'executed_actions.npy', np.stack(actions))
        metadata.update(completed=True, success_once=success_once, target_grasped_and_lifted=target_pick,
                        success_at_end=bool(info['success'].item()))
    finally:
        if exporter is not None:
            exporter.close()
        video.close()


def main(cli):
    if cli.max_steps < 1 or cli.max_steps > 500 or cli.stride < 1:
        raise ValueError('max-steps must be 1..500; stride must be positive')
    if cli.seed < 0 or cli.seed >= 2**31:
        raise ValueError('seed must be in [0, 2**31)')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if cli.device == 'auto' else cli.device)
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summaries, fingerprints = [], []
    for index, checkpoint_path in enumerate(cli.checkpoints):
        checkpoint_hash = file_sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        args, streams = config(checkpoint)
        if not -1 <= cli.query_index < args.num_queries:
            raise ValueError('query-index must be -1 or a valid action query index')
        if not -args.enc_layers <= cli.encoder_layer < args.enc_layers:
            raise ValueError('encoder-layer is outside the checkpoint encoder')
        torch.manual_seed(1)
        agent, _, stats = load_policy(checkpoint, device)
        mode = 'with_mask' if len(streams) == 2 else 'without_mask'
        folder = output / f'{index:02d}_{mode}'
        folder.mkdir()
        metadata = metadata_for(checkpoint_path, checkpoint_hash, streams, cli)
        print(f'{mode}: {checkpoint_path} -> {folder}', flush=True)
        if cli.dataset:
            run_dataset(agent, checkpoint, stats, streams, folder, metadata, cli)
        else:
            run_rollout(agent, checkpoint, stats, streams, folder, metadata, cli, fingerprints)
        summaries.append(metadata)
        (output / 'summary.json').write_text(json.dumps(summaries, indent=2))
        print(f'{mode}: finished; video {folder / "attention.mp4"}', flush=True)
        del agent, stats, checkpoint
    print(f'Outputs: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', default=DEFAULT_CHECKPOINTS)
    parser.add_argument('--dataset', type=Path, help='Optional H5 input; otherwise run closed-loop policy inference')
    parser.add_argument('--trajectory', default='traj_0', help='H5 trajectory key')
    parser.add_argument('--seed', type=int, default=2026092500)
    parser.add_argument('--target', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--stride', type=int, default=5, help='Refresh maps every N frames; video retains source frame rate')
    parser.add_argument('--query-index', type=int, default=0, help='Current chunk action index; -1 averages all queries')
    parser.add_argument('--encoder-layer', type=int, default=0, help='Encoder layer for mask-to-RGB relation; 0 is first')
    parser.add_argument('--smooth', action='store_true', help='Bilinear heatmap display; raw saved 7x7 maps are unchanged')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / f'act_attention_{datetime.now():%Y%m%d_%H%M%S}')
    main(parser.parse_args())
