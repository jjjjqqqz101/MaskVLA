#!/usr/bin/env python3
"""Freeze ACT RGB/state, intervene on masked RGB, and measure model responses.

Optional closed-loop rollouts test actual grasping from identical initial states;
their later observations are allowed to diverge. No model weights are changed.
"""

import argparse
import csv
from datetime import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from act.attention_maps import AttentionRecorder, compute_attention_maps
from act.mask_counterfactual import compare_predictions, region_scores, render_comparison
from evaluate_rgb_ablation import config, load_policy, make_env, rollout
from train_dual_rgb import dual_rgb, policy_state
from train_single_rgb import single_rgb
from visualize_act_attention import DEFAULT_CHECKPOINTS, RESIZE, ROOT, file_sha256, predict


CONDITIONS = ('cube1', 'cube2', 'cube3', 'empty', 'full')
HEAT_KEYS = ('action_original', 'action_masked', 'mask_to_original', 'decoder_original')


def sha(array):
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def resize_mask(mask):
    return np.asarray(Image.fromarray(np.asarray(mask, dtype=np.uint8)).resize(
        (224, 224), Image.Resampling.NEAREST)) > 0


def condition_mask(snapshot, condition):
    if condition.startswith('cube'):
        return snapshot['masks'][int(condition[-1]) - 1]
    shape = snapshot['raw_rgb'].shape[:2]
    return np.full(shape, condition == 'full', dtype=np.bool_)


def make_input(snapshot, condition, streams):
    original = torch.from_numpy(snapshot['raw_rgb'].copy())[None]
    mask = condition_mask(snapshot, condition)
    if streams == 2:
        masked = torch.where(torch.from_numpy(mask)[None, ..., None], original, 0)
        rgb = dual_rgb(original, masked, RESIZE)
    else:
        rgb = single_rgb(original, RESIZE)
    return dict(rgb=rgb, state=torch.from_numpy(snapshot['state'].copy())[None]), resize_mask(mask)


def capture_snapshots(checkpoint, agent, args, stats, cli):
    last_step = max(cli.reference_steps)
    video, policy, _ = make_env(checkpoint, last_step + 1)
    snapshots = []
    try:
        for seed in cli.seeds:
            video.reset(seed=seed)
            base = video.unwrapped
            base.target_idx[:] = cli.reference_target - 1
            history = []
            for step in range(last_step + 1):
                raw = base.get_obs(base.get_info())
                obs = policy.observation(raw)
                if step in cli.reference_steps:
                    camera = raw['sensor_data']['base_camera']
                    segmentation = camera['segmentation']
                    masks = np.stack([
                        (segmentation == cube.per_scene_id.to(segmentation.device).reshape(1, 1, 1, 1))
                        [0, ..., 0].cpu().numpy() for cube in base.cubes])
                    snapshot = dict(seed=seed, step=step,
                                    raw_rgb=camera['rgb'][0].cpu().numpy().copy(),
                                    masks=masks, state=policy_state(raw)[0].cpu().numpy().copy(),
                                    cube_positions=np.stack([cube.pose.p[0].cpu().numpy() for cube in base.cubes]),
                                    tcp_position=base.agent.tcp.pose.p[0].cpu().numpy().copy())
                    rebuilt, _ = make_input(snapshot, f'cube{cli.reference_target}', 2)
                    if not torch.equal(rebuilt['rgb'], obs['rgb'].cpu()) or not torch.equal(rebuilt['state'], obs['state'].cpu()):
                        raise RuntimeError('Intervention preprocessing differs from the policy wrapper')
                    snapshots.append(snapshot)
                    print(f'Frozen seed={seed}, reference frame={step}, target pixels={masks.sum(axis=(1, 2)).tolist()}', flush=True)
                if step == last_step:
                    break
                if args.temporal_agg or step % args.num_queries == 0:
                    prediction = predict(agent, obs, stats)
                if args.temporal_agg:
                    history.append((step, prediction))
                    history = [(t, p) for t, p in history if step - t < args.num_queries]
                    current = torch.stack([p[:, step - t] for t, p in history], dim=1)
                    weights = torch.exp(-0.01 * torch.arange(len(history), device=prediction.device))
                    action = (current * (weights / weights.sum())[None, :, None]).sum(dim=1)
                else:
                    action = prediction[:, step % args.num_queries]
                action = action * stats['action_std'] + stats['action_mean']
                if not torch.isfinite(action).all():
                    raise ValueError('Nonfinite reference trajectory action')
                video.step(action.cpu().numpy())
    finally:
        video.close()
    return snapshots


def load_snapshots(path):
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    keys = ('seed', 'step', 'raw_rgb', 'masks', 'state', 'cube_positions', 'tcp_position')
    if any(key not in arrays for key in keys):
        raise ValueError('Frozen archive is missing required arrays')
    count = len(arrays['step'])
    if not count or any(len(arrays[key]) != count for key in keys):
        raise ValueError('Frozen input arrays have inconsistent frame counts')
    if arrays['raw_rgb'].ndim != 4 or arrays['raw_rgb'].shape[-1] != 3 or arrays['raw_rgb'].dtype != np.uint8:
        raise ValueError('Expected uint8 raw RGB [F,H,W,3]')
    if arrays['masks'].shape != (count, 3, *arrays['raw_rgb'].shape[1:3]) or arrays['masks'].dtype != np.bool_:
        raise ValueError('Expected three boolean object masks per frozen input')
    if arrays['state'].shape != (count, 28) or not np.isfinite(arrays['state']).all():
        raise ValueError('Expected finite state [F,28]')
    snapshots = [{key: arrays[key][index].copy() for key in keys} for index in range(count)]
    pairs = [(int(row['seed']), int(row['step'])) for row in snapshots]
    if len(set(pairs)) != len(pairs) or any(step < 0 for _, step in pairs):
        raise ValueError('Frozen inputs must have unique, nonnegative seed/frame pairs')
    if 'provenance_json' in arrays:
        provenance = json.loads(str(arrays['provenance_json'].item()))
    else:
        source_manifest = path.parent / 'experiment.json'
        source = json.loads(source_manifest.read_text()) if source_manifest.is_file() else {}
        provenance = {key: source.get(key) for key in ('reference_target', 'reference_checkpoint')}
    if not isinstance(provenance, dict):
        raise ValueError('Frozen input provenance must be a JSON object')
    provenance = {key: provenance.get(key) for key in ('reference_target', 'reference_checkpoint')}
    return sorted(snapshots, key=lambda row: (int(row['seed']), int(row['step']))), provenance


def heat_maxima(rows):
    """One scale covers every intervention, action query, and relation layer."""
    maxima = np.zeros(len(HEAT_KEYS), dtype=np.float32)
    for row in rows:
        maps = row['maps']
        indices = maps['visual_token_indices']
        all_actions = row['decoder_attention'] @ maps['encoder_rollout']
        maxima[0] = max(maxima[0], float(all_actions[:, indices[0].ravel()].max()))
        maxima[3] = max(maxima[3], float(row['decoder_attention'][:, indices[0].ravel()].max()))
        if len(indices) == 2:
            maxima[1] = max(maxima[1], float(all_actions[:, indices[1].ravel()].max()))
            for layer in range(len(row['encoder_attention'])):
                relation = compute_attention_maps(row['encoder_attention'], row['decoder_attention'],
                                                 (maps['feature_height'], maps['feature_width']), 2,
                                                 row['target_mask'], encoder_layer=layer)
                maxima[2] = max(maxima[2], float(relation['mask_to_original'].max()))
    return maxima


def evaluate_snapshot(agent, stats, streams, snapshot, cli):
    rows, input_hashes = [], []
    fixed_mask = f'cube{cli.reference_target}'
    reference_input, _ = make_input(snapshot, fixed_mask, streams)
    reference = predict(agent, reference_input, stats)
    with AttentionRecorder(agent.model) as recorder:
        for condition in CONDITIONS:
            obs, mask = make_input(snapshot, condition, streams)
            before = (sha(obs['rgb']), sha(obs['state']))
            input_hashes.append((sha(obs['rgb'][:, 0]), sha(obs['state'])))
            recorder.clear()
            prediction = predict(agent, obs, stats)
            if prediction.shape != reference.shape or not torch.isfinite(prediction).all():
                raise ValueError('Invalid counterfactual prediction')
            if before != (sha(obs['rgb']), sha(obs['state'])):
                raise RuntimeError('Inference mutated frozen inputs')
            if condition == fixed_mask:
                torch.testing.assert_close(prediction, reference, atol=1e-6, rtol=1e-5)
            maps = recorder.maps(streams, mask, query_index=cli.query_index, encoder_layer=cli.encoder_layer)
            rgb = obs['rgb'][0].numpy().copy()
            row = dict(condition=condition, rgb=rgb, target_mask=mask, state=obs['state'][0].numpy().copy(),
                       normalized_actions=prediction[0].cpu().numpy().copy(),
                       actions=(prediction * stats['action_std'] + stats['action_mean'])[0].cpu().numpy().copy(),
                       encoder_attention=np.stack(recorder.encoder_attention)[:, 0],
                       decoder_attention=recorder.decoder_attention[0].copy(), maps=maps,
                       masked_rgb=rgb[1].copy() if streams == 2 else None,
                       original_rgb_sha256=input_hashes[-1][0], state_sha256=input_hashes[-1][1])
            rows.append(row)
    if len(set(input_hashes)) != 1:
        raise RuntimeError('Original RGB or state differs between interventions')
    repeated = predict(agent, reference_input, stats)
    repeat_error = float((repeated - reference).abs().max())
    torch.testing.assert_close(repeated, reference, atol=1e-6, rtol=1e-5)
    if streams == 1:
        for row in rows[1:]:
            np.testing.assert_allclose(row['normalized_actions'], rows[0]['normalized_actions'], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(row['decoder_attention'], rows[0]['decoder_attention'], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(row['maps']['action_original'], rows[0]['maps']['action_original'], atol=1e-6, rtol=1e-5)
    return rows, repeat_error


def save_recording(folder, rows, metadata):
    folder.mkdir()
    arrays = {key: np.stack([row[key] for row in rows]) for key in
              ('rgb', 'target_mask', 'state', 'encoder_attention', 'decoder_attention', 'heat_maxima')}
    arrays.update(steps=np.array([row['step'] for row in rows], dtype=np.int64),
                  predicted_normalized_actions=np.stack([row['normalized_actions'] for row in rows]),
                  predicted_actions=np.stack([row['actions'] for row in rows]))
    np.savez_compressed(folder / 'viser_frames.npz', **arrays)
    metadata.update(viser_frames_file='viser_frames.npz', viser_frames_version=1,
                    captured_frames=len(rows), feature_shape=list(rows[0]['maps']['action_original'].shape),
                    steps=int(rows[-1]['step']), fps=20, closed_loop=False,
                    heat_maxima_order=list(HEAT_KEYS),
                    heat_scale='Shared zero-to-maximum scale over conditions, queries, and layers at each reference frame.')
    write_json(folder / 'metadata.json', metadata)
    write_json(folder / 'frames.json', [row['metrics'] for row in rows])


def run_behavioral_rollouts(checkpoint, agent, args, stats, seeds, folder, cli, fingerprints):
    folder.mkdir()
    video, policy, _ = make_env(checkpoint, cli.rollout_steps)
    records = []
    try:
        for seed in seeds:
            for target in (1, 2, 3):
                movie = folder / f'seed{seed}_cube{target}.mp4'
                result = rollout(video, policy, agent, args, stats, seed, target, cli.rollout_steps,
                                 fingerprints, movie)
                records.append(result)
                write_json(folder / 'rollouts.json', records)
                print(f'Behavior seed={seed}, target=cube{target}, correct_pick={result["correct_pick"]}, '
                      f'grasped={result["grasped_cube_ids"]}', flush=True)
    finally:
        video.close()
    return records


def main(cli):
    if not cli.seeds or any(seed < 0 or seed >= 2**31 for seed in cli.seeds):
        raise ValueError('seeds must be in [0, 2**31)')
    if not cli.reference_steps or any(step < 0 or step > 499 for step in cli.reference_steps):
        raise ValueError('reference-steps must be in [0,499]')
    if not 0 <= cli.rollout_steps <= 500:
        raise ValueError('rollout-steps must be in [0,500]; zero skips behavioral testing')
    cli.seeds, cli.reference_steps = sorted(set(cli.seeds)), sorted(set(cli.reference_steps))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if cli.device == 'auto' else cli.device)
    first_checkpoint = torch.load(cli.checkpoints[0], map_location='cpu', weights_only=False)
    _, first_streams = config(first_checkpoint)
    if len(first_streams) != 2:
        raise ValueError('The first checkpoint must be a dual-RGB policy for the reference trajectory')
    torch.manual_seed(1)
    reference_agent, reference_args, reference_stats = load_policy(first_checkpoint, device)
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if cli.frozen_inputs:
        snapshots, provenance = load_snapshots(cli.frozen_inputs)
    else:
        snapshots = capture_snapshots(first_checkpoint, reference_agent, reference_args, reference_stats, cli)
        provenance = dict(reference_target=cli.reference_target,
                          reference_checkpoint=str(cli.checkpoints[0].resolve()))
    np.savez_compressed(output / 'frozen_inputs.npz', **{
        key: np.stack([row[key] for row in snapshots]) for key in snapshots[0]},
        provenance_json=np.asarray(json.dumps(provenance)))
    seeds = sorted({int(row['seed']) for row in snapshots})
    reports = output / 'comparisons'
    reports.mkdir()
    model_summaries, all_metrics, all_pairs = [], [], []
    fingerprints, cross_model_inputs = {}, {}
    for model_index, checkpoint_path in enumerate(cli.checkpoints):
        if model_index == 0:
            checkpoint, agent, args, stats = first_checkpoint, reference_agent, reference_args, reference_stats
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            torch.manual_seed(1)
            agent, args, stats = load_policy(checkpoint, device)
        _, stream_names = config(checkpoint)
        streams = len(stream_names)
        if not -1 <= cli.query_index < args.num_queries or not -args.enc_layers <= cli.encoder_layer < args.enc_layers:
            raise ValueError('Selected query or encoder layer is outside the checkpoint')
        mode = 'with_mask' if streams == 2 else 'without_mask'
        label = 'With mask' if streams == 2 else 'Without mask'
        checkpoint_hash = file_sha256(checkpoint_path)
        grouped, repeat_errors, model_pairs = {}, [], []
        for snapshot in snapshots:
            seed, step = int(snapshot['seed']), int(snapshot['step'])
            rows, repeat_error = evaluate_snapshot(agent, stats, streams, snapshot, cli)
            repeat_errors.append(repeat_error)
            fingerprint = (rows[0]['original_rgb_sha256'], rows[0]['state_sha256'])
            key = (seed, step)
            if key in cross_model_inputs and cross_model_inputs[key] != fingerprint:
                raise RuntimeError('Fixed inputs differ between the dual- and single-RGB models')
            cross_model_inputs[key] = fingerprint
            maxima = heat_maxima(rows)
            fixed_regions = np.stack([resize_mask(mask) for mask in snapshot['masks']])
            for row in rows:
                metrics = dict(model_index=model_index, model=mode, seed=seed, reference_step=step,
                               condition=row['condition'], out_of_distribution_control=row['condition'] in ('empty', 'full'),
                               target_pixels=int(row['target_mask'].sum()),
                               original_rgb_sha256=row['original_rgb_sha256'], state_sha256=row['state_sha256'],
                               masked_rgb_sha256=sha(row['masked_rgb']) if streams == 2 else None,
                               action_mass_original=row['maps']['action_mass_original'],
                               action_mass_masked=row['maps']['action_mass_masked'],
                               **region_scores(row['maps'], fixed_regions))
                row.update(step=step, metrics=metrics, heat_maxima=maxima)
                grouped.setdefault((seed, row['condition']), []).append(row)
                all_metrics.append(metrics)
            for left, right in combinations(rows, 2):
                pair = dict(model_index=model_index, model=mode, seed=seed, reference_step=step,
                            condition_a=left['condition'], condition_b=right['condition'],
                            **compare_predictions(left, right))
                all_pairs.append(pair)
                model_pairs.append(pair)
            render_comparison(rows, reports / f'{model_index:02d}_{mode}_seed{seed}_frame{step:04d}.png',
                              f'{label} | seed {seed} | frozen reference frame {step} | query {cli.query_index}')
            print(f'{mode} seed={seed}, frame={step}: RGB/state fixed; repeat max diff={repeat_error:.3g}', flush=True)
        for (seed, condition), rows in grouped.items():
            metadata = dict(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=checkpoint_hash,
                            weights='ema_agent', stream_order=list(stream_names), source='mask_counterfactual',
                            counterfactual_condition=condition, seed=seed,
                            target=int(condition[-1]) if condition.startswith('cube') else None,
                            reference_target=provenance.get('reference_target'),
                            reference_checkpoint=provenance.get('reference_checkpoint'),
                            display_label=f'{label} / seed {seed} / {condition}',
                            decoder_layer=0, encoder_relation_layer=cli.encoder_layer,
                            action_query=cli.query_index, fixed_rgb_state_verified=True,
                            action_units='First 7: absolute arm joint targets (rad); last: gripper command [-1,1].')
            save_recording(output / f'{model_index:02d}_{mode}_seed{seed}_{condition}', rows, metadata)
        natural_pairs = [row for row in model_pairs if row['condition_a'].startswith('cube') and row['condition_b'].startswith('cube')]
        summary = dict(model_index=model_index, model=mode, checkpoint=str(checkpoint_path.resolve()),
                       checkpoint_sha256=checkpoint_hash, frozen_observations=len(snapshots),
                       conditions=list(CONDITIONS), fixed_rgb_state_verified=True,
                       no_mask_negative_control_passed=True if streams == 1 else None,
                       hooks_preserve_predictions=True, max_repeat_prediction_difference=max(repeat_errors),
                       natural_mask_pair_mean_normalized_rmse=float(np.mean([row['normalized_chunk_rmse'] for row in natural_pairs])),
                       natural_mask_pair_mean_arm_rmse_rad=float(np.mean([row['arm_chunk_rmse_rad'] for row in natural_pairs])))
        if cli.rollout_steps:
            behavior = run_behavioral_rollouts(checkpoint, agent, args, stats, seeds,
                                              output / f'behavior_{model_index:02d}_{mode}', cli, fingerprints)
            summary['behavioral_rollouts'] = behavior
            summary['correct_pick_rate'] = float(np.mean([row['correct_pick'] for row in behavior]))
        model_summaries.append(summary)
        write_json(output / 'summary.json', model_summaries)
        write_csv(output / 'conditions.csv', all_metrics)
        write_csv(output / 'pairwise.csv', all_pairs)
        print(json.dumps(summary, indent=2), flush=True)
        if model_index == 0:
            del reference_agent, reference_stats, first_checkpoint
        del agent, stats, checkpoint
    manifest = dict(completed=True, source='mask_counterfactual', **provenance,
                    frozen_inputs_source=str(cli.frozen_inputs.resolve()) if cli.frozen_inputs else None,
                    snapshots=[dict(seed=int(row['seed']), step=int(row['step']),
                                    original_rgb_sha256=sha(row['raw_rgb']), state_sha256=sha(row['state'])) for row in snapshots],
                    conditions=list(CONDITIONS), query_index=cli.query_index, encoder_layer=cli.encoder_layer,
                    rollout_steps=cli.rollout_steps,
                    limits=['Actions are joint targets, not grasp coordinates.',
                            'Attention rollout is a coarse routing statistic, not causal pixel attribution.',
                            'Mask-to-RGB maps also change when query regions change; use action/raw attention responses as evidence.',
                            'Empty/full masks are diagnostic out-of-distribution controls.',
                            'Behavioral rollouts share initial inputs; later observations diverge.',
                            'A small number of scenes cannot establish generalization.'])
    write_json(output / 'experiment.json', manifest)
    print(f'Completed. Results: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', default=DEFAULT_CHECKPOINTS)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026092500])
    parser.add_argument('--reference-steps', type=int, nargs='+', default=[0, 100, 200])
    parser.add_argument('--reference-target', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--frozen-inputs', type=Path, help='Replay a saved frozen_inputs.npz without running reference simulation')
    parser.add_argument('--query-index', type=int, default=0, help='Attention action query; -1 averages all queries; metrics always use the full chunk')
    parser.add_argument('--encoder-layer', type=int, default=0)
    parser.add_argument('--rollout-steps', type=int, default=0, help='Additionally execute each real target from the same reset, up to 500 steps; zero skips this')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / f'act_mask_counterfactual_{datetime.now():%Y%m%d_%H%M%S}')
    main(parser.parse_args())
