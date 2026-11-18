"""Run bounded C/D data and model checks, without training or a simulator.

Example: python -m target_ablation_cd.verify --num-demos 2 --device cpu
An optional --checkpoint checks the new schema and optimizer/scheduler/EMA
update counts. No checkpoint is loaded into the models used for unit checks.
"""

import argparse
from dataclasses import fields
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import default_collate

from .data import DEFAULT_TRAIN, DEFAULT_VALIDATION, STATE_FIELDS, TargetDataset, mask_centroid
from .model import ACTPolicy, ModelConfig


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exact(left, right):
    torch.testing.assert_close(left, right, atol=0, rtol=0)


def clone_observation(obs):
    return {key: value.detach().clone() for key, value in obs.items()}


def check_observation_unchanged(obs, before):
    require(set(obs) == set(before), 'Inference changed observation keys')
    for key in before:
        exact(obs[key], before[key])


def check_centroids():
    masks = torch.zeros((4, 3, 5), dtype=torch.bool)
    masks[0, 0, 0] = True
    masks[1, 2, 4] = True
    masks[2, 0, 1] = masks[2, 2, 3] = True
    xy, valid = mask_centroid(masks)
    exact(xy, torch.tensor([[-1., -1.], [1., 1.], [0., 0.], [0., 0.]]))
    require(valid.tolist() == [True, True, True, False], 'Incorrect empty-mask validity')
    singleton_xy, singleton_valid = mask_centroid(torch.ones((1, 1, 1, 1), dtype=torch.bool))
    exact(singleton_xy, torch.zeros((1, 2)))
    require(bool(singleton_valid[0]), 'Visible singleton mask must be valid')
    return dict(passed=True, cases=5, empty_xy=[0, 0], empty_valid=False)


def dense_normalization(dataset):
    """Reference the original end-padded window definition on a small subset."""
    states, chunks = [], []
    with h5py.File(dataset.path, 'r') as source:
        for name in dataset.trajectory_names:
            trajectory = source[name]
            actions = torch.from_numpy(trajectory['actions'][:]).float()
            states.append(torch.cat([
                torch.from_numpy(trajectory[f'obs/{group}/{key}'][:len(actions)]).float()
                for group, key in STATE_FIELDS], dim=-1))
            for step in range(len(actions)):
                chunk = actions[step:step + dataset.num_queries]
                if len(chunk) < dataset.num_queries:
                    chunk = torch.cat((chunk, chunk[-1:].repeat(dataset.num_queries - len(chunk), 1)))
                chunks.append(chunk)
    differences = {}
    for prefix, values in (('state', torch.cat(states)), ('action', torch.cat(chunks))):
        expected = {f'{prefix}_mean': values.mean(0, keepdim=True),
                    f'{prefix}_std': values.std(0, keepdim=True).clamp_min(1e-2)}
        for key, value in expected.items():
            torch.testing.assert_close(dataset.norm_stats[key], value, atol=1e-6, rtol=2e-5)
            differences[key] = float((dataset.norm_stats[key] - value).abs().max())
    return differences


def check_data(args):
    datasets = []
    try:
        c = TargetDataset(args.train_data, 'C', num_demos=args.num_demos)
        datasets.append(c)
        d = TargetDataset(args.train_data, 'D', num_demos=args.num_demos)
        datasets.append(d)
        require(c.trajectory_names == d.trajectory_names and c.lengths == d.lengths,
                'C/D trajectories or action alignment differ')
        require(len(c) >= 2, 'The model check needs two training observations')
        for key in ('state_mean', 'state_std', 'action_mean', 'action_std'):
            exact(c.norm_stats[key], d.norm_stats[key])
        val = TargetDataset(args.validation_data, 'C', num_demos=1, norm_stats=c.norm_stats)
        datasets.append(val)
        for key in ('state_mean', 'state_std', 'action_mean', 'action_std'):
            exact(val.norm_stats[key], c.norm_stats[key])
        require(torch.isfinite(val[0]['actions']).all(), 'Validation actions are nonfinite')
        indices = sorted({0, 1, c.lengths[0] - 1, len(c) - 1})
        for index in indices:
            left, right = c[index], d[index]
            exact(left['observations']['rgb'][0], right['observations']['rgb'][0])
            exact(left['observations']['state'], right['observations']['state'])
            exact(left['actions'], right['actions'])
            binary = right['observations']['rgb'][1]
            require(binary.dtype == torch.uint8 and set(binary.unique().tolist()) <= {0, 255},
                    'D second stream must remain binary uint8 0/255')
            exact(binary[0], binary[1])
            exact(binary[1], binary[2])
            require(set(right['observations']) == {'rgb', 'state'}, 'D received centroid fields')
        batches = {condition: default_collate([dataset[0], dataset[1]])
                   for condition, dataset in (('C', c), ('D', d))}
        report = dict(passed=True, train_data=str(c.path), validation_data=str(val.path),
                      trajectories=c.num_traj, samples=len(c), checked_indices=indices,
                      original_rgb_state_actions_identical=True, binary_mask_verified=True,
                      validation_uses_training_stats=True,
                      weighted_vs_dense_max_errors=dense_normalization(c))
        return batches, report
    finally:
        for dataset in datasets:
            dataset.close()


def check_models(batches, device):
    torch.manual_seed(31)
    c = ACTPolicy(ModelConfig(condition='C')).to(device)
    torch.manual_seed(31)
    d = ACTPolicy(ModelConfig(condition='D')).to(device)
    c_state, d_state = c.state_dict(), d.state_dict()
    require(set(d_state) <= set(c_state), 'C/D shared parameter names differ')
    for name, value in d_state.items():
        exact(c_state[name], value)
    extra = sorted(set(c_state) - set(d_state))
    require(extra == ['model.center_pos.weight', 'model.center_proj.bias',
                      'model.center_proj.weight', 'model.missing_target.weight'],
            'C registered unexpected additional parameters')
    observations, repeat_errors = {}, {}
    for condition, policy in (('C', c), ('D', d)):
        obs = {key: value.to(device) for key, value in batches[condition]['observations'].items()}
        observations[condition] = obs
        before = clone_observation(obs)
        policy.eval()
        with torch.no_grad():
            first, second = policy.get_action(obs), policy.get_action(obs)
        require(first.shape == (2, 30, 8) and torch.isfinite(first).all(),
                f'{condition} predicted invalid action chunks')
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
        repeat_errors[condition] = float((first - second).abs().max())
        check_observation_unchanged(obs, before)
    invalid = clone_observation(observations['C'])
    invalid['target_valid'].zero_()
    invalid['target_xy'].zero_()
    invalid_before = clone_observation(invalid)
    changed = clone_observation(invalid)
    changed['target_xy'][:] = torch.tensor([[-1., 1.], [1., -1.]], device=device)
    changed_before = clone_observation(changed)
    with torch.no_grad():
        missing_first, missing_changed = c.get_action(invalid), c.get_action(changed)
    torch.testing.assert_close(missing_first, missing_changed, atol=1e-6, rtol=1e-5)
    check_observation_unchanged(invalid, invalid_before)
    check_observation_unchanged(changed, changed_before)
    obs = observations['C']
    require(bool(obs['target_valid'].all()), 'Model gradient check needs two visible targets')
    before = clone_observation(obs)
    actions = batches['C']['actions'].to(device)
    actions_before = actions.clone()
    c.train()
    c.zero_grad(set_to_none=True)
    torch.manual_seed(32)
    losses = c.compute_loss(obs, actions)
    require(set(losses) == {'loss', 'l1', 'kl'}, 'Unexpected complete ACT loss keys')
    require(all(torch.isfinite(value).all() for value in losses.values()), 'Nonfinite ACT loss')
    losses['loss'].backward()
    gradient = c.model.center_proj.weight.grad
    require(gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0,
            'model.center_proj.weight did not receive a finite nonzero gradient')
    check_observation_unchanged(obs, before)
    exact(actions, actions_before)
    return dict(passed=True, device=str(device), batch_size=2,
                shared_state_tensors=len(d_state), shared_initialization_identical=True,
                extra_centroid_parameters=extra, repeated_forward_max_errors=repeat_errors,
                invalid_xy_prediction_max_error=float((missing_first - missing_changed).abs().max()),
                observations_unchanged=True, center_proj_gradient_norm=float(gradient.norm()),
                losses={key: float(value.detach()) for key, value in losses.items()}, optimizer_updates=0)


def check_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    required = {'format_version', 'experiment', 'condition', 'model_config', 'training_config',
                'iteration', 'agent', 'ema_agent', 'norm_stats', 'optimizer', 'scheduler', 'ema',
                'state_fields', 'stream_order', 'input_contract', 'rng', 'best_metrics',
                'env_kwargs', 'train_signature', 'validation_signature', 'train_episode_seeds',
                'validation_episode_seeds', 'selection_eval_seeds'}
    require(isinstance(checkpoint, dict) and required <= set(checkpoint), 'Checkpoint schema is incomplete')
    require(checkpoint['experiment'] == 'target_ablation_cd' and checkpoint['format_version'] == 1,
            'Checkpoint is not target_ablation_cd format version 1')
    require({field.name for field in fields(ModelConfig)} <= set(checkpoint['model_config']),
            'Checkpoint model_config is incomplete')
    config = ModelConfig(**checkpoint['model_config'])
    require(config.condition == checkpoint['condition'] == checkpoint['training_config']['condition'],
            'Checkpoint condition fields disagree')
    iteration = int(checkpoint['iteration'])
    require(0 < iteration <= checkpoint['training_config']['total_iters'], 'Invalid checkpoint iteration')
    require(tuple(map(tuple, checkpoint['state_fields'])) == STATE_FIELDS, 'Checkpoint state whitelist differs')
    streams = ['original_rgb'] if config.condition == 'C' else ['original_rgb', 'binary_mask_rgb']
    require(list(checkpoint['stream_order']) == streams, 'Checkpoint stream order differs')
    contract = checkpoint['input_contract']
    require(contract['state_dim'] == 28 and contract['condition'] == config.condition
            and contract['rgb_shape'] == [len(streams), 3, 224, 224], 'Checkpoint input contract differs')
    weights, averaged = checkpoint['agent'], checkpoint['ema_agent']
    require(set(weights) == set(averaged), 'Agent and EMA parameter schemas differ')
    for key, value in weights.items():
        require(value.shape == averaged[key].shape, f'EMA shape differs: {key}')
        require(torch.isfinite(value).all() and torch.isfinite(averaged[key]).all(), f'Nonfinite weight: {key}')
    expected_shapes = {'model.input_proj_robot_state.weight': (config.hidden_dim, 28),
                       'model.action_head.weight': (8, config.hidden_dim),
                       'model.query_embed.weight': (config.num_queries, config.hidden_dim)}
    if config.condition == 'C':
        expected_shapes['model.center_proj.weight'] = (config.hidden_dim, 2)
    else:
        require('model.center_proj.weight' not in weights, 'D checkpoint unexpectedly contains centroid projection')
    for key, shape in expected_shapes.items():
        require(key in weights and weights[key].shape == shape, f'Unexpected parameter shape: {key}')
    for key, width in (('state_mean', 28), ('state_std', 28), ('action_mean', 8), ('action_std', 8)):
        value = checkpoint['norm_stats'][key]
        require(value.shape == (1, width) and torch.isfinite(value).all(), f'Invalid normalization: {key}')
        if key.endswith('std'):
            require((value > 0).all(), f'Nonpositive normalization: {key}')
    optimizer_states = checkpoint['optimizer']['state']
    require(bool(optimizer_states), 'Optimizer has no completed parameter updates')
    optimizer_steps = {int(value['step']) for value in optimizer_states.values()}
    scheduler_step = int(checkpoint['scheduler']['last_epoch'])
    ema_step = int(checkpoint['ema']['optimization_step'])
    require(optimizer_steps == {iteration} and scheduler_step == iteration and ema_step == iteration,
            f'Update counters differ: iteration={iteration}, optimizer={optimizer_steps}, '
            f'scheduler={scheduler_step}, ema={ema_step}')
    require(set(checkpoint['train_episode_seeds']).isdisjoint(checkpoint['validation_episode_seeds']),
            'Checkpoint training and validation seeds overlap')
    return dict(passed=True, path=str(Path(path).resolve()), condition=config.condition,
                iteration=iteration, optimizer_steps=sorted(optimizer_steps),
                scheduler_steps=scheduler_step, ema_steps=ema_step, format_version=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-demos', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu')
    parser.add_argument('--train-data', type=Path, default=DEFAULT_TRAIN)
    parser.add_argument('--validation-data', type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, help='Optional JSON report; omitted means stdout only')
    args = parser.parse_args(argv)
    report = {'passed': False, 'checks': {}}
    try:
        require(args.num_demos > 0, 'num-demos must be positive')
        torch.set_num_threads(2)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        require(device_name != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
        report['checks']['centroid'] = check_centroids()
        batches, report['checks']['data'] = check_data(args)
        report['checks']['model'] = check_models(batches, torch.device(device_name))
        if args.checkpoint:
            report['checks']['checkpoint'] = check_checkpoint(args.checkpoint)
        report['passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    rendered = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
