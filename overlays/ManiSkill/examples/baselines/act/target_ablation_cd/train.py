"""Train C/D ACT policies with independent outputs and a shared experiment protocol."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import random
import time

from diffusers.training_utils import EMAModel
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from .data import DEFAULT_TRAIN, DEFAULT_VALIDATION, STATE_FIELDS, TargetDataset
from .model import ACTPolicy, ModelConfig


HERE = Path(__file__).resolve().parent
FORMAT_VERSION = 1
EXPERIMENT = 'target_ablation_cd'
PATH_KEYS = ('train_data', 'validation_data', 'output')


class IterationSampler(Sampler):
    """Epoch permutations depend only on seed/iteration, including after resume."""

    def __init__(self, size, batch_size, start, stop, seed):
        self.size, self.batch_size = size, batch_size
        self.start, self.stop, self.seed = start, stop, seed
        self.batches_per_epoch = size // batch_size
        if self.batches_per_epoch < 1:
            raise ValueError('batch-size exceeds dataset length with drop_last=True')

    def __iter__(self):
        previous_epoch, order = None, None
        for iteration in range(self.start, self.stop):
            epoch, batch = divmod(iteration, self.batches_per_epoch)
            if epoch != previous_epoch:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                order = torch.randperm(self.size, generator=generator).tolist()
                previous_epoch = epoch
            first = batch * self.batch_size
            yield order[first:first + self.batch_size]

    def __len__(self):
        return max(0, self.stop - self.start)


def defaults():
    return dict(**asdict(ModelConfig()), train_data=str(DEFAULT_TRAIN),
                validation_data=str(DEFAULT_VALIDATION), num_demos=450,
                total_iters=200000, batch_size=256, seed=1, lr=1e-4,
                weight_decay=1e-4, num_workers=0, device='auto',
                log_every=100, save_every=10000, eval_every=20000,
                eval_scenes=10, eval_seed_start=2026100000,
                max_steps=500, temporal_agg=True, validation_batches=0,
                tensorboard=True, output=None)


def parser():
    result = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    result.add_argument('--condition', choices=('C', 'D'), help='C: RGB + visible-mask centroid; D: RGB + binary mask')
    result.add_argument('--train-data', type=Path)
    result.add_argument('--validation-data', type=Path)
    result.add_argument('--output', type=Path, help='New run directory; defaults under target_ablation_cd/runs')
    result.add_argument('--resume', type=Path, help='Resume a new-format checkpoint with its saved protocol')
    result.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    for name in ('num-demos', 'total-iters', 'batch-size', 'seed', 'num-workers', 'log-every',
                 'save-every', 'eval-every', 'eval-scenes', 'eval-seed-start', 'max-steps',
                 'validation-batches', 'hidden-dim', 'enc-layers', 'dec-layers',
                 'dim-feedforward', 'nheads', 'num-queries'):
        result.add_argument('--' + name, type=int)
    for name in ('lr', 'lr-backbone', 'weight-decay', 'kl-weight', 'dropout'):
        result.add_argument('--' + name, type=float)
    for name in ('temporal-agg', 'tensorboard'):
        result.add_argument('--' + name, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    result.add_argument('--stop-after', type=int, help='Stop after this many additional updates and save latest.pt; scheduler budget stays unchanged')
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def torch_load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def resolve_config(provided):
    provided = dict(provided)
    resume = provided.pop('resume', None)
    stop_after = provided.pop('stop_after', None)
    for key in PATH_KEYS:
        if key in provided:
            provided[key] = str(Path(provided[key]).resolve())
    checkpoint = None
    if resume is not None:
        checkpoint = torch_load(resume)
        if checkpoint.get('experiment') != EXPERIMENT or checkpoint.get('format_version') != FORMAT_VERSION:
            raise ValueError('Resume accepts only target_ablation_cd checkpoints, not old RGB/masked-RGB runs')
        config = dict(checkpoint['training_config'])
        for key, value in provided.items():
            if key not in ('device', 'num_workers', 'log_every') and value != config.get(key):
                raise ValueError(f'Resume cannot change {key}; start a new run to change the experiment protocol')
            config[key] = value
        if not Path(config['output']).is_dir():
            raise ValueError('The original run directory is missing; resume preserves its logs and checkpoints')
    else:
        if 'condition' not in provided:
            raise ValueError('A new run requires --condition C or --condition D')
        config = defaults()
        config.update(provided)
        if config['output'] is None:
            tag = f'{config["condition"]}_seed{config["seed"]}_{datetime.now():%Y%m%d_%H%M%S}'
            config['output'] = str(HERE / 'runs' / tag)
    for key in PATH_KEYS:
        config[key] = str(Path(config[key]).resolve())
    if config['total_iters'] < 2 or config['batch_size'] < 1 or config['num_demos'] < 0:
        raise ValueError('total-iters >= 2, batch-size >= 1, num-demos >= 0 (0 means all) are required')
    if not 0 <= config['seed'] < 2**31 or config['num_workers'] < 0:
        raise ValueError('Invalid seed or num-workers')
    if config['log_every'] < 1 or config['save_every'] < 1 or config['eval_every'] < 0:
        raise ValueError('log/save frequencies must be positive; eval-every=0 disables rollout selection')
    if config['eval_scenes'] < 1 or not 1 <= config['max_steps'] <= 500 or config['validation_batches'] < 0:
        raise ValueError('Invalid evaluation settings')
    if not 0 <= config['eval_seed_start'] < config['eval_seed_start'] + config['eval_scenes'] <= 2**31:
        raise ValueError('Invalid evaluation seed range')
    if config['lr'] <= 0 or config['lr_backbone'] < 0 or config['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')
    if stop_after is not None and stop_after < 1:
        raise ValueError('stop-after must be positive')
    if config['train_data'] == config['validation_data']:
        raise ValueError('Training and validation files must differ')
    return config, checkpoint, stop_after


def data_metadata(path, names):
    metadata_path = Path(path).with_suffix('.json')
    meta = json.loads(metadata_path.read_text())
    if meta['env_info']['env_id'] != 'TargetPickCube-v1':
        raise ValueError('Only TargetPickCube-v1 data is supported')
    kwargs = meta['env_info']['env_kwargs']
    if kwargs.get('control_mode') != 'pd_joint_pos':
        raise ValueError('Expected absolute pd_joint_pos demonstrations')
    episodes = {int(row['episode_id']): row for row in meta['episodes']}
    seeds = [int(episodes[int(name.split('_')[-1])]['episode_seed']) for name in names]
    stat = Path(path).stat()
    signature = dict(path=str(Path(path).resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                     metadata_path=str(metadata_path), trajectories=list(names))
    return kwargs, sorted(set(seeds)), signature


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


@torch.no_grad()
def validate(policy, loader, device, max_batches):
    was_training = policy.training
    policy.eval()
    total, count = 0.0, 0
    try:
        for index, batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            obs = {key: value.to(device) for key, value in batch['observations'].items()}
            target = batch['actions'].to(device)
            predicted = policy.get_action(obs)
            if not torch.isfinite(predicted).all():
                raise FloatingPointError('Nonfinite validation prediction')
            total += float(F.l1_loss(predicted, target)) * len(target)
            count += len(target)
    finally:
        policy.train(was_training)
    return dict(prior_l1=total / count, samples=count)


def main(argv=None):
    config, resume, stop_after = resolve_config(vars(parser().parse_args(argv)))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if config['device'] == 'auto' else config['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    device = torch.device(device_name)
    output = Path(config['output'])
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'checkpoints').mkdir()
    model_fields = set(asdict(ModelConfig()))
    model_config = ModelConfig(**{key: config[key] for key in model_fields})
    train = TargetDataset(config['train_data'], config['condition'], config['num_queries'],
                          config['num_demos'] or None, resume['norm_stats'] if resume else None)
    validation = TargetDataset(config['validation_data'], config['condition'], config['num_queries'],
                               norm_stats=train.norm_stats)
    env_kwargs, train_seeds, train_signature = data_metadata(config['train_data'], train.trajectory_names)
    val_kwargs, val_seeds, val_signature = data_metadata(config['validation_data'], validation.trajectory_names)
    if set(train_seeds) & set(val_seeds):
        raise ValueError('Training and validation episode seeds overlap')
    for key in ('sensor_configs', 'robot_uids', 'robot_init_qpos_noise', 'control_mode'):
        if env_kwargs.get(key) != val_kwargs.get(key):
            raise ValueError(f'Training/validation environment mismatch: {key}')
    eval_seeds = list(range(config['eval_seed_start'], config['eval_seed_start'] + config['eval_scenes'])) if config['eval_every'] else []
    if set(eval_seeds) & (set(train_seeds) | set(val_seeds)):
        raise ValueError('Rollout selection seeds overlap demonstration data')
    if resume and (resume['train_signature'] != train_signature or resume['validation_signature'] != val_signature):
        raise ValueError('Dataset identity changed since the checkpoint')
    if resume and (resume['env_kwargs'] != env_kwargs or resume['train_episode_seeds'] != train_seeds
                   or resume['validation_episode_seeds'] != val_seeds):
        raise ValueError('Dataset environment or episode metadata changed since the checkpoint')
    start = int(resume['iteration']) if resume else 0
    if start >= config['total_iters']:
        raise ValueError('This checkpoint has already reached its configured training budget')
    sampler = IterationSampler(len(train), config['batch_size'], start, config['total_iters'], config['seed'])
    loader_kwargs = dict(num_workers=config['num_workers'], pin_memory=device.type == 'cuda')
    if config['num_workers']:
        loader_kwargs.update(multiprocessing_context='spawn', persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train, batch_sampler=sampler,
                              generator=torch.Generator().manual_seed(config['seed'] + 7000), **loader_kwargs)
    val_loader = DataLoader(validation, batch_size=config['batch_size'], shuffle=False,
                            generator=torch.Generator().manual_seed(config['seed'] + 8000), **loader_kwargs)
    policy = ACTPolicy(model_config).to(device)
    ema_policy = deepcopy(policy).eval()
    parameter_groups = [
        dict(params=[p for name, p in policy.named_parameters() if 'backbone' not in name and p.requires_grad]),
        dict(params=[p for name, p in policy.named_parameters() if 'backbone' in name and p.requires_grad], lr=config['lr_backbone']),
    ]
    optimizer = torch.optim.AdamW(parameter_groups, lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max(1, int(config['total_iters'] * 2 / 3)), gamma=0.1)
    ema = EMAModel(parameters=policy.parameters(), power=0.75)
    best = dict(validation_l1=None, correct_pick=None, success_once=None)
    if resume:
        policy.load_state_dict(resume['agent'], strict=True)
        ema_policy.load_state_dict(resume['ema_agent'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        ema.load_state_dict(resume['ema'])
        ema.to(device)
        best.update(resume['best_metrics'])
        restore_rng(resume['rng'])
    elif device.type == 'cuda':
        ema.to(device)
    streams = ['original_rgb'] if config['condition'] == 'C' else ['original_rgb', 'binary_mask_rgb']
    contract = dict(condition=config['condition'], state_dim=28, state_fields=STATE_FIELDS,
                    stream_order=streams, rgb_shape=[len(streams), 3, 224, 224],
                    centroid_format='xy in [-1,1], from visible mask pixel means; bool valid' if config['condition'] == 'C' else None,
                    binary_mask_values=[0, 255] if config['condition'] == 'D' else None,
                    scene_encoder_tokens=52 if config['condition'] == 'C' else 100,
                    action_decoder_layer=0, action_units='7 absolute arm targets (rad), 1 gripper command',
                    parameter_count=sum(p.numel() for p in policy.parameters()))
    write_json(output / 'config.json', dict(training_config=config, model_config=asdict(model_config),
                                          input_contract=contract, train_signature=train_signature,
                                          validation_signature=val_signature, selection_eval_seeds=eval_seeds))
    print('INPUT_CONTRACT', json.dumps(contract), flush=True)
    print(f'Run: {output}; train samples={len(train)}; validation samples={len(validation)}; device={device}', flush=True)
    writer = SummaryWriter(str(output / 'tensorboard'), purge_step=start if resume else None) if config['tensorboard'] else None
    log = (output / 'metrics.jsonl').open('a', buffering=1)
    step = start

    def save(tag):
        ema.copy_to(ema_policy.parameters())
        state = dict(format_version=FORMAT_VERSION, experiment=EXPERIMENT,
                     condition=config['condition'], model_config=asdict(model_config), training_config=config,
                     iteration=step, agent=policy.state_dict(), ema_agent=ema_policy.state_dict(),
                     norm_stats={key: value.detach().cpu() for key, value in train.norm_stats.items()},
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), ema=ema.state_dict(),
                     rng=rng_state(), best_metrics=dict(best), env_kwargs=env_kwargs,
                     state_fields=STATE_FIELDS, stream_order=streams, input_contract=contract,
                     train_episode_seeds=train_seeds, validation_episode_seeds=val_seeds,
                     selection_eval_seeds=eval_seeds, train_signature=train_signature,
                     validation_signature=val_signature)
        destination = output / 'checkpoints' / f'{tag}.pt'
        temporary = destination.with_suffix('.pt.tmp')
        torch.save(state, temporary)
        os.replace(temporary, destination)

    def link_saved_checkpoint(tag):
        # A hard link preserves this checkpoint even when latest.pt is replaced.
        destination = output / 'checkpoints' / f'{tag}.pt'
        temporary = destination.with_suffix('.pt.tmp')
        if temporary.exists():
            temporary.unlink()
        os.link(output / 'checkpoints/latest.pt', temporary)
        os.replace(temporary, destination)

    policy.train()
    tick = time.monotonic()
    try:
        for batch in train_loader:
            obs = {key: value.to(device, non_blocking=True) for key, value in batch['observations'].items()}
            target = batch['actions'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = policy.compute_loss(obs, target)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise FloatingPointError(f'Nonfinite loss before update {step + 1}')
            losses['loss'].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            ema.step(policy.parameters())
            step += 1
            values = {key: float(value.detach()) for key, value in losses.items()}
            log.write(json.dumps(dict(iteration=step, **values, grad_norm=float(grad_norm), lr=optimizer.param_groups[0]['lr'])) + '\n')
            if writer:
                for key, value in values.items():
                    writer.add_scalar(f'train/{key}', value, step)
            if step == start + 1 or step % config['log_every'] == 0:
                print(f'{config["condition"]} update {step}/{config["total_iters"]}: '
                      f'loss={values["loss"]:.5f}, l1={values["l1"]:.5f}, kl={values["kl"]:.5f}, '
                      f'elapsed={time.monotonic() - tick:.1f}s', flush=True)
            stopping = stop_after is not None and step - start >= stop_after
            checking = step == config['total_iters'] or (config['eval_every'] and step % config['eval_every'] == 0)
            tags = []
            if checking:
                evaluation_rng = rng_state()
                ema.copy_to(ema_policy.parameters())
                validation_metrics = validate(ema_policy, val_loader, device, config['validation_batches'])
                log.write(json.dumps(dict(iteration=step, validation=validation_metrics)) + '\n')
                print('VALIDATION', json.dumps(validation_metrics), flush=True)
                if best['validation_l1'] is None or validation_metrics['prior_l1'] < best['validation_l1']:
                    best['validation_l1'] = validation_metrics['prior_l1']
                    tags.append('best_validation_l1')
                if writer:
                    writer.add_scalar('validation/prior_l1', validation_metrics['prior_l1'], step)
                if config['eval_every']:
                    from .evaluate import run_rollouts
                    result = run_rollouts(ema_policy, model_config, train.norm_stats, env_kwargs,
                                          eval_seeds, config['max_steps'], config['temporal_agg'])
                    write_json(output / f'evaluation_{step:07d}.json', result)
                    log.write(json.dumps(dict(iteration=step, evaluation=result['rates'])) + '\n')
                    print('EVALUATION', json.dumps(result['rates']), flush=True)
                    for key in ('correct_pick', 'success_once'):
                        value = result['rates'][key]
                        if best[key] is None or value > best[key]:
                            best[key] = value
                            tags.append(f'best_eval_{key}')
                    if writer:
                        for key, value in result['rates'].items():
                            writer.add_scalar(f'evaluation/{key}', value, step)
                restore_rng(evaluation_rng)
            if checking or stopping or step % config['save_every'] == 0:
                save('latest')
                for tag in tags:
                    link_saved_checkpoint(tag)
            if step == config['total_iters']:
                link_saved_checkpoint('final')
            if checking or stopping:
                write_json(output / 'status.json', dict(iteration=step, total_iters=config['total_iters'],
                                                       completed=step == config['total_iters']))
            if stopping:
                break
    finally:
        log.close()
        if writer:
            writer.close()
        train.close()
        validation.close()
    print(f'Saved {output / "checkpoints/latest.pt"}; completed={step == config["total_iters"]}', flush=True)


if __name__ == '__main__':
    main()
