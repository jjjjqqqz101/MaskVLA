"""Merge, audit, and split 500 successful fixed-camera target-picking demos."""
import copy
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from scripts.data_generation.replay_target_pick_cube_masks import check_projected_support, panel

ROOT = Path('demos/TargetPickCube-v1/motionplanning/target_pick_500_mask')
OUT = ROOT / 'dataset'


def require(test, message):
    if not test:
        raise RuntimeError(message)


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def validate(group, episode, frozen):
    count = len(group['actions'])
    target = episode['target_idx']
    require(count > 0 and len(group['env_states/target_idx']) == count + 1, 'Time alignment')
    require(np.all(group['env_states/target_idx'][:] == target), 'Target changed')
    require(int(group.attrs['target_idx']) == target, 'Target attribute')
    require(episode['target_cube'] == f'cube_{target}', 'Target actor')
    require(np.isfinite(group['actions'][:]).all(), 'Nonfinite actions')
    require(bool(group['success'][-1]) and bool(group['audit/full_success'][-1]), 'Final failure')
    require(not bool(group['truncated'][-1]), 'Truncated demonstration')
    held = group['audit/grasped_cubes'][:]
    grasp = group['audit/is_target_grasped'][:]
    lift = group['audit/target_lifted'][:]
    require(np.array_equal(held[:, target], grasp), 'Expert target contact mismatch')
    require(grasp[-1] and np.any(grasp & lift), 'Target was not grasped/lifted')
    require(not held[:, np.arange(3) != target].any(), 'Wrong object contact')
    require(not group['audit/wrong_object_grasped'][:].any(), 'Wrong object grasp')
    require(set(group['obs/extra']) == {'tcp_pose', 'goal_pos'}, 'Unexpected extra state field')
    require(set(group['obs/agent']) == {'qpos', 'qvel'}, 'Unexpected robot state field')
    require(group['obs/agent/qpos'].shape == (count + 1, 9), 'Panda qpos shape')
    for key in ['qpos', 'qvel']:
        require(np.isfinite(group[f'obs/agent/{key}'][:]).all(), 'Nonfinite robot state')
    ids = group['alignment/cube_seg_ids'][:]
    require(ids.shape == (count + 1, 3) and np.all(ids > 0), 'Invalid cube IDs')
    require(np.all(ids[:, 0] != ids[:, 1]) and np.all(ids[:, 0] != ids[:, 2]) and
            np.all(ids[:, 1] != ids[:, 2]), 'Nonunique cube IDs')
    expected = ids[:, target]
    require(np.array_equal(expected, group['alignment/target_seg_id'][:, 0]), 'Wrong target segmentation ID')
    rgb = group['obs/sensor_data/base_camera/rgb'][:]
    seg = group['obs/sensor_data/base_camera/segmentation'][:]
    mask = group['derived/base_camera/target_mask'][:]
    masked = group['derived/base_camera/masked_rgb'][:]
    require(rgb.shape == (count + 1, 224, 224, 3), 'Wrong RGB dimensions')
    require(seg.shape == mask.shape == (count + 1, 224, 224, 1), 'Wrong mask dimensions')
    require(rgb.dtype == masked.dtype == np.uint8, 'Wrong RGB dtype')
    require(np.array_equal(mask, seg == expected[:, None, None, None]), 'Mask identity mismatch')
    require(np.array_equal(masked, np.where(mask, rgb, 0)), 'Masked RGB mismatch')
    areas = mask.sum(axis=(1, 2, 3))
    require(np.all(areas > 0), 'Empty target mask')
    require(np.array_equal(areas, group['alignment/visible_pixels'][:]), 'Mask area mismatch')
    for key, value in frozen['camera'].items():
        actual = group[f'obs/sensor_param/base_camera/{key}'][:]
        require(np.array_equal(actual, np.broadcast_to(np.array(value, dtype=actual.dtype), actual.shape)), 'Camera changed')
    poses = group[f'env_states/actors/cube_{target}'][:, :7]
    for t in range(count + 1):
        check_projected_support(mask[t, ..., 0], poses[t], frozen['cube_half_size'],
                                np.array(frozen['camera']['extrinsic_cv']),
                                np.array(frozen['camera']['intrinsic_cv']))
    centers = np.array([group[f'env_states/actors/cube_{i}'][0, :3] for i in range(3)])
    distances = [np.linalg.norm(centers[i, :2] - centers[j, :2]) for i in range(3) for j in range(i)]
    require(min(distances) >= frozen['min_cube_center_distance'] - 1e-6, 'Overlapping initial cubes')
    require(set(group['env_states/actors']) == {'cube_0', 'cube_1', 'cube_2', 'goal_site', 'table-workspace'}, 'Unexpected scene actors')
    return dict(action_steps=count, frames=count + 1, min_mask_pixels=int(areas.min()),
                max_mask_pixels=int(areas.max()), initial_mask_pixels=int(areas[0]),
                target_idx=target, target_seg_id=int(expected[0]), target_label=f'cube{target + 1}',
                seed=episode['episode_seed'], all_checks_passed=True)


def subset(full, selected, metadata, name):
    info = copy.deepcopy(metadata)
    info['episodes'] = []
    with h5py.File(OUT / f'{name}.h5', 'w') as dest:
        for key, value in full.attrs.items():
            dest.attrs[key] = value
        for local, source_id in enumerate(selected):
            full.copy(full[f'traj_{source_id}'], dest, name=f'traj_{local}')
            episode = copy.deepcopy(metadata['episodes'][source_id])
            episode.update(episode_id=local, source_episode_id=source_id)
            info['episodes'].append(episode)
    info['split'] = dict(name=name, episode_ids_in_full_dataset=selected)
    (OUT / f'{name}.json').write_text(json.dumps(info, indent=2))


def main():
    OUT.mkdir(exist_ok=False)
    reports, episodes, attempts, previews, provenance = [], [], [], [], []
    seeds = set()
    metadata = None
    full_path = OUT / 'trajectory.rgb+segmentation.h5'
    with h5py.File(full_path, 'w') as full:
        for shard_idx in range(4):
            shard = ROOT / f'shard_{shard_idx}'
            scene = json.loads((shard / 'frozen_scene.json').read_text())
            scene.pop('start_seed')
            scene.pop('requested_demos')
            if shard_idx == 0:
                frozen = scene
                (OUT / 'frozen_scene.json').write_text(json.dumps(frozen, indent=2))
            require(scene == frozen, 'Shard scene/camera mismatch')
            info = json.loads((shard / 'trajectory.json').read_text())
            require(len(info['episodes']) == 125, 'Incomplete shard')
            if metadata is None:
                metadata = copy.deepcopy(info)
                metadata['episodes'] = []
            require(info['env_info'] == metadata['env_info'], 'Environment mismatch')
            attempts.extend([dict(shard=shard_idx, **x) for x in json.loads((shard / 'attempts.json').read_text())])
            provenance.append(dict(shard=shard_idx, path=str(shard), sha256=digest(shard / 'trajectory.h5')))
            with h5py.File(shard / 'trajectory.h5', 'r') as source:
                require(len(source) == 125, 'Wrong shard trajectory count')
                for local, episode in enumerate(info['episodes']):
                    global_id = len(episodes)
                    require(episode['episode_seed'] not in seeds, 'Duplicate seed')
                    seeds.add(episode['episode_seed'])
                    group = source[f"traj_{episode['episode_id']}"]
                    report = validate(group, episode, frozen)
                    report.update(episode_id=global_id, shard=shard_idx, source_episode_id=episode['episode_id'])
                    reports.append(report)
                    source.copy(group, full, name=f'traj_{global_id}')
                    saved = copy.deepcopy(episode)
                    saved.update(episode_id=global_id, source_shard=shard_idx, source_episode_id=episode['episode_id'])
                    metadata['episodes'].append(saved)
                    episodes.append(saved)
                    if local < 3:
                        camera = group['obs/sensor_data/base_camera']
                        raw = camera['rgb'][0]
                        mask = group['derived/base_camera/target_mask'][0, ..., 0]
                        masked = group['derived/base_camera/masked_rgb'][0]
                        previews.append(panel(raw, mask, masked, f"Ep {global_id + 1} | cube{report['target_idx'] + 1} | seed {report['seed']}"))
                    if (global_id + 1) % 25 == 0:
                        print(f'VALIDATED {global_id + 1}/500', flush=True)
        require(len(episodes) == 500, 'Expected exactly 500 successful demos')
        full.attrs['dataset_description'] = '500 successful target-specific MPLib demonstrations; fixed 224px camera; dual RGB'
        full.attrs['stream_order'] = 'original_rgb,masked_rgb'
    metadata['source_desc'] = 'MPLib target-specific expert; rejects wrong grasp or empty visible target mask; fixed camera'
    (OUT / 'trajectory.rgb+segmentation.json').write_text(json.dumps(metadata, indent=2))
    rng = np.random.default_rng(20260923)
    target_counts = np.bincount([x['target_idx'] for x in reports], minlength=3)
    quotas = np.floor(target_counts * .1).astype(int)
    for index in np.argsort(-(target_counts * .1 - quotas))[:50 - quotas.sum()]:
        quotas[index] += 1
    validation_ids = []
    for target in range(3):
        candidates = [x['episode_id'] for x in reports if x['target_idx'] == target]
        validation_ids.extend(rng.permutation(candidates)[:quotas[target]].tolist())
    validation_ids.sort()
    training_ids = [i for i in range(500) if i not in set(validation_ids)]
    require(len(validation_ids) == 50 and len(training_ids) == 450, 'Wrong split sizes')
    with h5py.File(full_path, 'r') as full:
        subset(full, training_ids, metadata, 'train')
        subset(full, validation_ids, metadata, 'validation')
    split = dict(seed=20260923, method='episode-level stratified by target_idx',
                 train_episode_ids=training_ids, validation_episode_ids=validation_ids,
                 train_target_counts=np.bincount([reports[i]['target_idx'] for i in training_ids], minlength=3).tolist(),
                 validation_target_counts=quotas.tolist())
    (OUT / 'split.json').write_text(json.dumps(split, indent=2))
    summary = dict(accepted_episodes=500, attempts=len(attempts), rejected_attempts=sum(not x['saved'] for x in attempts),
                   successful_episodes=500, wrong_grasp_steps=0, empty_mask_frames=0,
                   frames=sum(x['frames'] for x in reports), action_steps=sum(x['action_steps'] for x in reports),
                   target_counts=target_counts.tolist(), unique_seeds=len(seeds),
                   min_mask_pixels=min(x['min_mask_pixels'] for x in reports), max_mask_pixels=max(x['max_mask_pixels'] for x in reports),
                   camera_fixed_all_frames=True, mask_formula_checked_all_frames=True,
                   projected_target_support_checked_all_frames=True, state_target_leakage=False,
                   train_episodes=450, validation_episodes=50, all_checks_passed=True)
    (OUT / 'validation_report.json').write_text(json.dumps(dict(summary=summary, episodes=reports), indent=2))
    (OUT / 'attempts.json').write_text(json.dumps(attempts, indent=2))
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    with open(OUT / 'episodes.csv', 'w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    width, height = previews[0].size
    overview = Image.new('RGB', (2 * width, 6 * height), (22, 27, 36))
    for i, image in enumerate(previews):
        overview.paste(image, ((i % 2) * width, (i // 2) * height))
    overview.save(OUT / 'overview.png')
    preview_dir = OUT / 'previews'
    preview_dir.mkdir()
    for shard in range(4):
        choices = sorted((ROOT / f'shard_{shard}' / 'previews').iterdir())[:3]
        for choice in choices:
            shutil.copytree(choice, preview_dir / f'shard{shard}_{choice.name}')
    code_dir = OUT / 'source_snapshot'
    code_dir.mkdir()
    files = [
        'mani_skill/envs/tasks/tabletop/target_pick_cube.py',
        'mani_skill/envs/tasks/tabletop/pick_cube_cfgs.py',
        'mani_skill/examples/motionplanning/panda/solutions/target_pick_cube.py',
        'mani_skill/examples/motionplanning/panda/collect_target_pick_cube_masks.py',
        'examples/baselines/act/train_dual_rgb.py',
        'scripts/data_generation/finalize_target_pick_500.py',
    ]
    hashes = {}
    for index, filename in enumerate(files):
        path = Path(filename)
        shutil.copy2(path, code_dir / f'{index}_{path.name}')
        hashes[filename] = digest(path)
    (code_dir / 'sha256.json').write_text(json.dumps(hashes, indent=2))
    hashes = {name: digest(OUT / name) for name in ['trajectory.rgb+segmentation.h5', 'train.h5', 'validation.h5']}
    (OUT / 'sha256.json').write_text(json.dumps(hashes, indent=2))
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
