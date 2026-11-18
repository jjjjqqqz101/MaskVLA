#!/usr/bin/env python3
"""Synthetic conversion, exact legacy statistics, and multiprocessing checks."""
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from triple_rgb_dataset import TripleRGBMemmapDataset, STREAM_ORDER


def fixture(root):
    raw = root / "raw"
    shard = raw / "shards" / "test_target" / "shard_000"
    shard.mkdir(parents=True)
    split = dict(training_objects=["test_target"], heldout_objects=["heldout_target"])
    progress = dict(total=2, target=2, validation_demos=0, per_target={"test_target": 2})
    (raw / "split_manifest.json").write_text(json.dumps(split))
    (raw / "progress.json").write_text(json.dumps(progress))
    accepted, metadata = [], []
    rng = np.random.default_rng(31)
    with h5py.File(shard / "trajectory.h5", "w") as f:
        for i, steps in enumerate((3, 5)):
            g = f.create_group(f"traj_{i}")
            g.attrs.update(target_name="test_target", object_split="train")
            for field, dim in (("agent/qpos", 9), ("agent/qvel", 9), ("extra/tcp_pose", 7), ("extra/goal_pos", 3)):
                values = rng.normal(size=(steps + 1, dim)).astype(np.float64 if field == "agent/qpos" else np.float32)
                values[-1] = 999999  # must never influence training statistics
                g.create_dataset("obs/" + field, data=values)
            g.create_dataset("obs/extra/target_idx", data=np.full((steps + 1, 1), 123456))
            g.create_dataset("actions", data=rng.normal(size=(steps, 8)).astype(np.float64))
            rgb = rng.integers(0, 256, (steps + 1, 224, 224, 3), dtype=np.uint8)
            wrist = 255 - rgb
            mask = rng.integers(0, 2, (steps + 1, 224, 224, 1), dtype=np.uint8).astype(bool)
            g.create_dataset("obs/sensor_data/base_camera/rgb", data=rgb)
            g.create_dataset("obs/sensor_data/hand_camera/rgb", data=wrist)
            g.create_dataset("derived/base_camera/masked_rgb", data=rgb * mask)
            g.create_dataset("derived/base_camera/target_mask", data=mask)
            g.create_dataset("obs/sensor_data/base_camera/segmentation", data=np.where(mask, 17, 0).astype(np.int16))
            g.create_dataset("alignment/target_seg_id", data=np.full((steps + 1, 1), 17, dtype=np.int32))
            accepted.append(dict(trajectory=f"traj_{i}", target_name="test_target", selected_names=["test_target"],
                                 target_idx=0, grasped_and_lifted=True, wrong_grasp_ever=False, initial_visible_pixels=100,
                                 seed=i, frames=steps + 1))
            metadata.append(dict(episode_id=i, elapsed_steps=steps, success=True, episode_seed=i))
    (shard / "accepted.json").write_text(json.dumps(accepted))
    env = dict(env_id="TargetPickForty-v1", env_kwargs=dict(control_mode="pd_joint_pos", robot_uids="panda_wristcam"))
    (shard / "trajectory.json").write_text(json.dumps(dict(episodes=metadata, env_info=env)))
    return raw


def legacy_stats(state, actions, offsets, queries):
    states, chunks = [], []
    for start, end in zip(offsets[:-1], offsets[1:]):
        for ts in range(int(start), int(end)):
            states.append(torch.tensor(state[ts], dtype=torch.float32))
            act = torch.tensor(actions[ts:min(ts + queries, int(end))], dtype=torch.float32)
            if len(act) < queries:
                act = torch.cat([act, act[-1].repeat(queries - len(act), 1)])
            chunks.append(act)
    state, action = torch.stack(states), torch.cat(chunks)
    return dict(state_mean=state.mean(0, keepdim=True), state_std=state.std(0, keepdim=True).clamp_min(1e-2),
                action_mean=action.mean(0, keepdim=True), action_std=action.std(0, keepdim=True).clamp_min(1e-2),
                example_state=states[-1])


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="act-triple-test-") as directory:
        root = Path(directory)
        raw = fixture(root)
        converted = root / "converted"
        command = [sys.executable, str(Path(__file__).with_name("prepare_triple_rgb_dataset.py")),
                   "--raw-root", str(raw), "--output", str(converted), "--workers", "2", "--block-frames", "2",
                   "--expected-targets", "1", "--expected-per-target", "2", "--expected-shards", "1", "--expected-frames", "8"]
        subprocess.run(command, check=True)
        # Every output value must match the original source, without terminal frames or floating-point loss.
        state = np.load(converted / "state.npy", mmap_mode="r")
        actions = np.load(converted / "actions.npy", mmap_mode="r")
        rgb = np.load(converted / "rgb.npy", mmap_mode="r")
        offsets = np.load(converted / "episode_offsets.npy")
        assert state.dtype == actions.dtype == np.float64
        with h5py.File(next(raw.rglob("trajectory.h5")), "r") as source:
            for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
                g, n = source[f"traj_{i}"], int(end - start)
                expected_state = np.concatenate([g[f"obs/{field}"][:n] for field in ("agent/qpos", "agent/qvel", "extra/tcp_pose", "extra/goal_pos")], axis=-1)
                assert np.array_equal(state[start:end], expected_state)
                assert np.array_equal(actions[start:end], g["actions"][:])
                for stream, key in enumerate(("obs/sensor_data/base_camera/rgb", "obs/sensor_data/hand_camera/rgb", "derived/base_camera/masked_rgb")):
                    assert np.array_equal(rgb[start:end, stream], g[key][:n])
        for queries in (1, 4, 30):
            dataset = TripleRGBMemmapDataset(converted, num_queries=queries)
            assert len(dataset) == 8 and dataset.stream_order == STREAM_ORDER
            assert dataset.action_dim == 8 and dataset.state_dim == 28
            stats = legacy_stats(state, actions, offsets, queries)
            for key in stats:
                assert torch.equal(stats[key], dataset.norm_stats[key]), key
            for index in range(len(dataset)):
                item = dataset[index]
                observed = item["observations"]
                assert observed["rgb"].dtype == torch.uint8 and observed["rgb"].shape == (3, 3, 224, 224)
                assert np.array_equal(observed["rgb"].numpy(), rgb[index].transpose(0, 3, 1, 2))
                episode = np.searchsorted(offsets, index, side="right") - 1
                indices = np.minimum(np.arange(index, index + queries), offsets[episode + 1] - 1)
                expected = (torch.tensor(actions[indices], dtype=torch.float32) - stats["action_mean"]) / stats["action_std"]
                assert torch.equal(item["actions"], expected)
            restored = pickle.loads(pickle.dumps(dataset))
            assert restored._arrays is None and torch.equal(restored[-1]["actions"], dataset[-1]["actions"])
            loader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2, multiprocessing_context="spawn",
                                generator=torch.Generator().manual_seed(3), pin_memory=torch.cuda.is_available())
            batches = list(loader)
            assert sum(len(b["actions"]) for b in batches) == 8
            assert all(b["observations"]["rgb"].dtype == torch.uint8 for b in batches)
            if torch.cuda.is_available():
                assert all(b["observations"]["rgb"].is_pinned() for b in batches)
        subset = TripleRGBMemmapDataset(converted, num_queries=30, num_traj=1)
        assert len(subset) == 3
        for key, expected in legacy_stats(state[:3], actions[:3], offsets[:2], 30).items():
            assert torch.equal(subset.norm_stats[key], expected), key
        rejected = subprocess.run(command, capture_output=True)
        assert rejected.returncode != 0 and b"Output already exists" in rejected.stderr
        print(json.dumps(dict(ok=True, exact_legacy_statistics=True, lossless_float_promotion=True,
                              all_source_values_compared=True, boundary_padding=True, spawn_workers=2,
                              pinned_memory=torch.cuda.is_available(), destination_overwrite_rejected=True)))


if __name__ == "__main__":
    main()
