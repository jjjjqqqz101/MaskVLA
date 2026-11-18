"""Real-H5 full-window loader for pi0.5. No temporal padding enters flow loss."""
import dataclasses
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

from robocasa_bridge.paths import ROOT
ASSET_ID = "robocasa_pilot_v1"

def inside(path):
    path = Path(path).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Path leaves MASK_POLICY_ROOT: {path}")
    return path

def read_manifest(path):
    path = inside(path)
    manifest = json.loads(path.read_text())
    splits = {}
    for name in ("train", "validation", "test"):
        rows = manifest.get("splits", {}).get(name, [])
        if isinstance(rows, dict):
            rows = rows.get("paths", rows.get("episodes", []))
        if not rows:
            rows = [row for row in manifest.get("episodes", []) if row.get("split") == name]
        paths = []
        for row in rows:
            value = row if isinstance(row, str) else row.get("path", row.get("trajectory_path", row.get("h5_path")))
            if value is None:
                raise ValueError("Manifest episode needs path")
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            candidate = inside(candidate)
            if candidate.is_dir():
                candidate = candidate / "trajectory.h5"
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            paths.append(candidate)
        if len(set(paths)) != len(paths):
            raise ValueError(f"Duplicate episodes in split {name}")
        splits[name] = paths
    if not splits["train"]:
        raise ValueError("Manifest has no training episodes")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(splits[a]) & set(splits[b]):
            raise ValueError(f"Episode leakage between {a} and {b}")
    return manifest, splits, hashlib.sha256(path.read_bytes()).hexdigest()

class RawH5Dataset:
    def __init__(self, paths, horizon=10):
        self.paths = [inside(p) for p in paths]
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        self.index = []
        self.files = {}
        self.episode_reports = []
        for p in self.paths:
            summary = p.parent / "summary.json"
            if summary.exists() and json.loads(summary.read_text()).get("training_ready") is False:
                raise ValueError(f"Episode explicitly not training_ready: {p}")
            with h5py.File(p, "r") as f:
                t = len(f["actions/arm_gripper"])
                if f["actions/arm_gripper"].shape != (t, 7):
                    raise ValueError(f"Wrong 7D OSC action shape: {p}")
                for key in ("base_rgb", "wrist_rgb", "gt_target_mask_offline", "robot_state", "target_id", "subtask_index"):
                    if len(f["observations/" + key]) != t + 1:
                        raise ValueError(f"T+1 alignment failure {key}: {p}")
                if f["observations/robot_state"].shape != (t + 1, 23):
                    raise ValueError(f"Wrong robot_state shape: {p}")
                for key in ("base_rgb", "wrist_rgb"):
                    if f["observations/" + key].shape[1:] != (224, 224, 3):
                        raise ValueError(f"Wrong RGB shape: {p}")
                if f["observations/gt_target_mask_offline"].shape[1:] != (224, 224):
                    raise ValueError(f"Wrong mask shape: {p}")
                targets = f["observations/target_id"].asstr()[:t]
                contexts = f["observations/subtask_index"][:t]
                valid = []
                for i in range(max(0, t - self.horizon + 1)):
                    if targets[i] and np.all(targets[i:i+self.horizon] == targets[i]) and np.all(contexts[i:i+self.horizon] == contexts[i]):
                        valid.append(i)
                self.index.extend((p, i) for i in valid)
                self.episode_reports.append({"path": str(p), "actions": t,
                    "valid_full_windows": len(valid), "temporal_padding": False,
                    "excluded_tail_starts": min(self.horizon - 1, t)})
        if not self.index:
            raise ValueError("No complete, context-stable action windows in this split")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        p, frame = self.index[int(index)]
        if p not in self.files:
            self.files[p] = h5py.File(p, "r")
        f = self.files[p]
        base = f["observations/base_rgb"][frame]
        wrist = f["observations/wrist_rgb"][frame]
        mask = f["observations/gt_target_mask_offline"][frame].astype(bool)
        state = f["observations/robot_state"][frame].astype(np.float32)
        actions = f["actions/arm_gripper"][frame:frame+self.horizon].astype(np.float32)
        if len(actions) != self.horizon or not np.isfinite(actions).all() or not np.isfinite(state).all():
            raise ValueError("Invalid real full window")
        if np.max(np.abs(actions[:, :6])) > 1.0001 or not np.all(np.isin(actions[:, 6], [-1., 1.])):
            raise ValueError("Expected normalized native OSC delta controls and bipolar gripper")
        return {"observation/image": base, "observation/wrist_image": wrist,
                "observation/target_image": np.where(mask[..., None], base, 0).astype(np.uint8),
                "observation/state": state, "actions": actions}

    def close(self):
        for file in self.files.values():
            file.close()
        self.files.clear()

def fit_norm_stats(paths):
    from openpi.shared.normalize import NormStats
    states, actions = [], []
    for p in paths:
        with h5py.File(inside(p), "r") as f:
            t = len(f["actions/arm_gripper"])
            states.append(f["observations/robot_state"][:t].astype(np.float64))
            actions.append(f["actions/arm_gripper"][:].astype(np.float64))
    result = {}
    for key, chunks in (("state", states), ("actions", actions)):
        x = np.concatenate(chunks)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite training normalization data")
        result[key] = NormStats(mean=x.mean(0), std=np.maximum(x.std(0), 1e-6),
                               q01=np.quantile(x, .01, axis=0), q99=np.quantile(x, .99, axis=0))
    return result

class RawBatchLoader:
    def __init__(self, dataset, config, norm_stats, batch_size, seed=17, sharding=None):
        from openpi import transforms
        self.dataset = dataset
        self.config = dataclasses.replace(config, norm_stats=norm_stats, asset_id=ASSET_ID)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.sharding = sharding
        self.transform = transforms.compose([*self.config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=self.config.use_quantile_norm),
            *self.config.model_transforms.inputs])

    def data_config(self):
        return self.config

    def __iter__(self):
        return self

    def __next__(self):
        import jax
        from openpi.models.model import Observation
        rows = [self.transform(self.dataset[int(i)]) for i in self.rng.integers(len(self.dataset), size=self.batch_size)]
        batch = jax.tree.map(lambda *items: np.stack(items), *rows)
        observation = Observation.from_dict(batch)
        pair = (observation, batch["actions"])
        return jax.device_put(pair, self.sharding) if self.sharding is not None else pair
