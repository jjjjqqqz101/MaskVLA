"""Independent, lazy H5 inputs for centroid (C) and binary-mask (D) ACT.

Only original RGB, visible target masks, the approved robot state, and expert
actions are read. Target positions and object identity are never model inputs.
"""

from bisect import bisect_right
import operator
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Resize


STATE_FIELDS = (("agent", "qpos"), ("agent", "qvel"),
                ("extra", "tcp_pose"), ("extra", "goal_pos"))
_STATE_WIDTHS = (9, 9, 7, 3)
_DATA_ROOT = (Path(__file__).resolve().parents[4] / "demos/TargetPickCube-v1/"
              "motionplanning/target_pick_500_mask/dataset")
DEFAULT_TRAIN = _DATA_ROOT / "train.h5"
DEFAULT_VALIDATION = _DATA_ROOT / "validation.h5"
_RGB_RESIZE = Resize((224, 224), antialias=True)


def _condition(value):
    if value not in ("C", "D"):
        raise ValueError("condition must be C (centroid) or D (binary mask)")
    return value


def _boolean_mask(mask):
    mask = torch.as_tensor(mask)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 3 or min(mask.shape) < 1 or mask.dtype != torch.bool:
        raise ValueError("Expected a boolean target mask [B,H,W] or [B,H,W,1]")
    return mask


def mask_centroid(mask):
    """Return visible-pixel (x,y) centroids in [-1,1] and boolean validity.

    Empty masks return (0,0), valid=False. A one-pixel spatial dimension has
    coordinate zero. No object state or previous observation supplies missing
    coordinates. Coordinates refer to the original mask before image resizing.
    """
    mask = _boolean_mask(mask)
    _, height, width = mask.shape
    mass = mask.float()
    count = mass.sum(dim=(-2, -1))
    valid = count > 0
    divisor = count.clamp_min(1)
    x_axis = torch.arange(width, dtype=torch.float32, device=mask.device)
    y_axis = torch.arange(height, dtype=torch.float32, device=mask.device)
    x = (mass.sum(dim=1) * x_axis).sum(dim=1) / divisor
    y = (mass.sum(dim=2) * y_axis).sum(dim=1) / divisor
    x = 2 * x / (width - 1) - 1 if width > 1 else torch.zeros_like(x)
    y = 2 * y / (height - 1) - 1 if height > 1 else torch.zeros_like(y)
    xy = torch.stack((x, y), dim=-1)
    return torch.where(valid[:, None], xy, 0), valid


def extract_state(raw_obs):
    """Concatenate qpos, qvel, tcp_pose, and goal_pos in the established order."""
    parts = []
    for (group, key), width in zip(STATE_FIELDS, _STATE_WIDTHS):
        value = torch.as_tensor(raw_obs[group][key]).float()
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"Expected {group}/{key} [B,{width}]")
        parts.append(value)
    state = torch.cat(parts, dim=-1)
    if not torch.isfinite(state).all():
        raise ValueError("Robot state contains nonfinite values")
    return state


def encode_observation(rgb, mask, state, condition):
    """Encode raw batched observations without normalizing the 28-D state.

    C returns one RGB stream, target_xy [B,2], and target_valid [B]. D returns
    original RGB plus a white-on-black binary mask repeated over 3 channels.
    RGB uses the existing antialiased resize; masks use nearest interpolation.
    ImageNet normalization remains the responsibility of the ACT agent.
    """
    condition = _condition(condition)
    rgb = torch.as_tensor(rgb)
    mask = _boolean_mask(mask)
    state = torch.as_tensor(state).float()
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != torch.uint8:
        raise ValueError("Expected uint8 original RGB [B,H,W,3]")
    if tuple(rgb.shape[:3]) != tuple(mask.shape):
        raise ValueError("RGB and target mask batch/spatial dimensions must agree")
    if state.shape != (rgb.shape[0], 28) or not torch.isfinite(state).all():
        raise ValueError("Expected finite robot state [B,28]")
    mask = mask.to(rgb.device)
    original = _RGB_RESIZE(rgb.permute(0, 3, 1, 2))
    encoded = {"state": state}
    if condition == "C":
        encoded["rgb"] = original[:, None]
        encoded["target_xy"], encoded["target_valid"] = mask_centroid(mask)
    else:
        binary = F.interpolate(mask[:, None].float(), size=(224, 224), mode="nearest")
        binary = (binary.to(torch.uint8) * 255).expand(-1, 3, -1, -1)
        encoded["rgb"] = torch.stack((original, binary), dim=1)
    return encoded


def _moments_merge(accumulator, values, weights=None):
    values = values.double()
    if not torch.isfinite(values).all():
        raise ValueError("Training states/actions contain nonfinite values")
    if weights is None:
        count = values.shape[0]
        mean = values.mean(dim=0)
        m2 = ((values - mean) ** 2).sum(dim=0)
    else:
        weights = weights.double()[:, None]
        count = int(weights.sum().item())
        mean = (weights * values).sum(dim=0) / count
        m2 = (weights * (values - mean) ** 2).sum(dim=0)
    if accumulator is None:
        return count, mean, m2
    old_count, old_mean, old_m2 = accumulator
    total = old_count + count
    delta = mean - old_mean
    return (total, old_mean + delta * count / total,
            old_m2 + m2 + delta.square() * old_count * count / total)


def _normalize_stats(stats):
    result = {}
    for key, width in (("state_mean", 28), ("state_std", 28),
                       ("action_mean", 8), ("action_std", 8)):
        if key not in stats:
            raise ValueError(f"Missing normalization statistic: {key}")
        value = torch.as_tensor(stats[key]).detach().cpu().float().clone()
        if value.shape not in ((width,), (1, width)) or not torch.isfinite(value).all():
            raise ValueError(f"Expected finite {key} [1,{width}]")
        if key.endswith("std") and not (value > 0).all():
            raise ValueError(f"{key} must be positive")
        result[key] = value.reshape(1, width)
    if "example_state" in stats:
        result["example_state"] = torch.as_tensor(stats["example_state"]).detach().cpu().float().clone()
    return result


class TargetDataset(Dataset):
    """Read individual frames lazily, opening one H5 handle in each process.

    Training statistics reproduce the existing end-padded action-window
    weighting without storing every window. Pass training ``norm_stats`` when
    constructing validation data. Initialization inspects every trajectory's
    alignment; it never loads full RGB/mask trajectories into memory.
    """

    def __init__(self, path, condition, num_queries=30, num_demos=None, norm_stats=None):
        self.path = Path(path).expanduser().resolve()
        self.condition = _condition(condition)
        self.num_queries = operator.index(num_queries)
        if self.num_queries < 1:
            raise ValueError("num_queries must be positive")
        self._handle = None
        self._handle_pid = None
        with h5py.File(self.path, "r") as source:
            names = list(source)
            if not names or any(not name.startswith("traj_") or not name[5:].isdigit() for name in names):
                raise ValueError("Expected nonempty H5 containing traj_N groups")
            names.sort(key=lambda name: int(name[5:]))
            if num_demos is not None:
                num_demos = operator.index(num_demos)
                if not 0 < num_demos <= len(names):
                    raise ValueError("Requested demo count is unavailable")
                names = names[:num_demos]
            self.trajectory_names = names
            self.num_traj = len(names)
            self.lengths = [self._validate_trajectory(source[name], name) for name in names]
            self._ends = np.cumsum(self.lengths).tolist()
            self.norm_stats = (_normalize_stats(norm_stats) if norm_stats is not None
                               else self._compute_norm_stats(source))

    @staticmethod
    def _validate_trajectory(trajectory, name):
        actions = trajectory["actions"]
        if actions.ndim != 2 or actions.shape[1] != 8 or not len(actions) or actions.dtype.kind != "f":
            raise ValueError(f"{name}: expected nonempty floating actions [T,8]")
        frames = len(actions) + 1
        rgb = trajectory["obs/sensor_data/base_camera/rgb"]
        if (rgb.ndim != 4 or rgb.shape[0] != frames or rgb.shape[-1] != 3
                or min(rgb.shape[1:3]) < 1 or rgb.dtype != np.uint8):
            raise ValueError(f"{name}: expected uint8 RGB [T+1,H,W,3]")
        mask = trajectory["derived/base_camera/target_mask"]
        if mask.shape not in (rgb.shape[:3], (*rgb.shape[:3], 1)) or mask.dtype != np.bool_:
            raise ValueError(f"{name}: expected boolean mask aligned with RGB")
        for (group, key), width in zip(STATE_FIELDS, _STATE_WIDTHS):
            field = trajectory[f"obs/{group}/{key}"]
            if field.shape != (frames, width) or field.dtype.kind != "f":
                raise ValueError(f"{name}: expected floating {group}/{key} [T+1,{width}]")
        return len(actions)

    def _compute_norm_stats(self, source):
        state_moments = action_moments = None
        for name, length in zip(self.trajectory_names, self.lengths):
            trajectory = source[name]
            state = torch.cat([torch.from_numpy(trajectory[f"obs/{group}/{key}"][:length]).float()
                               for group, key in STATE_FIELDS], dim=-1)
            actions = torch.from_numpy(trajectory["actions"][:]).float()
            state_moments = _moments_merge(state_moments, state)
            # Each action appears in min(t+1,Q) windows. Tail padding adds
            # further copies of the last action for windows shorter than Q.
            weights = torch.arange(1, length + 1, dtype=torch.int64).clamp_max(self.num_queries)
            padded_windows = min(length, self.num_queries - 1)
            weights[-1] += padded_windows * self.num_queries - padded_windows * (padded_windows + 1) // 2
            action_moments = _moments_merge(action_moments, actions, weights)
        stats = {"example_state": state[-1].clone()}
        for prefix, (count, mean, m2) in (("state", state_moments), ("action", action_moments)):
            if count < 2:
                raise ValueError("At least two training samples are required for sample standard deviation")
            stats[f"{prefix}_mean"] = mean.float()[None]
            stats[f"{prefix}_std"] = (m2 / (count - 1)).clamp_min(0).sqrt().clamp_min(1e-2).float()[None]
        return stats

    def _source(self):
        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid:
            self.close()
            self._handle = h5py.File(self.path, "r")
            self._handle_pid = pid
        return self._handle

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Dataset index out of range")
        trajectory_index = bisect_right(self._ends, index)
        start = 0 if trajectory_index == 0 else self._ends[trajectory_index - 1]
        step = index - start
        trajectory = self._source()[self.trajectory_names[trajectory_index]]
        state = torch.cat([torch.from_numpy(trajectory[f"obs/{group}/{key}"][step:step + 1]).float()
                           for group, key in STATE_FIELDS], dim=-1)
        obs = encode_observation(
            trajectory["obs/sensor_data/base_camera/rgb"][step:step + 1],
            trajectory["derived/base_camera/target_mask"][step:step + 1], state, self.condition)
        obs["state"] = (obs["state"] - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
        actions = torch.from_numpy(trajectory["actions"][step:step + self.num_queries]).float()
        if not torch.isfinite(actions).all():
            raise ValueError("Expert actions contain nonfinite values")
        if len(actions) < self.num_queries:
            actions = torch.cat((actions, actions[-1:].expand(self.num_queries - len(actions), -1)), dim=0)
        actions = (actions - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        return {"observations": {key: value[0] for key, value in obs.items()}, "actions": actions}

    def close(self):
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None
        self._handle_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def __del__(self):
        self.close()
