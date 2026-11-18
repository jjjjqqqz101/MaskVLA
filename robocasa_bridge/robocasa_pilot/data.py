"""Read-only raw-HDF5 loader. Metadata/actions are cached, RGB is read on demand.

No transformed data is written. Action horizons never cross target boundaries.
Use num_workers=0 or create the dataset after worker creation; open HDF5 handles
are not shared across processes. Normalization is deliberately left to policies.
"""
from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import numpy as np
import h5py
from .protocol import STATE_DIM, ACTION_DIM, IMAGE_SHAPE, PROMPT, PROJECT, inside_root


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _h5_path(value, base=None):
    path = Path(value)
    if not path.is_absolute():
        path = (base or PROJECT) / path
    path = inside_root(path)
    if path.is_dir():
        path = inside_root(path / "trajectory.h5")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@dataclass
class Episode:
    path: Path
    metadata: dict
    states: np.ndarray
    actions: np.ndarray
    valid_frames: np.ndarray
    segment_stop: np.ndarray
    target_ids: tuple
    summary: dict = field(default_factory=dict)
    _file: object = field(default=None, repr=False)
    _pid: int = field(default=0, repr=False)

    @property
    def episode_id(self):
        return self.metadata.get("episode_id", self.path.parent.name)

    def __len__(self):
        return len(self.valid_frames)

    def _handle(self):
        if self._file is None or self._pid != os.getpid() or not self._file.id.valid:
            self.close()
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._file

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        state["_pid"] = 0
        return state

    def observation(self, frame):
        frame = int(frame)
        if frame < 0 or frame >= len(self.actions):
            raise IndexError("Training observations exclude the terminal frame")
        f = self._handle()
        base = np.asarray(f["observations/base_rgb"][frame], dtype=np.uint8)
        wrist = np.asarray(f["observations/wrist_rgb"][frame], dtype=np.uint8)
        mask = np.asarray(f["observations/gt_target_mask_offline"][frame]) != 0
        target = np.where(mask[..., None], base, 0).astype(np.uint8)
        return dict(base_rgb=base, wrist_rgb=wrist, target_mask=mask,
                    gt_target_mask_offline=mask, target_image=target,
                    robot_state=self.states[frame].copy(), prompt=PROMPT)

    def sample(self, frame, horizon=30):
        frame = int(frame)
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("Action horizon must be positive")
        if frame < 0 or frame >= len(self.actions) or self.segment_stop[frame] <= frame:
            raise IndexError("Not an active-target training start")
        obs = self.observation(frame)
        stop = min(frame + horizon, int(self.segment_stop[frame]))
        indices = np.minimum(frame + np.arange(horizon), stop - 1)
        actions = self.actions[indices].copy()
        is_pad = np.arange(horizon) >= stop - frame
        images = np.stack((obs["base_rgb"], obs["wrist_rgb"], obs["target_image"]))
        return dict(images=images.transpose(0, 3, 1, 2), state=obs["robot_state"],
                    actions=actions, is_pad=is_pad.astype(bool), target_mask=obs["target_mask"],
                    base_rgb=obs["base_rgb"], wrist_rgb=obs["wrist_rgb"], target_image=obs["target_image"],
                    episode_id=self.episode_id, frame_index=frame,
                    target_id=self.target_ids[frame], prompt=PROMPT)


def load_episode(path, *, require_ready=False):
    """Read metadata/state/action once; RGB remains lazily read-only.

    A manifest is an explicit quality gate. Additionally reject episodes whose
    adjacent summary explicitly says training_ready=false. Missing summaries
    can be rejected with require_ready=True.
    """
    path = _h5_path(path)
    summary_path = path.parent / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if summary.get("training_ready") is False or (require_ready and not summary.get("training_ready")):
        raise ValueError("Episode is not quality-approved for training: " + str(path))
    with h5py.File(path, "r") as f:
        metadata = json.loads(_decode(f.attrs["metadata_json"]))
        actions = np.asarray(f["actions/arm_gripper"], dtype=np.float32)
        all_states = np.asarray(f["observations/robot_state"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or len(actions) < 1:
            raise ValueError("Actions must be nonempty [T,7] OSC delta; no second delta transform")
        count = len(actions)
        if all_states.shape != (count + 1, STATE_DIM):
            raise ValueError("Robot states must align as [T+1,23]")
        if not np.isfinite(actions).all() or not np.isfinite(all_states).all():
            raise ValueError("Non-finite states or actions")
        if int(metadata.get("control_freq", 20)) != 20:
            raise ValueError("Pilot control frequency must be 20 Hz")
        for name in ("base_rgb", "wrist_rgb"):
            ds = f["observations/" + name]
            if ds.shape != (count + 1, *IMAGE_SHAPE) or ds.dtype != np.uint8:
                raise ValueError(name + " must be uint8[T+1,224,224,3]")
        if f["observations/gt_target_mask_offline"].shape != (count + 1, *IMAGE_SHAPE[:2]):
            raise ValueError("Mask/RGB alignment mismatch")
        subtasks = np.asarray(f["observations/subtask_index"], dtype=np.int64)
        targets = tuple(_decode(x) for x in f["observations/target_id"][:])
        if subtasks.shape != (count + 1,) or len(targets) != count + 1:
            raise ValueError("Target context must have T+1 observations")
        order = metadata.get("order", [])
        active = np.asarray([0 <= int(k) < len(order) and name == order[int(k)] and bool(name)
                             for k, name in zip(subtasks[:-1], targets[:-1])], dtype=bool)
        boundaries = np.flatnonzero((subtasks[1:count] != subtasks[:count - 1]) |
                                   np.asarray([targets[i + 1] != targets[i] for i in range(count - 1)])) + 1
        segment_stop = np.zeros(count, dtype=np.int64)
        start = 0
        for stop in [*boundaries.tolist(), count]:
            segment_stop[start:stop] = stop
            start = stop
        segment_stop[~active] = np.arange(count)[~active]
        valid = np.flatnonzero(active)
        if len(valid) == 0:
            raise ValueError("No active target training frames")
    return Episode(path, metadata, all_states[:-1], actions, valid, segment_stop, targets, summary)


def _manifest_rows(manifest, split):
    if isinstance(manifest, (str, Path)):
        manifest_path = _h5_path(manifest)
        payload = json.loads(manifest_path.read_text())
        base = manifest_path.parent
    else:
        payload = manifest
        base = PROJECT
    split = "validation" if split == "val" else str(split)
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a JSON object")
    if "splits" in payload:
        groups = payload["splits"]
        rows = groups.get(split, groups.get("val", []) if split == "validation" else [])
        seen = {}
        for group, values in groups.items():
            group = "validation" if group == "val" else group
            for row in values:
                raw = row.get("path", row.get("h5_path", row.get("trajectory"))) if isinstance(row, dict) else row
                key = str(_h5_path(raw, base))
                if key in seen and seen[key] != group:
                    raise ValueError("Manifest leaks an episode across splits: " + key)
                seen[key] = group
    else:
        rows = [row for row in payload.get("episodes", [])
                if ("validation" if row.get("split") == "val" else row.get("split")) == split]
    paths = []
    for row in rows:
        raw = row.get("path", row.get("h5_path", row.get("trajectory"))) if isinstance(row, dict) else row
        path = _h5_path(raw, base)
        if path in paths:
            raise ValueError("Duplicate episode in split: " + str(path))
        paths.append(path)
    return paths, split


class TrajectoryDataset:
    """A frame index plus cached state/action arrays; no repeated full scans."""
    def __init__(self, manifest, split="train", horizon=30, *, require_ready=False):
        paths, self.split = _manifest_rows(manifest, split)
        self.horizon = int(horizon)
        self.episodes = [load_episode(path, require_ready=require_ready) for path in paths]
        self.index = [(ei, int(frame)) for ei, ep in enumerate(self.episodes) for frame in ep.valid_frames]
        self._stats = None
        if not self.index:
            raise ValueError("Empty training split: " + self.split)

    def __len__(self):
        return len(self.index)

    def sample(self, index, horizon=None):
        ei, frame = self.index[int(index)]
        return self.episodes[ei].sample(frame, self.horizon if horizon is None else horizon)

    def __getitem__(self, index):
        return self.sample(index)

    def fit_stats(self):
        if self.split != "train":
            raise ValueError("Normalization must be fitted on TRAIN only")
        if self._stats is None:
            states = np.concatenate([ep.states[ep.valid_frames] for ep in self.episodes])
            actions = np.concatenate([ep.actions[ep.valid_frames] for ep in self.episodes])
            self._stats = dict(state_mean=states.astype(np.float64).mean(0).astype(np.float32),
                               state_std=np.maximum(states.astype(np.float64).std(0), 1e-6).astype(np.float32),
                               action_mean=actions.astype(np.float64).mean(0).astype(np.float32),
                               action_std=np.maximum(actions.astype(np.float64).std(0), 1e-6).astype(np.float32))
        return {k: v.copy() for k, v in self._stats.items()}

    def close(self):
        for ep in self.episodes:
            ep.close()


def build_index(manifest, split="train", horizon=30, **kwargs):
    return TrajectoryDataset(manifest, split, horizon, **kwargs)
