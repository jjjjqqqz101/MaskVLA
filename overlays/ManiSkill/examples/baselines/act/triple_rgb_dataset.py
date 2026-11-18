"""ACT dataset backed by lossless, action-aligned uint8 NumPy memmaps.

The RGB tensor is uint8 [stream, channel, height, width]. Divide by 255 and
apply ImageNet normalization in the model, exactly as the original ACT agent.
Only the approved 28 state values are present; provenance never becomes state.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SCHEMA = "act_triple_rgb_memmap_v1"
STREAM_ORDER = ("base_rgb", "wrist_rgb", "base_masked_rgb")
STATE_FIELDS = (("agent", "qpos"), ("agent", "qvel"),
                ("extra", "tcp_pose"), ("extra", "goal_pos"))
STATE_DIMS = (9, 9, 7, 3)


class TripleRGBMemmapDataset(Dataset):
    """Random-access CPU samples; safe with fork/spawn and pinned DataLoaders.

    Each process opens its own read-only mappings lazily. Returned arrays own
    their storage, so DataLoader never pins or mutates read-only mmap pages.
    Statistics duplicate the original dual-RGB loader's tensor ordering and
    reduction operations, including repeated final actions in every chunk.
    """

    def __init__(self, data_path, num_queries=30, num_traj=None,
                 include_depth=False, control_mode="pd_joint_pos", rgb_float=False):
        super().__init__()
        if include_depth:
            raise ValueError("This dataset has three RGB streams and no depth")
        if control_mode != "pd_joint_pos":
            raise ValueError("This dataset implements pd_joint_pos repeat-last padding only")
        if not isinstance(num_queries, int) or isinstance(num_queries, bool) or num_queries < 1:
            raise ValueError("num_queries must be a positive integer")
        self.data_path = str(Path(data_path).resolve())
        root = Path(self.data_path)
        self.manifest = json.loads((root / "manifest.json").read_text())
        m = self.manifest
        if m.get("schema") != SCHEMA or m.get("complete") is not True:
            raise ValueError("Dataset is incomplete or uses an unsupported schema")
        if tuple(m["stream_order"]) != STREAM_ORDER:
            raise ValueError("Unexpected RGB stream order")
        if m["state_fields"] != ["/".join(x) for x in STATE_FIELDS]:
            raise ValueError("Policy state whitelist differs from the approved 28 values")
        if m["control_mode"] != control_mode:
            raise ValueError("Dataset control mode differs from requested control mode")
        self.dataset_fingerprint = m["dataset_fingerprint"]
        self.stream_order = STREAM_ORDER
        self.state_dim, self.action_dim = 28, 8
        self.num_queries = num_queries
        self.control_mode = control_mode
        self.include_depth = False
        self.delta_control = False
        self.rgb_float = bool(rgb_float)
        self.obs_keys = ["state", "rgb"]
        self._arrays = None
        self._pid = None
        offsets = np.load(root / "episode_offsets.npy", allow_pickle=False)
        if offsets.dtype != np.int64 or offsets.shape != (m["num_episodes"] + 1,):
            raise ValueError("Invalid episode offsets")
        if offsets[0] != 0 or offsets[-1] != m["num_frames"] or np.any(np.diff(offsets) <= 0):
            raise ValueError("Offsets must cover every nonempty episode exactly once")
        self.num_traj = m["num_episodes"] if num_traj is None else num_traj
        if not isinstance(self.num_traj, int) or isinstance(self.num_traj, bool) or not 0 < self.num_traj <= m["num_episodes"]:
            raise ValueError("Requested demo count is unavailable")
        self.episode_offsets = offsets[:self.num_traj + 1].copy()
        self.num_frames = int(self.episode_offsets[-1])
        # Header inspection does not page any RGB data into RAM.
        expected = {"rgb": (m["num_frames"], 3, 224, 224, 3),
                    "state": (m["num_frames"], 28),
                    "actions": (m["num_frames"], 8)}
        for key, shape in expected.items():
            array = np.load(root / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype.str != m["arrays"][key]["dtype"]:
                raise ValueError(f"Invalid {key} shape/dtype")
            if key == "rgb" and array.dtype != np.uint8:
                raise ValueError("RGB must remain lossless uint8")
            del array
        self.norm_stats = self.get_norm_stats()
        digest = hashlib.sha256()
        digest.update(f"{self.dataset_fingerprint}:{self.num_traj}:{num_queries}".encode())
        for key in sorted(self.norm_stats):
            digest.update(key.encode())
            digest.update(self.norm_stats[key].numpy().tobytes())
        self.norm_stats_fingerprint = digest.hexdigest()
        # Keep no parent-process RGB mapping open before DataLoader forks.
        self.close()

    def _open(self):
        if self._arrays is None or self._pid != os.getpid():
            self.close()
            root = Path(self.data_path)
            self._arrays = {key: np.load(root / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                            for key in ("rgb", "state", "actions")}
            self._pid = os.getpid()
        return self._arrays

    def close(self):
        self._arrays = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"], state["_pid"] = None, None
        return state

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        arrays = self._open()
        episode = int(np.searchsorted(self.episode_offsets, index, side="right") - 1)
        end = min(index + self.num_queries, int(self.episode_offsets[episode + 1]))
        state = torch.from_numpy(np.array(arrays["state"][index], dtype=np.float32, copy=True))
        actions = torch.empty((self.num_queries, self.action_dim), dtype=torch.float32)
        n = end - index
        actions[:n] = torch.from_numpy(np.array(arrays["actions"][index:end], dtype=np.float32, copy=True))
        if n < self.num_queries:
            actions[n:] = actions[n - 1]
        state = (state - self.norm_stats["state_mean"][0]) / self.norm_stats["state_std"][0]
        actions = (actions - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        rgb = torch.from_numpy(arrays["rgb"][index].transpose(0, 3, 1, 2).copy())
        if self.rgb_float:
            rgb = rgb.float() / 255.0
        return {"observations": {"state": state, "rgb": rgb}, "actions": actions}

    def get_norm_stats(self):
        """Exact old loader calculation: start states and all padded chunks.

        The full image corpus is never loaded. At this dataset's size, the
        temporary contiguous action tensor is 126.7 MB at chunk length 30.
        Terminal observation states are deliberately excluded, as in the old
        loader's `self.slices = [(episode, t) for t in range(len(actions))]`.
        """
        root = Path(self.data_path)
        states = np.load(root / "state.npy", mmap_mode="r", allow_pickle=False)
        actions = np.load(root / "actions.npy", mmap_mode="r", allow_pickle=False)
        all_states = torch.from_numpy(np.array(states[:self.num_frames], dtype=np.float32, copy=True))
        all_actions = torch.empty((self.num_frames * self.num_queries, self.action_dim), dtype=torch.float32)
        query_offsets = torch.arange(self.num_queries)
        for start, end in zip(self.episode_offsets[:-1], self.episode_offsets[1:]):
            start, end = int(start), int(end)
            episode_actions = torch.from_numpy(np.array(actions[start:end], dtype=np.float32, copy=True))
            indices = (torch.arange(end - start)[:, None] + query_offsets[None, :]).clamp_max(end - start - 1)
            all_actions[start * self.num_queries:end * self.num_queries] = episode_actions[indices].reshape(-1, self.action_dim)
        stats = {
            "action_mean": all_actions.mean(dim=0, keepdim=True),
            "action_std": all_actions.std(dim=0, keepdim=True).clamp_min(1e-2),
            "state_mean": all_states.mean(dim=0, keepdim=True),
            "state_std": all_states.std(dim=0, keepdim=True).clamp_min(1e-2),
            "example_state": all_states[-1].clone(),
        }
        return stats
