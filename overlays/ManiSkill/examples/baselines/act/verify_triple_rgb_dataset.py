#!/usr/bin/env python3
"""Verify published memmaps, raw numeric fidelity, and bounded DataLoader use."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from prepare_triple_rgb_dataset import canonical, inventory, STATE_PATHS
from triple_rgb_dataset import TripleRGBMemmapDataset


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--hash-all-rgb", action="store_true")
    args = ap.parse_args()
    started = time.time()
    torch.set_num_threads(2)
    root = args.data_path
    m = json.loads((root / "manifest.json").read_text())
    episodes = json.loads((root / "episodes.json").read_text())
    source_inventory = json.loads((root / "source_inventory.json").read_text())
    raw = Path(m["source_root"])
    require(inventory(raw) == source_inventory, "Raw source inventory changed")
    expected_fingerprint = hashlib.sha256(canonical(dict(schema=m["schema"], source_inventory=source_inventory,
                                                        episodes=episodes, stream_order=tuple(m["stream_order"]),
                                                        state_paths=STATE_PATHS))).hexdigest()
    require(m["dataset_fingerprint"] == expected_fingerprint, "Manifest fingerprint differs")
    arrays = {key: np.load(root / f"{key}.npy", mmap_mode="r", allow_pickle=False)
              for key in ("state", "actions", "rgb")}
    latest_h5, handle = None, None
    hashed_rgb_frames, rgb_source_samples = 0, 0
    try:
        for rec in episodes:
            if latest_h5 != rec["source_h5"]:
                if handle is not None:
                    handle.close()
                latest_h5 = rec["source_h5"]
                handle = h5py.File(raw / latest_h5, "r")
            g = handle[rec["trajectory"]]
            start, stop, n = rec["start"], rec["stop"], rec["steps"]
            expected_state = np.concatenate([g[key][:n] for key in STATE_PATHS], axis=-1)
            require(np.array_equal(arrays["state"][start:stop], expected_state), "Raw policy state differs")
            require(np.array_equal(arrays["actions"][start:stop], g["actions"][:]), "Raw actions differ")
            for key in ("state", "actions"):
                digest = hashlib.sha256(arrays[key][start:stop].tobytes()).hexdigest()
                require(digest == rec["array_sha256"][key], f"Published {key} digest differs")
            for stream, stream_name in enumerate(m["stream_order"]):
                if args.hash_all_rgb:
                    digest = hashlib.sha256(arrays["rgb"][start:stop, stream].tobytes()).hexdigest()
                    require(digest == rec["array_sha256"][stream_name], "Published RGB digest differs")
                    hashed_rgb_frames += n
                for frame in sorted({0, n // 2, n - 1}):
                    require(np.array_equal(arrays["rgb"][start + frame, stream], g[m["stream_source_paths"][stream]][frame]),
                            "Published RGB source sample differs")
                    rgb_source_samples += 1
    finally:
        if handle is not None:
            handle.close()
    # Keep test samples small: this tests the actual multiprocessing/pinned path
    # without warming the entire 59.6 GB image corpus into process memory.
    dataset = TripleRGBMemmapDataset(root, num_queries=30)
    indices = np.unique(np.concatenate((np.linspace(0, len(dataset) - 1, 32, dtype=int),
                                        dataset.episode_offsets[1:][::40] - 1))).tolist()
    pin_memory = torch.cuda.is_available()
    loader = DataLoader(Subset(dataset, indices), batch_size=8, num_workers=2,
                        multiprocessing_context="spawn", pin_memory=pin_memory, prefetch_factor=2)
    seen = 0
    for batch in loader:
        rgb, state, actions = batch["observations"]["rgb"], batch["observations"]["state"], batch["actions"]
        require(rgb.dtype == torch.uint8 and rgb.shape[1:] == (3, 3, 224, 224), "Wrong training RGB layout/dtype")
        require(state.shape[1:] == (28,) and actions.shape[1:] == (30, 8), "Wrong training state/action shape")
        require(torch.isfinite(state).all() and torch.isfinite(actions).all(), "Nonfinite normalized batch")
        if pin_memory:
            require(rgb.is_pinned() and state.is_pinned() and actions.is_pinned(), "DataLoader output not pinned")
        seen += len(rgb)
    require(seen == len(indices), "DataLoader sample coverage differs")
    # Every episode end must repeat precisely its last raw action for all Q.
    for end in dataset.episode_offsets[1:]:
        index = int(end) - 1
        chunk = dataset[index]["actions"]
        expected = (torch.tensor(arrays["actions"][index], dtype=torch.float32) - dataset.norm_stats["action_mean"][0]) / dataset.norm_stats["action_std"][0]
        require(torch.equal(chunk, expected.repeat(30, 1)), "Final-action chunk padding differs")
    require(inventory(raw) == source_inventory, "Raw source inventory changed during verification")
    stats = {key: value.tolist() for key, value in dataset.norm_stats.items()}
    report = dict(ok=True, data_path=str(root), dataset_fingerprint=dataset.dataset_fingerprint,
                  norm_stats_fingerprint=dataset.norm_stats_fingerprint, num_episodes=dataset.num_traj,
                  num_frames=len(dataset), rgb_bytes=int(arrays["rgb"].nbytes),
                  rgb_dtype=str(arrays["rgb"].dtype), state_dtype=str(arrays["state"].dtype),
                  action_dtype=str(arrays["actions"].dtype), stream_order=list(dataset.stream_order),
                  state_fields=m["state_fields"], source_inventory_unchanged=True,
                  raw_state_actions_compared_all_frames=True, published_numeric_sha256_all_episodes=True,
                  published_rgb_sha256_all_frames=args.hash_all_rgb,
                  rgb_stream_frames_hashed=hashed_rgb_frames, rgb_source_frame_stream_samples=rgb_source_samples,
                  repeat_last_padding_all_episode_ends=True, spawn_workers=2, pinned_memory=pin_memory,
                  loader_sample_count=seen, norm_stats=stats, elapsed_seconds=time.time() - started)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "norm_stats"}), flush=True)


if __name__ == "__main__":
    main()
