"""Compute exact state and 10-step delta-action stats without decoding 18 GB of images."""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet

from openpi.shared import normalize
from openpi.training.config import get_config


MANISKILL_ROOT = Path(os.environ.get("MANISKILL_ROOT", Path(__file__).resolve().parents[2] / "ManiSkill")).expanduser()
DATASET = Path(os.environ.get("TARGET_PICK_FORTY_LEROBOT", MANISKILL_ROOT / "lerobot/maniskill/target_pick_forty_pi05_25x50")).expanduser()


def main():
    config = get_config("pi05_maniskill_25x50")
    manifest = json.loads((DATASET / "episode_manifest.json").read_text())
    assert len(manifest) == 1250
    horizon = config.model.action_horizon
    states, action_chunks = [], []
    total = 0
    for episode in manifest:
        index = episode["episode_index"]
        path = DATASET / f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"
        table = parquet.read_table(path, columns=["state", "actions"])
        state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        assert state.shape == action.shape == (episode["steps"], 8)
        indices = np.minimum(np.arange(len(action))[:, None] + np.arange(horizon)[None, :], len(action) - 1)
        chunk = action[indices].copy()
        chunk[:, :, :7] -= state[:, None, :7]
        states.append(state)
        action_chunks.append(chunk.reshape(-1, 8))
        total += len(state)
        if (index + 1) % 250 == 0:
            print(f"Read {index + 1}/1250 episodes", flush=True)
    assert total == 131988
    result = {}
    for name, values in (("state", states), ("actions", action_chunks)):
        running = normalize.RunningStats()
        running.update(np.concatenate(values, axis=0))
        result[name] = running.get_statistics()
    output = config.assets_dirs / config.data.repo_id
    normalize.save(output, result)
    print(f"Saved {output / 'norm_stats.json'}", flush=True)
    for name, stats in result.items():
        print(name, "q01", stats.q01.tolist(), "q99", stats.q99.tolist(), flush=True)


if __name__ == "__main__":
    main()
