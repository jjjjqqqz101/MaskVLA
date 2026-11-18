"""Resumable balanced collection for 25 train targets, 50 successes each."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py

from mani_skill.envs.tasks.tabletop.target_pick_forty import TRAIN_OBJECTS, HELDOUT_OBJECTS


COLLECTOR = "mani_skill.examples.motionplanning.panda.collect_target_pick_forty"
PROMPT = "Pick up and lift the object shown in the target image."


def shard_count(path):
    try:
        rows = json.loads((path / "accepted.json").read_text())
        with h5py.File(path / "trajectory.h5") as source:
            if all(row["trajectory"] in source and row["grasped_and_lifted"]
                   and not row["wrong_grasp_ever"] for row in rows):
                return len(rows)
    except (OSError, ValueError, KeyError):
        pass
    return 0


def save_progress(root):
    counts = {name: sum(shard_count(p) for p in (root / "shards" / name).glob("shard_*"))
              for name in TRAIN_OBJECTS}
    result = dict(total=sum(counts.values()), target=1250, per_target=counts,
                  training_objects=list(TRAIN_OBJECTS), heldout_objects=list(HELDOUT_OBJECTS),
                  prompt=PROMPT, frame_object_count=8, validation_demos=0)
    temporary = root / "progress.json.tmp"
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(root / "progress.json")
    return counts


def run(root, probe_only=False):
    root.mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    manifest = root / "split_manifest.json"
    expected = dict(training_objects=list(TRAIN_OBJECTS), heldout_objects=list(HELDOUT_OBJECTS),
                    prompt=PROMPT, frame_object_count=8, target_successes_per_train_object=50,
                    record_images=["base_camera.rgb", "hand_camera.rgb",
                                   "base_camera.target_mask", "base_camera.masked_rgb"])
    if manifest.exists() and json.loads(manifest.read_text()) != expected:
        raise RuntimeError("Existing split manifest differs; refusing to mix datasets")
    manifest.write_text(json.dumps(expected, indent=2))
    counts = save_progress(root)
    for index, name in enumerate(TRAIN_OBJECTS):
        target_count = 1 if probe_only else 50
        zero_success_shards = 0
        while counts[name] < target_count:
            folder = root / "shards" / name
            folder.mkdir(exist_ok=True)
            existing = [int(p.name.removeprefix("shard_")) for p in folder.glob("shard_*" )
                        if p.name.removeprefix("shard_").isdigit()]
            shard_id = max(existing, default=-1) + 1
            out = folder / f"shard_{shard_id:03d}"
            desired = min(1 if probe_only else 10, target_count - counts[name])
            seed = 2026092600 + index * 100000 + shard_id * 1000
            command = [sys.executable, "-m", COLLECTOR, "--output-dir", str(out),
                       "--target-name", name, "--num-demos", str(desired),
                       "--start-seed", str(seed), "--max-attempts", "100"]
            print(f"START {name} {counts[name]}/{target_count} shard={shard_id} seed={seed}", flush=True)
            with (root / "collection.log").open("a") as log:
                log.write(f"\nSTART {name} {counts[name]}/{target_count} shard={shard_id} seed={seed}\n")
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "MS_ASSET_DIR": os.environ.get("MS_ASSET_DIR", str(Path.home() / ".maniskill/data")),
                                             "CUDA_VISIBLE_DEVICES": os.environ.get("COLLECT_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))})
            gained = shard_count(out)
            counts = save_progress(root)
            print(f"RESULT {name} +{gained} => {counts[name]}/{target_count} rc={result.returncode}", flush=True)
            zero_success_shards = zero_success_shards + 1 if gained == 0 else 0
            if zero_success_shards >= 3:
                raise RuntimeError(f"Three consecutive empty shards for {name}; inspect collection.log")
            if shard_id >= 100:
                raise RuntimeError(f"Too many shards for {name}")
    print(f"DONE probe_only={probe_only} total={sum(counts.values())}/1250", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.probe_only)
