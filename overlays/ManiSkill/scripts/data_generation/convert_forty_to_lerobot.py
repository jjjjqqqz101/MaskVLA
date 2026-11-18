"""Convert the completed 25x50 ManiSkill collection into lossless LeRobot data."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


PROMPT = "Pick up and lift the object shown in the target image."
REPO_ID = "maniskill/target_pick_forty_pi05_25x50"
FEATURES = {
    "image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
    "target_image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
    "target_mask": {"dtype": "bool", "shape": (224, 224, 1), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
}


def sources(raw_root):
    for filename in ("split_manifest.json", "progress.json"):
        if not (raw_root / filename).is_file():
            raise FileNotFoundError(f"Optional raw collection unavailable: {raw_root / filename}. Datasets were not copied with the code; pass --raw-root for an existing collection.")
    split = json.loads((raw_root / "split_manifest.json").read_text())
    counts = json.loads((raw_root / "progress.json").read_text())
    train, heldout = split["training_objects"], split["heldout_objects"]
    if len(train) != 25 or len(heldout) != 15 or set(train) & set(heldout):
        raise RuntimeError("Invalid train/held-out object split")
    if counts["total"] != 1250 or any(counts["per_target"][x] != 50 for x in train):
        raise RuntimeError("Collection incomplete: need exactly 50 successes for each of 25 training targets")
    result = []
    for target in train:
        for shard in sorted((raw_root / "shards" / target).glob("shard_*")):
            rows = json.loads((shard / "accepted.json").read_text())
            for row in rows:
                if row["target_name"] != target or set(row["selected_names"]) & set(heldout):
                    raise RuntimeError(f"Invalid target or held-out distractor in {shard}")
                result.append((shard, row))
    if len(result) != 1250:
        raise RuntimeError(f"Expected 1250 accepted episodes, found {len(result)}")
    return split, result


def convert_episode(dataset, shard, row):
    with h5py.File(shard / "trajectory.h5", "r") as source:
        group = source[row["trajectory"]]
        if group.attrs["target_name"] != row["target_name"]:
            raise RuntimeError("Target metadata mismatch")
        if group.attrs["object_split"] != "train":
            raise RuntimeError("Non-training episode entered conversion")
        actions = group["actions"][:]
        qpos = group["obs/agent/qpos"][:]
        rgb = group["obs/sensor_data/base_camera/rgb"][:]
        wrist = group["obs/sensor_data/hand_camera/rgb"][:]
        masked = group["derived/base_camera/masked_rgb"][:]
        mask = group["derived/base_camera/target_mask"][:]
        steps = len(actions)
        if not all(len(x) == steps + 1 for x in (qpos, rgb, wrist, masked, mask)):
            raise RuntimeError("Frame/action alignment mismatch")
        if not np.array_equal(masked, rgb * mask):
            raise RuntimeError("RGB x Mask mismatch")
        if not np.array_equal(mask, group["obs/sensor_data/base_camera/segmentation"][:]
                              == group["alignment/target_seg_id"][:].reshape(-1, 1, 1, 1)):
            raise RuntimeError("Target segmentation mismatch")
        for t in range(steps):
            state = np.r_[qpos[t, :7], np.clip(1 - np.mean(qpos[t, 7:]) / 0.04, 0, 1)].astype(np.float32)
            action = np.r_[actions[t, :7], np.clip((1 - actions[t, 7]) / 2, 0, 1)].astype(np.float32)
            dataset.add_frame({"image": rgb[t], "wrist_image": wrist[t],
                               "target_image": masked[t], "target_mask": mask[t],
                               "state": state, "actions": action, "task": PROMPT})
        dataset.save_episode()
        return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    split, episodes = sources(args.raw_root)
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        parser.error(f"Optional LeRobot dependency unavailable: {error}. Run conversion with the configured openpi/LeRobot Python environment.")
    info_path = args.output_root / "meta" / "info.json"
    if info_path.exists():
        dataset = LeRobotDataset(REPO_ID, root=args.output_root)
    else:
        if args.output_root.exists() and any(args.output_root.iterdir()):
            raise RuntimeError("Output directory exists but contains no LeRobot metadata")
        dataset = LeRobotDataset.create(
            repo_id=REPO_ID, fps=20, root=args.output_root,
            robot_type="panda_wristcam", features=FEATURES,
            use_videos=False, image_writer_threads=8,
        )
    start = dataset.num_episodes
    if start > len(episodes):
        raise RuntimeError("LeRobot output has more episodes than the raw collection")
    manifest = args.output_root / "episode_manifest.json"
    records = [dict(episode_index=i, source_shard=str(shard), trajectory=row["trajectory"],
                    target_name=row["target_name"], selected_names=row["selected_names"],
                    seed=row["seed"], prompt=PROMPT)
               for i, (shard, row) in enumerate(episodes[:start])]
    manifest.write_text(json.dumps(records, indent=2))
    for index in range(start, len(episodes)):
        shard, row = episodes[index]
        steps = convert_episode(dataset, shard, row)
        records.append(dict(episode_index=index, source_shard=str(shard),
                            trajectory=row["trajectory"], target_name=row["target_name"],
                            selected_names=row["selected_names"], seed=row["seed"],
                            prompt=PROMPT, steps=steps))
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(records, indent=2))
        temporary.replace(manifest)
        print(f"CONVERTED {index+1}/1250 target={row['target_name']} steps={steps}", flush=True)
    (args.output_root / "split_manifest.json").write_text(json.dumps(split, indent=2))
    print(f"DONE {dataset.num_episodes} episodes, {len(dataset)} action frames", flush=True)


if __name__ == "__main__":
    main()
