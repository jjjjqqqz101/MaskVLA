"""Run 80 resumable ten-demo shards, then assemble one 800-demo training set."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py


LABELS = ("red_cube_1", "red_cube_2", "blue_cube", "cracker_box",
          "mug", "banana", "softball", "baseball")
MAX_DEMO_STEPS = 200
COLLECTOR = "mani_skill.examples.motionplanning.panda.collect_target_pick_eight"


def complete_shard(path):
    manifest = path / "accepted.json"
    h5_path = path / "trajectory.h5"
    if not manifest.exists() or not h5_path.exists():
        return False
    try:
        accepted = json.loads(manifest.read_text())
        if len(accepted) != 10:
            return False
        with h5py.File(h5_path) as source:
            return (len(source) == 10
                    and all(row["trajectory"] in source
                            and source[row["trajectory"]]["actions"].shape[0] <= MAX_DEMO_STEPS
                            and row["grasped_and_lifted"] and not row["wrong_grasp_ever"]
                            for row in accepted))
    except (OSError, ValueError, KeyError):
        return False


def collect(root, python):
    shards = root / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    selected = []

    def save_progress():
        counts = {label: sum(p.name.startswith(f"target_{i}_") for p in selected) * 10
                  for i, label in enumerate(LABELS)}
        (root / "progress.json").write_text(json.dumps({"counts": counts,
                                                          "total": sum(counts.values())}, indent=2))
        return sum(counts.values())

    for target in range(8):
        for shard in range(10):
            stem = f"target_{target}_shard_{shard:02d}"
            candidates = sorted(shards.glob(stem + "*"))
            done = next((path for path in candidates if complete_shard(path)), None)
            if done is not None:
                print(f"SKIP completed {done.name}", flush=True)
                selected.append(done)
                save_progress()
                continue
            retry = 0
            while True:
                name = stem if retry == 0 else f"{stem}_retry{retry}"
                path = shards / name
                if not path.exists():
                    break
                retry += 1
            seed = 2026101000 + target * 100000 + shard * 1000 + retry * 200
            cmd = [python, "-m", COLLECTOR, "--output-dir", str(path),
                   "--target-idx", str(target), "--num-demos", "10",
                   "--start-seed", str(seed), "--max-attempts", "200"]
            print(f"START {stem} seed={seed}", flush=True)
            with (root / "collection.log").open("a") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "MS_ASSET_DIR": os.environ.get("MS_ASSET_DIR", str(Path.home() / ".maniskill/data")),
                                             "CUDA_VISIBLE_DEVICES": os.environ.get("COLLECT_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))})
            if result.returncode or not complete_shard(path):
                raise RuntimeError(f"Shard failed: {path}. Inspect collection.log")
            selected.append(path)
            print(f"COMPLETE {stem} total={save_progress()}/800", flush=True)
    return selected


def merge(root, selected):
    final = root / "train.h5"
    partial = root / "train.h5.partial"
    if final.exists():
        summary = root / "summary.json"
        if summary.exists() and json.loads(summary.read_text()).get("total") == 800:
            print(f"ALREADY COMPLETE: {final}", flush=True)
            return
        raise FileExistsError(f"Training set already exists without valid summary: {final}")
    if partial.exists():
        partial.unlink()  # This runner owns the temporary merge output.
    episodes = []
    counts = {label: 0 for label in LABELS}
    template = None
    with h5py.File(partial, "w") as output:
        for shard_path in selected:
            manifest = json.loads((shard_path / "accepted.json").read_text())
            meta = json.loads((shard_path / "trajectory.json").read_text())
            if template is None:
                template = copy.deepcopy(meta)
                template["episodes"] = []
            meta_by_id = {f"traj_{row['episode_id']}": row for row in meta["episodes"]}
            with h5py.File(shard_path / "trajectory.h5") as source:
                for entry in manifest:
                    old = entry["trajectory"]
                    target = entry["target_idx"]
                    new = f"traj_{len(episodes)}"
                    source.copy(old, output, name=new)
                    group = output[new]
                    if int(group.attrs["target_idx"]) != target:
                        raise RuntimeError("Target attribute mismatch")
                    if group["derived/base_camera/masked_rgb"].shape[0] != group["actions"].shape[0] + 1:
                        raise RuntimeError("Frame/action alignment mismatch")
                    if group["actions"].shape[0] > MAX_DEMO_STEPS:
                        raise RuntimeError("Overlong trajectory entered final training set")
                    row = copy.deepcopy(meta_by_id[old])
                    row["episode_id"] = len(episodes)
                    row["grasp_success"] = True
                    row["target_idx"] = target
                    row["target_label"] = LABELS[target]
                    episodes.append(row)
                    counts[LABELS[target]] += 1
                    if len(episodes) % 50 == 0:
                        print(f"MERGED {len(episodes)}/800", flush=True)
        output.flush()
    if len(episodes) != 800 or set(counts.values()) != {100}:
        raise RuntimeError(f"Unexpected final counts: {counts}")
    template["episodes"] = episodes
    template["collection"] = dict(total=800, per_target=counts, training=800,
                                  validation=0, success_rule="target grasped and lifted; no wrong-object grasp")
    (root / "train.json").write_text(json.dumps(template, indent=2))
    (root / "summary.json").write_text(json.dumps(template["collection"], indent=2))
    partial.rename(final)
    print(f"DONE: {final} contains exactly 800 training demos", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = collect(args.output_dir, sys.executable)
    merge(args.output_dir, selected)


if __name__ == "__main__":
    main()
