#!/usr/bin/env python3
"""Convert approved raw ACT demonstrations to lossless action-frame memmaps.

Reads source files only. Builds in a unique sibling directory and publishes
only after byte comparisons, source-inventory validation, and fsync complete.
The final output path must not exist. Terminal observation frames are omitted.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import h5py
import numpy as np


SCHEMA = "act_triple_rgb_memmap_v1"
STREAM_PATHS = ("obs/sensor_data/base_camera/rgb", "obs/sensor_data/hand_camera/rgb",
                "derived/base_camera/masked_rgb")
STREAM_ORDER = ("base_rgb", "wrist_rgb", "base_masked_rgb")
STATE_PATHS = ("obs/agent/qpos", "obs/agent/qvel", "obs/extra/tcp_pose", "obs/extra/goal_pos")
STATE_DIMS = (9, 9, 7, 3)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"Source symlink is unexpected: {path}")
        if path.is_file():
            s = path.stat()
            result[path.relative_to(root).as_posix()] = dict(size=s.st_size, mtime_ns=s.st_mtime_ns,
                                                           ctime_ns=s.st_ctime_ns, inode=s.st_ino)
    return result


def scan_source(root, args):
    split = json.loads((root / "split_manifest.json").read_text())
    progress = json.loads((root / "progress.json").read_text())
    train, heldout = split["training_objects"], split["heldout_objects"]
    require(len(train) == args.expected_targets and len(set(train)) == len(train), "Invalid training targets")
    require(not set(train) & set(heldout), "Training/heldout target overlap")
    require({p.name for p in (root / "shards").iterdir() if p.is_dir()} == set(train), "Unexpected target directories")
    episodes, counts, shard_jobs, env_jsons, state_dtypes, action_dtypes = [], collections.Counter(), [], set(), set(), set()
    frame_offset = 0
    for target in train:
        for shard in sorted((root / "shards" / target).glob("shard_*")):
            require(shard.is_dir(), f"Expected shard directory: {shard}")
            accepted = json.loads((shard / "accepted.json").read_text())
            meta = json.loads((shard / "trajectory.json").read_text())
            metadata = {row["episode_id"]: row for row in meta["episodes"]}
            env_jsons.add(canonical(meta["env_info"]).decode())
            require(len(accepted) == len(metadata), f"Metadata count mismatch: {shard}")
            job_episodes = []
            with h5py.File(shard / "trajectory.h5", "r") as source:
                require(set(source) == {row["trajectory"] for row in accepted}, f"H5 trajectory mismatch: {shard}")
                for row in accepted:
                    name = row["trajectory"]
                    g = source[name]
                    episode = metadata[int(name.split("_")[-1])]
                    steps = len(g["actions"])
                    require(row["target_name"] == target == g.attrs["target_name"], f"Target mismatch: {shard}/{name}")
                    require(g.attrs["object_split"] == "train", "Only the approved training split is allowed")
                    require(row["selected_names"][row["target_idx"]] == target and set(row["selected_names"]) <= set(train), "Invalid scene targets")
                    require(row["grasped_and_lifted"] is True and row["wrong_grasp_ever"] is False and row["initial_visible_pixels"] > 0,
                            "Accepted success constraints differ")
                    require(episode["success"] is True and episode["episode_seed"] == row["seed"], "Episode success/seed mismatch")
                    require(episode["elapsed_steps"] == steps and row["frames"] == steps + 1, "Action/frame alignment mismatch")
                    require(g["actions"].shape == (steps, 8) and 0 < steps <= 500, "Invalid actions")
                    require(np.issubdtype(g["actions"].dtype, np.floating), "Actions must be floating point")
                    action_dtypes.add(g["actions"].dtype.str)
                    for key, dim in zip(STATE_PATHS, STATE_DIMS):
                        require(g[key].shape == (steps + 1, dim), f"Invalid state dimensions: {key}")
                        require(np.issubdtype(g[key].dtype, np.floating), f"State must be floating point: {key}")
                        state_dtypes.add(g[key].dtype.str)
                    for key in STREAM_PATHS:
                        require(g[key].shape == (steps + 1, 224, 224, 3) and g[key].dtype == np.uint8, f"Invalid RGB: {key}")
                    require(g["derived/base_camera/target_mask"].shape == (steps + 1, 224, 224, 1), "Invalid target mask dimensions")
                    require(g["obs/sensor_data/base_camera/segmentation"].shape == (steps + 1, 224, 224, 1), "Invalid segmentation dimensions")
                    require(g["alignment/target_seg_id"].shape == (steps + 1, 1), "Invalid target segmentation alignment")
                    rec = dict(episode_index=len(episodes), start=frame_offset, stop=frame_offset + steps, steps=steps,
                               source_h5=(shard / "trajectory.h5").relative_to(root).as_posix(), trajectory=name,
                               source_accepted=(shard / "accepted.json").relative_to(root).as_posix(),
                               accepted_record=row)
                    episodes.append(rec)
                    job_episodes.append(rec)
                    counts[target] += 1
                    frame_offset += steps
            shard_jobs.append(dict(source_h5=str(shard / "trajectory.h5"), episodes=job_episodes))
    expected_count = args.expected_targets * args.expected_per_target
    require(len(episodes) == expected_count and counts == dict.fromkeys(train, args.expected_per_target), "Wrong target/episode counts")
    require(progress["total"] == progress["target"] == expected_count and progress["validation_demos"] == 0, "Invalid progress metadata")
    require(progress["per_target"] == dict(counts), "Progress target counts differ")
    require(len(shard_jobs) == args.expected_shards, "Wrong shard count")
    require(frame_offset == args.expected_frames, "Wrong action frame count")
    require(len(env_jsons) == 1, "Source environment configurations differ")
    env = json.loads(next(iter(env_jsons)))
    require(env["env_id"] == "TargetPickForty-v1" and env["env_kwargs"]["control_mode"] == "pd_joint_pos"
            and env["env_kwargs"]["robot_uids"] == "panda_wristcam", "Unexpected source environment")
    # np.result_type can promote float32 to float64 without losing any values.
    state_dtype = np.result_type(*[np.dtype(x) for x in state_dtypes])
    action_dtype = np.result_type(*[np.dtype(x) for x in action_dtypes])
    return episodes, shard_jobs, frame_offset, env, split, state_dtype, action_dtype


def convert_shard(job, staging, block_frames):
    root = Path(staging)
    arrays = {key: np.load(root / f"{key}.npy", mmap_mode="r+", allow_pickle=False)
              for key in ("rgb", "state", "actions")}
    records = []
    with h5py.File(job["source_h5"], "r") as source:
        for rec in job["episodes"]:
            g, start, stop = source[rec["trajectory"]], rec["start"], rec["stop"]
            state = np.concatenate([g[key][:rec["steps"]] for key in STATE_PATHS], axis=-1)
            actions = g["actions"][:]
            require(np.isfinite(state).all() and np.isfinite(actions).all(), "Nonfinite policy state/actions")
            arrays["state"][start:stop] = state
            arrays["actions"][start:stop] = actions
            require(np.array_equal(arrays["state"][start:stop], state), "State copy differs")
            require(np.array_equal(arrays["actions"][start:stop], actions), "Action copy differs")
            digests = {"state": hashlib.sha256(arrays["state"][start:stop].tobytes()).hexdigest(),
                       "actions": hashlib.sha256(arrays["actions"][start:stop].tobytes()).hexdigest()}
            for frame in sorted({0, rec["steps"] // 2, rec["steps"] - 1}):
                mask = g["derived/base_camera/target_mask"][frame]
                segmentation = g["obs/sensor_data/base_camera/segmentation"][frame]
                target_id = np.asarray(g["alignment/target_seg_id"][frame]).reshape(1, 1, 1)
                require(np.array_equal(mask, segmentation == target_id), "Stored target mask and segmentation differ")
                require(np.array_equal(g[STREAM_PATHS[2]][frame], g[STREAM_PATHS[0]][frame] * mask),
                        "Stored masked RGB is not aligned base RGB multiplied by target mask")
            for stream, key in enumerate(STREAM_PATHS):
                digest = hashlib.sha256()
                for offset in range(0, rec["steps"], block_frames):
                    n = min(block_frames, rec["steps"] - offset)
                    raw = g[key][offset:offset + n]
                    view = arrays["rgb"][start + offset:start + offset + n, stream]
                    view[:] = raw
                    require(np.array_equal(view, raw), f"RGB byte comparison failed: {rec['trajectory']} {key} {offset}")
                    digest.update(view.tobytes())
                digests[STREAM_ORDER[stream]] = digest.hexdigest()
            records.append(dict(episode_index=rec["episode_index"], array_sha256=digests))
    # Each worker writes disjoint episode ranges. The parent fsyncs once after
    # every worker exits, avoiding repeated whole-file msyncs during conversion.
    return records


def publish_noreplace(staging, output):
    # Linux renameat2 prevents races from replacing a newly created destination.
    import ctypes
    import errno
    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), str(output))
    else:
        raise OSError(errno.ENOSYS, "Atomic no-replace publication requires Linux renameat2")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--block-frames", type=int, default=26)
    ap.add_argument("--expected-targets", type=int, default=25)
    ap.add_argument("--expected-per-target", type=int, default=50)
    ap.add_argument("--expected-shards", type=int, default=150)
    ap.add_argument("--expected-frames", type=int, default=131988)
    args = ap.parse_args()
    require(args.workers > 0 and args.block_frames > 0, "workers/block-frames must be positive")
    started = time.time()
    raw = args.raw_root.resolve(strict=True)
    output = args.output.absolute()
    require(not output.exists(), f"Output already exists: {output}")
    require(raw != output and raw not in output.parents and output not in raw.parents, "Output must be separate from raw source")
    before = inventory(raw)
    episodes, jobs, frames, env, split, state_dtype, action_dtype = scan_source(raw, args)
    rgb_bytes = frames * 3 * 224 * 224 * 3
    output.parent.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(output.parent).free > rgb_bytes + frames * (28 * state_dtype.itemsize + 8 * action_dtype.itemsize) + 1024 ** 3,
            "Not enough disk space for the uncompressed lossless memmaps")
    staging = output.with_name(output.name + f".preparing-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    staging.mkdir()
    shapes = {"rgb": (frames, 3, 224, 224, 3), "state": (frames, 28), "actions": (frames, 8)}
    dtypes = {"rgb": np.dtype("uint8"), "state": state_dtype, "actions": action_dtype}
    for key in shapes:
        mapped = np.lib.format.open_memmap(staging / f"{key}.npy", mode="w+", dtype=dtypes[key], shape=shapes[key])
        del mapped
    np.save(staging / "episode_offsets.npy", np.array([r["start"] for r in episodes] + [frames], dtype=np.int64), allow_pickle=False)
    print(json.dumps(dict(event="conversion_started", staging=str(staging), output=str(output), workers=args.workers,
                          num_episodes=len(episodes), num_frames=frames, rgb_bytes=rgb_bytes)), flush=True)
    checksums = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(convert_shard, job, str(staging), args.block_frames) for job in jobs]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            checksums.extend(future.result())
            print(json.dumps(dict(event="shard_complete", shards_complete=completed, shards_total=len(jobs),
                                  episodes_complete=len(checksums), elapsed_seconds=round(time.time() - started, 2))), flush=True)
    checksums.sort(key=lambda row: row["episode_index"])
    require([r["episode_index"] for r in checksums] == list(range(len(episodes))), "Missing episode conversion checksums")
    for rec, checked in zip(episodes, checksums):
        rec.update(array_sha256=checked["array_sha256"])
    after = inventory(raw)
    require(before == after, "Source inventory changed during conversion; refusing publication")
    write_json(staging / "episodes.json", episodes)
    write_json(staging / "source_inventory.json", before)
    write_json(staging / "source_split_manifest.json", split)
    fingerprint = hashlib.sha256(canonical(dict(schema=SCHEMA, source_inventory=before, episodes=episodes,
                                               stream_order=STREAM_ORDER, state_paths=STATE_PATHS))).hexdigest()
    manifest = dict(schema=SCHEMA, complete=True, dataset_fingerprint=fingerprint,
                    created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source_root=str(raw),
                    num_episodes=len(episodes), num_frames=frames, num_source_shards=len(jobs),
                    source_observation_frames=frames + len(episodes), terminal_observations_copied=False,
                    stream_order=list(STREAM_ORDER), stream_source_paths=list(STREAM_PATHS),
                    state_fields=[x.removeprefix("obs/") for x in STATE_PATHS], state_field_dims=list(STATE_DIMS),
                    state_dim=28, action_dim=8, control_mode="pd_joint_pos", environment=env,
                    arrays={key: dict(file=f"{key}.npy", shape=list(shapes[key]), dtype=dtypes[key].str) for key in shapes},
                    padding="repeat final raw action within its episode", numeric_conversion="lossless NumPy promotion only; no controller/gripper conversion",
                    source_inventory_stable=True, byte_comparisons="Every copied source state/action value and RGB byte",
                    mask_alignment_samples="First, middle and last action frame of every episode; target segmentation and base RGB x target mask",
                    elapsed_seconds=time.time() - started)
    write_json(staging / "manifest.json", manifest)
    for path in staging.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    publish_noreplace(staging, output)
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps(dict(event="conversion_complete", output=str(output), dataset_fingerprint=fingerprint,
                          elapsed_seconds=time.time() - started)), flush=True)


if __name__ == "__main__":
    main()
