"""Collect successful, target-balanced three-view trajectories for pi0.5."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs
from mani_skill.envs.tasks.tabletop.target_pick_forty import TRAIN_OBJECTS
from mani_skill.examples.motionplanning.panda.solutions.target_pick_forty import solve
from mani_skill.utils.wrappers import RecordEpisode


MAX_DEMO_STEPS = 200


class GraspAudit(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = self.unwrapped
        self.target_idx = int(info["target_idx"].item())
        self.names = tuple(base.selected_names)
        self.ids = np.asarray([int(obj.per_scene_id.item()) for obj in base.cubes], dtype=np.int32)
        if len(self.names) != 8 or len(set(self.names)) != 8 or len(set(self.ids)) != 8:
            raise RuntimeError("Scene must have eight unique objects and segmentation IDs")
        if set(self.names).difference(TRAIN_OBJECTS):
            raise RuntimeError("Held-out object entered a training scene")
        self.visible_pixels = []
        self.wrong_ever = False
        self.held_and_lifted = False
        self._inspect(obs)
        return obs, info

    def _inspect(self, obs):
        seg = obs["sensor_data"]["base_camera"]["segmentation"]
        self.visible_pixels.append(int((seg == int(self.ids[self.target_idx])).sum().item()))

    def step(self, action):
        if len(self.visible_pixels) > MAX_DEMO_STEPS:
            raise RuntimeError(f"Expert exceeded {MAX_DEMO_STEPS} steps")
        result = self.env.step(action)
        info = result[-1]
        if int(info["target_idx"].item()) != self.target_idx:
            raise RuntimeError("Target identity changed mid-trajectory")
        self.wrong_ever |= bool(info["wrong_object_grasped"].item())
        self.held_and_lifted |= bool(info["is_target_grasped"].item()
                                     and info["target_lifted"].item())
        self._inspect(result[0])
        return result


def add_masks(group, ids, target):
    camera = group["obs/sensor_data/base_camera"]
    seg, rgb = camera["segmentation"][:], camera["rgb"][:]
    hand = group["obs/sensor_data/hand_camera/rgb"]
    if len(rgb) != len(hand) or len(rgb) != len(group["actions"]) + 1:
        raise RuntimeError("Camera/action time alignment error")
    mask = seg == int(ids[target])
    masked = np.where(mask, rgb, 0).astype(np.uint8)
    derived = group.create_group("derived/base_camera")
    for name, data in (("target_mask", mask), ("masked_rgb", masked)):
        derived.create_dataset(name, data=data, compression="gzip", compression_opts=4,
                               chunks=(1, *data.shape[1:]))
    alignment = group.create_group("alignment")
    alignment.create_dataset("target_seg_id", data=np.full((len(rgb), 1), ids[target], np.int32))
    alignment.create_dataset("object_seg_ids", data=np.tile(ids, (len(rgb), 1)))
    alignment.create_dataset("visible_pixels", data=mask.sum(axis=(1, 2, 3)).astype(np.int32))
    if not np.array_equal(masked, rgb * mask):
        raise RuntimeError("Masked RGB disagrees with RGB and target mask")
    return len(rgb), int(mask[0].sum()), int(mask.sum(axis=(1, 2, 3)).min())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-name", choices=TRAIN_OBJECTS, required=True)
    parser.add_argument("--num-demos", type=int, default=10)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, default=200)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    env = gym.make(
        "TargetPickForty-v1", robot_uids="panda_wristcam",
        obs_mode="rgb+segmentation", control_mode="pd_joint_pos",
        sim_backend="physx_cpu", render_mode="rgb_array", max_episode_steps=500,
        sensor_configs=dict(width=224, height=224, fov=float(np.pi / 2), near=.01,
                            far=100, shader_pack="minimal"),
    )
    audit = GraspAudit(env)
    recorder = RecordEpisode(
        audit, args.output_dir, trajectory_name="trajectory", save_trajectory=True,
        save_video=False, save_on_reset=False, clean_on_close=False,
        source_type="motionplanning",
        source_desc="Eight objects sampled from 25 train targets; wrist camera; target mask; no wrong grasp",
    )
    accepted, attempts = [], []
    try:
        for attempt in range(args.max_attempts):
            seed = args.start_seed + attempt
            error = None
            try:
                result = solve(recorder, target_name=args.target_name, seed=seed)
            except Exception as exc:
                result = -1
                error = f"{type(exc).__name__}: {exc}"
            info = result[-1] if not isinstance(result, int) else None
            visible = getattr(audit, "visible_pixels", [])
            valid = bool(
                info is not None and visible and info["is_target_grasped"].item()
                and info["target_lifted"].item() and not audit.wrong_ever
                and audit.held_and_lifted and visible[0] >= 20
                and len(visible) - 1 <= MAX_DEMO_STEPS
            )
            attempts.append(dict(seed=seed, accepted=valid, error=error,
                                 initial_visible_pixels=visible[0] if visible else None,
                                 min_visible_pixels=min(visible) if visible else None))
            if not valid:
                recorder.flush_trajectory(save=False)
                print(f"REJECT target={args.target_name} seed={seed} error={error}", flush=True)
                continue
            recorder.flush_trajectory()
            key = f"traj_{recorder._episode_id}"
            group = recorder._h5_file[key]
            group.attrs["target_idx"] = audit.target_idx
            group.attrs["target_name"] = args.target_name
            group.attrs["selected_names"] = json.dumps(audit.names)
            group.attrs["object_split"] = "train"
            group.attrs["seed"] = seed
            frames, initial_pixels, min_pixels = add_masks(group, audit.ids, audit.target_idx)
            recorded_target = np.asarray(group["env_states/target_idx"][:]).reshape(-1)
            if not np.all(recorded_target == audit.target_idx):
                raise RuntimeError("Recorded target indices differ from requested target")
            accepted.append(dict(trajectory=key, seed=seed, target_name=args.target_name,
                                 target_idx=audit.target_idx, selected_names=audit.names,
                                 frames=frames, initial_visible_pixels=initial_pixels,
                                 min_visible_pixels=min_pixels, wrong_grasp_ever=False,
                                 grasped_and_lifted=True))
            recorder._h5_file.flush()
            (args.output_dir / "attempts.json").write_text(json.dumps(attempts, indent=2))
            (args.output_dir / "accepted.json").write_text(json.dumps(accepted, indent=2))
            print(f"SAVED target={args.target_name} {len(accepted)}/{args.num_demos} "
                  f"seed={seed} frames={frames}", flush=True)
            if len(accepted) == args.num_demos:
                break
    finally:
        recorder.close()
    (args.output_dir / "attempts.json").write_text(json.dumps(attempts, indent=2))
    (args.output_dir / "accepted.json").write_text(json.dumps(accepted, indent=2))
    if len(accepted) != args.num_demos:
        raise RuntimeError(f"Collected {len(accepted)}/{args.num_demos} for {args.target_name}")
    print(f"COMPLETE target={args.target_name} accepted={len(accepted)} attempts={len(attempts)}", flush=True)


if __name__ == "__main__":
    main()
