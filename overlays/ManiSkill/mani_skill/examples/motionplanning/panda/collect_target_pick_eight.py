"""Collect successful, target-balanced RGB+mask ACT demos in resumable shards."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions.target_pick_eight import solve
from mani_skill.utils.wrappers import RecordEpisode


LABELS = ("red_cube_1", "red_cube_2", "blue_cube", "cracker_box",
          "mug", "banana", "softball", "baseball")
MAX_DEMO_STEPS = 200


class GraspAudit(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = self.unwrapped
        self.target_idx = int(info["target_idx"].item())
        self.ids = np.asarray([int(x.per_scene_id.item()) for x in base.cubes], dtype=np.int32)
        if len(set(self.ids)) != len(LABELS):
            raise RuntimeError("Object segmentation IDs are not unique")
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
    seg = camera["segmentation"][:]
    rgb = camera["rgb"][:]
    mask = (seg == int(ids[target]))
    masked = np.where(mask, rgb, 0).astype(np.uint8)
    derived = group.create_group("derived/base_camera")
    for name, data in (("target_mask", mask), ("masked_rgb", masked)):
        derived.create_dataset(name, data=data, compression="gzip", compression_opts=4,
                               chunks=(1, *data.shape[1:]))
    alignment = group.create_group("alignment")
    alignment.create_dataset("target_seg_id", data=np.full((len(rgb), 1), ids[target], np.int32))
    alignment.create_dataset("object_seg_ids", data=np.tile(ids, (len(rgb), 1)))
    alignment.create_dataset("visible_pixels", data=mask.sum(axis=(1, 2, 3)).astype(np.int32))
    return len(rgb), int(mask[0].sum()), int(mask.sum(axis=(1, 2, 3)).min())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-idx", type=int, choices=range(8), required=True)
    parser.add_argument("--num-demos", type=int, default=10)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, default=200)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    env = gym.make(
        "TargetPickEight-v1", obs_mode="rgb+segmentation", control_mode="pd_joint_pos",
        sim_backend="physx_cpu", render_mode="rgb_array", max_episode_steps=500,
        sensor_configs=dict(width=224, height=224, fov=float(np.pi / 2), near=.01,
                            far=100, shader_pack="minimal"),
    )
    audit = GraspAudit(env)
    recorder = RecordEpisode(
        audit, args.output_dir, trajectory_name="trajectory", save_trajectory=True,
        save_video=False, save_on_reset=False, clean_on_close=False,
        source_type="motionplanning",
        source_desc="Eight-object target-conditioned grasp and lift; no wrong object grasp",
    )
    accepted, attempts = [], []
    try:
        for attempt in range(args.max_attempts):
            seed = args.start_seed + attempt
            error = None
            try:
                result = solve(recorder, target_idx=args.target_idx, seed=seed)
            except Exception as exc:
                result = -1
                error = f"{type(exc).__name__}: {exc}"
            info = result[-1] if not isinstance(result, int) else None
            valid = bool(
                info is not None
                and info["is_target_grasped"].item()
                and info["target_lifted"].item()
                and not audit.wrong_ever
                and audit.held_and_lifted
                and audit.visible_pixels[0] >= 20
                and len(audit.visible_pixels) - 1 <= MAX_DEMO_STEPS
            )
            attempts.append(dict(seed=seed, accepted=valid, error=error,
                                 initial_visible_pixels=audit.visible_pixels[0],
                                 min_visible_pixels=min(audit.visible_pixels)))
            if not valid:
                recorder.flush_trajectory(save=False)
                print(f"REJECT target={args.target_idx} seed={seed} error={error}", flush=True)
                continue
            recorder.flush_trajectory()
            key = f"traj_{recorder._episode_id}"
            group = recorder._h5_file[key]
            group.attrs["target_idx"] = args.target_idx
            group.attrs["target_label"] = LABELS[args.target_idx]
            group.attrs["seed"] = seed
            frames, initial_pixels, min_pixels = add_masks(group, audit.ids, args.target_idx)
            recorded_target = np.asarray(group["env_states/target_idx"][:]).reshape(-1)
            if not np.all(recorded_target == args.target_idx):
                raise RuntimeError("Recorded target indices differ from requested target")
            accepted.append(dict(trajectory=key, seed=seed, target_idx=args.target_idx,
                                 target_label=LABELS[args.target_idx], frames=frames,
                                 initial_visible_pixels=initial_pixels,
                                 min_visible_pixels=min_pixels, wrong_grasp_ever=False,
                                 grasped_and_lifted=True))
            recorder._h5_file.flush()
            (args.output_dir / "attempts.json").write_text(json.dumps(attempts, indent=2))
            (args.output_dir / "accepted.json").write_text(json.dumps(accepted, indent=2))
            print(f"SAVED target={args.target_idx} {len(accepted)}/{args.num_demos} "
                  f"seed={seed} frames={frames}", flush=True)
            if len(accepted) == args.num_demos:
                break
    finally:
        recorder.close()
    (args.output_dir / "attempts.json").write_text(json.dumps(attempts, indent=2))
    (args.output_dir / "accepted.json").write_text(json.dumps(accepted, indent=2))
    if len(accepted) != args.num_demos:
        raise RuntimeError(f"Collected {len(accepted)}/{args.num_demos} for target {args.target_idx}")
    print(f"COMPLETE target={args.target_idx} accepted={len(accepted)} attempts={len(attempts)}", flush=True)


if __name__ == "__main__":
    main()
