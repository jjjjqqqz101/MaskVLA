"""Collect successful target-conditioned MPLib demos with per-step audits."""

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions.target_pick_cube import solve
from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.wrappers import RecordEpisode


class GraspAudit(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.target_idx = int(info["target_idx"].item())
        self.rows = []
        self.initial_poses = {
            cube.name: cube.pose.raw_pose[0].tolist() for cube in self.unwrapped.cubes
        }
        self.initial_image = self.env.render()[0].cpu().numpy()
        self.latest_info = info
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        info = result[-1]
        if int(info["target_idx"].item()) != self.target_idx:
            raise RuntimeError("Target identity changed within an episode")
        held = [bool(self.unwrapped.agent.is_grasping(c).item()) for c in self.unwrapped.cubes]
        row = dict(
            is_target_grasped=bool(info["is_target_grasped"].item()),
            target_lifted=bool(info["target_lifted"].item()),
            wrong_object_grasped=bool(info["wrong_object_grasped"].item()),
            full_success=bool(info["full_success"].item()),
            grasped_cubes=held,
        )
        if row["is_target_grasped"] != held[self.target_idx]:
            raise RuntimeError("Target grasp metric disagrees with per-object contacts")
        self.rows.append(row)
        self.latest_info = info
        return result


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


class LabelledRecorder(RecordEpisode):
    display_episode = 1

    def capture_image(self, infos=None):
        rgb = super().capture_image(infos)
        height, width = rgb.shape[:2]
        image = Image.new("RGB", (width, height + 64), (20, 26, 35))
        image.paste(Image.fromarray(rgb), (0, 64))
        draw = ImageDraw.Draw(image)
        target = self.base_env.target_idx.item()
        draw.text((15, 7), f"Episode {self.display_episode:02d} | Target: cube{target + 1} | target_idx={target}", font=font(20), fill="white")
        info = self.env.latest_info
        text = "  ".join(f"{label}: {int(info[key].item())}" for label, key in [
            ("grasp", "is_target_grasped"), ("lift", "target_lifted"),
            ("wrong", "wrong_object_grasped"), ("success", "full_success"),
        ])
        draw.text((15, 35), text, font=font(17), fill=(210, 225, 240))
        return np.asarray(image)


def labelled_initial_scene(base, audit, episode, seed):
    rgb = audit.initial_image
    height, width = rgb.shape[:2]
    image = Image.new("RGB", (width, height + 128), (20, 26, 35))
    image.paste(Image.fromarray(rgb), (0, 64))
    draw = ImageDraw.Draw(image)
    target = audit.target_idx
    draw.text((14, 10), f"Episode {episode:02d} | seed {seed} | Target: cube{target + 1}", font=font(22), fill="white")
    draw.text((14, 39), "Three identical red cubes; yellow circle marks the target.", font=font(16), fill=(210, 225, 240))
    params = base._human_render_cameras["render_camera"].get_params()
    extrinsic = params["extrinsic_cv"][0].cpu().numpy()
    intrinsic = params["intrinsic_cv"][0].cpu().numpy()
    for index in range(3):
        position = np.array(audit.initial_poses[f"cube_{index}"][:3])
        projected = intrinsic @ (extrinsic @ np.append(position, 1))[:3]
        x, y = projected[:2] / projected[2]
        y += 64
        color = (255, 215, 70) if index == target else (235, 240, 255)
        draw.ellipse((x-18, y-18, x+18, y+18), outline=color, width=3)
        draw.text((x-7, y-14), str(index + 1), font=font(22), fill=color, stroke_width=2, stroke_fill=(20, 26, 35))
        label_x = 15 + index * (width // 3)
        draw.text((label_x, height + 75), f"cube{index + 1}", font=font(21), fill=color)
        draw.text((label_x, height + 103), f"code: cube_{index}", font=font(16), fill=color)
    return image


def save_manifest(output, episodes, attempts):
    (output / "episodes.json").write_text(json.dumps(episodes, indent=2))
    (output / "attempts.json").write_text(json.dumps(attempts, indent=2))
    fields = [
        "episode", "trajectory", "seed", "target_idx", "target_cube", "target_label",
        "steps", "is_target_grasped", "target_lifted", "wrong_object_grasped_ever",
        "full_success", "video", "image",
    ]
    with (output / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-demos", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=100)
    args = parser.parse_args()
    if args.num_demos < 1 or args.max_attempts < args.num_demos:
        parser.error("Require 1 <= num-demos <= max-attempts")
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        "TargetPickCube-v1", obs_mode="rgb", control_mode="pd_joint_pos",
        sim_backend="physx_cpu", render_mode="rgb_array", max_episode_steps=500,
        human_render_camera_configs=dict(width=640, height=640, shader_pack="default"),
    )
    audit = GraspAudit(env)
    recorder = LabelledRecorder(
        audit, output, trajectory_name="trajectory", save_trajectory=True,
        save_video=True, save_on_reset=False, video_fps=20,
        source_type="motionplanning",
        source_desc="MPLib Panda expert selects cubes[target_idx]; rejects any wrong-object grasp",
    )
    episodes, attempts = [], []
    first_image = None
    try:
        for seed in range(args.start_seed, args.start_seed + args.max_attempts):
            recorder.display_episode = len(episodes) + 1
            error = None
            try:
                result = solve(recorder, seed=seed)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                result = -1
            rows = audit.rows
            wrong_ever = any(row["wrong_object_grasped"] for row in rows)
            lifted_while_grasped = any(row["is_target_grasped"] and row["target_lifted"] for row in rows)
            valid = (
                not isinstance(result, int)
                and bool(result[-1]["full_success"].item())
                and bool(result[-1]["is_target_grasped"].item())
                and not bool(result[3].item())
                and lifted_while_grasped and not wrong_ever
            )
            attempts.append(dict(seed=seed, target_idx=audit.target_idx, saved=valid, steps=len(rows), wrong_object_grasped_ever=wrong_ever, error=error))
            if not valid:
                recorder.flush_trajectory(save=False)
                recorder.flush_video(save=False)
                save_manifest(output, episodes, attempts)
                print(f"REJECT seed={seed}, target_idx={audit.target_idx}, error={error}", flush=True)
                continue

            number = len(episodes) + 1
            target = audit.target_idx
            name = f"episode_{number:02d}_cube{target + 1}"
            recorder.flush_trajectory()
            recorder.flush_video(name=name)
            image = labelled_initial_scene(env.unwrapped, audit, number, seed)
            image.save(output / f"{name}.png")
            if first_image is None:
                first_image = image
                image.save(output / "cube_labels.png")

            key = f"traj_{recorder._episode_id}"
            group = recorder._h5_file[key]
            group.attrs["target_idx"] = target
            group.attrs["target_label"] = f"cube{target + 1}"
            group.attrs["seed"] = seed
            audit_group = group.create_group("audit")
            for field in rows[0]:
                audit_group.create_dataset(field, data=np.asarray([row[field] for row in rows], dtype=bool))
            saved_targets = group["env_states/target_idx"][:]
            if not np.all(saved_targets == target):
                raise RuntimeError("Recorded target IDs do not match the expert target")
            entry = dict(
                episode=number, trajectory=key, seed=seed, target_idx=target,
                target_cube=f"cube_{target}", target_label=f"cube{target + 1}",
                steps=len(rows), is_target_grasped=True,
                target_lifted=bool(result[-1]["target_lifted"].item()),
                target_lifted_while_grasped=True,
                wrong_object_grasped_ever=False, full_success=True,
                video=f"{name}.mp4", image=f"{name}.png",
                initial_cube_poses=audit.initial_poses,
            )
            episodes.append(entry)
            recorder._json_data["episodes"][-1].update(
                target_idx=target, target_cube=f"cube_{target}",
                target_label=f"cube{target + 1}", video=entry["video"],
                wrong_object_grasped_ever=False,
            )
            dump_json(recorder._json_path, recorder._json_data, indent=2)
            recorder._h5_file.flush()
            save_manifest(output, episodes, attempts)
            print(f"SAVED {number}/{args.num_demos}: seed={seed}, target=cube{target + 1} (idx={target}), steps={len(rows)}", flush=True)
            if number == args.num_demos:
                break
    finally:
        recorder.close()
    if len(episodes) != args.num_demos:
        raise RuntimeError(f"Only saved {len(episodes)} demos after {len(attempts)} attempts")
    versions = {name: version(name) for name in ["numpy", "mplib", "opencv-python", "sapien", "torch"]}
    (output / "generation.json").write_text(json.dumps(dict(
        num_demos=len(episodes), num_attempts=len(attempts), versions=versions,
        control_mode="pd_joint_pos", sim_backend="physx_cpu", obs_mode="rgb",
        label_mapping={f"cube{i+1}": dict(target_idx=i, actor=f"cube_{i}") for i in range(3)},
    ), indent=2))
    print(f"DONE: {len(episodes)} verified demos in {output}", flush=True)


if __name__ == "__main__":
    main()
