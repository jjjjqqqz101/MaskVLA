"""Render real TargetPickCube scenes; annotations are only in saved previews.

Headless: python -m mani_skill.examples.demo_target_pick_cube --output-dir outputs/target_pick_cube
Desktop:  python -m mani_skill.examples.demo_target_pick_cube --gui
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mani_skill.envs


def annotate_scene(base, rgb, seed):
    height, width = rgb.shape[:2]
    canvas = Image.new("RGB", (width, height + 96), (22, 27, 36))
    canvas.paste(Image.fromarray(rgb), (0, 56))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = small_font = ImageFont.load_default()
    target = base.target_idx.item()
    draw.text((18, 15), f"TargetPickCube-v1  |  seed {seed}  |  target = cube_{target}", fill="white", font=font)
    draw.text((18, height + 67), "Yellow ring: target (preview only). Green sphere: destination.", fill=(213, 223, 236), font=small_font)
    params = base._human_render_cameras["render_camera"].get_params()
    extrinsic = params["extrinsic_cv"][0].cpu().numpy()
    intrinsic = params["intrinsic_cv"][0].cpu().numpy()
    for i, cube in enumerate(base.cubes):
        position = cube.pose.p[0].cpu().numpy()
        point_camera = extrinsic @ np.append(position, 1)
        projected = intrinsic @ point_camera[:3]
        x, y = (projected[:2] / projected[2]).tolist()
        y += 56
        color = (255, 213, 70) if i == target else (230, 240, 255)
        radius = 23 if i == target else 17
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=3)
        label = str(i)
        draw.text(
            (x - 7, y - 13), label, fill=color, font=font,
            stroke_width=2, stroke_fill=(22, 27, 36),
        )
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/target_pick_cube"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.num_scenes < 1:
        parser.error("--num-scenes must be positive")
    env = gym.make(
        "TargetPickCube-v1",
        obs_mode="rgb",
        sim_backend="physx_cpu",
        control_mode="pd_joint_pos",
        render_mode="human" if args.gui else "rgb_array",
        human_render_camera_configs=dict(width=768, height=768, shader_pack="default"),
    )
    try:
        if args.gui:
            env.reset(seed=args.seed)
            print(f"target_idx={env.unwrapped.target_idx.item()}; press N for a new layout.")
            viewer = env.unwrapped.render_human()
            seed = args.seed
            while not viewer.closed:
                env.unwrapped.render_human()
                if viewer.window.key_press("n"):
                    seed += 1
                    env.reset(seed=seed)
                    print(f"seed={seed}, target_idx={env.unwrapped.target_idx.item()}")
            return

        args.output_dir.mkdir(parents=True, exist_ok=True)
        previews = []
        metadata = []
        seen = set()
        # For the default three-scene preview, illustrate all three target IDs.
        for seed in range(args.seed, args.seed + max(100, args.num_scenes)):
            env.reset(seed=seed)
            base = env.unwrapped
            target = base.target_idx.item()
            if len(seen) < min(3, args.num_scenes) and target in seen:
                continue
            seen.add(target)
            rgb = env.render()[0].cpu().numpy()
            imageio.imwrite(args.output_dir / f"scene_seed_{seed}_raw.png", rgb)
            preview = annotate_scene(base, rgb, seed)
            preview.save(args.output_dir / f"scene_seed_{seed}.png")
            previews.append(preview)
            metadata.append(dict(
                seed=seed,
                target_idx=target,
                cubes={c.name: c.pose.raw_pose[0].tolist() for c in base.cubes},
                goal_pos=base.goal_site.pose.p[0].tolist(),
            ))
            if len(previews) == args.num_scenes:
                break
        if len(previews) != args.num_scenes:
            raise RuntimeError("Could not produce the requested number of scenes")
        gallery = Image.new("RGB", (previews[0].width * len(previews), previews[0].height))
        for i, preview in enumerate(previews):
            gallery.paste(preview, (i * preview.width, 0))
        gallery.save(args.output_dir / "target_pick_cube_scenes.png")
        with imageio.get_writer(args.output_dir / "target_pick_cube_resets.mp4", fps=20) as video:
            for preview in previews:
                frame = np.array(preview)
                for _ in range(50):
                    video.append_data(frame)
        (args.output_dir / "scenes.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2))
        print(f"Saved scene images and reset slideshow to {args.output_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
