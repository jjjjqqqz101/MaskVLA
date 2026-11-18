"""Re-render saved TargetPickCube states with RGB and instance segmentation.

No dynamics are advanced: obs[t] is rendered from source env_states[t], paired
with the unchanged action[t]. Contact/success evidence comes from the original
expert audit, since set_state_dict alone does not recompute contact forces.
"""

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import sapien
import torch
from PIL import Image, ImageDraw, ImageFont

import mani_skill.envs


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_at(group, frame):
    return {
        key: state_at(value, frame) if isinstance(value, h5py.Group)
        else torch.as_tensor(value[frame:frame + 1])
        for key, value in group.items()
    }


def leaves(tree, prefix=""):
    for key, value in tree.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            yield from leaves(value, name)
        else:
            yield name, value.detach().cpu().numpy()[0]


def append_frame(group, path, frame, count, value):
    if frame == 0:
        parent, _, name = path.rpartition("/")
        destination = group.require_group(parent) if parent else group
        destination.create_dataset(
            name, shape=(count, *value.shape), dtype=value.dtype,
            chunks=(1, *value.shape), compression="gzip", compression_opts=4,
        )
    group[path][frame] = value


def check_projected_support(mask, pose, half_size, extrinsic, intrinsic):
    """Independent geometry check: visible pixels lie in the actor's projection."""
    corners = np.asarray(list(itertools.product([-half_size, half_size], repeat=3)))
    transform = sapien.Pose(p=pose[:3], q=pose[3:7]).to_transformation_matrix()
    world = corners @ transform[:3, :3].T + transform[:3, 3]
    camera = np.c_[world, np.ones(8)] @ extrinsic.T
    require(np.all(camera[:, 2] > 0), "Cube crosses the camera plane")
    pixels = camera @ intrinsic.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    y, x = np.nonzero(mask)
    if len(x):
        xy = np.c_[x, y]
        require(np.all(xy >= pixels.min(axis=0) - 2) and
                np.all(xy <= pixels.max(axis=0) + 2),
                "Segmentation pixels fall outside the target's projected bounds")


def font(size=13):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def panel(rgb, mask, masked_rgb, title):
    height, width = rgb.shape[:2]
    image = Image.new("RGB", (width * 3, height + 44), (22, 27, 36))
    draw = ImageDraw.Draw(image)
    draw.text((6, 2), title, fill="white", font=font())
    for index, (name, array) in enumerate([
        ("Original RGB", rgb), ("Target mask", mask.astype(np.uint8) * 255),
        ("Masked RGB", masked_rgb),
    ]):
        draw.text((index * width + 6, 23), name, fill=(200, 215, 230), font=font())
        image.paste(Image.fromarray(array), (index * width, 44))
    return image


def save_pngs(directory, rgb, mask, masked_rgb):
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(directory / "original.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(directory / "mask.png")
    Image.fromarray(masked_rgb).save(directory / "masked_rgb.png")


def run(source, output, count):
    source = source.resolve()
    metadata = json.loads(source.with_suffix(".json").read_text())
    episodes = metadata["episodes"][:count]
    require(len(episodes) == count, "Insufficient source episodes")
    require(metadata["env_info"]["env_id"] == "TargetPickCube-v1", "Wrong task")
    output.mkdir(parents=True, exist_ok=False)
    source_hash = sha256(source)
    kwargs = dict(metadata["env_info"]["env_kwargs"])
    kwargs.update(obs_mode="rgb+segmentation", num_envs=1, sim_backend="physx_cpu")
    kwargs["max_episode_steps"] = metadata["env_info"]["max_episode_steps"]
    env = gym.make("TargetPickCube-v1", **kwargs)
    base = env.unwrapped
    reports, previews = [], []
    destination = output / "trajectory.rgb+segmentation.h5"
    try:
        with h5py.File(source, "r") as src, h5py.File(destination, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["replay_method"] = "render each saved state; no dynamics step"
            dst.attrs["source_sha256"] = source_hash
            for episode in episodes:
                index = episode["episode_id"]
                name = f"traj_{index}"
                original = src[name]
                group = dst.create_group(name)
                for key, value in original.attrs.items():
                    group.attrs[key] = value
                for key in original:
                    if key != "obs":
                        src.copy(original[key], group, name=key)
                steps = len(original["actions"])
                frames = steps + 1
                target = int(episode["target_idx"])
                label = f"cube{target + 1}"
                require(int(original.attrs["target_idx"]) == target, "Attribute target mismatch")
                require(episode["target_cube"] == f"cube_{target}", "Actor name mismatch")
                require(episode["target_label"] == label, "Display label mismatch")
                require(np.all(original["env_states/target_idx"][:] == target), "State target changed")
                require(len(original["env_states/target_idx"]) == frames, "State length mismatch")
                held = original["audit/grasped_cubes"][:]
                grasp = original["audit/is_target_grasped"][:]
                lift = original["audit/target_lifted"][:]
                wrong = original["audit/wrong_object_grasped"][:]
                require(np.array_equal(grasp, held[:, target]), "Expert contact target mismatch")
                require(not held[:, np.arange(3) != target].any() and not wrong.any(), "Wrong cube grasped")
                require(np.any(grasp & lift) and bool(original["audit/full_success"][-1]), "Expert did not succeed")
                grasp_frame = int(np.flatnonzero(grasp)[0] + 1)
                lift_frame = int(np.flatnonzero(grasp & lift)[0] + 1)
                snapshots = {0: ["initial"], grasp_frame: ["grasp"]}
                snapshots.setdefault(lift_frame, []).append("lift")
                folder = output / f"episode_{index + 1:02d}_{label}"
                folder.mkdir()
                env.reset(**episode["reset_kwargs"])
                ids = np.array([int(c.per_scene_id.item()) for c in base.cubes], dtype=np.int32)
                require(len(set(ids.tolist())) == 3 and np.all(ids > 0), "Invalid actor IDs")
                require([c.name for c in base.cubes] == [f"cube_{i}" for i in range(3)], "Actor order mismatch")
                target_id = int(ids[target])
                areas, exact_rgb, max_rgb_error = [], 0, 0
                with imageio.get_writer(folder / "preview.mp4", fps=20, codec="libx264", macro_block_size=1) as video:
                    for frame in range(frames):
                        base.set_state_dict(state_at(original["env_states"], frame))
                        require(int(base.target_idx.item()) == target, "Restored target differs from expert")
                        for state_name, restored in leaves(base.get_state_dict()):
                            require(np.allclose(restored, original[f"env_states/{state_name}"][frame], rtol=0, atol=1e-6),
                                    f"State restoration mismatch at {name}/{frame}/{state_name}")
                        obs = base.get_obs()
                        for obs_name, value in leaves(obs):
                            append_frame(group, f"obs/{obs_name}", frame, frames, value)
                        sensors = obs["sensor_data"]
                        require(list(sensors) == ["base_camera"], "This exporter expects the original base_camera")
                        rgb = sensors["base_camera"]["rgb"][0].cpu().numpy()
                        seg = sensors["base_camera"]["segmentation"][0].cpu().numpy()
                        require(seg.shape == (*rgb.shape[:2], 1), "RGB/segmentation shape mismatch")
                        mask = seg[..., 0] == target_id
                        masked = np.where(mask[..., None], rgb, 0).astype(np.uint8)
                        require(not np.any(mask & np.isin(seg[..., 0], ids[np.arange(3) != target])), "Distractor in mask")
                        camera = obs["sensor_param"]["base_camera"]
                        check_projected_support(mask, original[f"env_states/actors/cube_{target}"][frame, :7],
                                                base.cube_half_size, camera["extrinsic_cv"][0].cpu().numpy(),
                                                camera["intrinsic_cv"][0].cpu().numpy())
                        delta = np.abs(rgb.astype(np.int16) - original["obs/sensor_data/base_camera/rgb"][frame].astype(np.int16))
                        exact_rgb += int(not delta.any())
                        max_rgb_error = max(max_rgb_error, int(delta.max()))
                        # These metadata/derived arrays are deliberately outside obs/extra.
                        append_frame(group, "alignment/target_seg_id", frame, frames, np.array([target_id], dtype=np.int32))
                        append_frame(group, "alignment/cube_seg_ids", frame, frames, ids)
                        append_frame(group, "derived/base_camera/target_mask", frame, frames, mask[..., None])
                        append_frame(group, "derived/base_camera/masked_rgb", frame, frames, masked)
                        areas.append(int(mask.sum()))
                        preview = panel(rgb, mask, masked, f"Ep {index + 1:02d} | {label} | idx={target} | frame {frame}")
                        video.append_data(np.asarray(preview))
                        if frame == 0:
                            save_pngs(folder, rgb, mask, masked)
                            previews.append(preview.copy())
                        for stage in snapshots.get(frame, []):
                            if stage != "initial":
                                save_pngs(folder / stage, rgb, mask, masked)
                group["alignment"].create_dataset("visible_pixels", data=np.array(areas, dtype=np.int32))
                require(np.array_equal(group["actions"][:], original["actions"][:]), "Actions changed")
                report = dict(episode=index + 1, trajectory=name, target_idx=target,
                              target_cube=f"cube_{target}", target_label=label,
                              cube_seg_ids=ids.tolist(), target_seg_id=target_id,
                              action_steps=steps, observation_frames=frames,
                              identity_aligned_frames=frames, projection_checks_passed=frames,
                              rgb_identical_to_source_frames=exact_rgb, rgb_max_channel_error=max_rgb_error,
                              visible_pixel_min=min(areas), visible_pixel_max=max(areas),
                              empty_mask_frames=np.flatnonzero(np.array(areas) == 0).tolist(),
                              expert_grasp_steps=int(grasp.sum()), wrong_object_grasp_steps=0,
                              expert_full_success=True, initial_frame=0,
                              grasp_frame=grasp_frame, lift_frame=lift_frame,
                              folder=folder.name)
                (folder / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
                reports.append(report)
                print(json.dumps(report), flush=True)
    finally:
        env.close()
    require(sha256(source) == source_hash, "Source file changed")
    out_metadata = copy.deepcopy(metadata)
    out_metadata["env_info"]["env_kwargs"] = kwargs
    out_metadata["episodes"] = episodes
    out_metadata["replay"] = dict(method="saved-state rendering", source=str(source), source_sha256=source_hash,
                                  contact_metrics="copied from source expert audit", derived_data="outside obs")
    destination.with_suffix(".json").write_text(json.dumps(out_metadata, indent=2) + "\n")
    report = dict(episodes=len(reports), action_steps=sum(x["action_steps"] for x in reports),
                  observation_frames=sum(x["observation_frames"] for x in reports),
                  identity_aligned_frames=sum(x["identity_aligned_frames"] for x in reports),
                  rgb_identical_to_source_frames=sum(x["rgb_identical_to_source_frames"] for x in reports),
                  empty_mask_frames=sum(len(x["empty_mask_frames"]) for x in reports),
                  wrong_object_grasp_steps=0, source_sha256=source_hash,
                  source_unchanged=True, per_episode=reports)
    (output / "alignment_report.json").write_text(json.dumps(report, indent=2) + "\n")
    width, height = previews[0].size
    overview = Image.new("RGB", (width * 2, height * ((len(previews) + 1) // 2)), (22, 27, 36))
    for i, preview in enumerate(previews):
        overview.paste(preview, ((i % 2) * width, (i // 2) * height))
    overview.save(output / "overview.png")
    print("COMPLETE", json.dumps({k: v for k, v in report.items() if k != "per_episode"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    run(args.traj_path, args.output_dir, args.count)
