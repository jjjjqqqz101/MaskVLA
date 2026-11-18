"""Display-only scene labels and the sole Original RGB input for single-stream ACT."""

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mani_skill.utils import common


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


class SingleRGBVideoWrapper(gym.Wrapper):
    """Render labels beside the exact resized uint8 inputs returned to ACT.

    This wrapper never changes observations or the physical scene. It snapshots
    policy RGB before normalization; render() only draws onto separate images.
    Frame t shows obs[t] and contacts at state t, including the terminal frame.
    """

    policy_title = "Original RGB: ACT stream 0"

    def __init__(self, env):
        super().__init__(env)
        self.policy_rgb = None
        self.latest_info = {}
        self.episode_number = 0
        self.frame_number = 0
        self.last_frame_metadata = []

    def _remember(self, observation, info):
        rgb = observation["rgb"]
        if tuple(rgb.shape[1:]) != (1, 3, 224, 224) or rgb.dtype != torch.uint8:
            raise ValueError("Video expects ACT inputs [N,1,3,224,224] uint8")
        self.policy_rgb = rgb.detach().clone()
        self.latest_info = info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.episode_number += 1
        self.frame_number = 0
        self._remember(observation, info)
        return observation, info

    def step(self, action):
        result = self.env.step(action)
        self.frame_number += 1
        self._remember(result[0], result[-1])
        return result

    def render(self):
        if self.policy_rgb is None:
            raise RuntimeError("Reset before rendering single RGB video")
        base = self.unwrapped
        scene_images = common.to_numpy(self.env.render())
        if scene_images.ndim == 3:
            scene_images = scene_images[None]
        streams = common.to_numpy(self.policy_rgb).transpose(0, 1, 3, 4, 2)
        camera = base._human_render_cameras["render_camera"].get_params()
        extrinsics = common.to_numpy(camera["extrinsic_cv"])
        intrinsics = common.to_numpy(camera["intrinsic_cv"])
        grasped = np.stack([common.to_numpy(base.agent.is_grasping(cube)) for cube in base.cubes], axis=1)
        targets = common.to_numpy(base.target_idx)
        poses = np.stack([common.to_numpy(cube.pose.p) for cube in base.cubes], axis=1)
        labels = getattr(base, "object_labels", tuple(f"cube {i + 1}" for i in range(len(base.cubes))))
        frames = []
        self.last_frame_metadata = []
        for env_index in range(len(streams)):
            target = int(targets[env_index])
            holding = np.flatnonzero(grasped[env_index]).tolist()
            wrong = any(index != target for index in holding)
            target_grasped = target in holding
            lifted = bool(common.to_numpy(self.latest_info["target_lifted"]).reshape(-1)[env_index])
            success = bool(common.to_numpy(self.latest_info["success"]).reshape(-1)[env_index])
            holding_text = ", ".join(labels[i] for i in holding) if holding else "none"
            status = "WRONG OBJECT" if wrong else "TARGET HELD" if target_grasped else "NO OBJECT HELD"
            gold, green, red = (255, 210, 60), (80, 230, 140), (255, 95, 95)
            status_color = red if wrong else green if target_grasped else (205, 215, 230)
            canvas = Image.new("RGB", (896, 640), (19, 24, 33))
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 10), f"Episode {self.episode_number:03d} | Frame {self.frame_number:04d} | Target: {labels[target]} (target_idx={target})", font=font(24), fill=gold)
            draw.text((16, 45), f"Holding: {holding_text} | {status} | target lifted: {int(lifted)} | success: {int(success)}", font=font(22), fill=status_color)
            titles = ["Scene: numbered object identities", self.policy_title]
            for index, title in enumerate(titles):
                draw.text((index * 448 + 12, 110), title, font=font(19), fill="white")

            # Preserve arbitrary scene aspect ratios when placing projected labels.
            scene = Image.fromarray(scene_images[env_index]).convert("RGB")
            scale = min(448 / scene.width, 448 / scene.height)
            size = (round(scene.width * scale), round(scene.height * scale))
            offset = ((448 - size[0]) // 2, (448 - size[1]) // 2)
            canvas.paste(scene.resize(size, Image.Resampling.BILINEAR), (offset[0], 144 + offset[1]))
            projected = []
            for index in range(len(base.cubes)):
                point = intrinsics[env_index] @ (extrinsics[env_index] @ np.r_[poses[env_index, index], 1])
                if point[2] <= 0:
                    projected.append(None)
                    continue
                x, y = point[:2] / point[2]
                x = float(x * size[0] / scene.width + offset[0])
                y = float(y * size[1] / scene.height + offset[1] + 144)
                projected.append((x, y))
                if not (0 <= x < 448 and 144 <= y < 592):
                    continue
                color = gold if index == target else (230, 235, 245)
                radius = 13 if index == target else 9
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=3)

            # Separate numbered badges prevent overlap when cubes are close together.
            anchors = []
            for index, position in enumerate(projected):
                if position is None:
                    continue
                x, y = position
                if not (0 <= x < 448 and 144 <= y < 592):
                    continue
                ax = max(24, min(424, x))
                ay = max(166, min(570, y - 32))
                while any(abs(ax-px) < 42 and abs(ay-py) < 38 for px, py in anchors):
                    ay = ay - 40 if ay > 206 else ay + 80
                anchors.append((ax, ay))
                color = red if index in holding and index != target else green if index in holding else gold if index == target else (230, 235, 245)
                draw.line((x, y, ax, ay), fill=color, width=2)
                draw.ellipse((ax-17, ay-17, ax+17, ay+17), fill=(19, 24, 33), outline=color, width=3)
                draw.text((ax, ay), str(index + 1), font=font(23), fill=color, anchor="mm")

            # Display exact 224x224 policy pixels at 2x nearest-neighbor scale.
            # No labels or outlines are painted onto the policy input panel.
            for stream in range(1):
                panel = Image.fromarray(streams[env_index, stream]).resize((448, 448), Image.Resampling.NEAREST)
                canvas.paste(panel, ((stream + 1) * 448, 144))
            draw.text((14, 607), "Gold circle = target | Green badge = held target | Red badge = held distractor", font=font(16), fill=(205, 215, 230))
            self.last_frame_metadata.append(dict(target_idx=target, holding=holding, wrong_object_grasped=wrong,
                                                frame=self.frame_number, episode=self.episode_number))
            frames.append(np.asarray(canvas))
        return np.stack(frames)
