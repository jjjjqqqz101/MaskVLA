"""Eight target objects for balanced ACT demonstration collection."""

import json

import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.envs.tasks.tabletop.target_pick_cube_ycb import TargetPickCubeYCBEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("TargetPickEight-v1", max_episode_steps=500, asset_download_ids=["ycb"])
class TargetPickEightEnv(TargetPickCubeYCBEnv):
    """Two red cubes, one blue cube, three YCB goods and two sized YCB balls."""

    num_cubes = 8  # Includes the five non-cube actors for inherited target logic.
    object_labels = ("红方块 1", "红方块 2", "蓝方块", "饼干盒", "杯子", "香蕉",
                     "大球（垒球）", "小球（棒球）")
    model_specs = (("003_cracker_box", None), ("025_mug", 0.70),
                   ("011_banana", None), ("054_softball", 0.76),
                   ("055_baseball", 0.58))
    # Eight of nine cells are used per episode. Shuffle actor-to-cell assignment
    # and add bounded jitter; all objects change positions without intersecting.
    layout_slots = ((-0.08, -0.205), (-0.08, 0.0), (-0.08, 0.205),
                    (0.05, -0.205), (0.05, 0.0), (0.05, 0.205),
                    (0.18, -0.205), (0.18, 0.0), (0.18, 0.205))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.cubes = []
        for index in range(3):
            color = [0.1, 0.25, 1, 1] if index == 2 else [1, 0, 0, 1]
            self.cubes.append(actors.build_cube(
                self.scene, half_size=self.cube_half_size, color=color,
                name=f"cube_{index}",
                initial_pose=sapien.Pose(p=[0, (index - 1) * 0.08, self.cube_half_size])))

        metadata = json.loads((ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").read_text())
        self.object_rest_z = [self.cube_half_size] * 3
        for model_id, override_scale in self.model_specs:
            model = metadata[model_id]
            scale = override_scale if override_scale is not None else model.get("scales", [1.0])[0]
            model_dir = ASSET_DIR / "assets/mani_skill2_ycb/models" / model_id
            builder = self.scene.create_actor_builder()
            builder.add_multiple_convex_collisions_from_file(
                filename=str(model_dir / "collision.ply"), scale=[scale] * 3,
                density=model.get("density", 1000))
            builder.add_visual_from_file(filename=str(model_dir / "textured.obj"), scale=[scale] * 3)
            rest_z = -model["bbox"]["min"][2] * scale
            builder.initial_pose = sapien.Pose(p=[0, 0, rest_z])
            self.cubes.append(builder.build(name=model_id))
            self.object_rest_z.append(rest_z)

        self.goal_site = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[0, 1, 0, 1],
            name="goal_site", body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0.2]))
        self._hidden_objects.append(self.goal_site)
        self.target_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            slots = torch.tensor(self.layout_slots, dtype=torch.float32)
            placements = torch.stack([slots[torch.randperm(len(slots))[:self.num_cubes]]
                                      for _ in range(b)])
            placements += (torch.rand_like(placements) - 0.5) * 0.018
            for index, obj in enumerate(self.cubes):
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = placements[:, index]
                xyz[:, 2] = self.object_rest_z[index]
                quat = (randomization.random_quaternions(b, lock_x=True, lock_y=True)
                        if index < 3 else None)
                obj.set_pose(Pose.create_from_pq(p=xyz, q=quat))
            requested_target = options.get("target_idx")
            if requested_target is None:
                self.target_idx[env_idx] = torch.randint(self.num_cubes, (b,))
            else:
                if not 0 <= int(requested_target) < self.num_cubes:
                    raise ValueError("target_idx must be in 0..7")
                self.target_idx[env_idx] = int(requested_target)
            # Keep the legacy ACT state shape, but grasp-and-lift has no goal site.
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 2] = 0.2
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz))

    def evaluate(self):
        metrics = super().evaluate()
        success = (metrics["is_target_grasped"] & metrics["target_lifted"]
                   & ~metrics["wrong_object_grasped"])
        metrics["success"] = success
        metrics["full_success"] = success
        return metrics
