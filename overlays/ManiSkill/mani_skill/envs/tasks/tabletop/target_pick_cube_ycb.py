"""Six selectable targets: three training-style cubes and three YCB objects.

This is an evaluation-only scene. The original TargetPickCube-v1 environment
and its 450 demonstration trajectories are left untouched.
"""

import json

import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.envs.tasks.tabletop.target_pick_cube import TargetPickCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("TargetPickCubeYCB-v1", max_episode_steps=500, asset_download_ids=["ycb"])
class TargetPickCubeYCBEnv(TargetPickCubeEnv):
    """Keep the ACT observation contract while adding selectable YCB objects."""

    model_ids = ("003_cracker_box", "025_mug", "011_banana")
    object_labels = ("cube 1", "cube 2", "cube 3", "饼干盒", "杯子", "香蕉")
    num_cubes = 6
    ycb_xy = ((-0.20, -0.08), (-0.18, 0.18), (0.04, -0.20))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubes = [
            actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=[1, 0, 0, 1],
                name=f"cube_{i}",
                initial_pose=sapien.Pose(p=[0, (i - 1) * 0.08, self.cube_half_size]),
            )
            for i in range(3)
        ]
        metadata_path = ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
        metadata = json.loads(metadata_path.read_text())
        self.object_rest_z = [self.cube_half_size] * 3
        for model_id, xy in zip(self.model_ids, self.ycb_xy):
            model = metadata[model_id]
            scale = model.get("scales", [1.0])[0]
            rest_z = -model["bbox"]["min"][2] * scale
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
            builder.initial_pose = sapien.Pose(p=[xy[0], xy[1], rest_z])
            self.cubes.append(builder.build(name=model_id))
            self.object_rest_z.append(rest_z)
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0.2]),
        )
        self._hidden_objects.append(self.goal_site)
        self.target_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _sample_three_cubes(self, batch_size):
        # The central cube region leaves room for the three larger YCB models.
        positions = torch.empty((batch_size, 3, 2), device=self.device)
        remaining = torch.arange(batch_size, device=self.device)
        pairs = torch.triu_indices(3, 3, 1, device=self.device)
        for _ in range(1000):
            candidate = torch.rand((len(remaining), 3, 2), device=self.device)
            candidate[..., 0] = candidate[..., 0] * 0.14 - 0.06
            candidate[..., 1] = candidate[..., 1] * 0.12 - 0.06
            distances = torch.linalg.norm(candidate[:, pairs[0]] - candidate[:, pairs[1]], dim=-1)
            valid = (distances > self.min_cube_center_distance).all(dim=1)
            positions[remaining[valid]] = candidate[valid]
            remaining = remaining[~valid]
            if not len(remaining):
                return positions
        raise RuntimeError("Could not sample three non-overlapping cubes")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xy = self._sample_three_cubes(b)
            for i, cube in enumerate(self.cubes[:3]):
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = xy[:, i]
                xyz[:, 2] = self.cube_half_size
                quat = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                cube.set_pose(Pose.create_from_pq(p=xyz, q=quat))
            for i, obj in enumerate(self.cubes[3:]):
                xyz = torch.tensor((*self.ycb_xy[i], self.object_rest_z[i + 3])).repeat(b, 1)
                obj.set_pose(Pose.create_from_pq(p=xyz))
            self.target_idx[env_idx] = torch.randint(self.num_cubes, (b,))
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.16 - 0.08
            rest_z = torch.tensor(self.object_rest_z)[self.target_idx[env_idx]]
            goal_xyz[:, 2] = rest_z + 0.08 + torch.rand(b) * 0.18
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz))

    def evaluate(self):
        grasped = torch.stack([self.agent.is_grasping(obj) for obj in self.cubes], dim=1)
        target_mask = torch.arange(self.num_cubes, device=self.device)[None] == self.target_idx[:, None]
        target_pos = self.target_pose.p
        rest_z = torch.tensor(self.object_rest_z, device=self.device)[self.target_idx]
        target_lifted = target_pos[:, 2] > rest_z + self.lift_height
        wrong = (grasped & ~target_mask).any(dim=1)
        placed = torch.linalg.norm(target_pos - self.goal_site.pose.p, dim=1) <= self.goal_thresh
        static = self.agent.is_static(0.2)
        success = placed & static & ~wrong
        return dict(success=success, full_success=success,
                    is_target_grasped=self._select_target(grasped),
                    target_lifted=target_lifted, wrong_object_grasped=wrong,
                    is_obj_placed=placed, is_robot_static=static,
                    target_idx=self.target_idx.clone())
