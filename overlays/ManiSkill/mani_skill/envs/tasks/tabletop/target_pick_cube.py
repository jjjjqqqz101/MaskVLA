"""Three identical cubes with an episode-specific target, without visual prompts."""

from typing import Optional

import sapien
import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("TargetPickCube-v1", max_episode_steps=100)
class TargetPickCubeEnv(PickCubeEnv):
    """Move the designated one of three identical red cubes to the goal.

    Every reset samples non-overlapping cube positions, yaw angles, a goal, and
    an independent target index. Visual observations contain no target prompt.
    Target identity is available in info and privileged state observations, and
    is recorded in the trajectory environment state for faithful replay.

    Success means the target is within goal_thresh of the goal, the robot is
    static, and no other cube is currently grasped. Grasp/lift metrics describe
    the current step, not whether they occurred earlier in the episode.
    """

    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    _sample_video_link = None
    num_cubes = 3

    def __init__(self, *args, lift_height=0.05, **kwargs):
        if lift_height <= 0:
            raise ValueError("lift_height must be positive")
        self.lift_height = float(lift_height)
        super().__init__(*args, **kwargs)

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
                initial_pose=sapien.Pose(
                    p=[
                        self.cube_spawn_center[0],
                        self.cube_spawn_center[1] + (i - 1) * 0.08,
                        self.cube_half_size,
                    ]
                ),
            )
            for i in range(self.num_cubes)
        ]
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

    @property
    def min_cube_center_distance(self):
        # Circumscribed xy circles remain separated at every sampled yaw.
        return 2 * (2**0.5) * self.cube_half_size + 0.01

    def _sample_cube_xy(self, batch_size):
        """Rejection sample complete layouts; never return an invalid fallback."""
        xy = torch.empty((batch_size, self.num_cubes, 2), device=self.device)
        remaining = torch.arange(batch_size, device=self.device)
        center = torch.tensor(self.cube_spawn_center, device=self.device)
        pairs = torch.triu_indices(
            self.num_cubes, self.num_cubes, 1, device=self.device
        )
        for _ in range(1000):
            candidate = (
                torch.rand((len(remaining), self.num_cubes, 2), device=self.device)
                * (2 * self.cube_spawn_half_size)
                - self.cube_spawn_half_size
                + center
            )
            distances = torch.linalg.norm(
                candidate[:, pairs[0]] - candidate[:, pairs[1]], dim=-1
            )
            valid = (distances > self.min_cube_center_distance).all(dim=1)
            xy[remaining[valid]] = candidate[valid]
            remaining = remaining[~valid]
            if len(remaining) == 0:
                return xy
        raise RuntimeError("Could not sample separated cubes; enlarge the spawn region")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xy = self._sample_cube_xy(b)
            for i, cube in enumerate(self.cubes):
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = xy[:, i]
                xyz[:, 2] = self.cube_half_size
                q = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                cube.set_pose(Pose.create_from_pq(p=xyz, q=q))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * (2 * self.cube_spawn_half_size)
                - self.cube_spawn_half_size
                + torch.tensor(self.cube_spawn_center)
            )
            goal_xyz[:, 2] = torch.rand(b) * self.max_goal_height + self.cube_half_size
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz))
            self.target_idx[env_idx] = torch.randint(self.num_cubes, (b,))

    def _select_target(self, values: torch.Tensor):
        return values[torch.arange(self.num_envs, device=self.device), self.target_idx]

    @property
    def target_pose(self):
        return Pose.create(
            self._select_target(torch.stack([c.pose.raw_pose for c in self.cubes], dim=1))
        )

    def evaluate(self):
        grasped = torch.stack([self.agent.is_grasping(c) for c in self.cubes], dim=1)
        target_mask = (
            torch.arange(self.num_cubes, device=self.device)[None]
            == self.target_idx[:, None]
        )
        is_target_grasped = self._select_target(grasped)
        wrong_object_grasped = (grasped & ~target_mask).any(dim=1)
        target_pos = self.target_pose.p
        target_lifted = target_pos[:, 2] > self.cube_half_size + self.lift_height
        is_obj_placed = (
            torch.linalg.norm(target_pos - self.goal_site.pose.p, dim=1)
            <= self.goal_thresh
        )
        is_robot_static = self.agent.is_static(0.2)
        success = is_obj_placed & is_robot_static & ~wrong_object_grasped
        return dict(
            success=success,
            full_success=success,
            is_target_grasped=is_target_grasped,
            target_lifted=target_lifted,
            wrong_object_grasped=wrong_object_grasped,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            target_idx=self.target_idx.clone(),
        )

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose, goal_pos=self.goal_site.pose.p)
        if self.obs_mode_struct.use_state:
            obs.update(
                cube_poses=torch.stack(
                    [c.pose.raw_pose for c in self.cubes], dim=1
                ).flatten(1),
                target_idx=self.target_idx[:, None].clone(),
                target_pose=self.target_pose.raw_pose,
            )
        return obs

    def get_state_dict(self):
        state = super().get_state_dict()
        # Keep a feature dimension for the recorder and replay batching code.
        state["target_idx"] = self.target_idx[:, None].clone()
        return state

    def set_state_dict(self, state: dict, env_idx: Optional[torch.Tensor] = None):
        if "target_idx" not in state:
            raise ValueError(
                "TargetPickCube replay requires dictionary states containing target_idx"
            )
        physical_state = dict(state)
        target_idx = torch.as_tensor(
            physical_state.pop("target_idx"), device=self.device
        ).reshape(-1)
        indices = (
            torch.arange(self.num_envs, device=self.device)
            if env_idx is None
            else torch.as_tensor(
                env_idx, device=self.device, dtype=torch.long
            ).reshape(-1)
        )
        if len(target_idx) != len(indices):
            raise ValueError("target_idx must have one value per restored environment")
        if not torch.all(
            (target_idx >= 0)
            & (target_idx < self.num_cubes)
            & (target_idx == target_idx.long())
        ):
            raise ValueError("target_idx must contain integers in {0, 1, 2}")
        super().set_state_dict(physical_state, env_idx)
        self.target_idx[indices] = target_idx.long()
