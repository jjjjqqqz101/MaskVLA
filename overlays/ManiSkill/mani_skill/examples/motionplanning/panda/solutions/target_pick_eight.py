"""Motion-planning grasp-and-lift expert for TargetPickEight-v1."""

import numpy as np
import sapien
import mplib

# The isolated data-collection runtime uses mplib 0.2, whose pose API differs
# from the version used by the project's original motion-planning wrapper.
if mplib.__version__.startswith("0.2"):
    _Planner = mplib.Planner

    class _CompatiblePlanner(_Planner):
        @staticmethod
        def _pose(value):
            value = np.asarray(value, dtype=np.float64)
            return mplib.Pose(value[:3], value[3:])

        def set_base_pose(self, pose):
            return super().set_base_pose(self._pose(pose))

        def plan_screw(self, goal_pose, current_qpos, *, use_point_cloud=False, **kwargs):
            return super().plan_screw(self._pose(goal_pose), current_qpos, **kwargs)

        def plan_qpos_to_pose(self, goal_pose, current_qpos, *,
                              use_point_cloud=False, planner_name=None, **kwargs):
            return super().plan_pose(self._pose(goal_pose), current_qpos, **kwargs)

    mplib.Planner = _CompatiblePlanner

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


def solve(env, target_idx, seed, use_obstacle_cloud=True):
    """Return a successful final step tuple, or -1 for a rejected attempt."""
    env.reset(seed=seed, options={"target_idx": target_idx})
    base = env.unwrapped
    if base.num_envs != 1 or base.robot_uids != "panda":
        raise ValueError("Eight-object expert requires one Panda CPU environment")
    target = base.cubes[target_idx]
    planner = PandaArmMotionPlanningSolver(
        env, debug=False, vis=False, base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False, print_env_info=False,
        joint_vel_limits=0.7, joint_acc_limits=0.7)
    try:
        if use_obstacle_cloud:
            for index, obj in enumerate(base.cubes):
                if index == target_idx:
                    continue
                mesh = obj.get_first_collision_mesh()
                if mesh is None:
                    continue
                points = np.asarray(mesh.vertices)
                if len(points) > 300:
                    points = points[np.linspace(0, len(points) - 1, 300, dtype=int)]
                planner.add_collision_pts(points)
        approaching = np.array([0, 0, -1])
        closing = base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            get_actor_obb(target), approaching=approaching,
            target_closing=closing, depth=0.025)
        grasp_pose = base.agent.build_grasp_pose(
            approaching, grasp_info["closing"], grasp_info["center"])
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.10])
        for pose in (reach_pose, grasp_pose):
            result = planner.move_to_pose_with_screw(pose)
            if isinstance(result, int):
                return -1
        result = planner.close_gripper(t=12)
        if not bool(result[-1]["is_target_grasped"].item()):
            return -1
        lift_pose = sapien.Pose([0, 0, 0.12]) * grasp_pose
        result = planner.move_to_pose_with_screw(lift_pose, refine_steps=3)
        if isinstance(result, int):
            return -1
        if not (bool(result[-1]["is_target_grasped"].item())
                and bool(result[-1]["target_lifted"].item())
                and not bool(result[-1]["wrong_object_grasped"].item())):
            return -1
        return result
    finally:
        planner.close()
