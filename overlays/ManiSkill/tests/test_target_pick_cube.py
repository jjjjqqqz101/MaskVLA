"""Run with python -m unittest tests.test_target_pick_cube.

Set TARGET_PICK_BACKEND=physx_cuda for the batched GPU checks.
"""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs
from mani_skill.utils.wrappers import RecordEpisode


class TargetPickCubeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = os.environ.get("TARGET_PICK_BACKEND", "physx_cpu")
        cls.num_envs = 8 if cls.backend == "physx_cuda" else 1
        cls.env = gym.make(
            "TargetPickCube-v1",
            num_envs=cls.num_envs,
            sim_backend=cls.backend,
            render_backend="none",
            obs_mode="state_dict",
            control_mode="pd_joint_pos",
        )
        cls.base = cls.env.unwrapped

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=7)

    def test_random_layouts_and_seed_reproducibility(self):
        self.assertEqual([c.name for c in self.base.cubes], ["cube_0", "cube_1", "cube_2"])
        seen = set()
        previous = None
        for seed in range(30):
            _, info = self.env.reset(seed=seed)
            seen.update(info["target_idx"].tolist())
            poses = torch.stack([c.pose.p for c in self.base.cubes], dim=1)
            self.assertTrue(torch.allclose(poses[..., 2], torch.full_like(poses[..., 2], 0.02)))
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                dist = torch.linalg.norm(poses[:, i, :2] - poses[:, j, :2], dim=1)
                self.assertTrue((dist > self.base.min_cube_center_distance).all())
            self.assertTrue((poses[..., :2].abs() <= 0.1).all())
            if previous is not None:
                self.assertFalse(torch.equal(previous, poses))
            previous = poses.clone()
        self.assertEqual(seen, {0, 1, 2})
        self.env.reset(seed=41)
        saved = copy.deepcopy(self.base.get_state_dict())
        self.env.reset(seed=41)
        torch.testing.assert_close(saved["target_idx"], self.base.target_idx[:, None])
        for name, value in saved["actors"].items():
            torch.testing.assert_close(value, self.base.get_state_dict()["actors"][name])

    def test_state_restore_and_partial_reset(self):
        saved = copy.deepcopy(self.base.get_state_dict())
        self.env.reset(seed=99)
        self.base.set_state_dict(saved)
        restored = self.base.get_state_dict()
        torch.testing.assert_close(restored["target_idx"], saved["target_idx"])
        self.assertIn("target_idx", saved)  # setter must not mutate its input
        for name, value in saved["actors"].items():
            torch.testing.assert_close(value, restored["actors"][name])
        if self.num_envs > 1:
            indices = torch.tensor([1, 4], device=self.base.device)
            untouched = torch.tensor([0, 2, 3, 5, 6, 7], device=self.base.device)
            self.env.reset(seed=88, options={"env_idx": indices})
            after = self.base.get_state_dict()
            torch.testing.assert_close(after["target_idx"][untouched], saved["target_idx"][untouched])
            for name in saved["actors"]:
                torch.testing.assert_close(after["actors"][name][untouched], saved["actors"][name][untouched])
            def subset(value):
                return {k: subset(v) for k, v in value.items()} if isinstance(value, dict) else value[indices]
            self.base.set_state_dict(subset(saved), indices)
            torch.testing.assert_close(self.base.get_state_dict()["target_idx"], saved["target_idx"])
            for name in saved["actors"]:
                torch.testing.assert_close(self.base.get_state_dict()["actors"][name], saved["actors"][name])

    def test_invalid_target_restore_is_rejected(self):
        state = copy.deepcopy(self.base.get_state_dict())
        for bad in [-1, 3, 0.5]:
            invalid = dict(state)
            invalid["target_idx"] = torch.full((self.num_envs, 1), bad, device=self.base.device)
            with self.assertRaises(ValueError):
                self.base.set_state_dict(invalid)
        with self.assertRaisesRegex(ValueError, "dictionary states"):
            self.base.set_state_dict({k: v for k, v in state.items() if k != "target_idx"})

    def test_target_selection_and_success_metrics(self):
        # Inject contact outcomes to check all targets, wrong grasps, and simultaneous grasps.
        for target in range(3):
            state = copy.deepcopy(self.base.get_state_dict())
            state["target_idx"].fill_(target)
            state["actors"][f"cube_{target}"][:, :3] = torch.tensor([0.0, 0.0, 0.15], device=self.base.device)
            state["actors"]["goal_site"][:, :3] = state["actors"][f"cube_{target}"][:, :3]
            self.base.set_state_dict(state)
            for held in [set(), {target}, {(target + 1) % 3}, {target, (target + 1) % 3}]:
                def grasped(cube):
                    return torch.full((self.num_envs,), int(cube.name.rsplit("_", 1)[1]) in held, dtype=torch.bool, device=self.base.device)
                with patch.object(self.base.agent, "is_grasping", side_effect=grasped), patch.object(
                    self.base.agent, "is_static", return_value=torch.ones(self.num_envs, dtype=torch.bool, device=self.base.device)
                ):
                    info = self.base.evaluate()
                wrong = bool(held - {target})
                self.assertTrue((info["is_target_grasped"] == (target in held)).all())
                self.assertTrue((info["wrong_object_grasped"] == wrong).all())
                self.assertTrue(info["target_lifted"].all())
                self.assertTrue((info["success"] == (not wrong)).all())
                self.assertTrue(torch.equal(info["success"], info["full_success"]))

    def test_trajectory_records_and_restores_target(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RecordEpisode(self.env, directory, save_video=False, save_trajectory=True, trajectory_name="check")
            _, info = recorder.reset(seed=4)
            target = info["target_idx"].clone()
            action = torch.cat([self.base.agent.robot.qpos[:, :7], torch.ones((self.num_envs, 1), device=self.base.device)], dim=1)
            for _ in range(3):
                recorder.step(action)
            recorder.flush_trajectory()
            with h5py.File(Path(directory) / "check.h5", "r") as data:
                for i in range(self.num_envs):
                    trajectory = data[f"traj_{i}"]
                    ids = trajectory["env_states/target_idx"][:]
                    self.assertEqual(ids.shape, (4, 1))
                    np.testing.assert_array_equal(ids, np.full((4, 1), target[i].item()))
                def initial(group):
                    return {k: initial(v) for k, v in group.items()} if isinstance(group, h5py.Group) else group[0:1]
                state = initial(data["traj_0/env_states"])
            self.env.reset(seed=23)
            indices = torch.tensor([0], device=self.base.device)
            self.base.set_state_dict(state, indices)
            self.assertEqual(self.base.target_idx[0].item(), target[0].item())
            # Explicitly release the recorder file without closing the shared environment.
            recorder._h5_file.close()


if __name__ == "__main__":
    unittest.main()
