"""Eight objects per episode, sampled from 25 training or 15 held-out targets."""

import json

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.envs.tasks.tabletop.target_pick_eight import TargetPickEightEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


TRAIN_OBJECTS = (
    "red_cube_1", "red_cube_2", "blue_cube", "003_cracker_box", "025_mug",
    "011_banana", "054_softball", "055_baseball", "002_master_chef_can",
    "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle",
    "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box",
    "010_potted_meat_can", "013_apple", "014_lemon", "015_peach",
    "016_pear", "017_orange", "018_plum", "026_sponge",
    "036_wood_block", "061_foam_brick",
)
HELDOUT_OBJECTS = (
    "012_strawberry", "024_bowl", "038_padlock", "040_large_marker",
    "050_medium_clamp", "053_mini_soccer_ball", "056_tennis_ball",
    "057_racquetball", "058_golf_ball", "065-a_cups", "065-d_cups",
    "070-a_colored_wood_blocks", "071_nine_hole_peg_test",
    "073-a_lego_duplo", "077_rubiks_cube",
)
ALL_OBJECTS = TRAIN_OBJECTS + HELDOUT_OBJECTS
assert len(TRAIN_OBJECTS) == 25 and len(HELDOUT_OBJECTS) == 15
assert len(set(ALL_OBJECTS)) == 40

SCALE_OVERRIDES = {"025_mug": 0.70, "054_softball": 0.76, "055_baseball": 0.58}


@register_env("TargetPickForty-v1", max_episode_steps=500, asset_download_ids=["ycb"])
class TargetPickFortyEnv(TargetPickEightEnv):
    """Preserve the eight-object table, with target-conditioned random contents."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    num_cubes = 8

    def _load_scene(self, options: dict):
        split = options.get("object_split", "train")
        if split not in ("train", "heldout"):
            raise ValueError("object_split must be train or heldout")
        pool = TRAIN_OBJECTS if split == "train" else HELDOUT_OBJECTS
        target_name = options.get("target_name")
        if target_name is None:
            target_name = str(self._episode_rng.choice(pool))
        if target_name not in pool:
            raise ValueError(f"Target {target_name} is not in the {split} split")

        # Collection defaults to training-only distractors. Interactive tests may
        # explicitly include held-out distractors without changing that default.
        include_heldout = options.get("include_heldout_distractors", False)
        if not isinstance(include_heldout, bool):
            raise ValueError("include_heldout_distractors must be bool")
        pool_for_distractors = ALL_OBJECTS if split == "heldout" or include_heldout else TRAIN_OBJECTS
        distractor_pool = tuple(x for x in pool_for_distractors if x != target_name)
        if split == "train" and include_heldout:
            # The UI's "include unseen" switch guarantees at least one unseen distractor.
            required_unseen = str(self._episode_rng.choice(HELDOUT_OBJECTS))
            remaining = tuple(x for x in distractor_pool if x != required_unseen)
            distractors = [required_unseen, *self._episode_rng.choice(remaining, size=6, replace=False).tolist()]
        else:
            distractors = self._episode_rng.choice(distractor_pool, size=7, replace=False).tolist()
        self.selected_names = tuple(self._episode_rng.permutation([target_name, *distractors]).tolist())
        self.selected_split = split
        self.selected_target_name = target_name
        self.selected_target_idx = self.selected_names.index(target_name)

        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        metadata = json.loads((ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").read_text())
        self.cubes = []
        self.object_rest_z = []
        for name in self.selected_names:
            if name in ("red_cube_1", "red_cube_2", "blue_cube"):
                color = [0.1, 0.25, 1, 1] if name == "blue_cube" else [1, 0, 0, 1]
                obj = actors.build_cube(
                    self.scene, half_size=self.cube_half_size, color=color,
                    name=name, initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]))
                rest_z = self.cube_half_size
            else:
                model = metadata[name]
                scale = SCALE_OVERRIDES.get(name, model.get("scales", [1.0])[0])
                model_dir = ASSET_DIR / "assets/mani_skill2_ycb/models" / name
                builder = self.scene.create_actor_builder()
                builder.add_multiple_convex_collisions_from_file(
                    filename=str(model_dir / "collision.ply"), scale=[scale] * 3,
                    density=model.get("density", 1000))
                builder.add_visual_from_file(filename=str(model_dir / "textured.obj"), scale=[scale] * 3)
                rest_z = -model["bbox"]["min"][2] * scale
                builder.initial_pose = sapien.Pose(p=[0, 0, rest_z])
                obj = builder.build(name=name)
            self.cubes.append(obj)
            self.object_rest_z.append(rest_z)

        self.goal_site = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[0, 1, 0, 1],
            name="goal_site", body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0.2]))
        self._hidden_objects.append(self.goal_site)
        self.target_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        local_options = dict(options, target_idx=self.selected_target_idx)
        super()._initialize_episode(env_idx, local_options)
