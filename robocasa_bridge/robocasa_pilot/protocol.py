"""Frozen robot/image/action contract shared by pilot collection and policies."""
from pathlib import Path

from robocasa_bridge.paths import ROOT

PROJECT = ROOT
PILOT_VERSION = "pilot_v1"
SCHEMA = "mask_policy.robocasa.pilot.v1"
STATE_DIM = 23
ACTION_DIM = 7
CONTROL_HZ = HZ = 20
IMAGE_SIZE = 224
IMAGE_SHAPE = (224, 224, 3)
STREAMS = ("base_rgb", "wrist_rgb", "base_masked_rgb")
STATE_FIELDS = ("arm_joint_pos[7]", "arm_joint_vel[7]", "eef_world_position[3]",
                "eef_world_quaternion_xyzw[4]", "gripper_joint_pos[2]")
ACTION_FIELDS = ("dx_normalized", "dy_normalized", "dz_normalized", "drot_x_normalized",
                 "drot_y_normalized", "drot_z_normalized", "gripper_close")
ACTION_CONTRACT = {"type": "OSC_POSE", "input_type": "delta", "reference_frame": "world",
                   "position_scale_m": .05, "rotation_scale_rad": .5,
                   "gripper_open": -1., "gripper_close": 1., "policy_action_dim": ACTION_DIM}
PROMPT = TARGET_PROMPT = "Pick up the object shown in the target image and place it into the green tray."
DATA_ROOT = PROJECT / "data/robocasa/pilot_v1"
OUTPUT_ROOT = PROJECT / "outputs/robocasa/pilot_v1"
CHECKPOINT_ROOT = PROJECT / "ckpt/robocasa/pilot_v1"
MASK_SOURCE = "sim_gt_offline_annotation"
TARGET_IMAGE_SEMANTICS = "current base_rgb multiplied by current same-frame target_mask; not first-frame reference"
TASK = "single_target_full_pick_place"


def inside_root(value):
    """Reject escaping paths, including symlinks; never create directories."""
    path = Path(value).expanduser().resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("All pilot paths must remain inside MASK_POLICY_ROOT")
    return path


def protocol_dict():
    return dict(schema=SCHEMA, pilot_version=PILOT_VERSION, task=TASK,
                state_dim=STATE_DIM, action_dim=ACTION_DIM, control_hz=CONTROL_HZ,
                state_fields=list(STATE_FIELDS), action_fields=list(ACTION_FIELDS),
                action_contract=dict(ACTION_CONTRACT), image_size=IMAGE_SIZE,
                streams=list(STREAMS), mask_source=MASK_SOURCE, prompt=PROMPT,
                target_image_semantics=TARGET_IMAGE_SEMANTICS,
                success="target grasp, lift, place inside tray, release and stable physical settling; no wrong-object grasp",
                distractors="optional; never required to be moved",
                alignment="obs[t],action[t],obs[t+1]; terminal observation excluded from training starts")


def validate_observation(obs):
    import numpy as np
    for name in ("base_rgb", "wrist_rgb"):
        image = np.asarray(obs[name])
        if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
            raise ValueError(name + " must be uint8[224,224,3]")
    mask = np.asarray(obs.get("target_mask", obs.get("gt_target_mask_offline")))
    if mask.shape != IMAGE_SHAPE[:2]:
        raise ValueError("target_mask must be [224,224]")
    state = np.asarray(obs.get("robot_state", obs.get("state")), dtype=np.float32)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError("robot_state must be finite float[23]")
    return state, mask.astype(bool)
