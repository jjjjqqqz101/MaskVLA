"""Read-only replay of completed pilot rollouts with exact geometry v2 scoring.

No environment, renderer, expert, controller, stepping, or raw-data writes are
used. MuJoCo forward reconstructs kinematics and the recorded contact state.
The legacy results and immutable trajectory files are never overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from .geometry import (
    SCORING_VERSION, LIFT_THRESHOLD_M, STABLE_STEPS, collision_bounds,
    geometry_conditions, stable_placement, _id,
)
from .protocol import DATA_ROOT, OUTPUT_ROOT, inside_root
from .recorder import write_json


@dataclass
class Recording:
    metadata: dict
    summary: dict
    model: object
    qpos: np.ndarray
    qvel: np.ndarray
    actions: np.ndarray
    states: np.ndarray
    object_body_id: int
    tray_body_id: int
    object_qpos_id: int
    object_dof_id: int
    eef_site_id: int
    contact_geom_ids: tuple
    contact_geom_names: tuple
    fingerpad_ids: tuple


def load_recording(episode_id):
    """Load a closed, hash-verified pilot recording without opening images."""
    if Path(episode_id).name != episode_id:
        raise ValueError("Expected an episode ID, not a path")
    directory = inside_root(DATA_ROOT / "episodes" / episode_id)
    metadata = json.loads((directory / "metadata.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if metadata["episode_id"] != episode_id or summary["episode_id"] != episode_id:
        raise ValueError("Episode identity mismatch")
    if metadata.get("robot") != "PandaOmron" or metadata.get("distractor_count") != 0:
        raise NotImplementedError("Offline fingerpad scoring supports this fixed PandaOmron single-target pilot only")
    model_path = inside_root(metadata["compiled_model_file"])
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["compiled_model_sha256"]:
        raise ValueError("Compiled model hash mismatch")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    with h5py.File(directory / "trajectory.h5", "r") as source:
        qpos = source["observations/qpos"][:]
        qvel = source["observations/qvel"][:]
        actions = source["actions/arm_gripper"][:]
        states = source["observations/robot_state"][:]
    steps = len(actions)
    if (qpos.shape != (steps + 1, model.nq) or qvel.shape != (steps + 1, model.nv)
            or actions.shape != (steps, 7) or states.shape != (steps + 1, 23)
            or summary["steps"] != steps or steps < 1):
        raise ValueError("Recording alignment/shape mismatch")
    if not all(np.isfinite(x).all() for x in (qpos, qvel, actions, states)):
        raise ValueError("Nonfinite recording")
    if not np.isin(actions[:, 6], [-1., 1.]).all():
        raise ValueError("Expected recorded binary gripper commands")
    target = metadata["target_id"]
    joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, target + "_joint0")
    if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Target must have a free joint")
    object_body = int(model.jnt_bodyid[joint])
    tray_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "collection_tray_joint0")
    tray_body = int(model.jnt_bodyid[tray_joint])
    # The compiled body's contact-enabled descendant geoms are exactly the
    # fixed pilot object's collision geoms; visual/region geoms have both masks 0.
    descendants = {object_body}
    for body in range(object_body + 1, model.nbody):
        if int(model.body_parentid[body]) in descendants:
            descendants.add(body)
    contact_ids = tuple(g for g in range(model.ngeom)
                        if int(model.geom_bodyid[g]) in descendants
                        and (model.geom_contype[g] or model.geom_conaffinity[g]))
    contact_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                          for g in contact_ids)
    if not contact_ids or any(n is None or not n.startswith(target + "_") for n in contact_names):
        raise ValueError("Ambiguous target collision geometry")
    # Match robosuite ManipulationEnv._check_grasp, which uses fingerpad groups,
    # rather than merely any collision with either entire finger.
    pads = tuple(_id(model, mujoco.mjtObj.mjOBJ_GEOM, "gripper0_right_finger%d_pad_collision" % i)
                 for i in (1, 2))
    return Recording(metadata, summary, model, qpos, qvel, actions, states,
                     object_body, tray_body, int(model.jnt_qposadr[joint]),
                     int(model.jnt_dofadr[joint]),
                     _id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site"),
                     contact_ids, contact_names, pads)


def _bilateral_contact(data, contact_geom_ids, fingerpad_ids):
    target_ids = set(contact_geom_ids)
    touching = set()
    for contact in data.contact:
        a, b = int(contact.geom1), int(contact.geom2)
        if a in target_ids and b in fingerpad_ids:
            touching.add(b)
        if b in target_ids and a in fingerpad_ids:
            touching.add(a)
    return all(pad in touching for pad in fingerpad_ids)


def score_recording(recording):
    """Recompute lift, release, clearance, containment and 10-frame stability."""
    r = recording
    model = r.model
    data = mujoco.MjData(model)
    data.qpos[:] = r.qpos[0]
    data.qvel[:] = r.qvel[0]
    mujoco.mj_forward(model, data)
    initial_z = float(data.xpos[r.object_body_id, 2])
    lift_verified = False
    first_lift_frame = None
    lift_max = 0.
    stable_steps = max_stable = 0
    first_success_frame = None
    for frame in range(1, len(r.actions) + 1):
        data.qpos[:] = r.qpos[frame]
        data.qvel[:] = r.qvel[frame]
        mujoco.mj_forward(model, data)
        target_position = data.xpos[r.object_body_id]
        eef_position = data.site_xpos[r.eef_site_id]
        if np.max(np.abs(eef_position - r.states[frame, 14:17])) > 2e-6:
            raise ValueError("Recorded EEF and reconstructed qpos are misaligned")
        lift = float(target_position[2] - initial_z)
        lift_max = max(lift_max, lift)
        bilateral = _bilateral_contact(data, r.contact_geom_ids, r.fingerpad_ids)
        if lift > LIFT_THRESHOLD_M and bilateral:
            lift_verified = True
            if first_lift_frame is None:
                first_lift_frame = frame
        lower, upper = collision_bounds(model, data, r.contact_geom_ids, r.tray_body_id)
        distance = float(np.linalg.norm(eef_position - target_position))
        conditions = geometry_conditions(lower, upper, distance)
        speed = float(np.linalg.norm(r.qvel[frame, r.object_dof_id:r.object_dof_id + 3]))
        gripper = float(r.actions[frame - 1, 6])
        stable = stable_placement(conditions, speed, gripper)
        stable_steps = stable_steps + 1 if stable else 0
        max_stable = max(max_stable, stable_steps)
        if first_success_frame is None and lift_verified and stable_steps >= STABLE_STEPS:
            first_success_frame = frame
    return dict(
        scoring_version=SCORING_VERSION,
        simulator_success=bool(first_success_frame is not None and r.summary.get("error") is None),
        lift_verified=lift_verified,
        target_lift_max_m=lift_max,
        stable_placement_steps=stable_steps,
        max_stable_placement_steps=max_stable,
        first_success_frame=first_success_frame,
        first_bilateral_lift_frame=first_lift_frame,
        terminal=dict(
            collision_bounds_tray_min_m=lower.tolist(),
            collision_bounds_tray_max_m=upper.tolist(),
            geometry_conditions=conditions,
            eef_distance_m=distance,
            object_linear_speed_m_s=speed,
            gripper_command=gripper,
            bilateral_fingerpad_contact=bilateral,
        ),
        contact_geom_names=list(r.contact_geom_names),
        contact_geom_source="compiled_target_body_contact_enabled_descendant_geoms",
        privileged_offline_only=True,
        thresholds=dict(xy_m=[.143, .108], bottom_m=[-.006, .035],
                        upper_m=.27, eef_clearance_m=.07, linear_speed_m_s=.035,
                        bilateral_lift_m=.06, stable_steps=10),
    )


def rescore_episode(episode_id):
    return score_recording(load_recording(episode_id))


def rescore_results(results_path):
    path = inside_root(results_path)
    original_bytes = path.read_bytes()
    original = json.loads(original_bytes)
    rows = []
    for row in original["rollouts"]:
        score = rescore_episode(row["episode_id"])
        changed = dict(row)
        changed["original_scoring"] = {
            k: row.get(k) for k in (
                "simulator_success", "lift_verified", "target_lift_max_m",
                "stable_placement_steps", "scoring_source", "status",
            )
        }
        for key in ("simulator_success", "lift_verified", "target_lift_max_m", "stable_placement_steps"):
            changed[key] = score[key]
        changed["status"] = "completed" if score["simulator_success"] else "failed"
        changed["scoring_source"] = "simulator_privileged_offline_geometry_v2"
        changed["geometry_v2"] = score
        rows.append(changed)
    return dict(
        schema="robocasa.pilot.geometry_v2.results.v1",
        scoring_version=SCORING_VERSION,
        original_results=str(path),
        original_results_sha256=hashlib.sha256(original_bytes).hexdigest(),
        health=original.get("health"),
        rollouts=rows,
        total=len(rows),
        success_count=sum(row["simulator_success"] for row in rows),
        original_success_count=sum(row["original_scoring"]["simulator_success"] for row in rows),
        no_raw_or_legacy_result_writes=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = inside_root(args.results)
    output = inside_root(args.output)
    if output == source or not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError("Write a new geometry v2 file under the pilot outputs directory")
    if not output.name.endswith("_geometry_v2_results.json"):
        raise ValueError("Output must end with _geometry_v2_results.json")
    if output.exists():
        raise FileExistsError("Preserve existing results: " + str(output))
    report = rescore_results(source)
    write_json(output, report)
    print(json.dumps(dict(output=str(output), total=report["total"],
                          success_count=report["success_count"],
                          original_success_count=report["original_success_count"])), flush=True)


if __name__ == "__main__":
    main()
