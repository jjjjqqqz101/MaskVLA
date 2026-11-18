"""Regression checks against genuine recorded mesh geometry and failure cases.

All counterfactual translations/actions below modify private in-memory arrays
only. They are not simulation rollouts and are never written to raw data.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from .geometry import (
    collision_bounds, collision_bounds_in_tray, geometry_conditions,
    stable_placement, SCORING_VERSION,
)
from .geometry_rescore import load_recording, score_recording, rescore_results
from .protocol import DATA_ROOT, OUTPUT_ROOT, inside_root
from .recorder import write_json


def frame_geometry(recording, frame):
    r = recording
    data = mujoco.MjData(r.model)
    data.qpos[:] = r.qpos[frame]
    data.qvel[:] = r.qvel[frame]
    mujoco.mj_forward(r.model, data)
    env = SimpleNamespace(
        sim=SimpleNamespace(model=SimpleNamespace(_model=r.model),
                            data=SimpleNamespace(_data=data)),
        obj_body_id={"collection_tray": r.tray_body_id},
        objects={r.metadata["target_id"]: SimpleNamespace(contact_geoms=r.contact_geom_names)},
    )
    lower, upper = collision_bounds_in_tray(env, r.metadata["target_id"])
    distance = float(np.linalg.norm(data.site_xpos[r.eef_site_id] - data.xpos[r.object_body_id]))
    conditions = geometry_conditions(lower, upper, distance)
    speed = float(np.linalg.norm(r.qvel[frame, r.object_dof_id:r.object_dof_id + 3]))
    gripper = float(r.actions[frame - 1, 6])
    return data, dict(frame=frame, minimum=lower.tolist(), maximum=upper.tolist(),
                     conditions=conditions, eef_distance=distance, speed=speed,
                     gripper=gripper, stable=stable_placement(conditions, speed, gripper))


def audit_expert_terminal_geometry(manifest_path):
    path = inside_root(manifest_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads(path.read_text())
    rows = manifest["episodes"]
    if len(rows) != 70 or len({r["episode_id"] for r in rows}) != 70:
        raise AssertionError("Expected exactly the frozen 70 unique expert episodes")
    report = []
    for row in rows:
        r = load_recording(row["episode_id"])
        if r.metadata["source"] != "scripted_expert_privileged_state" or not r.summary["training_ready"]:
            raise AssertionError("Unqualified expert in manifest")
        checks = []
        for frame in range(len(r.actions) - 9, len(r.actions) + 1):
            _, check = frame_geometry(r, frame)
            checks.append(check)
        if not all(x["stable"] for x in checks):
            raise AssertionError("Geometry v2 rejected accepted expert terminal stability: " + row["episode_id"])
        report.append(dict(episode_id=row["episode_id"], object=r.metadata["object_name"],
                           split=row["split"], passed_last10_stable=True, terminal=checks[-1]))
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise AssertionError("Manifest changed during read-only audit")
    return dict(schema="robocasa.pilot.expert70.geometry_v2.audit.v1",
                scoring_version=SCORING_VERSION, manifest=str(path),
                manifest_sha256=digest, total=70, passed=70, failed=0,
                scope="terminal exact mesh containment, clearance, open command, speed and 10 consecutive frames; original expert lift qualification is preserved",
                raw_and_manifest_unchanged=True, episodes=report)


def run_regression(results_path):
    source = inside_root(results_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    original = json.loads(source.read_text())
    by_seed = {r["seed"]: r for r in original["rollouts"]}
    if len(by_seed) != 10 or 910000 not in by_seed or 911001 not in by_seed:
        raise AssertionError("Use the full formal3000_ex30 regression batch")
    apple = load_recording(by_seed[910000]["episode_id"])
    apple_checks = []
    for frame in (250, 600):
        data, check = frame_geometry(apple, frame)
        target = apple.metadata["target_id"]
        bbox = mujoco.mj_name2id(apple.model, mujoco.mjtObj.mjOBJ_GEOM, target + "_reg_bbox")
        tray_rot = data.xmat[apple.tray_body_id].reshape(3, 3)
        object_rot = data.xmat[apple.object_body_id].reshape(3, 3)
        center = tray_rot.T @ (data.xpos[apple.object_body_id]
                 + object_rot @ apple.model.geom_pos[bbox] - data.xpos[apple.tray_body_id])
        half = np.abs(tray_rot.T @ object_rot) @ (
            np.asarray(apple.metadata["selected_objects"][0]["bbox_size_m"]) / 2)
        old_bottom = float(center[2] - half[2])
        if not check["stable"] or not old_bottom < -.006 or not check["minimum"][2] > -.006:
            raise AssertionError("Apple OBB false negative not reproduced")
        apple_checks.append(dict(**check, old_obb_bottom=old_bottom))
    # Unsupported shapes must explicitly fail; never silently use an OBB.
    try:
        collision_bounds(apple.model, data, ["collection_tray_bottom"], apple.tray_body_id)
    except NotImplementedError:
        unsupported_rejected = True
    else:
        raise AssertionError("Nonmesh silently accepted")

    genuine = rescore_results(source)
    old_successes = [r for r in genuine["rollouts"] if r["original_scoring"]["simulator_success"]]
    if len(old_successes) != 5 or not all(r["simulator_success"] for r in old_successes):
        raise AssertionError("An original successful banana/can/soap/mug was rejected")
    for seed in (910001, 911000, 913000, 914000):
        result = next(r for r in genuine["rollouts"] if r["seed"] == seed)
        if result["simulator_success"] or result["lift_verified"]:
            raise AssertionError("A genuine failed grasp was accepted")
    banana = load_recording(by_seed[911001]["episode_id"])
    data, _ = frame_geometry(banana, len(banana.actions))
    # Shift the full object trajectory half a metre along the tray's local X.
    outside_qpos = banana.qpos.copy()
    outside_qpos[:, banana.object_qpos_id:banana.object_qpos_id + 3] += (
        data.xmat[banana.tray_body_id].reshape(3, 3) @ np.array([.5, 0., 0.]))
    outside = score_recording(replace(banana, qpos=outside_qpos))
    if outside["simulator_success"] or outside["terminal"]["geometry_conditions"]["xy"]:
        raise AssertionError("An object outside the tray was accepted")
    # A recorded settled pose cannot qualify if every command is closed.
    closed_actions = banana.actions.copy()
    closed_actions[:, 6] = 1.
    closed = score_recording(replace(banana, actions=closed_actions))
    if closed["simulator_success"] or closed["max_stable_placement_steps"] != 0:
        raise AssertionError("A closed-gripper sequence was accepted")
    # Keep the object at its genuine final settled pose throughout: placement
    # still lasts >=10 frames, but there is no >6cm bilateral lift.
    never_lifted_qpos = banana.qpos.copy()
    never_lifted_qpos[:, banana.object_qpos_id:banana.object_qpos_id + 7] = (
        banana.qpos[-1, banana.object_qpos_id:banana.object_qpos_id + 7])
    never_lifted_qvel = banana.qvel.copy()
    never_lifted_qvel[:, banana.object_dof_id:banana.object_dof_id + 6] = 0.
    no_lift = score_recording(replace(banana, qpos=never_lifted_qpos, qvel=never_lifted_qvel))
    if no_lift["simulator_success"] or no_lift["lift_verified"] or no_lift["max_stable_placement_steps"] < 10:
        raise AssertionError("The bilateral lift prerequisite is not enforced")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise AssertionError("Original results changed")
    return dict(schema="robocasa.pilot.geometry_v2.regression.v1", passed=True,
                scoring_version=SCORING_VERSION, original_results=str(source),
                original_results_sha256=digest, total=genuine["total"],
                original_success_count=genuine["original_success_count"],
                exact_geometry_success_count=genuine["success_count"],
                apple_obb_false_negative=apple_checks,
                original_successes_preserved=[dict(object=r["object"], seed=r["seed"])
                                             for r in old_successes],
                genuine_failed_grasps_rejected=[910001, 911000, 913000, 914000],
                outside_tray_rejected=not outside["simulator_success"],
                closed_gripper_rejected=not closed["simulator_success"],
                no_lift_rejected=not no_lift["simulator_success"],
                no_lift_counterexample_stable_steps=no_lift["max_stable_placement_steps"],
                unsupported_nonmesh_rejected=unsupported_rejected,
                legacy_results_unchanged=True, counterfactuals_in_memory_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expert-manifest")
    parser.add_argument("--expert-output")
    args = parser.parse_args()
    output = inside_root(args.output)
    if not output.is_relative_to(OUTPUT_ROOT) or output.exists():
        raise ValueError("Use a new validation output under pilot outputs")
    report = run_regression(args.results)
    write_json(output, report)
    print(json.dumps(dict(regression_passed=report["passed"], output=str(output),
                          old_success=report["original_success_count"],
                          geometry_v2_success=report["exact_geometry_success_count"])), flush=True)
    if args.expert_manifest:
        if not args.expert_output:
            parser.error("--expert-output required with --expert-manifest")
        expert_output = inside_root(args.expert_output)
        if not expert_output.is_relative_to(OUTPUT_ROOT) or expert_output.exists():
            raise ValueError("Use a new expert audit output under pilot outputs")
        audit = audit_expert_terminal_geometry(args.expert_manifest)
        write_json(expert_output, audit)
        print(json.dumps(dict(expert_terminal_audit_passed=audit["passed"],
                              total=audit["total"], output=str(expert_output))), flush=True)


if __name__ == "__main__":
    main()
