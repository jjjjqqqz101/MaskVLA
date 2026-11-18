"""Offline numerical diagnosis of genuine learned-policy raw rollouts.

Privileged qpos/object positions are read only after execution. They are never
fed into policy inference or used to intervene in a rollout.
"""
import argparse
import json
from pathlib import Path
import h5py
import mujoco
import numpy as np
from .protocol import DATA_ROOT, inside_root
from .recorder import write_json

def episode_diagnosis(episode_id, summary=None):
    directory=inside_root(DATA_ROOT/"episodes"/episode_id)
    metadata=json.loads((directory/"metadata.json").read_text())
    if summary is None:
        summary=json.loads((directory/"summary.json").read_text())
    model=mujoco.MjModel.from_binary_path(str(inside_root(metadata["compiled_model_file"])))
    def position_index(name):
        joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name+"_joint0")
        if joint<0 or model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Expected recorded free object joint "+name)
        return int(model.jnt_qposadr[joint])
    target_index=position_index(metadata["target_id"])
    tray_index=position_index("collection_tray")
    with h5py.File(directory/"trajectory.h5","r") as f:
        states=f["observations/robot_state"][:]
        actions=f["actions/arm_gripper"][:]
        qpos=f["observations/qpos"][:]
        pixels=f["observations/gt_target_mask_offline"][:].sum(axis=(1,2))
    target=qpos[:,target_index:target_index+3]
    tray=qpos[:,tray_index:tray_index+3]
    eef=states[:,14:17]
    distances=np.linalg.norm(eef-target,axis=1)
    close=actions[:,6]>0
    changes=np.flatnonzero(np.diff(close.astype(int)))+1
    close_frames=np.flatnonzero(close)
    late=actions[max(0,len(actions)-100):,:6]
    success=bool(summary["simulator_success"])
    diagnosis=("complete_success" if success else
               "lifted_without_stable_placement" if summary["lift_verified"] else
               "no_verified_bilateral_lift")
    return dict(episode_id=episode_id,object=metadata["object_name"],
        source=metadata["source"],steps=len(actions),summary_success=success,
        diagnosis=diagnosis,diagnostic_inputs_are_privileged_offline_only=True,
        mask_source="sim_gt_oracle_for_policy_evaluation",
        lift_verified=bool(summary["lift_verified"]),
        target_lift_max_m=float(summary["target_lift_max_m"]),
        qpos_lift_max_m=float((target[:,2]-target[0,2]).max()),
        stable_placement_steps=int(summary["stable_placement_steps"]),
        target_world_displacement_m=(target[-1]-target[0]).tolist(),
        target_to_tray_final_world_xyz_m=(target[-1]-tray[-1]).tolist(),
        eef_to_object_origin_distance_m=dict(initial=float(distances[0]),minimum=float(distances.min()),
                                            final=float(distances[-1])),
        distance_note="Object free-joint origin is a diagnostic reference, not a calibrated grasp center",
        eef_world_range_m=(eef.max(axis=0)-eef.min(axis=0)).tolist(),
        eef_path_length_m=float(np.linalg.norm(np.diff(eef,axis=0),axis=1).sum()),
        gripper=dict(close_fraction=float(close.mean()) if len(close) else 0.,
                     first_close_frame=int(close_frames[0]) if len(close_frames) else None,
                     switch_frames=changes.tolist()),
        action_saturation_fraction_abs_ge_0_99=np.mean(np.abs(actions[:,:6])>=.99,axis=0).tolist(),
        late100_action_abs_mean=np.abs(late).mean(axis=0).tolist() if len(late) else [],
        mask=dict(initial_pixels=int(pixels[0]),minimum_pixels=int(pixels.min()),
                  final_pixels=int(pixels[-1]),empty_fraction=float((pixels==0).mean())),
        error=summary.get("error"))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    results_path=inside_root(args.results)
    result=json.loads(results_path.read_text())
    rows=[episode_diagnosis(row["episode_id"],row) for row in result["rollouts"]]
    report=dict(schema="robocasa.learned_policy.offline_diagnosis.v1",
        results=str(results_path),checkpoint=result["health"]["checkpoint"],
        total=len(rows),success_count=sum(row["summary_success"] for row in rows),episodes=rows)
    write_json(inside_root(args.output),report)
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    main()
