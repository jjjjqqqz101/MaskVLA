"""Verified NumPy adapters for new ACT / OpenPI configurations, not old checkpoints."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import h5py
from .paths import episode_dir
from .dataset import dump_json

STREAMS={"act_dual":["base_rgb","base_masked_rgb"],
         "act_triple":["base_rgb","wrist_rgb","base_masked_rgb"]}
def validate_raw(path):
    with h5py.File(path,"r") as f:
        m=json.loads(f.attrs["metadata_json"]);t=len(f["actions/native"])
        if t<1:raise ValueError("Empty trajectories cannot train")
        for name,ds in f["observations"].items():
            if len(ds)!=t+1:raise ValueError(f"Observation/action alignment broken: {name}")
        if len(f["actions/arm_gripper"])!=t:raise ValueError("Action streams differ in length")
        for key in ("base_rgb","wrist_rgb"):
            if f["observations/"+key].shape[1:]!=(224,224,3):raise ValueError("Policy image shape mismatch")
        target_ids=[x.decode() if isinstance(x,bytes) else str(x) for x in f["observations/target_id"][:]]
        indices=f["observations/subtask_index"][:]
        for name,idx in zip(target_ids,indices):
            expected=m["order"][int(idx)] if int(idx)<len(m["order"]) else ""
            if name!=expected:raise ValueError("Subtask target identity is not aligned")
        masks=f["observations/gt_target_mask_offline"]
        for target in set(target_ids[:-1])-{""}:
            frames=np.array([i for i,x in enumerate(target_ids[:-1]) if x==target],int)
            if not any(int(np.count_nonzero(masks[int(i)]))>=10 for i in frames):
                raise ValueError("Visual data quality rejected: target never visible in base mask: "+target)
        states=f["observations/robot_state"][:-1].astype(np.float64)
        actions=f["actions/arm_gripper"][:].astype(np.float64)
        if states.shape[1]!=m["policy_state_dim"] or actions.shape[1]!=7:
            raise ValueError("State/action dimensions disagree with metadata")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():raise ValueError("Nonfinite training data")
        return m,dict(state_mean=states.mean(0),state_std=np.maximum(states.std(0),1e-6),
            action_mean=actions.mean(0),action_std=np.maximum(actions.std(0),1e-6)),t

def batch(path,indices=(0,),kind="act_triple",horizon=30,normalize=True,norm_stats=None):
    m,stats,t=validate_raw(path);indices=list(indices)
    if norm_stats is not None:stats=norm_stats
    if kind not in STREAMS:raise ValueError("Choose act_dual or act_triple")
    if not indices or any(type(i) is not int or not 0<=i<t for i in indices):raise ValueError("Invalid sample indices")
    images=[];states=[];actions=[];pads=[]
    with h5py.File(path,"r") as f:
        for i in indices:
            base=f["observations/base_rgb"][i];wrist=f["observations/wrist_rgb"][i]
            mask=f["observations/gt_target_mask_offline"][i].astype(bool)
            masked=np.where(mask[...,None],base,0).astype(np.uint8)
            streams={"base_rgb":base,"wrist_rgb":wrist,"base_masked_rgb":masked}
            images.append(np.stack([streams[k].transpose(2,0,1) for k in STREAMS[kind]]))
            states.append(f["observations/robot_state"][i])
            current=int(f["observations/subtask_index"][i])
            if current>=len(m["order"]):
                raise ValueError("Terminal observations without an active target are not training samples")
            stop=min(t,i+horizon)
            # An action chunk belongs to one target skill. Never supervise blue
            # actions while the conditioning image still masks red.
            contexts=f["observations/subtask_index"][i:stop]
            boundaries=np.flatnonzero(contexts!=current)
            if len(boundaries):stop=i+int(boundaries[0])
            chunk=f["actions/arm_gripper"][i:stop]
            n=len(chunk);pad=np.arange(horizon)>=n
            chunk=np.concatenate([chunk,np.repeat(chunk[-1:],horizon-n,axis=0)],axis=0)
            actions.append(chunk);pads.append(pad)
    state=np.asarray(states,np.float32);action=np.asarray(actions,np.float32)
    if normalize:
        state=(state-stats["state_mean"])/stats["state_std"]
        action=(action-stats["action_mean"])/stats["action_std"]
    return dict(rgb=np.asarray(images,np.uint8),state=state.astype(np.float32),
                actions=action.astype(np.float32),is_pad=np.asarray(pads,bool)),m,stats

def openpi_sample(path,index=0,horizon=10):
    # Uses the local ManiSkill adapter's key naming only, not its robot schema.
    # A new OpenPI config must set this robot's state/action dimensions and stats.
    data,m,stats=batch(path,[index],"act_triple",horizon,normalize=False)
    rgb=data["rgb"][0].transpose(0,2,3,1)
    with h5py.File(path,"r") as f:
        target=f["observations/target_id"][index]
        target=target.decode() if isinstance(target,bytes) else str(target)
    label=next((x["name"] for x in m.get("selected_objects",[]) if x["id"]==target),target)
    prompt=f"Pick up the {label.replace('_',' ')} and place it into the green tray."
    return {"observation/image":rgb[0],"observation/wrist_image":rgb[1],
        "observation/target_image":rgb[2],"observation/state":data["state"][0],
        "prompt":prompt,"actions":data["actions"][0],"action_is_pad":data["is_pad"][0]},m,stats

def prepare(eid):
    p=episode_dir(eid);path=p/"trajectory.h5";m,stats,t=validate_raw(path)
    with h5py.File(path,"r") as f:
        valid=np.flatnonzero(f["observations/subtask_index"][:-1]<len(m["order"]))
    if not len(valid):raise ValueError("No active-target training samples")
    samples=[int(valid[0]),int(valid[-1])]
    dual,_,_=batch(path,samples,"act_dual",30)
    triple,_,_=batch(path,samples,"act_triple",30)
    pi,_,_=openpi_sample(path,int(valid[0]),10)
    with h5py.File(path,"r") as f:
        # Native full actions must match the documented fixed-base arm mapping.
        native=f["actions/native"][:]
        parts=m["action_contract"]["native_parts"]
        arm=f["actions/arm_gripper"][:]
        a,b=parts["right"];g0,g1=parts["right_gripper"]
        if not np.allclose(native[:,a:b],arm[:,:6]) or not np.allclose(native[:,g0:g1],arm[:,6:7]):
            raise ValueError("Native/policy action mapping mismatch")
        for name in ("base","torso"):
            a,b=parts[name]
            if not np.allclose(native[:,a:b],0):raise ValueError("A fixed-base demonstration contains base/torso commands")
        if not np.allclose(native[:,-1],-1):raise ValueError("Fixed-base demonstrations must stay in arm control mode")
    for value in (dual["state"],dual["actions"],triple["state"],triple["actions"],pi["actions"]):
        if not np.isfinite(value).all():raise ValueError("Adapter produced nonfinite values")
    np.savez_compressed(p/"training_sample.npz",dual_rgb=dual["rgb"],triple_rgb=triple["rgb"],
        state=triple["state"],actions=triple["actions"],is_pad=triple["is_pad"],
        pi_base=pi["observation/image"],pi_wrist=pi["observation/wrist_image"],
        pi_target=pi["observation/target_image"],pi_state=pi["observation/state"],pi_actions=pi["actions"])
    manifest=dict(schema="mask_policy.robocasa.training_manifest.v1",episode_id=eid,
        raw_schema=m["schema"],raw_file="trajectory.h5",raw_sha256=sha256_file(path),
        selected_objects=m.get("selected_objects",[]),task=m.get("task"),
        steps=t,active_target_training_steps=int(len(valid)),observations=t+1,state_dim=m["policy_state_dim"],action_dim=7,
        state_fields=m["robot_state_fields"],action_contract=m["action_contract"],
        masks=dict(source="sim_gt_offline_annotation",single_current_target=True,
                   use="offline training annotation; not SAM2/Cutie inference"),
        normalization={k:v.tolist() for k,v in stats.items()},
        normalization_scope="per_episode_preview_only; use fit_training_stats on TRAIN episodes for real multi-episode training",
        cross_subtask_action_chunks=False,
        act_dual=dict(stream_order=STREAMS["act_dual"],validated_batch_shape=list(dual["rgb"].shape)),
        act_triple=dict(stream_order=STREAMS["act_triple"],validated_batch_shape=list(triple["rgb"].shape)),
        pi05=dict(validated_state_shape=list(pi["observation/state"].shape),
                  validated_actions_shape=list(pi["actions"].shape),
                  lerobot_exported=False,
                  status="NumPy/OpenPI sample adapter verified; new robot-specific OpenPI config and target-version dataset conversion still required",
                  incompatible_with="Existing ManiSkill 8D state/action checkpoint config"),
        alignment="obs[t],action[t],obs[t+1]; is_pad excludes final repeated horizon padding",
        valid_for_new_training_only=True,weights_trained=False,adapter_validation_passed=True)
    dump_json(p/"training_manifest.json",manifest)
    return manifest

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):digest.update(block)
    return digest.hexdigest()

def fit_training_stats(episode_ids):
    states=[];actions=[]
    for eid in episode_ids:
        path=episode_dir(eid)/"trajectory.h5"
        meta,_,_=validate_raw(path)
        with h5py.File(path,"r") as f:
            valid=f["observations/subtask_index"][:-1]<len(meta["order"])
            states.append(f["observations/robot_state"][:-1][valid])
            actions.append(f["actions/arm_gripper"][:][valid])
    state=np.concatenate(states);action=np.concatenate(actions)
    return dict(state_mean=state.mean(0),state_std=np.maximum(state.std(0),1e-6),
                action_mean=action.mean(0),action_std=np.maximum(action.std(0),1e-6))

def main():
    parser=argparse.ArgumentParser();parser.add_argument("episode_id");args=parser.parse_args()
    result=prepare(args.episode_id);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
