"""Self-describing raw trajectories: observations[t], action[t], observations[t+1]."""
import json,time,hashlib
from datetime import datetime
import numpy as np
import h5py
from .paths import DATA,OUTPUT,episode_dir

SCHEMA="mask_policy.robocasa.raw.v1"
OBS_KEYS=("base_rgb","wrist_rgb","overview_rgb","robot_state","qpos","qvel",
          "integration_state","gt_target_mask_offline","frame_id","sim_time",
          "timestamp","subtask_index","target_id","stage")
STR_KEYS={"target_id","stage"}
def dump_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x)))

class Recorder:
    def __init__(self,engine,source,label=""):
        self.id="rc_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.directory=episode_dir(self.id);self.output=episode_dir(self.id,True)
        self.directory.mkdir(parents=True);self.output.mkdir(parents=True)
        self.metadata={**engine.metadata(),**engine.ensure_compiled_model()}
        self.metadata.update(schema=SCHEMA,episode_id=self.id,source=source,label=str(label)[:200],
            created_at=time.time(),mask_source="sim_gt_offline_annotation",
            alignment="T native/arm actions, T+1 observations and simulation states. action[t] maps obs[t] to obs[t+1].",
            policies_available=False,training_adapter_status="new RoboCasa training configurations required")
        dump_json(self.directory/"metadata.json",self.metadata)
        (self.directory/"model.xml").write_text(engine.env.sim.model.get_xml())
        engine.save_snapshot(self.directory/"initial_snapshot.json")
        self.h5=h5py.File(self.directory/"trajectory.h5","w")
        self.h5.attrs["schema"]=SCHEMA
        self.h5.attrs["metadata_json"]=json.dumps(self.metadata,ensure_ascii=False)
        self.obs=self.h5.create_group("observations")
        self.actions=self.h5.create_group("actions")
        self.length=0;self.obs_length=0
        import imageio.v2 as imageio
        self.video=imageio.get_writer(str(self.output/"execution.mp4"),fps=20,codec="libx264",
                                     quality=7,macro_block_size=16,ffmpeg_params=["-movflags","+faststart"])
        self.append_obs(engine.observation())
        self.video.append_data(engine.frame_array())

    def _append(self,group,key,value):
        if isinstance(value,str):
            arr=np.asarray(value,dtype=h5py.string_dtype("utf-8"))
        else:arr=np.asarray(value)
        if key not in group:
            shape=arr.shape
            kwargs={"compression":"lzf"} if arr.ndim>=2 else {}
            group.create_dataset(key,shape=(0,*shape),maxshape=(None,*shape),
                                 dtype=arr.dtype,chunks=(1,*shape),**kwargs)
        ds=group[key];ds.resize(ds.shape[0]+1,axis=0);ds[-1]=arr

    def append_obs(self,row):
        for key in OBS_KEYS:self._append(self.obs,key,row[key])
        self.obs_length+=1

    def update_context(self,row):
        for key in ("gt_target_mask_offline","subtask_index","target_id","stage"):
            self.obs[key][-1]=row[key]

    def append_transition(self,native,arm,next_row,frame):
        self._append(self.actions,"native",np.asarray(native,np.float32))
        self._append(self.actions,"arm_gripper",np.asarray(arm,np.float32))
        self.append_obs(next_row);self.video.append_data(frame);self.length+=1
        if self.length%20==0:self.h5.flush()

    def close(self,engine,quality="stopped",note="",error=None):
        self.h5.flush();self.h5.close();self.video.close()
        scoring=engine.scoring()
        summary=dict(episode_id=self.id,status="completed" if scoring["simulator_success"] else "stopped" if quality=="stopped" else "failed",
            source=self.metadata["source"],steps=self.length,observations=self.obs_length,
            user_quality=quality,note=str(note)[:1000],error=error,**scoring,
            training_ready=bool(scoring["simulator_success"] and scoring["order_valid"] and self.length>0),
            raw_schema=SCHEMA,mask_source="sim_gt_offline_annotation",
            video_url=f"/artifacts/{self.id}/execution.mp4",
            raw_url=f"/artifacts/{self.id}/trajectory.h5")
        dump_json(self.directory/"summary.json",summary);dump_json(self.output/"summary.json",summary)
        return summary

def list_episodes():
    result=[]
    root=DATA/"episodes"
    for p in sorted(root.glob("rc_*"),reverse=True)[:200]:
        try:
            m=json.loads((p/"metadata.json").read_text())
            s=json.loads((p/"summary.json").read_text()) if (p/"summary.json").exists() else {}
            result.append(dict(id=p.name,source=m.get("source"),created_at=m.get("created_at"),
                label=m.get("label",""),status=s.get("status","recording"),quality=s.get("user_quality"),steps=s.get("steps",0),
                simulator_success=s.get("simulator_success",False),order_valid=s.get("order_valid",True),
                training_ready=s.get("training_ready",False),can_replay=bool(s and s.get("steps",0)>0 and m.get("compiled_model_sha256")),
                video_url=s.get("video_url"),order=m.get("order"),seed=m.get("seed"),
                task=m.get("task","ordered_counter_to_tray_v1"),selected_objects=m.get("selected_objects",[]),
                scripted_demo_available=m.get("scripted_demo_available",not bool(m.get("selected_objects"))),
                raw_url=s.get("raw_url"),metadata_url=f"/artifacts/{p.name}/metadata.json",
                manifest_url=f"/artifacts/{p.name}/training_manifest.json" if (p/"training_manifest.json").exists() else None,
                replay_states_url=f"/artifacts/{p.name}/replay_states.mp4" if (episode_dir(p.name,True)/"replay_states.mp4").exists() else None,
                replay_actions_url=f"/artifacts/{p.name}/replay_actions.mp4" if (episode_dir(p.name,True)/"replay_actions.mp4").exists() else None))
        except (OSError,ValueError):continue
    return result
