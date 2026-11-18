"""Pilot raw trajectories; all storage stays in the approved project root."""
import json,time,hashlib,uuid
from pathlib import Path
import h5py,numpy as np
from robocasa_bridge.dataset import OBS_KEYS
from robocasa_bridge.paths import ROOT
DATA=ROOT/'data/robocasa/pilot_v1'
OUTPUT=ROOT/'outputs/robocasa/pilot_v1'
def write_json(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Path outside approved project')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
    tmp.replace(path)
class Recorder:
    def __init__(self,engine,eid,extra):
        self.id=eid;self.directory=DATA/'episodes'/eid;self.output=OUTPUT/'episodes'/eid
        self.directory.mkdir(parents=True,exist_ok=False);self.output.mkdir(parents=True,exist_ok=False)
        self.metadata={**engine.metadata(),**engine.ensure_compiled_model(),**extra,
            'schema':'mask_policy.robocasa.pilot.raw.v1','episode_id':eid,
            'source':extra.get('source','scripted_expert_privileged_state'),'task':'single_target_full_pick_place',
            'target_id':engine.order[0],'mask_source':extra.get('mask_source','sim_gt_offline_annotation'),
            'alignment':'action[t] maps observation[t] to observation[t+1]',
            'created_at':time.time(),'policy_input_state_dim':23,'policy_action_dim':7}
        self.metadata['pilot_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
        write_json(self.directory/'metadata.json',self.metadata)
        engine.save_snapshot(self.directory/'initial_snapshot.json')
        self.h5=h5py.File(self.directory/'trajectory.h5','w')
        self.h5.attrs['metadata_json']=json.dumps(self.metadata);self.h5.attrs['schema']=self.metadata['schema']
        self.obs=self.h5.create_group('observations');self.actions=self.h5.create_group('actions')
        self.length=0;self.obs_length=0;self.video=None
        import imageio.v2 as imageio
        self.video=imageio.get_writer(str(self.output/'execution.mp4'),fps=5,codec='libx264',quality=7,macro_block_size=16)
        self.append_obs(engine.observation());self.video.append_data(engine.frame_array())
    def _append(self,group,key,value):
        arr=np.asarray(value,dtype=h5py.string_dtype('utf-8')) if isinstance(value,str) else np.asarray(value)
        if key not in group:
            kwargs={'compression':'lzf'} if arr.ndim>=2 else {}
            group.create_dataset(key,shape=(0,*arr.shape),maxshape=(None,*arr.shape),dtype=arr.dtype,chunks=(1,*arr.shape),**kwargs)
        ds=group[key];ds.resize(ds.shape[0]+1,axis=0);ds[-1]=arr
    def append_obs(self,row):
        for key in OBS_KEYS:self._append(self.obs,key,row[key])
        self.obs_length+=1
    def update_context(self,row):
        for key in ('gt_target_mask_offline','subtask_index','target_id','stage'):self.obs[key][-1]=row[key]
    def append_transition(self,native,arm,row,engine):
        self._append(self.actions,'native',np.asarray(native,np.float32));self._append(self.actions,'arm_gripper',np.asarray(arm,np.float32))
        self.append_obs(row);self.length+=1
        if self.length%4==0:self.video.append_data(engine.frame_array())
        if self.length%20==0:self.h5.flush()
    def close(self,engine,error=None):
        pixels=self.obs['gt_target_mask_offline'][:].sum(axis=(1,2));states=self.obs['robot_state'][:];actions=self.actions['arm_gripper'][:]
        aligned=self.obs_length==self.length+1
        qc=dict(alignment=aligned,finite=bool(np.isfinite(states).all() and np.isfinite(actions).all()),
            initial_target_pixels=int(pixels[0]),min_target_pixels=int(pixels.min()),max_target_pixels=int(pixels.max()),
            empty_mask_fraction=float((pixels==0).mean()),action_bound_ok=bool(np.abs(actions).max()<=1.00001) if len(actions) else False)
        score=engine.scoring()
        accepted=bool(score['simulator_success'] and engine.lift_verified and aligned and qc['finite'] and qc['action_bound_ok'] and pixels[0]>=20 and error is None)
        summary=dict(episode_id=self.id,source=self.metadata['source'],status='completed' if accepted else 'failed',
            training_ready=bool(accepted and self.metadata['source']=='scripted_expert_privileged_state'),steps=self.length,observations=self.obs_length,error=error,
            split=self.metadata['split'],object_name=self.metadata['object_name'],seed=self.metadata['seed'],
            lift_verified=engine.lift_verified,qc=qc,**score)
        self.h5.flush();self.h5.close();self.video.close()
        write_json(self.directory/'summary.json',summary);write_json(self.output/'summary.json',summary)
        return summary
