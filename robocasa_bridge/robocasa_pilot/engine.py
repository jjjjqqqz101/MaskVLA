"""Single-target pilot; privileged expert separated from policy execution."""
import copy,hashlib,json,threading,time,uuid
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from robocasa_bridge.engine import Engine as BaseEngine
from .recorder import DATA,OUTPUT,Recorder,write_json
class PilotEngine(BaseEngine):
    def __init__(self):
        super().__init__(lambda *a:None,threading.Event())
        self.lift_verified=False;self.lift_max=0.;self._placed_steps=0;self._template=None
    def create(self,request,seed=170001):
        super().scene(dict(seed=seed,objects=[request]))
        self.active_subtask=0
        tray=self.env.objects['collection_tray'];pose=self.env.sim.data.get_joint_qpos(tray.joints[0]).copy()
        pose[:3]-=self.env.counter_x*.16
        self.env.sim.data.set_joint_qpos(tray.joints[0],pose);self.env.sim.data.qvel[:]=0
        self.env.sim.forward();self.latest_obs=self.env._get_observations(force_update=True)
        self._template=copy.deepcopy(self.snapshot())
        self.template_pose=self.env.sim.data.get_joint_qpos(self.env.objects[self.order[0]].joints[0]).copy()
        self.template_tray=self.position('collection_tray').copy()
        self.template_robot=self.eef().copy()
    def reset_trial(self,seed,xy_range=.035,yaw_range=.65):
        if self.record:raise ValueError('Cannot reset while recording')
        self.restore_snapshot(copy.deepcopy(self._template));self.seed=int(seed)
        rng=np.random.default_rng(seed);name=self.order[0];obj=self.env.objects[name]
        pose=self.template_pose.copy()
        delta=rng.uniform(-xy_range,xy_range,2)
        pose[:3]+=self.env.counter_x*delta[0]+self.env.counter_y*delta[1]
        base=Rotation.from_quat(pose[3:][[1,2,3,0]])
        rot=Rotation.from_euler('z',rng.uniform(-yaw_range,yaw_range))*base
        pose[3:]=rot.as_quat()[[3,0,1,2]]
        self.env.sim.data.set_joint_qpos(obj.joints[0],pose);self.env.sim.data.qvel[:]=0
        self.env.sim.forward()
        for c in self._controllers().values():
            if hasattr(c,'update'):c.update(force=True)
            if hasattr(c,'reset_goal'):c.reset_goal()
        self.completed=[];self.order_valid=True;self._stable={};self._placed_steps=0
        self.active_subtask=0;self.frame_id=0;self.stage='initializing';self.lift_verified=False;self.lift_max=0.
        self.gripper=-1.;self._segmentation_cache=None
        native,arm=self.native_action([0,0,0],[0,0,0],-1)
        for _ in range(12):self.step(native,arm)
        self.initial_object_z=float(self.position(name)[2]);self.frame_id=0;self.stage='ready'
        self.latest_obs=self.env._get_observations(force_update=True);self._segmentation_cache=None
        self.initial_rgb_hashes=self.rgb_hashes();self.initial=self.snapshot()
        return self.observation()
    def metadata(self):
        row=super().metadata();row.update(scoring_version='collision_mesh_v2',placement_geometry='compiled target collision vertices in tray coordinates',task='single_target_full_pick_place',distractor_count=0,
            instruction='Pick up the object shown in the target image and place it into the green tray.',
            target_condition='current-frame base RGB multiplied by current-frame target mask',
            initialization='seeded XY/yaw writes only before recording; physical settling; no state writes during expert/policy',
            initial_rgb_sha256=self.rgb_hashes())
        return row
    def _placed(self,name):
        from .geometry import collision_bounds_in_tray,geometry_conditions
        lower,upper=collision_bounds_in_tray(self.env,name)
        distance=float(np.linalg.norm(self.eef()-self.position(name)))
        return all(geometry_conditions(lower,upper,distance).values())
    def update_score(self):
        if not self.env:return
        name=self.order[0]
        if hasattr(self,'initial_object_z'):
            lift=float(self.position(name)[2])-self.initial_object_z;self.lift_max=max(self.lift_max,lift)
            if lift>.06:
                obj=self.env.objects[name]
                self.lift_verified |= bool(self.env._check_grasp(self.env.robots[0].gripper['right'],obj.contact_geoms))
        inside=self._placed(name)
        vel=self.env.sim.data.get_joint_qvel(self.env.objects[name].joints[0])
        stable=inside and np.linalg.norm(vel[:3])<.035 and self.gripper<0
        self._placed_steps=self._placed_steps+1 if stable else 0
        if self._placed_steps>=10:self.completed=[name]
    def scoring(self):
        name=self.order[0]
        success=bool(self.env is not None and self._placed_steps>=10 and self.lift_verified)
        return dict(simulator_success=success,scoring_version='collision_mesh_v2',order_valid=True,ordered_completed=[name] if success else [],
            scoring_source='simulator_privileged_offline_evaluation',stable_placement_steps=self._placed_steps,
            target_lift_max_m=float(self.lift_max),full_policy_action_scope='approach_grasp_lift_transport_release_retreat')
    def step(self,native,arm,publish_frame=False):
        self._check_stop();self.env.step(np.asarray(native,float));self._segmentation_cache=None
        self.env.sim.forward();self.latest_obs=self.env._get_observations(force_update=True)
        self.frame_id+=1;self.update_score()
        if self.record:self.record.append_transition(native,arm,self.observation(),self)
    def start_pilot_record(self,eid,extra):
        self.stage='approach';self.record=Recorder(self,eid,extra)
    def finish_pilot_record(self,error=None):
        r=self.record;self.record=None
        return r.close(self,error)
    def ensure_compiled_model(self):
        if self._model_binary is None:
            import mujoco
            d=DATA/'models';d.mkdir(parents=True,exist_ok=True)
            tmp=d/('.candidate_'+uuid.uuid4().hex+'.mjb');mujoco.mj_saveModel(self.env.sim.model._model,str(tmp),None)
            sha=self._sha256_file(tmp);target=d/(sha+'.mjb')
            if target.exists():tmp.unlink()
            else:tmp.replace(target)
            self._model_binary=(sha,target)
        sha,path=self._model_binary
        return dict(compiled_model_sha256=sha,compiled_model_file=str(path),compiled_model_format='MuJoCo MJB 3.3.1')
    def move(self,goal,gripper,stage,orientation=None,max_steps=100,tolerance=.007):
        self.stage=stage;goal=np.asarray(goal);ori=self.initial_ori if orientation is None else orientation
        hits=0
        for _ in range(max_steps):
            error=goal-self.eef();current=self.env.sim.data.site_xmat[self.env.robots[0].eef_site_id['right']].reshape(3,3)
            rot=Rotation.from_matrix(ori@current.T).as_rotvec()
            native,arm=self.native_action(np.clip(error*.65,-.018,.018),np.clip(rot*.45,-.12,.12),gripper)
            self.gripper=float(gripper);self.step(native,arm)
            hits=hits+1 if np.linalg.norm(error)<tolerance and np.linalg.norm(rot)<.45 else 0
            if hits>=3:return
        raise RuntimeError(f'Waypoint timeout {stage}: pos={np.linalg.norm(error):.4f} rot={np.linalg.norm(rot):.3f}')
    def hold(self,gripper,count,stage):
        self.stage=stage;self.gripper=float(gripper)
        for _ in range(count):
            native,arm=self.native_action([0,0,0],[0,0,0],gripper);self.step(native,arm)
    def expert(self,candidate=0):
        name=self.order[0];obj=self.env.objects[name];info=self.selected_objects[0]
        origin=self.position(name).copy();r=self.env.sim.data.body_xmat[self.env.obj_body_id[name]].reshape(3,3).copy()
        center=origin+r@((np.asarray(obj.bottom_offset)+np.asarray(obj.top_offset))/2)
        axis=int(np.argmin(np.asarray(info['bbox_size_m'])[:2]));closing=r[:,axis].copy();closing[2]=0;closing/=np.linalg.norm(closing)
        z=np.array([0.,0.,-1.])
        choices=[np.stack([x,np.cross(z,x),z],axis=1) for x in (closing,-closing)]
        choices.sort(key=lambda o:np.linalg.norm(Rotation.from_matrix(o@self.initial_ori.T).as_rotvec()))
        ori=choices[candidate%2]
        # Site is calibrated near the finger contact center; candidates vary insertion depth.
        height=max(.006,.5*float(info['bbox_size_m'][2])-.025)
        grasp=center+np.array([0,0,height-(candidate//2)*.004])
        self.move(grasp+[0,0,.16],-1,'approach',ori)
        self.move(grasp,-1,'descend',ori,tolerance=.004)
        self.hold(1,16,'close')
        self.move(grasp+[0,0,.18],1,'lift',ori)
        self.hold(1,6,'lift_verify')
        if not self.lift_verified:raise RuntimeError('Target not lifted with bilateral grasp')
        # Long/handled objects otherwise drive joint7 into its lower limit near the tray.
        carry_ori=ori
        if name.endswith(('_bar_soap','_mug')):
            carry_ori=Rotation.from_euler('z',-np.pi/2).as_matrix()@self.initial_ori
            self.move(self.eef().copy(),1,'reorient',carry_ori,max_steps=150,tolerance=.01)
        # Preserve the physically measured grasp offset when choosing an object-bottom release height.
        rnow=self.env.sim.data.body_xmat[self.env.obj_body_id[name]].reshape(3,3)
        bottom_offset=float((rnow@np.asarray(obj.bottom_offset))[2])
        tray=self.position('collection_tray').copy()
        desired_origin=tray+np.array([0,0,.012-bottom_offset])
        grasp_offset=self.eef()-self.position(name)
        release=desired_origin+grasp_offset
        self.move(release+[0,0,.18],1,'transport',carry_ori,max_steps=150,tolerance=.015)
        self.move(release,1,'place',carry_ori,tolerance=.005)
        self.hold(1,5,'place_settle');self.hold(-1,16,'release')
        self.move(release+[0,0,.20],-1,'retreat',carry_ori,tolerance=.015)
        self.hold(-1,20,'settle')
        if not self.scoring()['simulator_success']:raise RuntimeError('Final stable tray placement was not confirmed')
        return self.scoring()
