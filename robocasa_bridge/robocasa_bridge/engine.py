"""Single-thread-owned RoboCasa simulation, contact demonstrations and replay."""
import io,json,time,hashlib,os,copy,re,uuid
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .paths import OUTPUT, DATA, ROBOCASA_ROOT, ROBOSUITE_ROOT, episode_dir
from .scene import make_env,validate_order,OBJECTS
from .catalog import resolve_selection,validate_resolved
from .dataset import Recorder,dump_json,list_episodes

CAMERAS={"base_rgb":"policy_base","wrist_rgb":"robot0_eye_in_hand","overview_rgb":"task_overview"}
CONTROLLER_MEMORY=("goal_pos","goal_ori","goal_qpos","goal_vel","goal_qvel",
    "initial_joint","initial_ref_pos","initial_ref_ori_mat","origin_pos","origin_ori",
    "_goal_update_mode")
class Stopped(RuntimeError):pass

class Engine:
    def __init__(self,publish,stop_event):
        self.env=None;self.record=None;self.publish_callback=publish;self.stop_event=stop_event
        self.active_subtask=None;self.frame_id=0;self.stage="idle";self.seed=17;self.order=list(OBJECTS)
        self.completed=[];self.order_valid=True;self._stable={};self._inside_prev={}
        self.initial=None;self.initial_rgb_hashes=None;self.last_summary=None;self.last_replay=None
        self._model_binary=None;self._segmentation_cache=None;self.rendering={};self.latest_obs={};self.gripper=-1.
        self.scene_info=None;self.geometry={};self.selected_objects=[]

    def close(self):
        if self.record:self.finish_record("stopped","Simulation closed")
        if self.env:
            self.env.close();self.env=None

    def _check_stop(self):
        if self.stop_event.is_set():raise Stopped("User requested stop")

    def integration_state(self):
        import mujoco
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        model=self.env.sim.model._model;data=self.env.sim.data._data
        out=np.empty(mujoco.mj_stateSize(model,spec))
        mujoco.mj_getState(model,data,out,spec)
        return out

    def _controllers(self):
        return self.env.robots[0].composite_controller.part_controllers

    def snapshot(self):
        c={}
        for name,obj in self._controllers().items():
            vals={}
            for key in CONTROLLER_MEMORY:
                val=getattr(obj,key,None)
                if val is not None:vals[key]=val if isinstance(val,str) else np.asarray(val).tolist()
            c[name]=vals
        grippers={a:np.asarray(g.current_action).tolist() for a,g in self.env.robots[0].gripper.items()}
        return dict(integration_state=self.integration_state().tolist(),controllers=c,grippers=grippers,
            frame_id=self.frame_id,env_timestep=int(self.env.timestep),cur_time=float(self.env.cur_time),
            rng=copy.deepcopy(self.env.rng.bit_generator.state),completed=list(self.completed),
            order_valid=self.order_valid,stage=self.stage,gripper=self.gripper,
            active_subtask=self.active_subtask,stable=dict(self._stable),inside_prev=dict(self._inside_prev))

    def save_snapshot(self,path):dump_json(path,self.snapshot())

    def restore_snapshot(self,snap):
        self._segmentation_cache=None
        import mujoco
        mujoco.mj_setState(self.env.sim.model._model,self.env.sim.data._data,
                          np.asarray(snap["integration_state"],float),mujoco.mjtState.mjSTATE_INTEGRATION)
        self.env.sim.forward()
        for controller in self._controllers().values():
            if hasattr(controller,"update"):controller.update(force=True)
            if hasattr(controller,"reset_goal"):controller.reset_goal()
        for name,vals in snap.get("controllers",{}).items():
            if name in self._controllers():
                baseline=getattr(self,"_controller_baseline",{}).get(name,{})
                merged={**baseline,**vals}
                for key,value in merged.items():
                    setattr(self._controllers()[name],key,value if isinstance(value,str) else np.asarray(value,float))
        for arm,value in snap.get("grippers",{}).items():
            self.env.robots[0].gripper[arm].current_action=np.asarray(value,float)
        self.frame_id=int(snap["frame_id"]);self.env.timestep=int(snap.get("env_timestep",0))
        self.env.cur_time=float(snap.get("cur_time",self.env.sim.data.time))
        if snap.get("rng"):self.env.rng.bit_generator.state=snap["rng"]
        self.completed=list(snap.get("completed",[]));self.order_valid=bool(snap.get("order_valid",True))
        self.active_subtask=snap.get("active_subtask")
        self._stable=dict(snap.get("stable",{}));self._inside_prev=dict(snap.get("inside_prev",{}))
        self.stage=snap.get("stage","idle");self.gripper=float(snap.get("gripper",-1))
        self.env.sim.forward()
        self.latest_obs=self.env._get_observations(force_update=True)

    def scene(self,params):
        if self.record:raise ValueError("Stop and save the recording before changing scenes")
        if params.get("restore"):
            if self.env is None or self.initial is None:raise ValueError("Generate a scene first")
            self.restore_snapshot(self.initial)
            hashes=self.rgb_hashes()
            if hashes!=self.initial_rgb_hashes:raise RuntimeError("Initial RGB verification failed after scene restore")
            self.scene_info.update(restored=True,initial_verified=True)
            # Snapshot stages describe recorded context, not the lifecycle of
            # a newly prepared scene with no active Recorder.
            self.stage="scene_preview"
            self.publish();return
        reuse=params.get("reuse_episode")
        meta=None
        if reuse:
            meta=json.loads((episode_dir(reuse)/"metadata.json").read_text())
            params=dict(seed=meta["seed"],order=meta["order"],layout_id=meta["layout_id"],style_id=meta["style_id"])
        seed=int(params.get("seed",17))
        if not 0<=seed<2**32:raise ValueError("seed must be unsigned 32-bit")
        if reuse:
            selected=validate_resolved(meta["selected_objects"]) if meta.get("selected_objects") else []
        elif params.get("preset")=="classic2" or "objects" not in params:
            selected=[]
        else:
            selected=resolve_selection(params["objects"],seed)
        targets=[x["id"] for x in selected] if selected else list(OBJECTS)
        order=validate_order(params.get("order"),targets)
        if self.env:self.env.close()
        self.env=None;self._model_binary=None;self._segmentation_cache=None
        self.seed=seed;self.selected_objects=selected;self.order=order
        self.completed=[];self.order_valid=True;self._stable={};self._inside_prev={}
        self.active_subtask=None;self.frame_id=0;self.stage="scene_preview";self.gripper=-1.
        layout=int(params.get("layout_id",os.environ.get("ROBOCASA_LAYOUT_ID","11")))
        style=int(params.get("style_id",os.environ.get("ROBOCASA_STYLE_ID","1")))
        self.env=make_env(self.seed,self.order,layout,style,selected_objects=self.selected_objects)
        self.env.sim.forward()
        self.latest_obs=self.env._get_observations(force_update=True)
        self._controllers()["right"].update(force=True)
        self._controller_baseline=self.snapshot()["controllers"]
        self.initial_ori=np.array(self.env.sim.data.site_xmat[self.env.robots[0].eef_site_id["right"]]).reshape(3,3).copy()
        self._build_geom_ids()
        ctx=self.env.sim._render_context_offscreen
        ctx.gl_ctx.make_current()
        from OpenGL import GL
        def text(v):return v.decode() if isinstance(v,bytes) else str(v)
        self.rendering=dict(backend="MuJoCo EGL",vendor=text(GL.glGetString(GL.GL_VENDOR)),
            renderer=text(GL.glGetString(GL.GL_RENDERER)),version=text(GL.glGetString(GL.GL_VERSION)))
        self.rendering["gpu_verified"]="NVIDIA" in self.rendering["vendor"] and "llvmpipe" not in self.rendering["renderer"].lower()
        # Do not retain an old context across MJB replacement: robosuite's
        # destructor frees GL resources in the CURRENT context. A late local
        # reference would destroy new-context texture/buffer IDs on return.
        del ctx
        if not self.rendering["gpu_verified"]:raise RuntimeError("RoboCasa requires verified NVIDIA GPU rendering")
        self.scene_info=dict(seed=self.seed,order=list(self.order),layout_id=layout,style_id=style,
            robot="PandaOmron",control_freq=20,restored=bool(reuse),initial_verified=bool(reuse),
            task=self.env.task_mode,counter=self.env.counter.name,
            selected_objects=copy.deepcopy(self.selected_objects),
            scripted_demo_available=not bool(self.selected_objects),
            scripted_demo_reason=None if not self.selected_objects else "脚本专家仅针对默认红蓝方块验证；官方物品场景请手动控制和采集",
            instruction=("按顺序将 "+", ".join(x.get("name_zh",x["id"]) for x in self.selected_objects)+" 放入绿色托盘") if self.selected_objects else
                "按指定顺序将红色和蓝色方块放入绿色托盘",
            fixed_base=True,policy_state_dim=int(self.observation()["robot_state"].size),
            policy_action_dim=7,native_action_dim=int(self.env.action_dim),rendering=dict(self.rendering))
        if reuse:
            directory=episode_dir(reuse)
            compiled_sha=meta.get("compiled_model_sha256","")
            if not re.fullmatch(r"[0-9a-f]{64}",compiled_sha):
                raise ValueError("This historical diagnostic episode has no exact compiled model; collect a new episode for deterministic replay")
            binary=DATA/"models"/(compiled_sha+".mjb")
            if binary.is_symlink() or not binary.is_file():
                raise ValueError("Compiled replay model is missing")
            if self._sha256_file(binary)!=compiled_sha:
                raise ValueError("Compiled replay model checksum mismatch")
            import mujoco
            from robosuite.utils.binding_utils import MjSim
            self.env.close()
            self.env.sim=MjSim(mujoco.MjModel.from_binary_path(str(binary)))
            self.env.initialize_time(self.env.control_freq)
            self.env.deterministic_reset=True
            try:self.env.reset()
            finally:self.env.deterministic_reset=False
            self._model_binary=(compiled_sha,binary)
            self._build_geom_ids()
            self.restore_snapshot(json.loads((directory/"initial_snapshot.json").read_text()))
            expected=meta.get("initial_rgb_sha256")
            if expected and self.rgb_hashes()!=expected:raise RuntimeError("Recorded episode initial RGB mismatch")
        # Episode snapshots can legitimately start during manual recording
        # or a scripted skill. A scene prepared from them is nevertheless idle.
        self.stage="scene_preview"
        self.initial=self.snapshot();self.initial_rgb_hashes=self.rgb_hashes()
        self.scene_info["initial_rgb_sha256"]=dict(self.initial_rgb_hashes)
        self.scene_info["initial_object_poses"]=self.object_poses()
        self.last_summary=None;self.last_replay=None
        self.publish()

    def _build_geom_ids(self):
        # Include precisely one object's subtree, including nested visuals.
        self.geom_ids={}
        model=self.env.sim.model
        for name in self.order:
            bodies={int(self.env.obj_body_id[name])}
            for body in range(model.nbody):
                if int(model.body_parentid[body]) in bodies:bodies.add(body)
            self.geom_ids[name]=np.flatnonzero(np.isin(model.geom_bodyid,list(bodies))).tolist()

    def object_poses(self):
        return {name:np.r_[self.env.sim.data.body_xpos[self.env.obj_body_id[name]],
            self.env.sim.data.body_xquat[self.env.obj_body_id[name]]].tolist()
            for name in (*self.order,"collection_tray")}

    @staticmethod
    def _sha256_file(path):
        digest=hashlib.sha256()
        with Path(path).open("rb") as f:
            for block in iter(lambda:f.read(8*1024*1024),b""):digest.update(block)
        return digest.hexdigest()

    def ensure_compiled_model(self):
        if self._model_binary is None:
            import mujoco
            directory=DATA/"models";directory.mkdir(parents=True,exist_ok=True)
            tmp=directory/(".candidate_"+uuid.uuid4().hex+".mjb")
            mujoco.mj_saveModel(self.env.sim.model._model,str(tmp),None)
            sha=self._sha256_file(tmp);target=directory/(sha+".mjb")
            if target.exists():tmp.unlink()
            else:tmp.replace(target)
            self._model_binary=(sha,target)
        sha,path=self._model_binary
        return dict(compiled_model_sha256=sha,compiled_model_file="models/"+path.name,
                    compiled_model_format="MuJoCo MJB 3.3.1; deduplicated across episodes")

    def _rgb(self,key):
        return np.ascontiguousarray(self.latest_obs[CAMERAS[key]+"_image"],dtype=np.uint8)

    def rgb_hashes(self):
        return {k:hashlib.sha256(self._rgb(k).tobytes()).hexdigest() for k in CAMERAS}

    def subtask_index(self):
        return self.active_subtask if self.active_subtask is not None else len(self.completed)

    def target(self):
        index=self.subtask_index()
        return self.order[index] if 0<=index<len(self.order) else ""

    def observation(self):
        env=self.env;robot=env.robots[0];controller=self._controllers()["right"]
        pos=np.array(env.sim.data.site_xpos[robot.eef_site_id["right"]])
        mat=np.array(env.sim.data.site_xmat[robot.eef_site_id["right"]]).reshape(3,3)
        from scipy.spatial.transform import Rotation
        quat=Rotation.from_matrix(mat).as_quat()
        gripper_qpos=env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]].copy()
        state=np.r_[env.sim.data.qpos[controller.qpos_index],env.sim.data.qvel[controller.qvel_index],
                    pos,quat,gripper_qpos].astype(np.float32)
        row={k:self._rgb(k) for k in CAMERAS}
        # Privileged annotations are explicitly offline. Learned-policy readers
        # whitelist robot_state and RGB; they never consume qpos / sim states.
        if self._segmentation_cache is None:
            raw=env.sim.render(camera_name=CAMERAS["base_rgb"],width=224,height=224,segmentation=True)
            import robosuite.macros as macros
            from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
            self._segmentation_cache=raw[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION],:,1].copy()
        raw=self._segmentation_cache
        name=self.target()
        mask=np.isin(raw,self.geom_ids.get(name,[])).astype(np.uint8) if name else np.zeros((224,224),np.uint8)
        row.update(robot_state=state,qpos=env.sim.data.qpos.copy(),qvel=env.sim.data.qvel.copy(),
            integration_state=self.integration_state(),gt_target_mask_offline=mask,frame_id=np.int64(self.frame_id),
            sim_time=np.float64(env.sim.data.time),timestamp=np.float64(time.time()),
            subtask_index=np.int32(self.subtask_index()),target_id=name,stage=self.stage)
        return row

    def metadata(self):
        state_fields=["arm_joint_pos[7]","arm_joint_vel[7]","eef_world_position[3]","eef_world_quaternion_xyzw[4]","gripper_joint_pos[2]"]
        return dict(seed=self.seed,order=list(self.order),layout_id=int(self.env.layout_id),style_id=int(self.env.style_id),
            robot="PandaOmron",control_freq=20,camera_map=CAMERAS,camera_resolution={"base_rgb":[224,224],"wrist_rgb":[224,224],"overview_rgb":[448,448]},
            image_orientation="robosuite opencv canonical; RGB and segmentation use identical vertical convention",
            camera_calibration={key:dict(name=name,
                initial_world_position=self.env.sim.data.cam_xpos[self.env.sim.model.camera_name2id(name)].tolist(),
                initial_world_rotation=self.env.sim.data.cam_xmat[self.env.sim.model.camera_name2id(name)].reshape(3,3).tolist(),
                fovy_deg=float(self.env.sim.model.cam_fovy[self.env.sim.model.camera_name2id(name)]))
                for key,name in CAMERAS.items()},
            robot_state_fields=state_fields,policy_state_dim=int(self.observation()["robot_state"].size),
            action_contract=dict(type="OSC_POSE",input_type="delta",reference_frame="world",policy_action_dim=7,
                policy_action_fields=["dx_normalized","dy_normalized","dz_normalized","drot_x_normalized","drot_y_normalized","drot_z_normalized","gripper_close"],
                position_scale_m=.05,rotation_scale_rad=.5,gripper_open=-1,gripper_close=1,
                native_action_dim=int(self.env.action_dim),
                native_parts={k:list(v) for k,v in self.env.robots[0].composite_controller._action_split_indexes.items()},
                locked_commands=["base","torso","base_mode"]),
            policy_input_whitelist=["observations/base_rgb","observations/wrist_rgb","observations/robot_state","observations/gt_target_mask_offline (training annotation only)"],
            privileged_not_policy_inputs=["observations/qpos","observations/qvel","observations/integration_state","object body IDs/poses","simulator success"],
            initial_rgb_sha256=self.rgb_hashes(),rendering=dict(self.rendering),msaa_samples=0,
            task=self.env.task_mode,selected_objects=copy.deepcopy(self.selected_objects),
            initial_object_poses=self.object_poses(),scripted_demo_available=not bool(self.selected_objects),
            selection_schema="official_tabletop_selection.v1" if self.selected_objects else "classic2.v1",
            initialization="Reset-only object placement and kinematic IK, followed by physical settling; no pose writes during demonstrations",
            egl_device_index=int(os.environ.get("MUJOCO_EGL_DEVICE_ID","0")),
            physical_cuda_device=os.environ.get("CUDA_VISIBLE_DEVICES"),
            versions=self.versions())

    def versions(self):
        import importlib.metadata as m,subprocess
        result={k:m.version(k) for k in ("robocasa","robosuite","mujoco","numpy","h5py")}
        for name, checkout in (("robocasa", ROBOCASA_ROOT), ("robosuite", ROBOSUITE_ROOT)):
            result[name + "_commit"] = None
            if (checkout / ".git").exists():
                result[name + "_commit"] = subprocess.check_output(
                    ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
                ).strip()
        return result

    def eef(self):
        return self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id["right"]].copy()

    def position(self,name):
        return self.env.sim.data.body_xpos[self.env.obj_body_id[name]].copy()

    def _placed(self,name):
        tray_id=self.env.obj_body_id["collection_tray"];cube_id=self.env.obj_body_id[name]
        center=self.env.sim.data.body_xpos[tray_id]
        rot=self.env.sim.data.body_xmat[tray_id].reshape(3,3)
        rel=rot.T@(self.env.sim.data.body_xpos[cube_id]-center)
        if self.selected_objects:
            obj=self.env.objects[name]
            local_center=(np.asarray(obj.bottom_offset)+np.asarray(obj.top_offset))/2
            objrot=self.env.sim.data.body_xmat[cube_id].reshape(3,3)
            rel=rot.T@(self.env.sim.data.body_xpos[cube_id]+objrot@local_center-center)
            size=next(x["bbox_size_m"] for x in self.selected_objects if x["id"]==name)
            half=np.abs(rot.T@objrot)@(np.asarray(size)/2)
            return bool((np.abs(rel[:2])+half[:2]<np.array([.14,.105])+.003).all()
                and -.006<rel[2]-half[2]<.035 and rel[2]+half[2]<.27
                and np.linalg.norm(self.eef()-self.position(name))>.07)
        return bool(abs(rel[0])<.12 and abs(rel[1])<.085 and .012<rel[2]<.075
                    and np.linalg.norm(self.eef()-self.position(name))>.07)

    def update_score(self):
        for name in self.order:
            inside=self._placed(name)
            self._stable[name]=self._stable.get(name,0)+1 if inside else 0
            if self._stable[name]>=3 and name not in self.completed:
                expected=self.order[len(self.completed)] if len(self.completed)<len(self.order) else None
                if name!=expected:self.order_valid=False
                self.completed.append(name)

    def scoring(self):
        all_inside=bool(self.env is not None and all(self._placed(n) for n in self.order))
        return dict(ordered_completed=list(self.completed),order_valid=self.order_valid,
            simulator_success=bool(all_inside and self.completed==self.order and self.order_valid),
            scoring_source="simulator_privileged_offline_evaluation")

    def frame_array(self):
        canvas=Image.new("RGB",(896,960),(20,42,34));draw=ImageDraw.Draw(canvas)
        draw.text((12,8),f"RoboCasa | frame {self.frame_id} | {self.stage} | offline collection",fill="white")
        imgs=[self._rgb("overview_rgb"),self._rgb("base_rgb"),self._rgb("wrist_rgb")]
        row=self.observation()
        imgs.append(np.where(row["gt_target_mask_offline"][...,None].astype(bool),imgs[1],0))
        labels=["Kitchen overview RGB","Base RGB","Wrist RGB","Offline GT target RGB (not SAM2/Cutie)"]
        for i,(img,label) in enumerate(zip(imgs,labels)):
            x=(i%2)*448;y=40+(i//2)*460
            draw.text((x+10,y),label,fill="white")
            canvas.paste(Image.fromarray(img).resize((448,432)),(x,y+25))
        return np.array(canvas)

    def publish(self):
        if self.env:
            im=Image.fromarray(self.frame_array());buf=io.BytesIO();im.save(buf,format="PNG")
            self.publish_callback(self.public_status(),buf.getvalue())
        else:self.publish_callback(self.public_status(),None)

    def public_status(self):
        score=self.scoring() if self.env else dict(simulator_success=False,order_valid=True,ordered_completed=[])
        ep=(dict(id=self.record.id,source=self.record.metadata["source"],status="recording",steps=self.record.length)
            if self.record else self.last_summary)
        return dict(scene=self.scene_info,recording=self.record is not None,episode=ep,
            frame_id=self.frame_id,updated_at=time.time(),frame_url="/frame.png" if self.env else None,
            gripper=self.gripper,phase=self.stage,video_url=(self.last_replay or self.last_summary or {}).get("video_url"),
            quality=(self.last_summary or {}).get("user_quality"),user_quality=(self.last_summary or {}).get("user_quality"),
            progress=dict(completed=len(self.completed),total=len(self.order),current_target=self.target()),
            target_override=self.target() if self.active_subtask is not None else None,
            scripted_demo_available=not bool(self.selected_objects),
            scripted_demo_reason=None if not self.selected_objects else "仅默认红蓝方块支持已验证脚本专家",**score,
            replay=self.last_replay,rendering=self.rendering,policies_available=False)

    def start_record(self,params=None,source="manual"):
        if self.env is None:raise ValueError("Generate a scene first")
        if self.record:raise ValueError("A recording is already active")
        self.stage="manual_recording" if source=="manual" else "scripted_expert"
        self.record=Recorder(self,source,(params or {}).get("label",""))
        self.last_summary=None;self.publish()

    def finish_record(self,quality="stopped",note="",error=None):
        if not self.record:raise ValueError("No active recording")
        rec=self.record;self.record=None
        self.last_summary=rec.close(self,quality,note,error)
        if rec.length:
            try:
                from .training import prepare
                prepare(rec.id)
                self.last_summary["training_manifest_url"]=f"/artifacts/{rec.id}/training_manifest.json"
            except Exception as exc:
                self.last_summary["training_adapter_error"]=str(exc)
                self.last_summary["training_ready"]=False
                self.last_summary["data_quality_issues"]=[str(exc)]
            dump_json(rec.directory/"summary.json",self.last_summary)
            dump_json(rec.output/"summary.json",self.last_summary)
        self.active_subtask=None;self.stage="recording_saved"
        self.publish_callback(self.public_status(),None)
        return self.last_summary

    def native_action(self,translation,rotation,gripper):
        trans=np.asarray(translation,float);rot=np.asarray(rotation,float)
        if trans.shape!=(3,) or rot.shape!=(3,) or not np.isfinite(np.r_[trans,rot]).all():
            raise ValueError("Translation and rotation must be finite three-vectors")
        if (np.abs(trans)>.05+1e-9).any() or (np.abs(rot)>.5+1e-9).any():
            raise ValueError("Per-step command exceeds 0.05m or 0.5rad")
        if gripper not in (-1,1,-1.,1.):raise ValueError("Gripper must be -1=open or +1=close")
        arm=np.r_[trans/.05,rot/.5,float(gripper)].astype(np.float32)
        parts={"right":arm[:6],"right_gripper":arm[6:7],"base":np.zeros(3),"torso":np.zeros(1),"base_mode":-1.}
        native=self.env.robots[0].create_action_vector(parts)
        return np.asarray(native,np.float64),arm

    def step(self,native,arm,publish_frame=True):
        self._check_stop()
        if self.record:self.record.update_context(self.observation())
        self.env.step(np.asarray(native,float))
        self._segmentation_cache=None
        self.env.sim.forward()
        self.latest_obs=self.env._get_observations(force_update=True)
        self.frame_id+=1;self.update_score()
        if self.record or publish_frame:
            frame=self.frame_array()
            if self.record:self.record.append_transition(native,arm,self.observation(),frame)
            if publish_frame:
                buf=io.BytesIO();Image.fromarray(frame).save(buf,format="PNG",compress_level=1)
                self.publish_callback(self.public_status(),buf.getvalue())

    def control(self,params):
        if self.env is None:raise ValueError("Generate a scene first")
        steps=params.get("steps",1)
        if type(steps) is not int or not 1<=steps<=10:raise ValueError("steps must be 1..10")
        self.gripper=float(params.get("gripper",self.gripper))
        native,arm=self.native_action(params.get("translation",[0,0,0]),params.get("rotation",[0,0,0]),self.gripper)
        for _ in range(steps):self.step(native,arm)

    def _move(self,goal,gripper,stage,max_steps=110,tolerance=.008):
        from scipy.spatial.transform import Rotation
        self.stage=stage;hits=0
        for _ in range(max_steps):
            self._check_stop()
            error=np.asarray(goal)-self.eef()
            current=self.env.sim.data.site_xmat[self.env.robots[0].eef_site_id["right"]].reshape(3,3)
            rot=Rotation.from_matrix(self.initial_ori@current.T).as_rotvec()
            native,arm=self.native_action(np.clip(error*.65,-.018,.018),np.clip(rot*.4,-.1,.1),gripper)
            self.gripper=gripper;self.step(native,arm)
            hits=hits+1 if np.linalg.norm(error)<tolerance else 0
            if hits>=3:return
        raise RuntimeError(f"Expert waypoint timeout: {stage}, remaining error {np.linalg.norm(error):.3f}m")

    def _hold(self,gripper,count,stage):
        self.stage=stage
        for _ in range(count):
            native,arm=self.native_action([0,0,0],[0,0,0],gripper)
            self.gripper=gripper;self.step(native,arm)

    def select_target(self,params):
        if self.env is None:raise ValueError("Generate a scene first")
        name=params.get("target_id")
        if name is not None and name not in self.order:raise ValueError("target_id must identify one selected scene object")
        self.active_subtask=self.order.index(name) if name is not None else None
        if self.record:self.record.update_context(self.observation())
        self.publish()

    def scripted_demo(self,params):
        if self.env is None:raise ValueError("Generate a scene first")
        if self.selected_objects:
            raise ValueError("Scripted expert is validated only for the default red/blue cubes; use manual collection for official objects")
        if self.record:raise ValueError("Stop current recording first")
        self.scene({"restore":True})
        self.active_subtask=0
        self.start_record({"label":"Contact-based privileged scripted expert; not ACT/pi0.5/VLM"},"scripted_expert_privileged_state")
        try:
            for index,name in enumerate(self.order):
                self.active_subtask=index
                obj=self.position(name)
                self._move(obj+[0,0,.16],-1,f"{name}:approach")
                self._move(obj+[0,0,.006],-1,f"{name}:descend",tolerance=.005)
                self._hold(1,16,f"{name}:close")
                self._move(obj+[0,0,.18],1,f"{name}:lift")
                if self.position(name)[2]<obj[2]+.06:
                    raise RuntimeError(f"{name} was not physically lifted; demonstration rejected")
                tray=self.position("collection_tray")
                rot=self.env.sim.data.body_xmat[self.env.obj_body_id["collection_tray"]].reshape(3,3)
                slot=tray+rot@np.array([-.060 if index==0 else .060,0,.024])
                self._move(slot+[0,0,.16],1,f"{name}:transport")
                self._move(slot+[0,0,.018],1,f"{name}:place",tolerance=.006)
                self._hold(-1,14,f"{name}:release")
                self._move(slot+[0,0,.20],-1,f"{name}:retreat")
                self._hold(-1,10,f"{name}:settle")
                if name not in self.completed:raise RuntimeError(f"{name} placement was not confirmed by simulator")
            self.finish_record("success","Scripted contact demonstration completed")
        except BaseException as e:
            if self.record:self.finish_record("stopped" if isinstance(e,Stopped) else "failure",str(e),None if isinstance(e,Stopped) else str(e))
            raise

    def replay(self,params):
        import h5py,mujoco,imageio.v2 as imageio
        eid=params.get("episode_id");mode=params.get("mode","states")
        if mode not in ("states","actions"):raise ValueError("Replay mode must be states or actions")
        if self.record:raise ValueError("Stop recording before replay")
        directory=episode_dir(eid);meta=json.loads((directory/"metadata.json").read_text())
        self.scene({"reuse_episode":eid})
        snap=json.loads((directory/"initial_snapshot.json").read_text())
        out=episode_dir(eid,True);out.mkdir(parents=True,exist_ok=True)
        video=out/f"replay_{mode}.mp4"
        writer=imageio.get_writer(str(video),fps=20,codec="libx264",quality=7,macro_block_size=16,
                                 ffmpeg_params=["-movflags","+faststart"])
        max_error=0.;max_arm_error=0.;max_eef_error=0.;max_object_xyz=0.;max_object_angle=0.
        mask_equal_frames=0;count=0
        initial_hashes=self.rgb_hashes();initial_match=initial_hashes==meta["initial_rgb_sha256"]
        def build_report(completed=True,error=None):
            return dict(episode_id=eid,mode=mode,frames=count,max_abs_qpos_error=max_error,
                initial_rgb_match=initial_match,verified=bool(completed and initial_match and max_error<(1e-9 if mode=="states" else 1e-4)),
                tolerance=1e-9 if mode=="states" else 1e-4,
                video_url=f"/artifacts/{eid}/replay_{mode}.mp4",source="recorded_states" if mode=="states" else "reexecuted_native_controls",
                initial_rgb_sha256=initial_hashes,expected_initial_rgb_sha256=meta["initial_rgb_sha256"],
                max_arm_joint_error_rad=max_arm_error,max_eef_position_error_m=max_eef_error,
                max_object_position_error_m=max_object_xyz,max_object_rotation_error_rad=max_object_angle,
                target_mask_exact_frames=mask_equal_frames,target_mask_comparison_frames=count-1,
                recorded_context_restored=True,
                task_outcome=self.scoring(),
                task_outcome_kind="offline scoring of replayed states; not autonomous execution" if mode=="states" else "offline scoring after native actions; replay_completed indicates full-sequence completion",
                controller_restore="saved memory; missing static legacy fields use deterministic same-config initialization baseline",
                replay_completed=completed,error=error,
                strict_trajectory_verified=bool(completed and initial_match and max_error<(1e-9 if mode=="states" else 1e-4)),
                task_reproduced=bool(completed and initial_match and self.scoring()["simulator_success"]) if mode=="actions" else None)
        try:
            with h5py.File(directory/"trajectory.h5","r") as f:
                states=f["observations/integration_state"];qpos=f["observations/qpos"]
                self.active_subtask=int(f["observations/subtask_index"][0])
                self.stage=f"replay_{mode}"
                writer.append_data(self.frame_array());count+=1
                for i in range(len(f["actions/native"])):
                    self._check_stop()
                    context_index=i+1 if mode=="states" else i
                    self.active_subtask=int(f["observations/subtask_index"][context_index])
                    if mode=="states":
                        mujoco.mj_setState(self.env.sim.model._model,self.env.sim.data._data,states[i+1],mujoco.mjtState.mjSTATE_INTEGRATION)
                        self._segmentation_cache=None
                        self.env.sim.forward();self.frame_id=int(f["observations/frame_id"][i+1])
                        self.latest_obs=self.env._get_observations(force_update=True)
                        self.update_score()
                    else:
                        self.step(f["actions/native"][i],f["actions/arm_gripper"][i],publish_frame=False)
                    # Recorded target context is restored for every displayed
                    # post-action frame, including the exact red/blue boundary.
                    self.active_subtask=int(f["observations/subtask_index"][i+1])
                    self.stage="replay_"+mode
                    if i%5==0:self.publish()
                    actual=self.env.sim.data.qpos;expected=qpos[i+1]
                    error=float(np.max(np.abs(actual-expected)))
                    max_error=max(max_error,error)
                    arm_idx=self._controllers()["right"].qpos_index
                    max_arm_error=max(max_arm_error,float(np.max(np.abs(actual[arm_idx]-expected[arm_idx]))))
                    max_eef_error=max(max_eef_error,float(np.linalg.norm(self.eef()-f["observations/robot_state"][i+1,14:17])))
                    for name in (*self.order,"collection_tray"):
                        lo,hi=self.env.sim.model.get_joint_qpos_addr(self.env.objects[name].joints[0])
                        max_object_xyz=max(max_object_xyz,float(np.linalg.norm(actual[lo:lo+3]-expected[lo:lo+3])))
                        a=actual[lo+3:hi];b=expected[lo+3:hi]
                        cosine=min(1.,abs(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))))
                        max_object_angle=max(max_object_angle,float(2*np.arccos(cosine)))
                    mask_equal_frames+=int(np.array_equal(self.observation()["gt_target_mask_offline"],f["observations/gt_target_mask_offline"][i+1]))
                    writer.append_data(self.frame_array());count+=1
            report=build_report()
        except Stopped as exc:
            report=build_report(False,str(exc));report["stopped"]=True
        except BaseException as exc:
            report=build_report(False,str(exc));report["stopped"]=False
        finally:writer.close()
        dump_json(out/f"replay_{mode}.json",report)
        self.last_replay=report
        self.stage="replay_finished" if report["replay_completed"] else "replay_stopped" if report.get("stopped") else "replay_failed"
        self.publish_callback(self.public_status(),None)
        if report.get("error") and not report.get("stopped"):
            raise RuntimeError(report["error"])
        return report
