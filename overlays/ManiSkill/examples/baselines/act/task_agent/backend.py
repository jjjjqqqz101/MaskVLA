"""Persistent RGB-only task execution with ACT/pi0.5 acquisition and MPLib placement.

VLM selections and perceived masks are the only target information used by the
controller. Simulator object state is written to a separate diagnostic file;
it is never included in snapshots, decisions, policy inputs or placement poses.
"""
from __future__ import annotations

import base64
import collections
from dataclasses import fields
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

from deployment_paths import (act_checkpoint, act_triple_checkpoint,
                              task_runs_root, TRIPLE_CHECKPOINT_RELATIVE, project_path)
from .mask_dropout import MaskDropout,normalize_mask_dropout
from .policies import POLICY_IDS, policy_record


ZONE = {
    'name': 'green_collection_area',
    'description': '绿色收纳区',
    'center': [-0.23, 0.0],
    'half_sizes': [0.10, 0.25],
    # Keep calibrated release points well inside the visible collection area.
    'slots': [[-0.23, -0.12], [-0.23, 0.12], [-0.23, -0.04], [-0.23, 0.04]],
    'capacity': 4,
}
OVERVIEW_CAMERA = {
    'width': 448, 'height': 448, 'fov': 1.0,
    'eye': [0.0, -0.65, 0.85], 'target': [-0.045, 0.0, 0.025],
}
_ENV_REGISTERED = False
ACT_POLICIES = ('act', 'act_triple')
TRIPLE_CHECKPOINT = TRIPLE_CHECKPOINT_RELATIVE.as_posix()
TRIPLE_OPTIMIZER_STEP = 160000


class BackendStopped(RuntimeError):
    """Cooperative cancellation before another simulator action."""


class _SkillInterrupted(RuntimeError):
    def __init__(self, reason):
        self.reason = str(reason)
        super().__init__(self.reason)


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def pi_state_from_observation(obs):
    qpos = array(obs['agent']['qpos'])[0]
    return np.r_[qpos[:7], np.clip(1 - np.mean(qpos[7:9]) / .04, 0, 1)].astype(np.float32)


def hold_action_from_observation(obs):
    qpos = array(obs['agent']['qpos'])[0]
    openness = float(np.clip(np.mean(qpos[7:9]) / .04, 0, 1))
    return np.r_[qpos[:7], 2 * openness - 1].astype(np.float32)


def sampling_seed(seed, frame):
    return int(np.random.SeedSequence([int(seed), int(frame)]).generate_state(1)[0])


def deposit_position(slot_index, height=.08):
    if type(slot_index) is not int or not 0 <= slot_index < ZONE['capacity']:
        raise ValueError('The green collection area has four calibrated placement slots')
    return np.array([*ZONE['slots'][slot_index], float(height)], dtype=np.float64)


def sam_prompt_bbox(box, width, height):
    """Expand a visual target hint without changing the recorded model box."""
    from perception_mask.core import validate_box
    original=validate_box(box,width,height,'pixels')
    x0,y0,x1,y1=original
    margin=max(4.0,.15*max(x1-x0,y1-y0))
    return [max(0.,x0-margin),max(0.,y0-margin),
            min(float(width),x1+margin),min(float(height),y1+margin)]


def _register_scene():
    global _ENV_REGISTERED
    if _ENV_REGISTERED:
        return
    import sapien
    from mani_skill.envs.tasks.tabletop.target_pick_eight import TargetPickEightEnv
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.building import actors
    from mani_skill.utils.registration import register_env

    @register_env('TaskCollection-v1', max_episode_steps=10000, asset_download_ids=['ycb'])
    class TaskCollectionEnv(TargetPickEightEnv):
        SUPPORTED_ROBOTS = ['panda', 'panda_wristcam']

        @property
        def _default_sensor_configs(self):
            # This fixed visual observation camera never replaces the two
            # trained policy cameras. Its pose is independent of scene actors.
            overview = CameraConfig('overview_camera',
                sapien_utils.look_at(eye=OVERVIEW_CAMERA['eye'], target=OVERVIEW_CAMERA['target']),
                OVERVIEW_CAMERA['width'], OVERVIEW_CAMERA['height'], OVERVIEW_CAMERA['fov'],
                .01, 100, shader_pack='minimal')
            return [*super()._default_sensor_configs, overview]

        def _load_scene(self, options):
            super()._load_scene(options)
            # A real rendered marker in a known, calibrated free area. It has
            # no collision and does not move or teleport any task object.
            self.collection_marker = actors.build_box(
                self.scene, half_sizes=[*ZONE['half_sizes'], .001],
                color=[.12, .64, .26, 1], name='green_collection_area',
                body_type='kinematic', add_collision=False,
                initial_pose=sapien.Pose(p=[*ZONE['center'], .0012]))

    _ENV_REGISTERED = True


class _PublishedEnv:
    """MPLib's env facade; its follow_path uses this step implementation."""
    def __init__(self, owner):
        self.owner = owner

    @property
    def unwrapped(self):
        return self.owner.base

    def step(self, action):
        return self.owner._step(action)

    def __getattr__(self, name):
        return getattr(self.owner.raw, name)


class RobotBackend:
    def __init__(self, policy='pi05', seed=17, output=None, frame_callback=None,
                 stop_event=None, perception_url='http://127.0.0.1:18830',
                 pi05_url='http://127.0.0.1:18833', act_checkpoint=None,
                 max_grasp_steps=500, pace_hz=20, max_grasp_seconds=45,
                 mask_lost_timeout=8, mask_dropout=None, mask_dropout_control=None,
                 event_callback=None, defer_policy_loading=False, scene_spec=None, **unused):
        if unused:
            raise ValueError('Unsupported backend options: '+', '.join(sorted(unused)))
        if policy not in POLICY_IDS:
            raise ValueError('Policy must be '+', '.join(POLICY_IDS))
        if type(defer_policy_loading) is not bool:
            raise ValueError('defer_policy_loading must be boolean')
        from .scene_objects import normalize_scene_spec
        self.scene_spec=normalize_scene_spec(scene_spec)
        expected_objects=[item['id'] for item in self.scene_spec['objects']]
        if not 1 <= float(pace_hz) <= 60:
            raise ValueError('Control pacing must be between 1 and 60 Hz')
        for endpoint in (perception_url, pi05_url):
            parsed = urlparse(endpoint)
            if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1','localhost','::1')
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError('Model services must be local HTTP endpoints')
        import gymnasium as gym
        import imageio.v2 as imageio
        import torch
        import mani_skill.envs  # noqa: F401

        self.policy_name = policy
        self._policy_bound=False
        self._binding_attempted=False
        self._act_checkpoint=act_checkpoint
        self.seed = int(seed)
        self.output = project_path(output or task_runs_root(), 'backend output')
        self.output.mkdir(parents=True, exist_ok=True)
        self.frame_callback = frame_callback
        self.event_callback = event_callback
        self.stop_event = stop_event or threading.Event()
        self.perception_url, self.pi05_url = perception_url.rstrip('/'), pi05_url.rstrip('/')
        self.max_grasp_steps = max(1, min(int(max_grasp_steps), 2000))
        self.max_grasp_seconds = float(max_grasp_seconds)
        self.mask_lost_timeout = float(mask_lost_timeout)
        self.pace_hz = float(pace_hz)
        self.steps = 0
        self.reset_count = 0
        self.phase = 'idle'
        self.motion_phase = 'initializing'
        self.target_text = ''
        self.step_generation = 0
        self.control_source = 'initialization'
        self.perception = None
        self._tracking_frame = 0
        self._mask = None
        self._natural_mask = None
        self._last_target_bbox = None
        self._sam_prompt_bbox = None
        self._mask_metadata = {'state': 'idle', 'source': 'none', 'generation': 0}
        self._natural_mask_metadata = dict(self._mask_metadata)
        self._snapshot = None
        self._snapshot_lock = threading.RLock()
        self._decider = None
        self._closed = False
        self._last_gripper = 1.0
        self._act_agent = self._act_args = self._act_stats = None
        self._act_policy = None
        self._act_metadata = {}
        self.mask_dropout_config=normalize_mask_dropout(mask_dropout)
        self._mask_dropout=MaskDropout(self.mask_dropout_config,mask_dropout_control,self._dropout_event)
        self._mask_dropout_state=self._mask_dropout.tick(self.phase,self.step_generation,0)
        try:
            self.actions = (self.output/'actions.jsonl').open('w', buffering=1)
            self.diagnostics = (self.output/'diagnostic_gt.jsonl').open('w', buffering=1)
            self.writer = imageio.get_writer(self.output/'execution.mp4', fps=self.pace_hz,
                                            codec='libx264', quality=7, macro_block_size=16,
                                            ffmpeg_params=['-movflags','+faststart'])
            torch.set_num_threads(4)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if self.scene_spec['preset']=='classic8':
                _register_scene();env_id='TaskCollection-v1';scene_options={}
            else:
                from .scene_env import register_selectable_scene
                register_selectable_scene(ZONE,OVERVIEW_CAMERA)
                env_id='TaskCollectionSelectable-v1'
                scene_options=dict(scene_objects=self.scene_spec['objects'])
            self.raw = gym.make(env_id, robot_uids='panda_wristcam',
                obs_mode='rgb', control_mode='pd_joint_pos', sim_backend='physx_cpu',
                render_backend=os.environ.get('MASK_RENDER_BACKEND','gpu'),
                render_mode='rgb_array', reward_mode='sparse', max_episode_steps=10000,
                reconfiguration_freq=0,
                sensor_configs=dict(width=224,height=224,fov=float(np.pi/2),near=.01,far=100,
                    shader_pack='minimal',overview_camera=dict(width=OVERVIEW_CAMERA['width'],
                        height=OVERVIEW_CAMERA['height'],fov=OVERVIEW_CAMERA['fov'])),**scene_options)
            self.base = self.raw.unwrapped
            self.obs, _ = self.raw.reset(seed=self.seed)
            self.reset_count += 1
            aliases={'cube_0':'red_cube_1','cube_1':'red_cube_2','cube_2':'blue_cube'}
            self.selected_object_ids=[aliases.get(obj.name,obj.name) for obj in self.base.cubes]
            if self.selected_object_ids!=expected_objects:
                raise RuntimeError('Constructed scene objects do not match the requested scene specification')
            self._captured_at = time.monotonic()
            self.env = _PublishedEnv(self)
            self.initial_qpos = array(self.obs['agent']['qpos'])[0].copy()
            self.initial_tcp = self.base.agent.tcp.pose.sp
            if not defer_policy_loading:self._prepare_policy(policy)
            self._publish()
            # Let reset-created objects settle physically; no state restoration or
            # object movement is performed after the one scene reset.
            for _ in range(5):
                self._step(hold_action_from_observation(self.obs))
            self._save_image(self.output/'initial.png')
            (self.output/'backend_config.json').write_text(json.dumps(dict(
                **policy_record(policy),seed=self.seed,zone=ZONE,reset_count=self.reset_count,
                policy_device='deferred' if defer_policy_loading else str(self.device) if policy in ACT_POLICIES else 'existing_pi05_worker',
                defer_policy_loading=defer_policy_loading,
                scene_spec=self.scene_spec,selected_object_ids=self.selected_object_ids,
                render_backend=self.base.backend.render_backend,renderer=os.environ.get('MASK_SAPIEN_RENDERER','native'),
                scene_env_id=env_id,scene_layout_version=getattr(self.base,'layout_version','classic8-v1'),
                mask_source='vlm_sam2_cutie',perception_inputs='raw RGB, description and VLM bbox only',
                mask_loss_policy='zero-mask learned policy after initialized tracking; bounded timeout; hold before initial tracking',
                success_source='external_visual_supervisor',control_pace_hz=self.pace_hz,
                max_grasp_steps=self.max_grasp_steps,max_grasp_seconds=self.max_grasp_seconds,
                policy_cameras=dict(base=[224,224],wrist=[224,224]),
                sam_bbox_padding='per side max(4px, 15% of longest box side), clipped to image',
                mask_dropout=self.mask_dropout_config,
                visual_supervision_camera=OVERVIEW_CAMERA,
                home_condition='planned initial seven arm joints, observed max error <=0.02rad',
                placement='MPLib calibrated TCP slots; no object-pose offsets',
                **self._act_metadata),ensure_ascii=False,indent=2))
        except BaseException:
            self.close()
            raise

    @property
    def zone(self):
        return json.loads(json.dumps(ZONE))

    def _prepare_policy(self,policy):
        if policy=='act':
            default=act_checkpoint()
            self._load_act(project_path(self._act_checkpoint or default, 'ACT checkpoint'))
        elif policy=='act_triple':
            default=act_triple_checkpoint()
            self._load_act_triple(project_path(self._act_checkpoint or default, 'ACT triple checkpoint'))
        else:
            with urlopen(self.pi05_url+'/health',timeout=5) as response:health=json.load(response)
            if not health.get('supports_noise_seed'):
                raise RuntimeError('Start the current pi0.5 worker with deterministic noise support before a task')
        self._policy_bound=True

    def refresh_observation(self):
        """Capture current robot/RGB observations without any physics step."""
        self._check_stop()
        # TargetPickEight's RGB extras need no task evaluation info. Supplying
        # an empty dict avoids even evaluating simulator success/contact flags.
        self.obs=self.base.get_obs(info={})
        self._captured_at=time.monotonic()
        self._publish(record_video=False)
        self._save_image(self.output/'live.png')
        return self.snapshot()

    def bind_task(self,policy,seed,output,frame_callback=None,stop_event=None,
                  mask_dropout=None,mask_dropout_control=None,event_callback=None):
        if self._closed:raise RuntimeError('The prepared scene is already closed')
        if self._policy_bound or self._binding_attempted:
            raise RuntimeError('A prepared scene can be bound to only one task')
        if type(seed) is not int or seed!=self.seed:
            raise ValueError('Task seed must match the prepared scene seed')
        if Path(output).resolve()!=self.output:
            raise ValueError('Task output must be the prepared scene directory')
        if policy not in POLICY_IDS:raise ValueError('Policy must be '+', '.join(POLICY_IDS))
        config=normalize_mask_dropout(mask_dropout)
        self._binding_attempted=True
        # Retire preview bookkeeping before attaching the task's event stream.
        self._mask_dropout.close()
        self.policy_name=policy;self.frame_callback=frame_callback;self.event_callback=event_callback
        self.stop_event=stop_event or threading.Event()
        self.mask_dropout_config=config
        self._mask_dropout=MaskDropout(config,mask_dropout_control,self._dropout_event)
        self._mask_dropout_state=self._mask_dropout.tick(self.phase,self.step_generation,0)
        self._check_stop();self._prepare_policy(policy);self._check_stop()
        self.refresh_observation()
        config_path=self.output/'backend_config.json'
        recorded=json.loads(config_path.read_text())
        recorded.update(**policy_record(policy),policy_device=str(self.device) if policy in ACT_POLICIES else 'existing_pi05_worker',
                        defer_policy_loading=False,
                        prepared_scene_reused=True,mask_dropout=config,**getattr(self,'_act_metadata',{}))
        config_path.write_text(json.dumps(recorded,ensure_ascii=False,indent=2))
        return self

    def _load_act(self, checkpoint):
        import gymnasium as gym
        import torch
        from train_dual_rgb import Args, Agent
        import act.detr.backbone as backbone_module
        data = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if tuple(data.get('stream_order',())) != ('original_rgb','masked_rgb'):
            raise ValueError('Expected a dual original/masked ACT checkpoint')
        values = data['args']
        args = Args(**{f.name:values[f.name] for f in fields(Args) if f.name in values})
        if args.include_depth or args.control_mode!='pd_joint_pos':
            raise ValueError('Expected RGB-only pd_joint_pos ACT')
        use_cuda=torch.cuda.is_available()
        if use_cuda:
            free_bytes,_=torch.cuda.mem_get_info()
            use_cuda=free_bytes>=2*1024**3
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        adapter = SimpleNamespace(single_observation_space=gym.spaces.Dict({
            'state':gym.spaces.Box(-np.inf,np.inf,(28,),dtype=np.float32),
            'rgb':gym.spaces.Box(0,255,(2,3,224,224),dtype=np.uint8)}),
            single_action_space=gym.spaces.Box(-np.inf,np.inf,(8,),dtype=np.float32))
        original = backbone_module.is_main_process
        backbone_module.is_main_process = lambda: False
        try:
            agent = Agent(adapter,args).to(self.device).eval()
        finally:
            backbone_module.is_main_process = original
        agent.load_state_dict(data['ema_agent'],strict=True)
        self._act_stats = {k:torch.as_tensor(data['norm_stats'][k],device=self.device,dtype=torch.float32)
                           for k in ('state_mean','state_std','action_mean','action_std')}
        self._act_agent,self._act_args = agent,args
        self._act_metadata = dict(checkpoint_path=str(checkpoint.resolve()),
            optimizer_step=data.get('optimizer_step'),stream_order=list(data['stream_order']),
            weights_source='ema_agent',policy_temporal_aggregation=bool(args.temporal_agg),
            policy_device_metadata=self._act_device_metadata())

    def _act_device_metadata(self):
        import torch
        details=dict(device=str(self.device),torch_version=str(torch.__version__),
                     cuda_available=bool(torch.cuda.is_available()),cuda_runtime=torch.version.cuda)
        if self.device.type=='cuda':
            details['device_name']=torch.cuda.get_device_name(self.device)
            details['device_index']=self.device.index if self.device.index is not None else torch.cuda.current_device()
        return details

    def _load_act_triple(self, checkpoint):
        import torch
        from evaluate_triple_rgb import CheckpointPolicy, STREAM_ORDER, STATE_FIELDS
        use_cuda=torch.cuda.is_available()
        if use_cuda:
            free_bytes,_=torch.cuda.mem_get_info()
            use_cuda=free_bytes>=2*1024**3
        self.device=torch.device('cuda' if use_cuda else 'cpu')
        # This facade validates architecture, stream order, state whitelist,
        # normalization and strict EMA weights without creating an env wrapper.
        policy=CheckpointPolicy(checkpoint,device=self.device)
        if policy.metadata.get('optimizer_step')!=TRIPLE_OPTIMIZER_STEP:
            raise ValueError('act_triple requires the EMA checkpoint at optimizer step 160000')
        self._act_policy=policy
        self._act_agent,self._act_stats=policy.agent,policy.stats
        # Triple training has no temporal_agg argument. Match the original
        # task ACT controller's default temporal aggregation explicitly.
        self._act_args=SimpleNamespace(num_queries=30,temporal_agg=True)
        self._act_metadata=dict(checkpoint_path=str(checkpoint.resolve()),
            optimizer_step=policy.metadata['optimizer_step'],stream_order=list(STREAM_ORDER),
            state_fields=['/'.join(item) for item in STATE_FIELDS],goal_pos=[0.,0.,.2],
            weights_source='ema_agent',policy_temporal_aggregation=True,
            checkpoint_metadata=policy.metadata,policy_device_metadata=self._act_device_metadata(),
            normalization='checkpoint state z-score; Agent RGB /255 then ImageNet; checkpoint action inverse z-score')

    def _check_stop(self):
        if self.stop_event.is_set():
            raise BackendStopped('User requested task stop')
        if self._decider:
            decision = self._decider()
            if decision:
                raise _SkillInterrupted(decision)

    def set_context(self, phase, target_text='', step_generation=0):
        self.phase, self.target_text, self.step_generation = str(phase),str(target_text),int(step_generation)
        if self.phase in ('idle','goal_verify') and not self.target_text.strip():
            # A finished selection must not masquerade as the next target.
            # Keep activity-phase tracking intact, but retire it when there
            # is no active target to perceive or display.
            self._close_perception()
            self._mask=None
            self._natural_mask=None
            self._mask_metadata=dict(state='idle',source='none',generation=0)
            self._natural_mask_metadata=dict(self._mask_metadata)
            self._last_target_bbox=None
            self._sam_prompt_bbox=None
            self._tracking_frame=0
        self._publish(record_video=False)

    def _dropout_event(self,event,**values):
        row=dict(event=event,time=time.time(),frame_id=self.steps,phase=self.phase,**values)
        with (self.output/'mask_dropout.jsonl').open('a') as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        if self.event_callback:self.event_callback(event,**{k:v for k,v in row.items() if k not in ('event','time')})

    def _refresh_mask_dropout(self):
        engine=getattr(self,'_mask_dropout',None)
        if engine is None:return
        natural=self._natural_mask
        pixels=0 if natural is None else int(np.count_nonzero(natural))
        state=engine.tick(self.phase,self.step_generation,pixels)
        self._mask_dropout_state=state
        self._mask=None if natural is None else (np.zeros_like(natural) if state['active'] else natural.copy())
        metadata=dict(self._natural_mask_metadata)
        metadata.update(natural_state=metadata.get('state','idle'),natural_mask_pixels=pixels,effective_mask_pixels=state['effective_mask_pixels'],
            mask_pixels=state['effective_mask_pixels'],artificial_mask_dropout=state['active'])
        if state['active']:
            metadata.update(state='lost',intervention='dropout',intervention_active=True,
                            intervention_origin='task_mask_dropout',intervention_source=state['source'])
        self._mask_metadata=metadata

    def snapshot(self):
        with self._snapshot_lock:
            if self._snapshot is None:
                raise RuntimeError('No camera snapshot has been published')
            result = dict(self._snapshot)
            for key in ('base_rgb','wrist_rgb','overview_rgb','mask'):
                result[key] = result[key].copy() if result.get(key) is not None else None
            result['mask_metadata'] = json.loads(json.dumps(result['mask_metadata']))
            result['mask_dropout_state']=json.loads(json.dumps(result.get('mask_dropout_state',{})))
            for key in ('target_bbox_xyxy','sam_prompt_bbox_xyxy'):
                if result.get(key) is not None:result[key] = list(result[key])
            return result

    def _raw_images(self):
        cameras = self.obs['sensor_data']
        base = array(cameras['base_camera']['rgb'])[0].astype(np.uint8,copy=True)
        wrist = array(cameras['hand_camera']['rgb'])[0].astype(np.uint8,copy=True)
        return base,wrist

    def _publish(self, record_video=True):
        self._refresh_mask_dropout()
        base,wrist = self._raw_images()
        overview = array(self.obs['sensor_data']['overview_camera']['rgb'])[0].astype(np.uint8,copy=True)
        snapshot = dict(frame_id=self.steps,captured_at=self._captured_at,published_at=time.monotonic(),
            base_rgb=base,wrist_rgb=wrist,overview_rgb=overview,
            mask=None if self._mask is None else self._mask.copy(),
            target_bbox_xyxy=None if self._last_target_bbox is None else list(self._last_target_bbox),
            sam_prompt_bbox_xyxy=None if self._sam_prompt_bbox is None else list(self._sam_prompt_bbox),
            mask_metadata=json.loads(json.dumps(self._mask_metadata)),phase=self.phase,
            mask_dropout_state=dict(getattr(self,'_mask_dropout_state',{})),
            motion_phase=self.motion_phase,target_text=self.target_text,
            step_generation=self.step_generation,policy=self.policy_name,
            policy_loaded=getattr(self,'_policy_bound',True))
        with self._snapshot_lock:
            self._snapshot = snapshot
        if record_video:
            self.writer.append_data(self._frame(snapshot))
        if self.frame_callback:
            self.frame_callback(self.snapshot())
        if not record_video or self.steps % 5 == 0:
            self._save_image(self.output/'live.png')

    def _frame(self, snapshot):
        from PIL import Image,ImageDraw
        base,wrist = snapshot['base_rgb'],snapshot['wrist_rgb']
        masked = np.zeros_like(base) if snapshot['mask'] is None else np.where(snapshot['mask'][...,None],base,0)
        # All overlays are display-only: snapshot RGB arrays remain untouched.
        overview = snapshot.get('overview_rgb')
        # Four independently framed images with their titles directly above.
        # Only the overview is resized for display; policy arrays stay native.
        canvas = Image.new('RGB',(512,576),'#112920')
        draw=ImageDraw.Draw(canvas)
        model_label=self.policy_name if getattr(self,'_policy_bound',True) else 'scene preview | model not loaded'
        draw.text((8,8),f"step {self.steps} | {model_label} | mask {self._mask_metadata.get('state','?')}",fill='white')
        draw.text((8,28),f"{self.phase} / {self.motion_phase}",fill='white')
        dropout=snapshot.get('mask_dropout_state',{})
        if dropout.get('active'):
            draw.text((8,47),f"ARTIFICIAL MASK LOSS | {dropout.get('source')} | {dropout.get('remaining_seconds',0):.1f}s",fill='#ffb25b')
        overview_image=(Image.fromarray(overview).resize((224,224),Image.Resampling.BILINEAR)
                        if overview is not None else Image.new('RGB',(224,224),'black'))
        panels=[(16,88,'VLM overview RGB',overview_image),
                (272,88,'Policy base RGB',Image.fromarray(base)),
                (16,344,'Policy wrist RGB',Image.fromarray(wrist)),
                (272,344,'Perceived target RGB (Masked RGB)',Image.fromarray(masked))]
        for x,y,label,picture in panels:
            canvas.paste(picture,(x,y))
            draw.rectangle((x-1,y-1,x+224,y+224),outline='#587d69',width=1)
            draw.text((x,y-16),label,fill='white')
        box=snapshot.get('target_bbox_xyxy')
        if isinstance(box,(list,tuple)) and len(box)==4 and np.isfinite(box).all():
            x0,y0,x1,y1=box
            if 0<=x0<x1<=base.shape[1] and 0<=y0<y1<=base.shape[0]:
                draw.rectangle((272+x0,88+y0,272+x1,88+y1),outline='#32e6ff',width=2)
                draw.text((272+x0,max(88,88+y0-12)),'VLM bbox',fill='#32e6ff')
        return np.asarray(canvas)

    def _save_image(self, path):
        from PIL import Image
        if self._snapshot is None:
            return
        path = Path(path)
        temporary = path.with_name(path.stem+'.tmp.png')
        Image.fromarray(self._frame(self.snapshot())).save(temporary)
        temporary.replace(path)

    def _diagnostic_only(self):
        # This function's output goes only to disk. No control method reads it.
        try:
            rows=[]
            for obj in self.base.cubes:
                pos=array(obj.pose.p)[0]
                inside=bool(np.all(np.abs(pos[:2]-np.asarray(ZONE['center']))<=np.asarray(ZONE['half_sizes'])))
                rows.append(dict(name=obj.name,position=pos.tolist(),in_collection_area=inside,
                                 grasped=bool(self.base.agent.is_grasping(obj).item())))
            self.diagnostics.write(json.dumps(dict(step=self.steps,objects=rows,diagnostic_only=True))+'\n')
        except Exception as error:
            self.diagnostics.write(json.dumps(dict(step=self.steps,diagnostic_error=type(error).__name__))+'\n')

    def _step(self, action):
        self._check_stop()
        self._refresh_mask_dropout()
        started = time.monotonic()
        action = np.asarray(action,dtype=np.float32).reshape(-1)
        if action.shape!=(8,) or not np.isfinite(action).all():
            raise ValueError('Expected eight finite Panda joint/gripper targets')
        action = np.clip(action,self.raw.action_space.low,self.raw.action_space.high)
        result = self.raw.step(action)
        self.obs = result[0]
        self._captured_at = time.monotonic()
        self.steps += 1
        self._last_gripper = float(action[-1])
        if self.perception is not None:
            base,_ = self._raw_images()
            self._tracking_frame += 1
            try:
                perceived = self.perception.step(base,self._tracking_frame)
                self._accept_perception(perceived,base.shape[:2])
            except Exception as error:
                self._natural_mask = self._mask = np.zeros(base.shape[:2],dtype=bool)
                self._natural_mask_metadata = self._mask_metadata = dict(state='error',source='vlm_sam2_cutie',error=str(error))
                self._publish()
                raise
        self.actions.write(json.dumps(dict(step=self.steps,time=time.time(),phase=self.phase,
            motion_phase=self.motion_phase,source=self.control_source,action=action.tolist(),
            step_generation=self.step_generation,perception=self._mask_metadata),ensure_ascii=False)+'\n')
        if self.steps % 5 == 0:
            self._diagnostic_only()
        self._publish()
        remaining = 1/self.pace_hz - (time.monotonic()-started)
        if remaining > 0:
            time.sleep(remaining)
        self._check_stop()
        return result

    def _accept_perception(self, result, shape):
        mask = np.asarray(result.mask)
        if mask.shape!=shape or mask.dtype!=np.bool_:
            raise ValueError('Perception must provide an RGB-aligned bool mask')
        self._mask = mask.copy()
        self._mask_metadata = dict(result.metadata)
        if self._last_target_bbox is not None:
            # The shared service receives the padded prompt. Preserve that
            # provenance locally rather than relabeling it as a model output.
            self._mask_metadata.update(
                task_vlm_bbox_xyxy=list(self._last_target_bbox),
                vlm_bbox_xyxy=list(self._last_target_bbox),
                sam_prompt_bbox_xyxy=list(self._sam_prompt_bbox),
                bbox_initialization_source='task_vlm_bbox_with_sam_padding')
            events=[]
            for event in self._mask_metadata.get('events',[]):
                event=dict(event)
                if event.get('type') in ('vlm_bbox','initialized'):
                    event.update(task_vlm_bbox_xyxy=list(self._last_target_bbox),
                        sam_prompt_bbox_xyxy=list(self._sam_prompt_bbox),
                        bbox_source='task_vlm_bbox_with_sam_padding')
                    if event.get('type')=='vlm_bbox':
                        event.update(origin='upstream_task_vlm_padded_by_task_backend',
                            reason='Task VLM selected target; task backend padded SAM2 prompt')
                events.append(event)
            if 'events' in self._mask_metadata:self._mask_metadata['events']=events
        self._natural_mask=self._mask.copy()
        self._natural_mask_metadata=dict(self._mask_metadata)
        self._refresh_mask_dropout()

    def _close_perception(self):
        client,self.perception = self.perception,None
        if client:
            try:
                client.close()
            except Exception as error:
                with (self.output/'cleanup_errors.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(error=str(error),component='perception_session'))+'\n')

    def _start_perception(self, target):
        from perception_mask.client import PerceptionMaskClient
        forbidden={'object_id','actor_id','target_idx','gt_mask','segmentation','target_pose','world_position'}
        if not isinstance(target,dict) or forbidden.intersection(target):
            raise ValueError('A task skill accepts a visual description/bbox, not simulator identities or state')
        text = target.get('description')
        if not isinstance(text,str) or not text.strip():
            raise ValueError('A visible target description is required')
        self._close_perception()
        client = PerceptionMaskClient(self.perception_url,timeout=180)
        base,_ = self._raw_images()
        from perception_mask.core import validate_box
        self._last_target_bbox=validate_box(target['bbox_xyxy'],base.shape[1],base.shape[0],'pixels')
        self._sam_prompt_bbox=sam_prompt_bbox(self._last_target_bbox,base.shape[1],base.shape[0])
        try:
            result = client.start_from_vlm(base,text,self._sam_prompt_bbox,
                                           options={'loss_mode':'observe','single_target':True,
                                                    'single_target_bbox':list(self._last_target_bbox)},
                                           request_id=target.get('request_id'))
            self._accept_perception(result,base.shape[:2])
        except BaseException:
            try:client.close()
            finally:raise
        self.perception = client
        self._tracking_frame = 0
        self._publish(record_video=False)

    def _pi_prediction(self):
        if not getattr(self,'_policy_bound',True):raise RuntimeError('Scene policy has not been bound')
        base,wrist = self._raw_images()
        target = np.where(self._mask[...,None],base,0).astype(np.uint8)
        seed = sampling_seed(self.seed,self.steps)
        data = dict(state=pi_state_from_observation(self.obs).tolist(),noise_seed=seed)
        for key,image in [('base_rgb',base),('wrist_rgb',wrist),('target_rgb',target)]:
            data[key] = base64.b64encode(np.ascontiguousarray(image).tobytes()).decode('ascii')
        request = Request(self.pi05_url+'/infer',data=json.dumps(data).encode(),
                          headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=180) as response:
            result = json.load(response)
        if result.get('noise_seed')!=seed:
            raise RuntimeError('Pi worker did not honor deterministic sampling noise')
        chunk = np.asarray(result['actions'],dtype=np.float32)
        if chunk.ndim!=2 or chunk.shape[1]!=8 or not len(chunk) or not np.isfinite(chunk).all():
            raise RuntimeError('Invalid pi0.5 action output')
        return chunk

    def _act_prediction(self):
        if not getattr(self,'_policy_bound',True):raise RuntimeError('Scene policy has not been bound')
        if self.policy_name=='act_triple':
            return self._act_triple_prediction()
        import torch
        import torchvision.transforms as T
        from train_dual_rgb import dual_rgb,policy_state
        base,_ = self._raw_images()
        masked = np.where(self._mask[...,None],base,0).astype(np.uint8)
        obs = dict(rgb=dual_rgb(base[None],masked[None],T.Resize((224,224),antialias=True)).to(self.device),
                   state=policy_state(self.obs).to(self.device))
        obs['state'] = (obs['state']-self._act_stats['state_mean'])/self._act_stats['state_std']
        with torch.inference_mode():
            prediction = self._act_agent.get_action(obs)
        if prediction.shape!=(1,self._act_args.num_queries,8) or not torch.isfinite(prediction).all():
            raise RuntimeError('Invalid ACT action output')
        return prediction

    def _act_triple_prediction(self):
        import torch
        from evaluate_triple_rgb import policy_state,triple_rgb
        base,wrist=self._raw_images()
        # _mask is the effective current SAM2/Cutie mask, including active
        # interventions. Never read natural masks or simulator segmentation.
        mask=np.zeros(base.shape[:2],dtype=bool) if self._mask is None else self._mask.copy()
        approved=dict(agent={key:self.obs['agent'][key] for key in ('qpos','qvel')},
            extra=dict(tcp_pose=self.obs['extra']['tcp_pose'],goal_pos=np.array([[0.,0.,.2]],np.float32)))
        obs=dict(state=policy_state(approved),
            rgb=triple_rgb(base[None],wrist[None],mask[None,...,None]))
        # Preserve the shared ACT controller's normalized chunk interface;
        # grasp() applies inverse action normalization after aggregation.
        _,prediction=self._act_policy.predict(obs,return_normalized=True)
        if tuple(prediction.shape)!=(1,self._act_args.num_queries,8) or not torch.isfinite(prediction).all():
            raise RuntimeError('Invalid three-stream ACT action output')
        return prediction

    def grasp(self, target, should_stop=None, validate_mask=None):
        if not getattr(self,'_policy_bound',True):raise RuntimeError('Scene policy has not been bound')
        self._check_stop()
        start_step,started = self.steps,time.monotonic()
        self.control_source = self.policy_name
        self.motion_phase = 'grasp_lift'
        self._start_perception(target)
        # Reject semantic segmentation errors before any learned action. The
        # callback sees only public RGB and the perceived mask, never GT state.
        if validate_mask is not None:
            validation=validate_mask(self.snapshot())
            self._check_stop()
            if not isinstance(validation,dict) or type(validation.get('valid')) is not bool:
                raise ValueError('Visual mask validation must return a boolean valid field')
            if not validation['valid']:
                return dict(reason='mask_target_mismatch',steps=self.steps-start_step,
                    evidence=str(validation.get('reason','Visual mask validation rejected target')),
                    validation=validation)
        engine=getattr(self,'_mask_dropout',None)
        if engine is not None:
            engine.begin_attempt(self.step_generation)
            self._refresh_mask_dropout()
            self._publish(record_video=False)
        natural=getattr(self,'_natural_mask',self._mask)
        natural_metadata=getattr(self,'_natural_mask_metadata',self._mask_metadata)
        history=[];chunk=None;offset=0;last_generation=natural_metadata.get('generation',0);lost_since=None
        was_lost=None
        seen_tracking=natural is not None and natural.any() and natural_metadata.get('state')=='tracking'
        mask_shape=None if natural is None else natural.shape
        self._decider = should_stop
        try:
            for tick in range(self.max_grasp_steps):
                self._check_stop()
                self._refresh_mask_dropout()
                if time.monotonic()-started > self.max_grasp_seconds:
                    return dict(reason='grasp_wall_timeout',steps=self.steps-start_step)
                generation = self._mask_metadata.get('generation',0)
                if generation!=last_generation:
                    history=[];chunk=None;offset=0;last_generation=generation
                    seen_tracking=False;was_lost=None
                natural=getattr(self,'_natural_mask',self._mask)
                natural_metadata=getattr(self,'_natural_mask_metadata',self._mask_metadata)
                if natural is not None and natural.any() and natural_metadata.get('state')=='tracking':
                    seen_tracking=True;mask_shape=natural.shape
                lost = self._mask is None or not self._mask.any() or self._mask_metadata.get('state')!='tracking'
                if was_lost is not None and lost!=was_lost:
                    # Cached actions were conditioned on another visibility
                    # state. Re-query the learned policy on both transitions.
                    history=[];chunk=None;offset=0
                was_lost=lost
                if lost:
                    lost_since = time.monotonic() if lost_since is None else lost_since
                    if time.monotonic()-lost_since > self.mask_lost_timeout:
                        return dict(reason='mask_lost_timeout',steps=self.steps-start_step)
                    if not seen_tracking:
                        self.control_source='hold_for_perception'
                        self._step(hold_action_from_observation(self.obs))
                        continue
                    # Observe mode exposes genuine loss to the learned model.
                    # Short occlusions may resolve after lifting; freezing at
                    # first occlusion would prevent that physical recovery.
                    self._mask=np.zeros(mask_shape,dtype=bool)
                    self.control_source=self.policy_name+'_zero_mask'
                else:
                    seen_tracking=True;mask_shape=self._mask.shape
                    lost_since=None;self.control_source=self.policy_name
                if self.policy_name=='pi05':
                    if chunk is None or offset>=min(10,len(chunk)):
                        chunk=self._pi_prediction();offset=0
                    raw=chunk[offset];offset+=1
                    action=np.r_[raw[:7],1-2*np.clip(raw[7],0,1)]
                else:
                    import torch
                    q=self._act_args.num_queries
                    temporal_agg=getattr(self._act_args,'temporal_agg',True)
                    if temporal_agg or chunk is None or offset>=q:
                        chunk=self._act_prediction();offset=0
                    if temporal_agg:
                        history.append((tick,chunk));history=[(t,p) for t,p in history if tick-t<q]
                        current=torch.stack([p[:,tick-t] for t,p in history],dim=1)
                        weights=torch.exp(-.01*torch.arange(len(history),device=self.device))
                        raw=(current*(weights/weights.sum())[None,:,None]).sum(dim=1)
                    else:
                        raw=chunk[:,offset];offset+=1
                    action=array(raw*self._act_stats['action_std']+self._act_stats['action_mean'])[0]
                self._step(action)
            return dict(reason='grasp_step_timeout',steps=self.steps-start_step)
        except _SkillInterrupted as interrupted:
            return dict(reason=interrupted.reason,steps=self.steps-start_step)
        finally:
            self._decider=None
            if engine is not None:
                engine.end_attempt('grasp_finished')
                self._refresh_mask_dropout()
                self._publish(record_video=False)

    def _planner(self):
        from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
        planner=PandaArmMotionPlanningSolver(self.env,vis=False,debug=False,
            base_pose=self.base.agent.robot.pose,visualize_target_grasp_pose=False,
            print_env_info=False,joint_vel_limits=.55,joint_acc_limits=.55)
        planner.gripper_state=-1 if self._last_gripper<0 else 1
        return planner

    def _move(self, planner, pose, motion_phase):
        self.motion_phase=motion_phase;self.control_source='mplib'
        result=planner.move_to_pose_with_screw(pose,refine_steps=5)
        if isinstance(result,int):
            result=planner.move_to_pose_with_RRTConnect(pose,refine_steps=5)
        if isinstance(result,int):
            raise RuntimeError('motion_plan_failed:'+motion_phase)

    def _return_joint_home(self, planner, motion_phase):
        """Physically return the arm to its recorded initial joint branch."""
        self._check_stop()
        self.motion_phase=motion_phase;self.control_source='mplib_joint_home'
        current=array(self.obs['agent']['qpos'])[0].copy()
        goal=current.copy();goal[:7]=self.initial_qpos[:7]
        planner.gripper_state=planner.OPEN
        result=planner.planner.plan_qpos_to_qpos([goal],current.copy(),
            time_step=self.base.control_timestep,planning_time=2,
            planner_name='RRTConnect',use_point_cloud=planner.use_point_cloud)
        if result.get('status')!='Success':
            raise RuntimeError('initial_joint_home_plan_failed:'+str(result.get('status')))
        # The solver's existing env is PublishedEnv, so cancellation, fresh
        # cameras and tracking remain active for every physical trajectory step.
        planner.follow_path(result,refine_steps=10)
        self._check_stop()
        actual=array(self.obs['agent']['qpos'])[0,:7]
        max_error=float(np.max(np.abs(actual-self.initial_qpos[:7])))
        if not np.isfinite(max_error) or max_error>.02:
            raise RuntimeError(f'initial_joint_home_not_reached:max_error_rad={max_error:.6f},limit=0.020000')
        return max_error

    def transport_place(self, slot_index, should_stop=None):
        import sapien
        position=deposit_position(slot_index)
        start=self.steps;self._decider=should_stop
        planner=None
        try:
            self._check_stop();planner=self._planner();planner.gripper_state=planner.CLOSED
            tcp=self.base.agent.tcp.pose.sp
            raised=tcp.p.copy();raised[2]=max(float(raised[2]),.30)
            self._move(planner,sapien.Pose(raised,tcp.q),'transport_raise')
            self._move(planner,sapien.Pose([position[0],position[1],.30],tcp.q),'transport_to_green_zone')
            self._move(planner,sapien.Pose(position,tcp.q),'deposit_lower')
            # Settle at the calibrated release pose before opening. Keep a
            # closed command even when a held object's width separates fingers.
            self.control_source='hold_closed';self.motion_phase='deposit_settle_closed'
            settled_action=hold_action_from_observation(self.obs);settled_action[-1]=planner.CLOSED
            for _ in range(15):self._step(settled_action)
            self.control_source='scripted_release';self.motion_phase='release'
            planner.open_gripper(t=15)
            self._move(planner,sapien.Pose([position[0],position[1],.30],tcp.q),'retreat_raise')
            home_error=self._return_joint_home(planner,'retreat_joint_home')
            self.motion_phase='settle';self.control_source='hold'
            for _ in range(20):self._step(hold_action_from_observation(self.obs))
            return dict(reason='placement_motion_completed',steps=self.steps-start,slot=slot_index,
                        home_joint_max_error_rad=home_error)
        except _SkillInterrupted as interrupted:
            return dict(reason=interrupted.reason,steps=self.steps-start,slot=slot_index)
        finally:
            self._decider=None
            if planner is not None and hasattr(planner,'close'):
                planner.close()

    def hold(self, seconds=1.0, should_stop=None):
        deadline=time.monotonic()+max(0,float(seconds));start=self.steps;self._decider=should_stop
        previous=self.control_source;self.control_source='hold'
        try:
            while time.monotonic()<deadline:self._step(hold_action_from_observation(self.obs))
            return dict(reason='hold_completed',steps=self.steps-start)
        except _SkillInterrupted as interrupted:
            return dict(reason=interrupted.reason,steps=self.steps-start)
        finally:
            self._decider=None;self.control_source=previous

    def recover_open_home(self, should_stop=None):
        import sapien
        start=self.steps;self._decider=should_stop;planner=None
        try:
            self._check_stop();planner=self._planner()
            self.control_source='recovery';self.motion_phase='release'
            planner.open_gripper(t=10)
            tcp=self.base.agent.tcp.pose.sp
            raised=tcp.p.copy();raised[2]=max(float(raised[2]),.30)
            self._move(planner,sapien.Pose(raised,tcp.q),'recovery_raise')
            home_error=self._return_joint_home(planner,'recovery_joint_home')
            self._close_perception();self._mask=None
            self._natural_mask=None
            self._mask_metadata=dict(state='idle',source='none',generation=0)
            self._natural_mask_metadata=dict(self._mask_metadata)
            self._publish(record_video=False)
            return dict(reason='recovery_motion_completed',steps=self.steps-start,
                        home_joint_max_error_rad=home_error)
        except _SkillInterrupted as interrupted:
            return dict(reason=interrupted.reason,steps=self.steps-start)
        finally:
            self._decider=None
            if planner is not None and hasattr(planner,'close'):planner.close()

    def close(self):
        if self._closed:return
        self._closed=True
        self._decider=None
        engine=getattr(self,'_mask_dropout',None)
        if engine is not None:
            engine.close()
            self._refresh_mask_dropout()
            if getattr(self,'_snapshot',None) is not None:
                self._publish(record_video=False)
        try:self._save_image(self.output/'final.png')
        except Exception:pass
        self._close_perception()
        errors=[]
        for name in ('writer','raw','actions','diagnostics'):
            item=getattr(self,name,None)
            if item is not None:
                try:item.close()
                except Exception as error:errors.append(dict(component=name,error=str(error)))
        self._act_agent=self._act_args=self._act_stats=self._act_policy=None
        if errors:(self.output/'backend_cleanup.json').write_text(json.dumps(errors,indent=2))
