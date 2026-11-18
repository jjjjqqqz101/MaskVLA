"""Independent selectable scene; existing training environments stay untouched."""
import math

import numpy as np


LAYOUT_VERSION='aabb-upright-v1'
WORKSPACE_MIN=np.array([-.12,-.28],dtype=float)
WORKSPACE_MAX=np.array([.28,.28],dtype=float)
CENTER_MIN=np.array([-np.inf,-.22],dtype=float)
CENTER_MAX=np.array([.18,.22],dtype=float)
_REGISTERED=False


def _footprint(spec,yaw):
    lower=np.asarray(spec['bbox_min'],dtype=float)[:2]
    upper=np.asarray(spec['bbox_max'],dtype=float)[:2]
    corners=np.array([[lower[0],lower[1]],[lower[0],upper[1]],
                      [upper[0],lower[1]],[upper[0],upper[1]]])
    cosine,sine=math.cos(yaw),math.sin(yaw)
    rotated=corners@np.array([[cosine,sine],[-sine,cosine]])
    return rotated.min(axis=0),rotated.max(axis=0)


def pack_layout(specs,seed=None,rng=None,layout_attempts=64,candidate_attempts=256,gap=.01):
    """Bounded initial AABB packing, based only on static asset geometry.

    Return actor origins/yaw in the original requested order. No runtime poses,
    segmentation or scene targets are used for this initialization procedure.
    """
    if not 1<=len(specs)<=8:raise ValueError('Custom scenes require 1..8 objects')
    for value in (layout_attempts,candidate_attempts):
        if type(value) is not int or value<1:raise ValueError('Packing retry bounds must be positive integers')
    if not np.isfinite(gap) or gap<0:raise ValueError('Invalid packing clearance')
    for spec in specs:
        lower=np.asarray(spec['bbox_min'],dtype=float);upper=np.asarray(spec['bbox_max'],dtype=float)
        if lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all() or not (upper>lower).all():
            raise ValueError('Invalid static asset bounds')
        if spec['kind'] not in ('cube','ycb'):raise ValueError('Unknown asset kind')
    rng=rng if rng is not None else np.random.default_rng(seed)
    for _ in range(layout_attempts):
        footprints=[]
        for index,spec in enumerate(specs):
            yaw=float(rng.uniform(0,2*math.pi)) if spec['kind']=='cube' else 0.
            lower,upper=_footprint(spec,yaw)
            footprints.append((index,yaw,lower,upper))
        order=sorted(footprints,key=lambda item:-float(np.prod(item[3]-item[2])))
        placed=[];result=[None]*len(specs)
        for index,yaw,lower,upper in order:
            half=(upper-lower)/2;local_center=(upper+lower)/2
            low=np.maximum(WORKSPACE_MIN+half,CENTER_MIN)
            high=np.minimum(WORKSPACE_MAX-half,CENTER_MAX)
            if np.any(low>high):break
            for _ in range(candidate_attempts):
                center=np.asarray(rng.uniform(low,high),dtype=float)
                minimum,maximum=center-half,center+half
                if all(maximum[0]+gap<=a[0] or b[0]+gap<=minimum[0]
                       or maximum[1]+gap<=a[1] or b[1]+gap<=minimum[1] for a,b in placed):
                    origin=center-local_center
                    result[index]=dict(x=float(origin[0]),y=float(origin[1]),yaw=yaw,
                        aabb_min=minimum.tolist(),aabb_max=maximum.tolist())
                    placed.append((minimum,maximum));break
            else:break
        if all(item is not None for item in result):return result
    raise ValueError('所选物品无法在当前桌面区域无重叠摆放，请减少物品或更换组合')


def register_selectable_scene(zone,overview_camera):
    global _REGISTERED
    if _REGISTERED:return
    import sapien
    import torch
    from mani_skill import ASSET_DIR
    from mani_skill.envs.tasks.tabletop.target_pick_eight import TargetPickEightEnv
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.building import actors
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder.table import TableSceneBuilder
    from mani_skill.utils.structs.pose import Pose
    from .scene_objects import get_object_spec,normalize_scene_spec

    @register_env('TaskCollectionSelectable-v1',max_episode_steps=10000,asset_download_ids=['ycb'])
    class TaskCollectionSelectableEnv(TargetPickEightEnv):
        SUPPORTED_ROBOTS=['panda','panda_wristcam']
        layout_version=LAYOUT_VERSION

        def __init__(self,*args,scene_objects,**kwargs):
            canonical=normalize_scene_spec(dict(version=1,preset='custom',objects=scene_objects))
            self.scene_objects=canonical['objects']
            self.selected_names=tuple(item['id'] for item in self.scene_objects)
            self.num_cubes=len(self.selected_names)
            self.selected_target_idx=0
            self._static_specs=[get_object_spec(item['id']) for item in self.scene_objects]
            self.object_labels=tuple(spec['name'] for spec in self._static_specs)
            super().__init__(*args,**kwargs)

        @property
        def _default_sensor_configs(self):
            overview=CameraConfig('overview_camera',sapien_utils.look_at(
                eye=overview_camera['eye'],target=overview_camera['target']),
                overview_camera['width'],overview_camera['height'],overview_camera['fov'],.01,100,
                shader_pack='minimal')
            return [*super()._default_sensor_configs,overview]

        def _load_scene(self,options):
            self.table_scene=TableSceneBuilder(self,robot_init_qpos_noise=self.robot_init_qpos_noise)
            self.table_scene.build();self.cubes=[];self.object_rest_z=[]
            for item,spec in zip(self.scene_objects,self._static_specs):
                rest_z=spec['rest_z_m']
                if spec['kind']=='cube':
                    actor=actors.build_cube(self.scene,half_size=spec['half_size_m'],color=spec['color'],
                        name=item['id'],initial_pose=sapien.Pose(p=[0,0,rest_z]))
                else:
                    folder=ASSET_DIR/'assets/mani_skill2_ycb/models'/spec['asset_id']
                    builder=self.scene.create_actor_builder()
                    builder.add_multiple_convex_collisions_from_file(filename=str(folder/'collision.ply'),
                        scale=[item['scale']]*3,density=spec['density'])
                    builder.add_visual_from_file(filename=str(folder/'textured.obj'),scale=[item['scale']]*3)
                    builder.initial_pose=sapien.Pose(p=[0,0,rest_z]);actor=builder.build(name=item['id'])
                self.cubes.append(actor);self.object_rest_z.append(rest_z)
            self.goal_site=actors.build_sphere(self.scene,radius=self.goal_thresh,color=[0,1,0,1],
                name='goal_site',body_type='kinematic',add_collision=False,
                initial_pose=sapien.Pose(p=[0,0,.2]))
            self._hidden_objects.append(self.goal_site)
            self.target_idx=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
            self.collection_marker=actors.build_box(self.scene,half_sizes=[*zone['half_sizes'],.001],
                color=[.12,.64,.26,1],name='green_collection_area',body_type='kinematic',
                add_collision=False,initial_pose=sapien.Pose(p=[*zone['center'],.0012]))

        def _initialize_episode(self,env_idx,options):
            if self.num_envs!=1 or len(env_idx)!=1:raise ValueError('Selectable task scene supports one live environment')
            with torch.device(self.device):
                self.table_scene.initialize(env_idx)
                layout=pack_layout(self._static_specs,rng=self._episode_rng)
                for index,(actor,placement) in enumerate(zip(self.cubes,layout)):
                    xyz=torch.tensor([[placement['x'],placement['y'],self.object_rest_z[index]]],dtype=torch.float32)
                    yaw=placement['yaw']
                    quaternion=torch.tensor([[math.cos(yaw/2),0,0,math.sin(yaw/2)]],dtype=torch.float32)
                    actor.set_pose(Pose.create_from_pq(p=xyz,q=quaternion))
                # Inherited task metrics remain diagnostic-only; this index
                # never selects the planner/VLM target or policy mask.
                self.target_idx[env_idx]=0
                self.goal_site.set_pose(Pose.create_from_pq(p=torch.tensor([[0.,0.,.2]])))

    _REGISTERED=True
