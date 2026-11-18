"""A genuine RoboCasa kitchen with a fixed counter task and custom task objects."""
import os
import numpy as np
from pathlib import Path


OBJECTS = ("red_cube", "blue_cube")
def validate_order(order, targets=None):
    targets=list(OBJECTS if targets is None else targets)
    order=list(targets if order is None else order)
    if len(order)!=len(targets) or len(set(order))!=len(order) or set(order)!=set(targets):
        raise ValueError("Order must contain each selected object exactly once")
    if not 1<=len(order)<=6:raise ValueError("Select 1 to 6 tabletop targets")
    return order

def make_env(seed=17, order=None, layout_id=11, style_id=1, selected_objects=None):
    import robocasa
    import robosuite.macros as rs_macros
    from robocasa.environments.kitchen.kitchen import Kitchen
    from robocasa.models.fixtures import Counter
    from robosuite.models.objects import BoxObject, MujocoXMLObject
    from robosuite.controllers import load_composite_controller_config
    import xml.etree.ElementTree as ET
    def new_camera(name,pos,quat,fovy):
        return ET.Element("camera",dict(name=name,pos=" ".join(map(str,pos)),quat=" ".join(map(str,quat)),fovy=str(fovy)))
    from scipy.spatial.transform import Rotation

    # One canonical image orientation for RGB and segmentation, no wrapper double-flip.
    rs_macros.IMAGE_CONVENTION = "opencv"

    class OrderedCounterTray(Kitchen):
        def __init__(self, **kwargs):
            self.selected_objects=list(selected_objects or [])
            self.task_targets=[x["id"] for x in self.selected_objects] if self.selected_objects else list(OBJECTS)
            self.task_order = validate_order(order,self.task_targets)
            self.task_mode="official_tabletop_v1" if self.selected_objects else "ordered_counter_to_tray_v1"
            super().__init__(**kwargs)

        def _setup_kitchen_references(self):
            super()._setup_kitchen_references()
            candidates=[]
            for f in self.fixtures.values():
                if not isinstance(f,Counter) or f.has_opening or any(f.half_top):
                    continue
                for region in f.get_reset_regions(self,top_size=(.92,.55) if self.selected_objects else (.65,.32)).values():
                    candidates.append((f,region))
            if not candidates:
                raise RuntimeError("This fixed layout has no solid counter region large enough for the task")
            # A real top geom region excludes sink/stove cutouts.
            self.counter,self.reset_region=max(candidates,key=lambda v: float(v[1]["size"][0]*v[1]["size"][1]))
            self.fixture_refs["counter"] = self.counter
            self.init_robot_base_ref = self.counter

        def _create_objects(self):
            self.object_cfgs = []
            self.objects = {
                "red_cube": BoxObject("red_cube", size=[.022,.022,.022],
                                     rgba=[.85,.05,.03,1], density=400,
                                     friction=[1.,.01,.001]),
                "blue_cube": BoxObject("blue_cube", size=[.022,.022,.022],
                                      rgba=[.04,.15,.85,1], density=400,
                                      friction=[1.,.01,.001]),
                "collection_tray": MujocoXMLObject(
                    str(Path(__file__).resolve().parent / "assets/tray.xml"), name="collection_tray",
                    joints=[dict(type="free", damping=".005")],
                    obj_type="all", duplicate_collision_geoms=True),
            }
            if self.selected_objects:
                from .catalog import build_object
                tray=self.objects["collection_tray"]
                self.objects={x["id"]:build_object(x) for x in self.selected_objects}
                self.objects["collection_tray"]=tray
            self.model.merge_objects(list(self.objects.values()))
            p0, px, py, pz = map(np.asarray,self.counter.get_ext_sites(relative=False))
            self.counter_x = (px-p0)/np.linalg.norm(px-p0)
            self.counter_y = (py-p0)/np.linalg.norm(py-p0)
            from robocasa.utils.object_utils import get_pos_after_rel_offset
            self.counter_top=np.asarray(get_pos_after_rel_offset(self.counter,self.reset_region["offset"]),float)
            self.task_origin = self.counter_top.copy()
            margin=.03 if self.selected_objects else .10
            self.task_origin += self.counter_y * (-min(margin,max(0.,float(self.reset_region["size"][1])*.5-.14)))
            self.tray_initial = self.task_origin + self.counter_x*(.30 if self.selected_objects else .20) + [0,0,.008]
            self.tabletop_slots=[(x,y) for y in (-.10,.10) for x in (-.28,-.11,.06)]
            if self.selected_objects:
                # Explicit AABBs in counter coordinates prevent overlaps or
                # partial placement over the edge, even with six targets.
                region_size=np.asarray(self.reset_region["size"][:2],float)
                boxes=[]
                for info,(x,y) in zip(self.selected_objects,self.tabletop_slots):
                    size=np.asarray(info["bbox_size_m"][:2],float)
                    boxes.append((info["id"],np.array([x,y-margin]),size/2))
                boxes.append(("collection_tray",np.array([.30,-margin]),np.array([.15,.111])))
                for name,xy,half in boxes:
                    if (np.abs(xy)+half+.006>region_size/2).any():
                        raise ValueError(f"{name} does not fit the selected solid countertop")
                for i,(name,a,ha) in enumerate(boxes):
                    for other,b,hb in boxes[i+1:]:
                        if (np.abs(a-b)<ha+hb+.012).all():
                            raise ValueError(f"Tabletop placement overlaps: {name}, {other}")

            # World-fixed overview; unlike policy cameras it is not attached to the robot.
            eye=self.task_origin-self.counter_y*.8+np.array([0,0,.8])
            target=self.task_origin+np.array([0,0,.03])
            z=(eye-target);z/=np.linalg.norm(z)
            x=np.cross([0,0,1],z);x/=np.linalg.norm(x)
            y=np.cross(z,x)
            quat=Rotation.from_matrix(np.stack([x,y,z],axis=1)).as_quat()[[3,0,1,2]]
            self.model.worldbody.append(new_camera("task_overview", pos=eye, quat=quat, fovy="55"))
            # Calibrated diagonal view: avoids the robot's back/shoulder.
            # Validated on both complete pick-place trajectories, not just reset.
            target=self.task_origin-self.counter_x*(.07/3)+self.counter_y*(.07/3)+np.array([0,0,.052])
            eye=target+np.array([.55,.40,1.0])
            z=(eye-target);z/=np.linalg.norm(z)
            x=np.cross([0,0,1],z);x/=np.linalg.norm(x)
            y=np.cross(z,x)
            quat=Rotation.from_matrix(np.stack([x,y,z],axis=1)).as_quat()[[3,0,1,2]]
            self.model.worldbody.append(new_camera("policy_base", pos=eye, quat=quat, fovy="45"))

        def _setup_scene(self):
            super()._setup_scene()
            if self.deterministic_reset:
                return
            from scipy.spatial.transform import Rotation
            yaw=float(np.arctan2(self.counter_x[1],self.counter_x[0]))
            quat=Rotation.from_euler("z",yaw).as_quat()[[3,0,1,2]]
            positions={
                "red_cube":self.task_origin-self.counter_x*.20-self.counter_y*.02+[0,0,.024],
                "blue_cube":self.task_origin-self.counter_x*.07+self.counter_y*.09+[0,0,.024],
                "collection_tray":self.tray_initial,
            }
            if self.selected_objects:
                positions={"collection_tray":self.tray_initial}
                rot=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                for info,(x,y) in zip(self.selected_objects,self.tabletop_slots):
                    obj=self.objects[info["id"]]
                    bottom=np.asarray(obj.bottom_offset,float)
                    slot=self.task_origin+self.counter_x*x+self.counter_y*y
                    positions[info["id"]]=slot+[0,0,.004]-rot@bottom
            self.task_initial_poses={}
            for name,pos in positions.items():
                pose=np.r_[pos,quat]
                self.sim.data.set_joint_qpos(self.objects[name].joints[0],pose)
                self.task_initial_poses[name]=pose.tolist()

        def _check_success(self):
            # Ordered, stable placement is evaluated separately by the bridge.
            return False

        def get_ep_meta(self):
            meta=super().get_ep_meta()
            meta["bridge_task"]=self.task_mode
            meta["selected_objects"]=list(self.selected_objects)
            meta["task_order"]=list(self.task_order)
            meta["lang"]="Place "+", then ".join(self.task_order)+" into the green tray in order."
            return meta

    config=load_composite_controller_config(robot="PandaOmron")
    config["body_parts"]["right"]["input_ref_frame"]="world"
    config["body_parts"]["right"]["input_type"]="delta"
    env=OrderedCounterTray(robots="PandaOmron", controller_configs=config,
        layout_ids=[int(layout_id)],style_ids=[int(style_id)], seed=int(seed),
        initialization_noise=None, robot_spawn_deviation_pos_x=0.,
        robot_spawn_deviation_pos_y=0.,robot_spawn_deviation_rot=0.,
        use_camera_obs=True,use_object_obs=True,has_renderer=False,
        has_offscreen_renderer=True,renderer="mujoco",
        render_gpu_device_id=int(os.environ.get("ROBOCASA_RENDER_GPU",os.environ.get("MUJOCO_EGL_DEVICE_ID","0"))),
        control_freq=20,horizon=3000,ignore_done=True,randomize_cameras=False,
        camera_names=["policy_base","robot0_eye_in_hand","task_overview"],
        camera_heights=[224,224,448],camera_widths=[224,224,448])
    env.reset()
    initialize_workspace(env)
    # Multisample rasterization on this H20 introduces static RGB variation.
    # Disable MSAA and rebuild only the offscreen context before episode capture.
    env.sim.model.vis.quality.offsamples=0
    context=env.sim._render_context_offscreen
    if context is not None:
        context.con.free()
        context._set_mujoco_context_and_buffers()
    env.sim.forward()
    return env

def initialize_workspace(env):
    """Reset-only IK preparation; no object or robot pose writes during demos."""
    import mujoco
    from scipy.spatial.transform import Rotation
    # Restore the task objects to the explicitly selected solid top region after
    # RoboCasa's reset settling. These writes occur only before episode capture.
    env._setup_scene()
    env.sim.data.qvel[:]=0.
    env.sim.forward()
    from robocasa.utils.env_utils import set_robot_to_position
    base_pos=env.task_origin-env.counter_y*.55
    base_pos[2]=0.
    set_robot_to_position(env,base_pos)
    env.init_robot_base_pos=base_pos.copy()
    controller=env.robots[0].composite_controller.part_controllers["right"]
    sid=env.robots[0].eef_site_id["right"]
    desired=np.stack([env.counter_x,-env.counter_y,np.array([0,0,-1.])],axis=1)
    goal=env.task_origin+np.array([0,0,.28])
    model=env.sim.model._model;data=env.sim.data._data
    jpos=np.zeros((3,model.nv));jori=np.zeros((3,model.nv))
    ranges=env.sim.model.jnt_range[controller.joint_index]
    original=env.sim.data.qpos[controller.qpos_index].copy()
    best=(float("inf"),original.copy())
    seeds=[original]
    for yaw in (1.5,-1.5,2.6,-2.6):
        candidate=original.copy();candidate[0]=yaw
        seeds.append(candidate)
    for seed in seeds:
        env.sim.data.qpos[controller.qpos_index]=seed
        for _ in range(180):
            env.sim.forward()
            current=env.sim.data.site_xmat[sid].reshape(3,3)
            err=np.r_[goal-env.sim.data.site_xpos[sid],
                      Rotation.from_matrix(desired@current.T).as_rotvec()*.35]
            if np.linalg.norm(err)<best[0]:
                best=(float(np.linalg.norm(err)),env.sim.data.qpos[controller.qpos_index].copy())
            if np.linalg.norm(err)<.003:break
            mujoco.mj_jacSite(model,data,jpos,jori,sid)
            jac=np.vstack([jpos[:,controller.qvel_index],jori[:,controller.qvel_index]*.35])
            dq=jac.T@np.linalg.solve(jac@jac.T+.001*np.eye(6),err)
            env.sim.data.qpos[controller.qpos_index]=np.clip(
                env.sim.data.qpos[controller.qpos_index]+np.clip(dq,-.15,.15),
                ranges[:,0]+.02,ranges[:,1]-.02)
        if best[0]<.003:break
    env.sim.data.qpos[controller.qpos_index]=best[1]
    env.sim.data.qvel[:]=0.;env.sim.forward()
    residual=float(np.linalg.norm(goal-env.sim.data.site_xpos[sid]))
    if residual>.04:
        raise RuntimeError(f"Reset-only IK workspace residual {residual:.3f}m; counter={env.counter.name}, goal={goal}, base={base_pos}")
    for c in env.robots[0].composite_controller.part_controllers.values():
        if hasattr(c,"update"):c.update(force=True)
        if hasattr(c,"reset_goal"):c.reset_goal()
    hold=env.robots[0].create_action_vector({"right":np.zeros(6),"right_gripper":np.array([-1.]),
        "base":np.zeros(3),"torso":np.zeros(1),"base_mode":-1.})
    for _ in range(12):env.step(hold)
    env.sim.forward()
    # Fail early if a selected top region still contains an occluding fixture.
    for name in env.task_targets:
        z=env.sim.data.body_xpos[env.obj_body_id[name]][2]
        if z<env.task_origin[2]+(-.06 if env.selected_objects else .012):
            raise RuntimeError(f"Task object {name} did not settle on the kitchen counter")

