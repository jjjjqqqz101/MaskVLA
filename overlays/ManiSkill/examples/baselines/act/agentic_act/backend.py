"""Persistent simulated ACT acquisition followed by explicit MPLib placement.

This module never changes an actor pose, rewinds a state, or resets between
skills. The simulator is the verifier and supplies object identities/masks;
VLM perception is therefore explicitly ground-truth assisted in this prototype.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
import time

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sapien
import torch

from .contracts import StopRequested


DEFAULT_ZONES = {
    "zone_A": {"center": [0.0, -0.18], "radius": 0.045, "shape": "square", "label": "A"},
    "zone_B": {"center": [0.0, 0.18], "radius": 0.045, "shape": "square", "label": "B"},
    "zone_C": {"center": [0.18, 0.0], "radius": 0.045, "shape": "square", "label": "C"},
}


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _json(value):
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return _array(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class _RecordedEnv(gym.Wrapper):
    def __init__(self, env, owner):
        super().__init__(env)
        self.owner = owner

    def step(self, action):
        self.owner._check_stop()
        result = self.env.step(action)
        self.owner._after_step(action, result)
        self.owner._check_stop()
        return result


class RobotSession:
    def __init__(self, checkpoint=None, seed=2026092500, output=None, zones=None,
                 frame_callback=None, device="cuda", act_max_steps=500, stop_file=None):
        import mani_skill.envs  # noqa: F401
        from evaluate_rgb_ablation import load_policy
        from train_dual_rgb import DualRGBObservationWrapper

        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.repo = next((parent for parent in Path(__file__).resolve().parents
                          if (parent / "examples/baselines/act").is_dir()), Path.cwd())
        # Installed package is <repo>/examples/baselines/act/agentic_act/backend.py.
        self.act_root = self.repo / "examples/baselines/act"
        if not self.act_root.is_dir():
            self.repo = Path.cwd()
            self.act_root = self.repo / "examples/baselines/act"
        self.output = Path(output or self.repo / "outputs/agentic_act").resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.zones = _json(zones or DEFAULT_ZONES)
        self.frame_callback = frame_callback
        self.stop_file = Path(stop_file) if stop_file else None
        self.act_max_steps = int(act_max_steps)
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        default_ckpt = self.act_root / "runs/TargetPickCube-v1__train_dual_rgb__1__001/checkpoints/best_eval_success_once.pt"
        self.checkpoint = Path(checkpoint or default_ckpt).resolve()
        checkpoint_data = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if tuple(checkpoint_data.get("stream_order", ())) != ("original_rgb", "masked_rgb"):
            raise ValueError("Only the existing original+masked RGB ACT checkpoint is supported")
        self.agent, self.args, self.stats = load_policy(checkpoint_data, self.device)
        meta = (self.act_root / self.args.demo_path).resolve().with_suffix(".json")
        if not meta.is_file():
            meta = self.repo / "demos/TargetPickCube-v1/motionplanning/target_pick_500_mask/dataset/train.json"
        cfg = json.loads(meta.read_text())["env_info"]["env_kwargs"]
        kwargs = dict(obs_mode="rgb+segmentation", control_mode="pd_joint_pos",
                      reward_mode="sparse", sim_backend="physx_cpu", render_mode="rgb_array",
                      reconfiguration_freq=1, max_episode_steps=10000)
        for key in ("sensor_configs", "robot_uids", "robot_init_qpos_noise"):
            if key in cfg:
                kwargs[key] = cfg[key]
        self.raw = gym.make("TargetPickCube-v1", **kwargs)
        self.base = self.raw.unwrapped
        self.steps = 0
        self.reset_count = 0
        self.control_source = "initialization"
        self.phase = "initial"
        self.active_object = None
        self.active_region = None
        self.history = collections.deque(maxlen=12)
        self.events = []
        self.actions = (self.output / "actions.jsonl").open("w", buffering=1)
        self.writer = imageio.get_writer(self.output / "execution.mp4", fps=self.base.control_freq,
                                         codec="libx264", quality=7,
                                         ffmpeg_params=["-movflags", "+faststart"])
        self.env = _RecordedEnv(self.raw, self)
        self.policy = DualRGBObservationWrapper(self.env)
        self.policy.reset(seed=self.seed)
        self.reset_count += 1
        self.initial_qpos = _array(self.base.agent.robot.get_qpos())[0].copy()
        self.initial_tcp = self.base.agent.tcp.pose.sp
        self.initial_cube_positions = [_array(c.pose.p)[0].tolist() for c in self.base.cubes]
        self.current_goal = _array(self.base.goal_site.pose.p)[0].copy()
        self._sample_state()
        self._capture()
        self.image().save(self.output / "initial.png")
        self._event("session_started", checkpoint=str(self.checkpoint), seed=self.seed,
                    initial_cube_positions=self.initial_cube_positions,
                    verifier="simulator_contact_and_geometry", reset_count=self.reset_count)

    def _check_stop(self):
        if self.stop_file and self.stop_file.exists():
            raise StopRequested("Stop requested; no further simulator actions will be issued.")

    def _event(self, event, **kwargs):
        row = _json(dict(event=event, step=self.steps, time=time.time(), **kwargs))
        self.events.append(row)
        with (self.output / "backend_events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _sample_state(self):
        positions = np.array([_array(c.pose.p)[0] for c in self.base.cubes])
        held = [bool(self.base.agent.is_grasping(c).item()) for c in self.base.cubes]
        velocities = np.array([_array(c.linear_velocity)[0] for c in self.base.cubes])
        angular = np.array([_array(c.angular_velocity)[0] for c in self.base.cubes])
        self.history.append(dict(positions=positions, held=held, velocities=velocities, angular=angular))
        return positions, held, velocities

    def _after_step(self, action, result):
        self.steps += 1
        positions, held, velocities = self._sample_state()
        row = dict(step=self.steps, source=self.control_source, phase=self.phase,
                   object_id=self.active_object, region_id=self.active_region,
                   action=_array(action).reshape(-1).tolist(), cube_positions=positions.tolist(),
                   held=held, reset_count=self.reset_count)
        self.actions.write(json.dumps(row) + "\n")
        self._capture()

    def _capture(self):
        frame = np.asarray(self.image(annotated=True))
        self.writer.append_data(frame)
        if self.frame_callback:
            self.frame_callback(frame)
        if self.steps % 10 == 0:
            temporary = self.output / "live.tmp.png"
            Image.fromarray(frame).save(temporary)
            temporary.replace(self.output / "live.png")
            status = dict(step=self.steps, phase=self.phase, control_source=self.control_source,
                          object_id=self.active_object, region_id=self.active_region)
            temporary_status = self.output / "backend_status.tmp.json"
            temporary_status.write_text(json.dumps(status))
            temporary_status.replace(self.output / "backend_status.json")

    def _project(self, xyz, raw):
        params = raw["sensor_param"]["base_camera"]
        ext = _array(params["extrinsic_cv"])[0]
        intrinsic = _array(params["intrinsic_cv"])[0]
        cam = ext @ np.r_[xyz, 1.0]
        point = intrinsic @ cam[:3]
        return tuple((point[:2] / point[2] * 3).astype(int))

    def image(self, annotated=True):
        raw = self.base.get_obs(self.base.get_info())
        rgb = _array(raw["sensor_data"]["base_camera"]["rgb"])[0]
        image = Image.fromarray(rgb).resize((672, 672), Image.Resampling.NEAREST)
        if annotated:
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except OSError:
                font = ImageFont.load_default()
            for name, zone in self.zones.items():
                x, y = zone["center"]; r = zone.get("radius", .045)
                corners = [self._project([x + dx*r, y + dy*r, .002], raw)
                           for dx, dy in [(-1,-1),(-1,1),(1,1),(1,-1)]]
                color = {"zone_A": "cyan", "zone_B": "yellow", "zone_C": "lime"}.get(name, "white")
                draw.line(corners + [corners[0]], fill=color, width=3)
                label_x = sum(p[0] for p in corners) // 4 - 30
                label_y = max(p[1] for p in corners) + 8
                draw.text((label_x, label_y), name, font=font, fill=color, stroke_width=2, stroke_fill="black")
            for index, cube in enumerate(self.base.cubes):
                point = self._project(_array(cube.pose.p)[0], raw)
                label = (point[0]-22, point[1]-38)
                draw.line([(point[0], point[1]-4), (point[0], point[1]-16)], fill="white", width=1)
                draw.text(label, f"cube{index+1}", font=font, fill="white",
                          stroke_width=2, stroke_fill="black")
            draw.rectangle((0, 0, 672, 25), fill="black")
            draw.text((5, 3), f"step {self.steps} | {self.control_source}: {self.phase}", font=font, fill="white")
        return image

    def observe(self):
        positions = np.array([_array(c.pose.p)[0] for c in self.base.cubes])
        held = [bool(self.base.agent.is_grasping(c).item()) for c in self.base.cubes]
        objects = {}
        raw = self.base.get_obs(self.base.get_info())
        segmentation = raw["sensor_data"]["base_camera"]["segmentation"]
        for index, position in enumerate(positions):
            stable = len(self.history) >= 8 and all(
                np.linalg.norm(row["velocities"][index]) < .025 and
                np.linalg.norm(row["angular"][index]) < .2 and
                np.linalg.norm(row["positions"][index]-position) < .004
                for row in list(self.history)[-8:])
            on_table = abs(float(position[2])-self.base.cube_half_size) <= .012
            region = None
            transform = self.base.cubes[index].pose.sp.to_transformation_matrix()
            half = self.base.cube_half_size
            local = np.array([[x, y, z] for x in (-half, half)
                              for y in (-half, half) for z in (-half, half)])
            corners = local @ transform[:3, :3].T + transform[:3, 3]
            for region_id, zone in self.zones.items():
                if np.all(np.abs(corners[:, :2]-np.array(zone["center"])) <= zone.get("radius", .045)):
                    if on_table and not held[index]:
                        region = region_id
            objects[f"cube{index+1}"] = dict(position=position.tolist(), held=held[index],
                placed_region=region, stable=bool(stable and on_table and not held[index]),
                visible=bool((segmentation == int(self.base.cubes[index].per_scene_id.item())).any().item()))
        return dict(objects=objects, zones=self.zones, gripper_empty=not any(held),
                    step=self.steps, reset_count=self.reset_count,
                    verifier="simulator_contact_and_geometry", scene_identity=f"seed{self.seed}")

    def _planner(self):
        from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
        planner = PandaArmMotionPlanningSolver(self.env, vis=False, debug=False,
                    base_pose=self.base.agent.robot.pose, visualize_target_grasp_pose=False,
                    print_env_info=False, joint_vel_limits=.55, joint_acc_limits=.55)
        planner.gripper_state = planner.CLOSED if any(self.observe()["objects"][k]["held"] for k in self.observe()["objects"]) else planner.OPEN
        return planner

    def _move(self, planner, pose, phase):
        self.control_source = "mplib"
        self.phase = phase
        result = planner.move_to_pose_with_screw(pose, refine_steps=5)
        if isinstance(result, int):
            result = planner.move_to_pose_with_RRTConnect(pose, refine_steps=5)
        if isinstance(result, int):
            raise RuntimeError(f"motion_plan_failed:{phase}")
        return result

    def _retreat(self, planner):
        tcp = self.base.agent.tcp.pose.sp
        raised = tcp.p.copy(); raised[2] = max(float(raised[2]), .23)
        self._move(planner, sapien.Pose(raised, tcp.q), "retreat_raise")
        self._move(planner, self.initial_tcp, "retreat_home")
        self.control_source = "hold"; self.phase = "settle"
        q = _array(self.base.agent.robot.get_qpos())[0,:7]
        for _ in range(20):self.env.step(np.r_[q, 1.0])

    @torch.no_grad()
    def _act_pick(self, index):
        self.base.target_idx[:] = index
        self.control_source = "act"
        self.phase = "grasp_lift"
        history=[];q=self.args.num_queries
        consecutive=0
        for tick in range(self.act_max_steps):
            obs=self.policy.observation(self.base.get_obs(self.base.get_info()))
            inputs={key:value.clone().to(self.device) for key,value in obs.items()}
            inputs["state"]=(inputs["state"]-self.stats["state_mean"])/self.stats["state_std"]
            if self.args.temporal_agg or tick % q == 0:
                prediction=self.agent.get_action(inputs)
                if prediction.shape != (1,q,8) or not torch.isfinite(prediction).all():
                    raise RuntimeError("invalid_act_prediction")
            if self.args.temporal_agg:
                history.append((tick,prediction));history=[(t,p) for t,p in history if tick-t<q]
                current=torch.stack([p[:,tick-t] for t,p in history],dim=1)
                w=torch.exp(-.01*torch.arange(len(history),device=self.device));raw=(current*(w/w.sum())[None,:,None]).sum(dim=1)
            else:raw=prediction[:,tick%q]
            action=raw*self.stats["action_std"]+self.stats["action_mean"]
            self.env.step(action.cpu().numpy()[0])
            held=[bool(self.base.agent.is_grasping(c).item()) for c in self.base.cubes]
            if any(value for i,value in enumerate(held) if i!=index):
                return False,"wrong_object_grasped",tick+1
            lifted=float(self.base.cubes[index].pose.p[0,2])>.09
            consecutive=consecutive+1 if held[index] and lifted else 0
            if consecutive>=5:
                return True,"target_grasped_and_lifted",tick+1
        return False,"act_grasp_timeout",self.act_max_steps

    def execute(self, object_id, region_id, attempt=1, inject_failure=False):
        if object_id not in self.observe()["objects"] or region_id not in self.zones:
            raise ValueError("Unknown object or region")
        self._check_stop()
        start=self.steps;before=self.observe();self.active_object=object_id;self.active_region=region_id
        if inject_failure:
            result=dict(success=False,status="failed",reason="injected_skill_timeout",object_id=object_id,
                        region_id=region_id,attempt=attempt,injected=True,steps=0,state=before)
            self._event("skill_failed",**{k:v for k,v in result.items() if k!="state"});return result
        index=int(object_id[4:])-1
        if not before["gripper_empty"]:
            return dict(success=False,status="failed",reason="gripper_not_empty",state=before,steps=0)
        self._event("skill_started",object_id=object_id,region_id=region_id,attempt=attempt)
        planner=None;act_steps=0;acquired=False
        try:
            acquired,reason,act_steps=self._act_pick(index)
            if not acquired:raise RuntimeError(reason)
            self._event("act_acquired",object_id=object_id,act_steps=act_steps)
            planner=self._planner();planner.gripper_state=planner.CLOSED
            tcp=self.base.agent.tcp.pose.sp
            cube_pos=_array(self.base.cubes[index].pose.p)[0]
            offset=tcp.p-cube_pos
            lift=tcp.p.copy();lift[2]=max(float(lift[2]),.23)
            self._move(planner,sapien.Pose(lift,tcp.q),"transport_raise")
            zone=self.zones[region_id];above=np.array([*zone["center"],.20])+offset
            self._move(planner,sapien.Pose(above,tcp.q),"transport_to_zone")
            lower=np.array([*zone["center"],.038])+offset
            self._move(planner,sapien.Pose(lower,tcp.q),"lower_to_table")
            self.control_source="scripted_release";self.phase="release"
            planner.open_gripper(t=15)
            self._retreat(planner)
            state=self.observe();obj=state["objects"][object_id]
            success=obj["placed_region"]==region_id and obj["stable"] and state["gripper_empty"]
            reason="placed_and_released_stably" if success else "placement_verification_failed"
        except StopRequested:
            self._event("skill_stopped", object_id=object_id, region_id=region_id, attempt=attempt)
            raise
        except Exception as error:
            success=False;reason=str(error)
            if planner is None:planner=self._planner()
            try:
                self.control_source="recovery";self.phase="open_and_retreat"
                planner.open_gripper(t=10);self._retreat(planner)
            except StopRequested:
                raise
            except Exception as recovery_error:
                reason+=f"; recovery:{recovery_error}"
            state=self.observe()
        finally:
            if planner is not None:planner.close()
        result=dict(success=bool(success),status="succeeded" if success else "failed",reason=reason,
                    object_id=object_id,region_id=region_id,attempt=attempt,steps=self.steps-start,
                    act_steps=act_steps,act_acquired=acquired,state=state,reset_count=self.reset_count)
        self._event("skill_finished",**{k:v for k,v in result.items() if k!="state"})
        self.image().save(self.output/f"{object_id}_{region_id}_attempt{attempt}.png")
        return result

    def close(self):
        self.image().save(self.output/"final.png")
        (self.output/"final_state.json").write_text(json.dumps(self.observe(),indent=2))
        self.writer.close();self.actions.close();self.raw.close()
