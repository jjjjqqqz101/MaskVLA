"""Local web evaluation for 40-object ManiSkill π0.5 checkpoints."""

import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import os
import secrets
import signal
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from mani_skill.envs.tasks.tabletop.target_pick_forty import (
    ALL_OBJECTS, HELDOUT_OBJECTS, TRAIN_OBJECTS,
)


HERE = Path(__file__).resolve().parent
MANISKILL_ROOT = HERE.parents[2]
OPENPI_ROOT = Path(os.environ.get("OPENPI_ROOT", MANISKILL_ROOT.parent / "openpi")).expanduser()
CHECKPOINT_ROOT = Path(os.environ.get("PI05_CHECKPOINT_ROOT", HERE / "runs/openpi/pi05_maniskill_25x50/full_25x50_v1")).expanduser()
PROMPT = "Pick up and lift the object shown in the target image."
MAX_STEPS = 500
POLICY_PORT = 18773
PERCEPTION_URL = os.environ.get("MASK_PERCEPTION_URL", "http://127.0.0.1:18830")


def normalize_mask_config(value=None, defaults=None):
    """Validate user experiment settings independently of simulator state."""
    config = dict(mask_source="oracle", perception_url=PERCEPTION_URL, target_text="",
                  loss_mode="observe", max_pause_steps=50,
                  intervention=dict(kind="none", start=50, duration=30, dx=20, dy=0))
    if defaults:
        config.update(copy.deepcopy(defaults))
    if value is not None:
        if not isinstance(value, dict):
            raise ValueError("Mask 设置必须是对象")
        if set(value) - set(config):
            raise ValueError("包含未知 Mask 设置")
        for key, item in value.items():
            if key == "intervention":
                if not isinstance(item, dict):
                    raise ValueError("Mask 干预设置必须是对象")
                if set(item) - {"kind", "start", "duration", "dx", "dy"}:
                    raise ValueError("包含未知 Mask 干预设置")
                config[key].update(item)
            else:
                config[key] = item
    if config["mask_source"] not in ("oracle", "perception"):
        raise ValueError("请选择仿真真值或 VLM + SAM2 + Cutie")
    if not isinstance(config["perception_url"], str):
        raise ValueError("感知服务地址必须是字符串")
    endpoint = urlparse(config["perception_url"])
    if (endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "localhost", "::1")
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("感知服务必须是本机 HTTP 地址")
    if not isinstance(config["target_text"], str) or len(config["target_text"]) > 1000:
        raise ValueError("目标描述应少于 1000 字")
    config["target_text"] = config["target_text"].strip()
    if config["loss_mode"] not in ("observe", "pause", "reacquire"):
        raise ValueError("未知 Mask 丢失处理方式")
    if type(config["max_pause_steps"]) is not int or not 1 <= config["max_pause_steps"] <= MAX_STEPS:
        raise ValueError("暂停步数必须为 1 到 500 的整数")
    intervention = config["intervention"]
    if intervention["kind"] not in ("none", "dropout", "freeze", "shift", "occlusion"):
        raise ValueError("未知 Mask 干预方式")
    for key, low, high in (("start", 0, MAX_STEPS), ("duration", 1, MAX_STEPS),
                           ("dx", -224, 224), ("dy", -224, 224)):
        if type(intervention[key]) is not int or not low <= intervention[key] <= high:
            raise ValueError(f"{key} 超出有效整数范围 {low} 到 {high}")
    if config["mask_source"] == "oracle" and intervention["kind"] != "none":
        raise ValueError("Mask 丢失干预仅用于真实感知模式，请先切换 Mask 来源")
    return config


def target_description(target, config):
    text = config["target_text"]
    if text:
        return text
    if config["mask_source"] == "perception" and target in ("red_cube_1", "red_cube_2"):
        raise ValueError("两个红方块外观相同，请填写可见描述，例如“画面左侧的红方块”；感知流程不使用仿真编号")
    return target.split("_", 1)[-1].replace("_", " ") if target[:1].isdigit() else target.replace("_", " ")


def mask_diagnostics(mask, gt_mask, metadata):
    """Ground truth is scored here only and never passed to perception."""
    mask, gt_mask = np.asarray(mask, dtype=bool), np.asarray(gt_mask, dtype=bool)
    if mask.shape != gt_mask.shape or mask.ndim != 2:
        raise ValueError("感知 Mask 与桌面图像尺寸不一致")
    union = int(np.logical_or(mask, gt_mask).sum())
    return dict(metadata, perceived_mask_pixels=int(mask.sum()), gt_mask_pixels=int(gt_mask.sum()),
                mask_iou=float(np.logical_and(mask, gt_mask).sum() / union) if union else None,
                coverage=float(mask.mean()))


class PerceptionTrial:
    """RGB-only service boundary; simulator labels are not part of this API."""
    def __init__(self, config, target, client_factory=None):
        self.config = normalize_mask_config(config)
        self.target_text = target_description(target, self.config)
        if client_factory is None:
            from perception_mask.client import PerceptionMaskClient
            client_factory = PerceptionMaskClient
        self.client = client_factory(endpoint=self.config["perception_url"], timeout=180)
        self.started = False

    def apply(self, rgb, frame_index, force_reacquire=False):
        if not self.started:
            options = dict(loss_mode=self.config["loss_mode"],
                           intervention=copy.deepcopy(self.config["intervention"]))
            result = self.client.start(rgb, self.target_text, options)
            self.started = True
        else:
            result = self.client.step(rgb, frame_index, force_reacquire=force_reacquire)
        mask = np.asarray(result.mask, dtype=bool)
        if mask.shape != rgb.shape[:2]:
            raise ValueError("感知服务返回的 Mask 尺寸错误")
        metadata = copy.deepcopy(result.metadata)
        metadata.setdefault("source", "vlm_sam2_cutie")
        metadata.setdefault("state", "unknown")
        return mask, metadata

    def close(self):
        self.client.close()


def chunk_refresh_required(chunk, offset, previous_key, observation):
    perception = observation["perception"]
    key = (perception.get("generation", 0), perception.get("state", "tracking"),
           bool(observation["mask"].any()))
    return chunk is None or offset >= min(10, len(chunk)) or key != previous_key, key


def sampling_noise_seed(scene_seed, frame_index):
    """Match stochastic policy inputs by scene/control frame across conditions."""
    if type(scene_seed) is not int or not 0 <= scene_seed <= 2**32 - 1:
        raise ValueError("scene_seed must be an unsigned 32-bit integer")
    if type(frame_index) is not int or not 0 <= frame_index <= 2**32 - 1:
        raise ValueError("policy_frame_index must be an unsigned 32-bit integer")
    return int(np.random.SeedSequence([scene_seed, frame_index]).generate_state(1, dtype=np.uint32)[0])


def arr(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def scalar(value):
    return bool(arr(value).reshape(-1)[0])


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


class Lab:
    def __init__(self, output, gpu, mask_config=None, policy_port=POLICY_PORT):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.selected_gpu = gpu
        self.policy_port = policy_port
        self.mask_config = normalize_mask_config(mask_config)
        self.perception = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05-trial")
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.reacquire = threading.Event()
        self.status = dict(busy=False, message="请选择目标、场景物体范围并生成场景", error=None,
                           progress=0, scene=None, preview=None, frame_version=0, results=[],
                           loaded_checkpoint=None, loaded_gpu=None,
                           selected_gpu=gpu, max_steps=MAX_STEPS,
                           perception_supported=True,
                           mask_config=copy.deepcopy(self.mask_config), perception=None)
        for path in sorted(self.output.glob("trials/*/result.json"))[-100:]:
            try:
                self.status["results"].append(json.loads(path.read_text()))
            except (OSError, ValueError):
                pass
        self.env = None
        self.scene = None
        self.worker = None
        self.worker_checkpoint = None
        self.worker_gpu = None
        self.worker_log = None

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.status)

    def update(self, **fields):
        with self.lock:
            self.status.update(fields)

    def checkpoint_list(self):
        entries = []
        for path in CHECKPOINT_ROOT.iterdir() if CHECKPOINT_ROOT.exists() else []:
            if path.is_dir() and path.name.isdigit() and (path / "_CHECKPOINT_METADATA").is_file():
                entries.append(dict(name=path.name, path=str(path)))
        return sorted(entries, key=lambda item: int(item["name"]))

    def checkpoint_path(self, name):
        names = {x["name"]: x["path"] for x in self.checkpoint_list()}
        if name not in names:
            raise ValueError(f"所选权重不可用：{CHECKPOINT_ROOT}。可通过 PI05_CHECKPOINT_ROOT 指向已有权重。")
        return Path(names[name])

    def gpu_list(self):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                 "--format=csv,noheader,nounits"], text=True, timeout=10)
            gpus = []
            for line in output.splitlines():
                index, used, total = (int(value.strip()) for value in line.split(","))
                gpus.append(dict(index=index, used_mib=used, total_mib=total))
            if not gpus:
                raise ValueError("未检测到显卡")
            return sorted(gpus, key=lambda item: item["index"])
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise ValueError(f"读取显卡列表失败：{error}") from error

    def validate_gpu(self, gpu):
        if type(gpu) is not int or gpu not in {item["index"] for item in self.gpu_list()}:
            raise ValueError("请选择当前服务器上的有效 GPU")

    def submit(self, action, data):
        if action not in ("scene", "target", "gpu", "run", "mask"):
            raise ValueError("Unknown action")
        if action in ("scene", "run", "mask"):
            data["mask_config"] = normalize_mask_config(data.get("mask_config"), self.mask_config)
            target = data.get("target") if action == "scene" else (self.scene or {}).get("target")
            if target:
                target_description(target, data["mask_config"])
        if action == "scene":
            target = data.get("target")
            if target not in ALL_OBJECTS:
                raise ValueError("请选择 40 个物体中的一个目标")
            allow_heldout = data.get("allow_heldout")
            if type(allow_heldout) is not bool:
                raise ValueError("请选择未见物体是否可以进入场景")
            if target in HELDOUT_OBJECTS and not allow_heldout:
                raise ValueError("目标属于未见物体，请选择“出现未见物体”")
            seed = data.get("seed")
            if seed is None:
                seed = secrets.randbelow(2**31)
            if type(seed) is not int or not 0 <= seed < 2**31:
                raise ValueError("seed 必须是 0 到 2147483647 的整数")
            data["seed"] = seed
        elif action == "target":
            if self.scene is None or data.get("scene_id") != self.scene["id"]:
                raise ValueError("请先生成当前场景")
            if self.scene.get("used"):
                raise ValueError("该场景已测试，请刷新场景后再选择目标")
            if data.get("target") not in self.scene["objects"]:
                raise ValueError("目标必须是当前场景中的 8 个物体之一")
        elif action == "gpu":
            self.validate_gpu(data.get("gpu"))
        elif action == "mask":
            if self.scene is None or data.get("scene_id") != self.scene["id"]:
                raise ValueError("请先生成当前场景")
            if self.scene.get("used"):
                raise ValueError("该场景已测试，请刷新场景后配置 Mask")
        else:
            if self.scene is None or data.get("scene_id") != self.scene["id"]:
                raise ValueError("请先生成当前场景")
            if self.scene.get("used"):
                raise ValueError("这个场景已运行，请刷新场景后再次测试")
            self.checkpoint_path(data.get("checkpoint"))
            self.validate_gpu(data.get("gpu"))
        with self.lock:
            if self.status["busy"]:
                raise ValueError("已有操作正在运行")
            self.status.update(busy=True, error=None, progress=0, message="准备中…")
        self.cancel.clear()
        self.executor.submit(self.work, action, data.copy())

    def work(self, action, data):
        try:
            if action == "scene":
                self.new_scene(data)
            elif action == "target":
                self.select_target(data["target"])
            elif action == "gpu":
                self.select_gpu(data["gpu"])
            elif action == "mask":
                self.apply_mask_config(data["mask_config"])
            else:
                self.run_trial(data["checkpoint"], data["gpu"], data["mask_config"])
        except Exception as error:
            traceback.print_exc()
            self.update(error=str(error), message="操作失败；详细信息见服务日志")
        finally:
            self.update(busy=False)

    def ensure_env(self):
        if self.env is not None:
            return
        torch.set_num_threads(4)
        self.env = gym.make(
            "TargetPickForty-v1", robot_uids="panda_wristcam",
            obs_mode="rgb+segmentation", control_mode="pd_joint_pos",
            sim_backend="physx_cpu", render_mode="rgb_array", reward_mode="sparse",
            max_episode_steps=MAX_STEPS, reconfiguration_freq=0,
            sensor_configs=dict(width=224, height=224, fov=float(np.pi / 2),
                                near=0.01, far=100, shader_pack="minimal"),
        )

    def close_perception(self):
        if self.perception is not None:
            try:
                self.perception.close()
            except Exception as error:
                self.update(mask_cleanup_error=f"感知会话清理失败：{error}")
            finally:
                self.perception = None

    def observe(self, obs, frame_index=0, force_reacquire=False):
        base = self.env.unwrapped
        camera = obs["sensor_data"]["base_camera"]
        rgb = arr(camera["rgb"])[0].astype(np.uint8)
        wrist = arr(obs["sensor_data"]["hand_camera"]["rgb"])[0].astype(np.uint8)
        segmentation = arr(camera["segmentation"])[0, :, :, 0]
        target_id = int(base.cubes[base.selected_target_idx].per_scene_id.item())
        gt_mask = segmentation == target_id
        if self.mask_config["mask_source"] == "perception":
            if self.perception is None:
                raise RuntimeError("感知会话未初始化；不会使用仿真 Mask 回退")
            mask, metadata = self.perception.apply(rgb, frame_index, force_reacquire=force_reacquire)
        else:
            mask = gt_mask
            metadata = dict(state="tracking", source="simulator_instance_segmentation", generation=0,
                            quality=None, latency_ms=0, events=[])
        diagnostics = mask_diagnostics(mask, gt_mask, metadata)
        target = np.where(mask[:, :, None], rgb, 0).astype(np.uint8)
        qpos = arr(obs["agent"]["qpos"])[0]
        state = np.r_[qpos[:7], np.clip(1 - np.mean(qpos[7:]) / 0.04, 0, 1)].astype(np.float32)
        self.update(perception=copy.deepcopy(diagnostics))
        return dict(base=rgb, wrist=wrist, target=target, mask=mask, gt_mask=gt_mask,
                    state=state, perception=diagnostics, policy_frame_index=int(frame_index))

    def reset_perception(self, config, target):
        self.close_perception()
        self.mask_config = normalize_mask_config(config)
        self.update(mask_config=copy.deepcopy(self.mask_config), perception=None)
        if self.mask_config["mask_source"] == "perception":
            self.update(message="正在用 VLM 定位、SAM2 分割并初始化 Cutie…", preview=None)
            self.perception = PerceptionTrial(self.mask_config, target)

    def apply_mask_config(self, config):
        self.reset_perception(config, self.scene["target"])
        self.observation = self.observe(self.raw_obs)
        self.publish(self.frame(self.observation, 0, f"target: {self.scene['target']}"))
        self.update(message="Mask 设置已应用；抓取开始时会重新初始化跟踪")

    def frame(self, observation, step, note=""):
        panels = [observation["base"], observation["wrist"], observation["target"]]
        canvas = Image.new("RGB", (672, 280), "#14211e")
        for index, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel), (index * 224, 28))
        draw = ImageDraw.Draw(canvas)
        metadata = observation["perception"]
        bbox = metadata.get("vlm_bbox_xyxy")
        if (isinstance(bbox, (list, tuple)) and len(bbox) == 4
                and np.isfinite(np.asarray(bbox, dtype=float)).all()):
            x1, y1, x2, y2 = [float(value) for value in bbox]
            x1, x2, y1, y2 = np.clip([x1, x2, y1, y2], 0, 223)
            if x2 >= x1 and y2 >= y1:
                draw.rectangle((x1, y1 + 28, x2, y2 + 28), outline="#ffcb62", width=2)
        draw.text((7, 7), f"base RGB  |  wrist RGB  |  target RGB x mask", fill="white")
        draw.text((480, 7), f"step {step}/{MAX_STEPS}", fill="#9fe0b7")
        if note:
            draw.rectangle((0, 252, 671, 279), fill="#14211e")
            draw.text((7, 260), note[:100], fill="white")
        iou = metadata.get("mask_iou")
        details = (f"{metadata.get('source', '?')} | {metadata.get('state', '?')} | "
                   f"mask px={metadata.get('perceived_mask_pixels', 0)} | "
                   f"IoU={iou:.3f}" if iou is not None else
                   f"{metadata.get('source', '?')} | {metadata.get('state', '?')} | mask empty")
        draw.rectangle((0, 232, 671, 251), fill="#14211e")
        draw.text((7, 234), details[:110], fill="#9fe0b7")
        return np.asarray(canvas)

    def publish(self, frame):
        temporary = self.output / "live.tmp.png"
        Image.fromarray(frame).save(temporary)
        temporary.replace(self.output / "live.png")
        self.update(preview="/artifacts/live.png", frame_version=time.time_ns())

    def new_scene(self, data):
        self.ensure_env()
        self.close_perception()
        self.scene, self.observation, self.raw_obs = None, None, None
        self.update(scene=None, preview=None, perception=None)
        target = data["target"]
        split = "heldout" if target in HELDOUT_OBJECTS else "train"
        self.update(message="正在生成随机桌面场景…")
        obs, info = self.env.reset(seed=data["seed"], options={
            "reconfigure": True, "target_name": target, "object_split": split,
            "include_heldout_distractors": data["allow_heldout"],
        })
        base = self.env.unwrapped
        self.reset_perception(data["mask_config"], target)
        observation = self.observe(obs)
        pixels = int(observation["mask"].sum())
        if pixels == 0 and self.mask_config["mask_source"] == "oracle":
            raise RuntimeError("目标在桌面相机中不可见，请换一个 seed 刷新场景")
        scene_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{data['seed']}"
        selected = list(base.selected_names)
        scene = dict(id=scene_id, seed=data["seed"], target=target, split=split,
                     allow_heldout=data["allow_heldout"], objects=selected,
                     target_idx=base.selected_target_idx,
                     heldout_objects=[x for x in selected if x in HELDOUT_OBJECTS],
                     initial_target_pixels=pixels, used=False)
        folder = self.output / "scenes" / scene_id
        folder.mkdir(parents=True, exist_ok=False)
        dump(folder / "scene.json", scene)
        frame = self.frame(observation, 0, f"target: {target} | seed: {data['seed']}")
        Image.fromarray(frame).save(folder / "initial.png")
        self.publish(frame)
        self.scene = scene
        self.raw_obs = obs
        self.observation = observation
        self.update(scene=copy.deepcopy(scene), message=f"场景已生成：目标 {target}，共 8 个物体")

    def select_target(self, target):
        scene = self.scene
        index = scene["objects"].index(target)
        base = self.env.unwrapped
        base.selected_target_idx = index
        base.selected_target_name = target
        base.target_idx[:] = index
        scene["target"] = target
        scene["target_idx"] = index
        scene["split"] = "heldout" if target in HELDOUT_OBJECTS else "train"
        self.update(scene=copy.deepcopy(scene), preview=None, perception=None)
        self.reset_perception(self.mask_config, target)
        observation = self.observe(self.raw_obs)
        scene["initial_target_pixels"] = int(observation["mask"].sum())
        self.observation = observation
        frame = self.frame(observation, 0, f"target: {target} | seed: {scene['seed']}")
        folder = self.output / "scenes" / scene["id"]
        dump(folder / "scene.json", scene)
        Image.fromarray(frame).save(folder / "initial.png")
        self.publish(frame)
        self.update(scene=copy.deepcopy(scene),
                    message=f"当前场景目标已切换为 {target}；物体位置未改变")

    def select_gpu(self, gpu):
        if self.worker_gpu is not None and self.worker_gpu != gpu:
            self.stop_worker()
        self.selected_gpu = gpu
        self.update(selected_gpu=gpu, message=f"已选择 GPU {gpu}；下次抓取将在此显卡上推理")

    def stop_worker(self):
        if self.worker is not None and self.worker.poll() is None:
            self.worker.terminate()
            try:
                self.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait(timeout=5)
        self.worker = None
        self.worker_checkpoint = None
        self.worker_gpu = None
        self.update(loaded_checkpoint=None, loaded_gpu=None)
        if self.worker_log is not None:
            self.worker_log.close()
            self.worker_log = None

    def ensure_worker(self, checkpoint, gpu):
        if (self.worker is not None and self.worker.poll() is None
                and self.worker_checkpoint == checkpoint and self.worker_gpu == gpu):
            return
        python_bin = Path(os.environ.get("OPENPI_PYTHON", OPENPI_ROOT / ".venv/bin/python")).expanduser()
        if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
            raise FileNotFoundError(f"Optional openpi Python environment unavailable: {python_bin}. Configure OPENPI_ROOT / OPENPI_PYTHON before π0.5 inference.")
        if not (OPENPI_ROOT / "src/openpi/policies/maniskill_policy.py").is_file():
            raise FileNotFoundError(f"Missing ManiSkill openpi integration under {OPENPI_ROOT}; configure OPENPI_ROOT to the adapted openpi source.")
        self.stop_worker()
        self.update(message=f"正在加载 π0.5 第 {checkpoint.name} 步权重到 GPU {gpu}…")
        log_path = self.output / f"policy_worker_{checkpoint.name}_gpu{gpu}.log"
        self.worker_log = log_path.open("ab")
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS="cuda",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_COMPILATION_CACHE_DIR=str(self.output / "jax_cache"))
        # Select this project's ManiSkill adapter even when reusing an existing
        # compatible Python environment whose editable openpi points elsewhere.
        env['PYTHONPATH'] = os.pathsep.join([str(OPENPI_ROOT/'src'),
                            str(OPENPI_ROOT/'packages/openpi-client/src'),env.get('PYTHONPATH','')])
        command = [str(python_bin), str(HERE / "pi05_inference_worker.py"),
                   "--checkpoint", str(checkpoint), "--port", str(self.policy_port)]
        self.worker = subprocess.Popen(command, cwd=OPENPI_ROOT, env=env,
                                       stdout=self.worker_log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        self.worker_checkpoint = checkpoint
        self.worker_gpu = gpu
        started = time.monotonic()
        while time.monotonic() - started < 300:
            if self.cancel.is_set():
                self.stop_worker()
                raise RuntimeError("加载权重已停止")
            if self.worker.poll() is not None:
                self.stop_worker()
                raise RuntimeError(f"权重加载进程已退出，请查看 {log_path}")
            try:
                with urlopen(f"http://127.0.0.1:{self.policy_port}/health", timeout=1) as response:
                    status = json.load(response)
                if status.get("checkpoint") == str(checkpoint):
                    self.update(loaded_checkpoint=checkpoint.name, loaded_gpu=gpu)
                    return
            except (URLError, TimeoutError, ValueError):
                pass
            time.sleep(1)
        self.stop_worker()
        raise RuntimeError(f"加载权重超过 5 分钟，请查看 {log_path}")

    def infer(self, observation):
        noise_seed = sampling_noise_seed(self.scene["seed"], observation["policy_frame_index"])
        payload = dict(state=observation["state"].tolist(), noise_seed=noise_seed)
        for key, source in (("base_rgb", "base"), ("wrist_rgb", "wrist"), ("target_rgb", "target")):
            payload[key] = base64.b64encode(observation[source].tobytes()).decode("ascii")
        request = Request(f"http://127.0.0.1:{self.policy_port}/infer",
                          data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=180) as response:
                result = json.load(response)
        except Exception as error:
            raise RuntimeError(f"π0.5 推理失败：{error}; 请查看推理进程日志") from error
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 8 or len(actions) < 1 or not np.isfinite(actions).all():
            raise RuntimeError("π0.5 返回的动作格式无效")
        if result.get("noise_seed") != noise_seed:
            raise RuntimeError("π0.5 推理进程未确认固定采样噪声，请重启本任务推理进程后再做对照实验")
        return actions, float(result["inference_seconds"])

    def run_trial(self, checkpoint_name, gpu, mask_config=None):
        checkpoint = self.checkpoint_path(checkpoint_name)
        self.selected_gpu = gpu
        self.update(selected_gpu=gpu)
        scene = self.scene
        try:
            # First acquire and offload segmentation before loading the large JAX policy.
            self.reset_perception(mask_config or self.mask_config, scene["target"])
            observation = self.observe(self.raw_obs)
            self.publish(self.frame(observation, 0, f"target: {scene['target']}"))
            self.ensure_worker(checkpoint, gpu)
        except Exception:
            self.close_perception()
            raise
        if self.cancel.is_set():
            self.close_perception()
            raise RuntimeError("测试已停止")
        scene["used"] = True
        self.update(scene=copy.deepcopy(scene), message="π0.5 开始闭环抓取测试…")
        trial_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{scene['target']}_{checkpoint_name}"
        folder = self.output / "trials" / trial_id
        folder.mkdir(parents=True, exist_ok=False)
        self.observation = observation
        self.reacquire.clear()
        initial_input_hashes = {
            f"initial_{key}_sha256": hashlib.sha256(
                np.ascontiguousarray(observation[key]).tobytes()
            ).hexdigest() for key in ("base", "wrist", "state")
        }
        initial_input_hashes["initial_target_image_sha256"] = hashlib.sha256(
            np.ascontiguousarray(observation["target"]).tobytes()
        ).hexdigest()
        initial_input_hashes["initial_mask_sha256"] = hashlib.sha256(
            np.ascontiguousarray(observation["mask"]).tobytes()
        ).hexdigest()
        np.savez_compressed(folder / "initial_policy_input.npz", base=observation["base"],
                            wrist=observation["wrist"], target=observation["target"],
                            mask=observation["mask"], gt_mask=observation["gt_mask"], state=observation["state"])
        dump(folder / "mask_config.json", self.mask_config)
        image = self.frame(observation, 0, f"target: {scene['target']} | checkpoint: {checkpoint_name}")
        Image.fromarray(image).save(folder / "initial.png")
        writer = imageio.get_writer(folder / "video.mp4", fps=20, codec="libx264", quality=7,
                                   macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"])
        actions_record = []
        traces = []
        wrong_ever = False
        success_ever = False
        success_streak = 0
        stop_reason = "max_steps"
        inference_calls = 0
        inference_seconds = 0.0
        info = None
        chunk, chunk_offset, chunk_key = None, 0, None
        last_inference_step, last_inference_perception = None, None
        pause_steps = 0
        max_pause_streak = 0
        pause_streak = 0
        started = time.monotonic()
        writer.append_data(image)
        try:
            for step in range(MAX_STEPS):
                if self.cancel.is_set():
                    stop_reason = "cancelled"
                    break
                if self.reacquire.is_set() and self.perception is not None:
                    self.reacquire.clear()
                    observation = self.observe(self.raw_obs, frame_index=step, force_reacquire=True)
                    chunk = None
                control_perception = copy.deepcopy(observation["perception"])
                inference_refreshed = False
                lost = (self.mask_config["mask_source"] == "perception"
                        and observation["perception"].get("state") != "tracking")
                paused = lost and self.mask_config["loss_mode"] in ("pause", "reacquire")
                if paused:
                    # Holding current joint positions preserves physics, cancellation and frame updates.
                    policy_action = observation["state"].copy()
                    chunk = None
                    pause_streak += 1
                    if pause_streak > self.mask_config["max_pause_steps"]:
                        stop_reason = "mask_lost_pause_timeout"
                        break
                    pause_steps += 1
                    max_pause_streak = max(max_pause_streak, pause_streak)
                else:
                    pause_streak = 0
                    refresh, current_key = chunk_refresh_required(chunk, chunk_offset, chunk_key, observation)
                    if refresh:
                        chunk, seconds = self.infer(observation)
                        chunk_offset = 0
                        chunk_key = current_key
                        last_inference_step = step
                        last_inference_perception = copy.deepcopy(observation["perception"])
                        inference_refreshed = True
                        inference_calls += 1
                        inference_seconds += seconds
                    policy_action = chunk[chunk_offset]
                    chunk_offset += 1
                action = np.r_[policy_action[:7], 1 - 2 * np.clip(policy_action[7], 0, 1)].astype(np.float32)
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.raw_obs = obs
                observation = self.observe(obs, frame_index=step + 1)
                flags = {key: scalar(info[key]) for key in
                         ("is_target_grasped", "target_lifted", "wrong_object_grasped", "success")}
                wrong_ever |= flags["wrong_object_grasped"]
                success_ever |= flags["success"]
                success_streak = success_streak + 1 if flags["success"] and not wrong_ever else 0
                actions_record.append(action.tolist())
                traces.append(dict(step=step + 1, visible_target_pixels=int(observation["mask"].sum()),
                                   paused=paused, perception=copy.deepcopy(observation["perception"]),
                                   control_perception=control_perception,
                                   policy_input_perception=None if paused else copy.deepcopy(last_inference_perception),
                                   policy_input_frame=None if paused else last_inference_step,
                                   noise_seed=(None if paused else sampling_noise_seed(scene["seed"], last_inference_step)),
                                   policy_inference_refreshed=inference_refreshed, **flags))
                image = self.frame(observation, step + 1, f"target: {scene['target']} | success: {flags['success']} | wrong: {wrong_ever}")
                writer.append_data(image)
                if step % 5 == 0 or flags["success"]:
                    self.publish(image)
                    self.update(progress=(step + 1) / MAX_STEPS,
                                message=f"目标 {scene['target']} · {step + 1}/{MAX_STEPS} 步 · "
                                        f"抓起 {'是' if flags['success'] else '否'}")
                if success_streak >= 5:
                    stop_reason = "success_stable"
                    break
                if scalar(terminated) or scalar(truncated):
                    stop_reason = "environment_done"
                    break
        except Exception:
            stop_reason = "error"
            raise
        finally:
            writer.close()
            self.close_perception()
            Image.fromarray(image).save(folder / "final.png")
            np.save(folder / "actions.npy", np.asarray(actions_record, dtype=np.float32))
            dump(folder / "step_metrics.json", traces)
            result = dict(id=trial_id, scene_id=scene["id"], seed=scene["seed"],
                          target=scene["target"], target_split=scene["split"],
                          objects=scene["objects"], heldout_objects=scene["heldout_objects"],
                          allow_heldout=scene["allow_heldout"], checkpoint=str(checkpoint),
                          gpu=gpu, steps=len(actions_record), max_steps=MAX_STEPS,
                          stop_reason=stop_reason, success_once=success_ever,
                          success_at_end=bool(traces[-1]["success"]) if traces else False,
                          wrong_object_grasped_ever=wrong_ever,
                          correct_pick=success_ever and not wrong_ever,
                          inference_calls=inference_calls,
                          inference_seconds=round(inference_seconds, 3),
                          elapsed_seconds=round(time.monotonic() - started, 3),
                          prompt=PROMPT, mask_source=("vlm_sam2_cutie" if self.mask_config["mask_source"] == "perception"
                                                    else "simulator_instance_segmentation"),
                          mask_config=copy.deepcopy(self.mask_config),
                          mask_summary=self.mask_summary(traces), pause_steps=pause_steps,
                          max_pause_streak=max_pause_streak,
                          **initial_input_hashes,
                          sampling_mode="scene_frame_seeded_noise",
                          sampling_seed_rule="uint32(SeedSequence([scene.seed, control_frame_index]).generate_state(1)[0])",
                          sampling_noise_generator="numpy.default_rng(noise_seed).standard_normal((action_horizon, action_dim), dtype=float32)",
                          video=f"/artifacts/trials/{trial_id}/video.mp4",
                          initial=f"/artifacts/trials/{trial_id}/initial.png",
                          final=f"/artifacts/trials/{trial_id}/final.png",
                          metrics=f"/artifacts/trials/{trial_id}/result.json")
            dump(folder / "result.json", result)
            with self.lock:
                self.status["results"].append(result)
                self.status["results"] = self.status["results"][-100:]
            self.update(message=f"测试结束：{stop_reason}，已保存视频和记录", progress=1)

    @staticmethod
    def mask_summary(traces):
        frames = [row["perception"] for row in traces]
        ious = [row["mask_iou"] for row in frames if row.get("mask_iou") is not None]
        latencies = [float(row["latency_ms"]) for row in frames
                     if isinstance(row.get("latency_ms"), (int, float)) and row["latency_ms"] > 0]
        states, events, seen_events = {}, [], set()
        for row in frames:
            state = row.get("state", "unknown")
            states[state] = states.get(state, 0) + 1
            for event in row.get("events", []):
                key = json.dumps(event, sort_keys=True, ensure_ascii=False)
                if key not in seen_events:
                    seen_events.add(key)
                    events.append(copy.deepcopy(event))
        return dict(frames=len(frames), mean_iou=float(np.mean(ious)) if ious else None,
                    minimum_iou=float(min(ious)) if ious else None,
                    empty_mask_frames=sum(row.get("perceived_mask_pixels", 0) == 0 for row in frames),
                    states=states, generations=sorted({row.get("generation", 0) for row in frames}),
                    mean_perception_ms=float(np.mean(latencies)) if latencies else None,
                    processing_fps=1000 / float(np.mean(latencies)) if latencies else None,
                    latency_scope="per-frame service processing; excludes policy inference and simulation",
                    events=events)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format_string, *args):
        if self.path not in ("/api/status", "/artifacts/live.png"):
            super().log_message(format_string, *args)

    def reply(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self.reply({"error": "Origin mismatch"}, 403)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 16384:
                raise ValueError("请求数据过大")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("请求格式错误")
            route = urlparse(self.path).path
            if route == "/api/stop":
                self.server.lab.cancel.set()
            elif route == "/api/reacquire":
                if (not self.server.lab.snapshot()["busy"]
                        or self.server.lab.mask_config["mask_source"] != "perception"):
                    raise ValueError("仅在感知抓取运行中可请求重新定位")
                self.server.lab.reacquire.set()
            elif route in ("/api/scene", "/api/target", "/api/gpu", "/api/run", "/api/mask"):
                self.server.lab.submit(route[5:], data)
            else:
                raise ValueError("未知操作")
            self.reply({"ok": True})
        except (ValueError, TypeError, KeyError) as error:
            self.reply({"error": str(error)}, 400)

    def do_GET(self):
        route = unquote(urlparse(self.path).path)
        if route == "/api/status":
            return self.reply(self.server.lab.snapshot())
        if route == "/api/checkpoints":
            return self.reply(self.server.lab.checkpoint_list())
        if route == "/api/perception_health":
            try:
                from perception_mask.client import PerceptionMaskClient
                client = PerceptionMaskClient(endpoint=self.server.lab.mask_config["perception_url"], timeout=5)
                return self.reply(client.health())
            except Exception as error:
                return self.reply({"error": f"感知服务不可用：{error}"}, 503)
        if route == "/api/gpus":
            try:
                return self.reply(dict(gpus=self.server.lab.gpu_list(),
                                       selected_gpu=self.server.lab.selected_gpu))
            except ValueError as error:
                return self.reply({"error": str(error)}, 503)
        if route == "/api/catalog":
            return self.reply({"train": list(TRAIN_OBJECTS), "heldout": list(HELDOUT_OBJECTS)})
        path = HERE / "pi05_target_web.html" if route == "/" else None
        if route.startswith("/artifacts/"):
            path = (self.server.lab.output / route[len("/artifacts/"):]).resolve()
            if not path.is_relative_to(self.server.lab.output):
                return self.reply({"error": "Forbidden"}, 403)
        if path is None or not path.is_file():
            return self.reply({"error": "Not found"}, 404)
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        header = self.headers.get("Range")
        if header:
            try:
                spec = header.removeprefix("bytes=").split("-")
                start = int(spec[0]) if spec[0] else max(0, size - int(spec[1]))
                end = min(int(spec[1]), size - 1) if spec[0] and spec[1] else size - 1
                if start < 0 or start > end:
                    raise ValueError()
                code = 206
            except (ValueError, IndexError):
                return self.reply({"error": "Invalid range"}, 416)
        self.send_response(code)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with path.open("rb") as file:
                file.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = file.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18772)
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("PI05_GPU", "0")))
    parser.add_argument("--policy-port", type=int, default=POLICY_PORT)
    parser.add_argument("--mask-source", choices=("oracle", "perception"), default="oracle")
    parser.add_argument("--perception-url", default=PERCEPTION_URL)
    parser.add_argument("--target-text", default="")
    parser.add_argument("--mask-loss-mode", choices=("observe", "pause", "reacquire"), default="observe")
    parser.add_argument("--output-dir", type=Path,
                        default=MANISKILL_ROOT / "outputs/pi05_target_web_forty")
    args = parser.parse_args()
    if not 1024 <= args.policy_port <= 65535:
        parser.error("--policy-port must be between 1024 and 65535")
    lab = Lab(args.output_dir, args.gpu, dict(mask_source=args.mask_source, perception_url=args.perception_url,
                                          target_text=args.target_text, loss_mode=args.mask_loss_mode),
              policy_port=args.policy_port)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.lab = lab
    def shutdown_worker(_signum, _frame):
        lab.cancel.set()
        lab.close_perception()
        lab.stop_worker()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, shutdown_worker)
    signal.signal(signal.SIGHUP, shutdown_worker)
    signal.signal(signal.SIGINT, shutdown_worker)
    print(f"π0.5 target web: http://127.0.0.1:{args.port} | default GPU {args.gpu}", flush=True)
    try:
        server.serve_forever()
    finally:
        lab.stop_worker()
        lab.close_perception()


if __name__ == "__main__":
    main()
