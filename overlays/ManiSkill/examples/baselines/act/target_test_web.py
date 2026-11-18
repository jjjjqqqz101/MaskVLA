"""Local-only, interactive TargetPickCube ACT evaluation. No training or dataset loading."""
import argparse
import copy
import hashlib
import json
import mimetypes
import secrets
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

from train_dual_rgb import (Args, Agent, DualRGBObservationWrapper, STATE_FIELDS, STREAM_ORDER,
                            dual_rgb, policy_state as dual_policy_state)
from train_single_rgb import (Args as SingleArgs, Agent as SingleAgent,
                              policy_state, single_rgb)
from act.dual_rgb_video import DualRGBVideoWrapper
from act.single_rgb_video import SingleRGBVideoWrapper
from mani_skill.envs.tasks.tabletop.target_pick_cube_ycb import TargetPickCubeYCBEnv
from mani_skill.envs.tasks.tabletop.target_pick_eight import TargetPickEightEnv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def perception_config(data=None, defaults=None):
    """Validate experiment settings before a background job can start."""
    values = dict(mask_source='oracle', endpoint='http://127.0.0.1:18830',
                  target_text='', loss_mode='observe', max_pause_frames=60,
                  intervention=dict(kind='none', start=50, duration=30, dx=20, dy=0))
    for update in (defaults or {}, data or {}):
        if not isinstance(update, dict):
            raise ValueError('Mask 设置必须是 JSON 对象')
        values.update({k: v for k, v in update.items() if k in values and k != 'intervention'})
        if 'intervention' in update:
            if not isinstance(update['intervention'], dict):
                raise ValueError('Mask 干预设置必须是 JSON 对象')
            values['intervention'] = dict(values['intervention'], **update['intervention'])
    if values['mask_source'] not in ('oracle', 'perception'):
        raise ValueError('请选择仿真真值或 VLM + SAM2 + Cutie')
    if values['loss_mode'] not in ('observe', 'pause', 'reacquire'):
        raise ValueError('未知的 Mask 丢失处理方式')
    text = values['target_text']
    if not isinstance(text, str) or len(text) > 1000:
        raise ValueError('目标描述最多 1000 个字符')
    values['target_text'] = text.strip()
    if values['mask_source'] == 'perception' and not values['target_text']:
        raise ValueError('请描述画面中的目标，例如“桌面左边的红方块”；物体编号不会发送给 VLM')
    if not isinstance(values['endpoint'], str) or len(values['endpoint']) > 200:
        raise ValueError('感知服务地址无效')
    endpoint = urlparse(values['endpoint'])
    if endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('感知服务须为本机 HTTP 地址')
    if type(values['max_pause_frames']) is not int or not 1 <= values['max_pause_frames'] <= 500:
        raise ValueError('等待恢复步数须为 1 到 500 的整数')
    intervention = values['intervention']
    if intervention.get('kind') not in ('none', 'dropout', 'freeze', 'shift', 'occlusion'):
        raise ValueError('未知的 Mask 干预类型')
    for name, low, high in (('start', 0, 500), ('duration', 1, 500), ('dx', -224, 224), ('dy', -224, 224)):
        if type(intervention.get(name)) is not int or not low <= intervention[name] <= high:
            raise ValueError(f'Mask 干预 {name} 必须为 {low} 到 {high} 的整数')
    values['intervention'] = {k: intervention[k] for k in ('kind', 'start', 'duration', 'dx', 'dy')}
    return values


class TrackingObservationWrapper(gym.ObservationWrapper):
    """Keep RGB-only learned perception separate from evaluation-only GT labels.

    Construction/reset never contacts the model service. Each actual trial
    explicitly starts a fresh tracker after restoring its physical state.
    """

    def __init__(self, env, input_mode='dual_rgb', reacquire=None):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.input_mode = input_mode
        self.transforms = T.Resize((224, 224), antialias=True)
        self.client = None
        self.reacquire = reacquire or threading.Event()
        self.frame_index = 0
        self.last_metrics = {}
        self.last_rgb = self.last_mask = None
        self.mask_source = 'oracle'
        self.base_env.update_obs_space(self.observation(self.base_env._init_raw_obs))

    def close_perception(self):
        client, self.client = self.client, None
        if client is not None:
            client.close()

    def use_oracle(self):
        self.close_perception()
        self.mask_source = 'oracle'
        self.frame_index = 0
        self.reacquire.clear()

    def _rgb(self, observation):
        rgb = array(observation['sensor_data']['base_camera']['rgb'])
        if rgb.shape[0] != 1:
            raise ValueError('当前交互感知实验只支持单场景')
        return np.asarray(rgb[0], dtype=np.uint8)

    def start_perception(self, observation, target_text, options, client):
        self.close_perception()
        self.mask_source = 'perception'
        self.client = client
        self.frame_index = 0
        # Only these RGB pixels and human-visible description leave this process.
        result = client.start(self._rgb(observation), target_text, options)
        return self._build(observation, result)

    def observation(self, observation):
        result = None
        if self.mask_source == 'perception':
            if self.client is None:
                raise RuntimeError('感知会话已关闭，请重新开始测试')
            self.frame_index += 1
            force = self.reacquire.is_set()
            self.reacquire.clear()
            result = self.client.step(self._rgb(observation), self.frame_index, force_reacquire=force)
        return self._build(observation, result)

    def _build(self, observation, result):
        camera = observation['sensor_data']['base_camera']
        ids = torch.stack([obj.per_scene_id for obj in self.base_env.cubes], dim=1)
        target_ids = ids.gather(1, self.base_env.target_idx[:, None]).to(camera['segmentation'].device)
        gt = camera['segmentation'] == target_ids[:, None, None, :]
        gt_array = array(gt)[0, :, :, 0].astype(bool)
        raw = self._rgb(observation)
        if result is None:
            effective = gt_array
            metadata = dict(source='oracle', state='oracle', generation=0, quality=1.0, latency_ms=0)
        else:
            effective = np.asarray(result.mask)
            if effective.shape != raw.shape[:2] or effective.dtype != np.bool_:
                raise ValueError('感知服务必须返回与 RGB 对齐的 bool H×W Mask；已停止，未回退到真值')
            metadata = dict(result.metadata)
            metadata['source'] = 'vlm_sam2_cutie'
        union = int(np.logical_or(effective, gt_array).sum())
        self.last_metrics = dict(metadata, frame_index=self.frame_index,
            mask_source=self.mask_source, policy_mask_pixels=int(effective.sum()),
            gt_mask_pixels=int(gt_array.sum()),
            gt_iou=float(np.logical_and(effective, gt_array).sum() / union) if union else None)
        self.last_rgb, self.last_mask = raw.copy(), effective.copy()
        mask = torch.as_tensor(effective, device=camera['rgb'].device)[None, :, :, None]
        masked = torch.where(mask, camera['rgb'], 0)
        state_fn = dual_policy_state if self.input_mode == 'dual_rgb' else policy_state
        images = (dual_rgb(camera['rgb'], masked, self.transforms) if self.input_mode == 'dual_rgb'
                  else single_rgb(masked, self.transforms))
        return dict(state=state_fn(observation), rgb=images)


def hold_joint_action(observation):
    """Hold Panda joint targets without consuming an ACT action chunk."""
    qpos = array(observation['state'])[0, :9]
    openness = float(np.clip(np.mean(qpos[7:9]) / .04, 0, 1))
    return np.r_[qpos[:7], 2 * openness - 1].astype(np.float32)[None]


class MaskedOnlyObservationWrapper(gym.ObservationWrapper):
    """Give a single-RGB ACT checkpoint only target RGB multiplied by its mask."""

    def __init__(self, env):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.transforms = T.Resize((224, 224), antialias=True)
        self.base_env.update_obs_space(self.observation(self.base_env._init_raw_obs))

    def observation(self, observation):
        camera = observation['sensor_data']['base_camera']
        ids = torch.stack([obj.per_scene_id for obj in self.base_env.cubes], dim=1)
        target_ids = ids.gather(1, self.base_env.target_idx[:, None])
        target_ids = target_ids.to(camera['segmentation'].device)
        mask = camera['segmentation'] == target_ids[:, None, None, :]
        masked_rgb = torch.where(mask, camera['rgb'], 0)
        return dict(state=policy_state(observation),
                    rgb=single_rgb(masked_rgb, self.transforms))


class MaskedOnlyVideoWrapper(SingleRGBVideoWrapper):
    policy_title = 'Target RGB x Mask: sole ACT image'


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(array(value)).tobytes()).hexdigest()


def file_digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    return copy.deepcopy(value)


def assert_tree_equal(a, b, path='state'):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise RuntimeError(f'{path}: state keys changed')
        for key in a:
            assert_tree_equal(a[key], b[key], f'{path}/{key}')
    # PhysX root-link pose round trips can differ by a few float32 ULPs.
    # Policy RGB and policy state are still checked bit-for-bit below.
    elif not np.allclose(array(a), array(b), rtol=0, atol=1e-7):
        raise RuntimeError(f'{path}: initial state restore mismatch')


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


class Lab:
    def __init__(self, output, dataset_json, env_id='TargetPickCube-v1', input_mode='dual_rgb',
                 perception_defaults=None):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.env_id = env_id
        self.input_mode = input_mode
        self.stream_order = STREAM_ORDER if input_mode == 'dual_rgb' else ('original_rgb',)
        self.mask_stream = 1 if input_mode == 'dual_rgb' else 0
        self.perception_defaults = perception_config(defaults=perception_defaults)
        self.target_labels = (TargetPickEightEnv.object_labels if env_id == 'TargetPickEight-v1'
                              else TargetPickCubeYCBEnv.object_labels if env_id == 'TargetPickCubeYCB-v1'
                              else ('cube 1', 'cube 2', 'cube 3'))
        if not dataset_json.is_file():
            raise FileNotFoundError(f'Optional dataset metadata unavailable: {dataset_json}. Pass --dataset-json for an existing dataset; new datasets were not copied with the code.')
        self.recorded = json.loads(dataset_json.read_text())['env_info']['env_kwargs']
        meta = json.loads(dataset_json.read_text())
        self.train_seeds = {e.get('episode_seed') for e in meta['episodes']}
        self.dataset_json = str(dataset_json.resolve())
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='simulation')
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.reacquire = threading.Event()
        self.status = dict(busy=False, message='请选择权重并生成场景', scene=None,
                           targets=list(self.target_labels), env_id=self.env_id,
                           input_mode=input_mode, progress=0, preview=None,
                           error=None, results=[], comparison=None,
                           perception_supported=True, perception_api_version=1,
                           perception_config=self.perception_defaults, perception=None,
                           perception_preview=None, perception_mask_preview=None)
        self.env = self.policy = self.video = self.base = None
        self.model = None
        self.loaded_key = None
        self.scene = None
        # Completed records remain available after a server restart.
        for p in sorted(self.output.glob('trials/*/result.json'))[-100:]:
            try:
                self.status['results'].append(json.loads(p.read_text()))
            except (ValueError, OSError):
                pass

    def update(self, **kwargs):
        with self.lock:
            self.status.update(kwargs)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.status)

    def url(self, path):
        return '/artifacts/' + path.relative_to(self.output).as_posix()

    def checkpoints(self):
        # Folder names are user-editable; the model contract is checked at load.
        # Preserve default three-cube discovery, including user-renamed runs.
        # The input contract is still validated before a checkpoint is loaded.
        paths = sorted(p for p in (HERE / 'runs').rglob('*.pt') if p.is_file())
        if self.input_mode == 'masked_only':
            paths = [p for p in paths if '__train_single_rgb__' in p.parent.parent.name
                     and 'TargetPickEight-v1' in p.parent.parent.name]
        return [dict(path=str(p.relative_to(HERE)), label=f'{p.parent.parent.name} / {p.name}',
                     bytes=p.stat().st_size) for p in paths]

    def checkpoint_path(self, name):
        path = (HERE / str(name)).resolve()
        allowed = {str((HERE / x['path']).resolve()) for x in self.checkpoints()}
        if str(path) not in allowed:
            raise ValueError('权重路径已失效或文件夹已改名，请点击“刷新权重列表”后重新选择 checkpoint')
        return path

    def submit(self, action, data):
        if action not in ('scene', 'target', 'run', 'compare'):
            raise ValueError('Unknown operation')
        target = data.get('target', 1)
        if type(target) is not int or not 1 <= target <= len(self.target_labels):
            raise ValueError(f'目标 ID 必须为 1 到 {len(self.target_labels)}')
        if action == 'scene':
            seed = data.get('seed')
            if seed is None:
                seed = secrets.randbelow(2**31)
            if type(seed) is not int or not 0 <= seed < 2**31:
                raise ValueError('seed 必须为 0 到 2147483647 的整数')
            data['seed'] = seed
        else:
            if self.scene is None:
                raise ValueError('请先生成场景')
            if data.get('scene_id') != self.scene['id']:
                raise ValueError('场景已经变化，请刷新后重试')
        if action in ('run', 'compare'):
            self.checkpoint_path(data.get('checkpoint', ''))
            data['perception'] = perception_config(data.get('perception'), self.perception_defaults)
            if action == 'compare' and data['perception']['mask_source'] == 'perception':
                raise ValueError('感知模式请逐个目标输入可见特征描述并测试；全部目标对照继续使用仿真真值')
        with self.lock:
            if self.status['busy']:
                raise ValueError('已有任务正在运行，请等待或停止当前测试')
            self.status.update(busy=True, error=None, progress=0, message='准备中…')
        self.cancel.clear()
        self.executor.submit(self.work, action, dict(data, target=target))

    def work(self, action, data):
        try:
            if action == 'scene':
                self.new_scene(data['seed'], data['target'])
            elif action == 'target':
                self.restore(data['target'])
                self.publish(self.video.render()[0])
                self.update(message=f'已恢复初始场景 · 目标 {self.target_labels[data["target"] - 1]}')
            else:
                self.load_model(data['checkpoint'])
                targets = [data['target']] if action == 'run' else list(range(1, len(self.target_labels) + 1))
                group = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                records = []
                for target in targets:
                    if self.cancel.is_set():
                        break
                    records.append(self.run_trial(target, group, data['perception']))
                    if records[-1].get('error'):
                        break
                if action == 'compare':
                    comparison = dict(group=group, scene_id=self.scene['id'], seed=self.scene['seed'],
                                      checkpoint=str(self.ckpt), trials=[r['id'] for r in records],
                                      complete=len(records) == len(targets) and all(r.get('complete') for r in records),
                                      same_initial_rgb=len({r['initial_rgb_sha256'] for r in records}) == 1,
                                      same_initial_state=len({r['initial_state_sha256'] for r in records}) == 1,
                                      different_masks=len({r['initial_masked_rgb_sha256'] for r in records}) == len(targets),
                                      correct_pick_count=sum(r['correct_pick'] for r in records))
                    write_json(self.output / f'comparison_{group}.json', comparison)
                    self.update(comparison=comparison)
                self.update(message=('测试已停止，已保存完成的帧和结果' if self.cancel.is_set()
                                     else '测试失败，视频与逐步指标已保存' if any(r.get('error') for r in records)
                                     else '测试完成，视频与结果已保存'))
        except Exception as exc:
            traceback.print_exc()
            self.update(error=str(exc), message='操作失败，请查看错误信息')
        finally:
            self.update(busy=False)

    def ensure_env(self):
        if self.env is not None:
            return
        torch.set_num_threads(4)
        torch.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        kwargs = dict(obs_mode='rgb+segmentation', control_mode='pd_joint_pos',
                      sim_backend='physx_cpu', reward_mode='sparse', render_mode='rgb_array',
                      max_episode_steps=500, reconfiguration_freq=0)
        for key in ('sensor_configs', 'robot_uids', 'robot_init_qpos_noise'):
            if key in self.recorded:
                kwargs[key] = self.recorded[key]
        self.env = gym.make(self.env_id, **kwargs)
        self.base = self.env.unwrapped
        video_type = DualRGBVideoWrapper if self.input_mode == 'dual_rgb' else MaskedOnlyVideoWrapper
        self.policy = TrackingObservationWrapper(self.env, self.input_mode, self.reacquire)
        self.video = video_type(self.policy)

    def raw_camera_rgb(self):
        info = self.base.get_info()
        return self.base.get_obs(info)['sensor_data']['base_camera']['rgb'].detach().clone()

    def observe(self):
        info = self.base.get_info()
        obs = self.policy.observation(self.base.get_obs(info))
        self.video._remember(obs, info)
        return obs, info

    def new_scene(self, seed, target):
        self.ensure_env()
        self.policy.use_oracle()
        self.video.reset(seed=seed)
        self.base.target_idx[:] = target - 1
        obs, _ = self.observe()
        self.initial = clone_tree(self.base.get_state_dict())
        self.controller = clone_tree(self.base.agent.get_controller_state())
        raw_state = obs['state'].clone()
        # Compare like with like: PhysX pose serialization/restoration can change
        # derived TCP coordinates by a float32 ULP. Establish the policy baseline
        # AFTER the same restore path used for every subsequent target trial.
        # Keep self.initial unchanged: repeatedly serializing the restored state
        # could accumulate root-link pose roundoff.
        obs, _ = self._apply_initial_state(target)
        roundtrip_error = float((obs['state'] - raw_state).abs().max())
        self.initial_rgb = (obs['rgb'][:, 0].clone() if self.input_mode == 'dual_rgb'
                            else self.raw_camera_rgb())
        self.initial_state = obs['state'].clone()
        scene_id = f'{datetime.now():%Y%m%d_%H%M%S_%f}_seed{seed}'
        folder = self.output / 'scenes' / scene_id
        folder.mkdir(parents=True)
        torch.save(dict(env_state=self.initial, controller=self.controller), folder / 'initial_state.pt')
        self.scene = dict(id=scene_id, seed=seed, target=target, train_seed=seed in self.train_seeds,
                          object_labels=list(self.target_labels),
                          object_positions=[array(c.pose.p)[0].tolist() for c in self.base.cubes],
                          goal_position=array(self.base.goal_site.pose.p)[0].tolist(),
                          initial_rgb_sha256=digest(self.initial_rgb),
                          initial_state_sha256=digest(self.initial_state),
                          initial_observation_after_restore=True,
                          initial_state_roundtrip_max_error=roundtrip_error,
                          dataset_json=self.dataset_json, sensor_configs=self.recorded.get('sensor_configs'),
                          state_file=str(folder / 'initial_state.pt'))
        if self.env_id == 'TargetPickCube-v1':
            self.scene['cube_positions'] = self.scene['object_positions']
        write_json(folder / 'scene.json', self.scene)
        self.update(scene=copy.deepcopy(self.scene), comparison=None)
        frame = self.video.render()[0]
        Image.fromarray(frame).save(folder / 'initial.png')
        self.publish(frame)
        self.update(message=f'新场景已生成 · seed {seed} · 请选择目标')

    def _apply_initial_state(self, target):
        state = clone_tree(self.initial)
        state['target_idx'][:] = target - 1
        self.base.set_state_dict(state)
        self.base.agent.set_controller_state(clone_tree(self.controller))
        assert_tree_equal(state, self.base.get_state_dict())
        self.base.target_idx[:] = target - 1
        return self.observe()

    def restore(self, target, require_visible=True):
        # Reset controller, counters, forces and simulator contacts, then restore
        # the saved physical state. No simulation step is taken before obs[0].
        self.policy.use_oracle()
        self.video.reset(seed=self.scene['seed'])
        obs, info = self._apply_initial_state(target)
        current_rgb = obs['rgb'][:, 0] if self.input_mode == 'dual_rgb' else self.raw_camera_rgb()
        if not torch.equal(current_rgb, self.initial_rgb):
            raise RuntimeError('恢复后 Original RGB 不一致，已阻止测试')
        if not torch.equal(obs['state'], self.initial_state):
            difference = (obs['state'] - self.initial_state).abs()
            raise RuntimeError(f'恢复后 policy state 不一致，已阻止测试；最大差值 {difference.max().item():.3g}，'
                               f'不同分量 {torch.nonzero(difference).tolist()}')
        if require_visible and not torch.count_nonzero(obs['rgb'][:, self.mask_stream]):
            raise RuntimeError('目标初始 Mask 为空，无法执行有效测试')
        self.scene['target'] = target
        self.update(scene=copy.deepcopy(self.scene))
        return obs, info

    def publish(self, frame):
        tmp = self.output / 'live.tmp.png'
        Image.fromarray(frame).save(tmp)
        tmp.replace(self.output / 'live.png')
        self.update(preview='/artifacts/live.png', frame_version=time.time_ns())
        metrics = self.policy.last_metrics
        self.update(perception=copy.deepcopy(metrics))
        if self.policy.mask_source == 'perception' and self.policy.last_rgb is not None:
            rgb, mask = self.policy.last_rgb, self.policy.last_mask
            overlay = rgb.copy()
            overlay[mask] = (overlay[mask] * .55 + np.array([40, 220, 140]) * .45).astype(np.uint8)
            image = Image.fromarray(overlay)
            # The current mask bounds are not the VLM's original grounding box.
            box = metrics.get('vlm_bbox_xyxy')
            if isinstance(box, (list, tuple)) and len(box) == 4:
                ImageDraw.Draw(image).rectangle(tuple(float(v) for v in box), outline='#ffcf43', width=2)
            for image, name in ((image, 'perception_live.png'),
                                (Image.fromarray(mask.astype(np.uint8) * 255), 'perception_mask.png')):
                temporary = self.output / (name + '.tmp')
                image.save(temporary, format='PNG')
                temporary.replace(self.output / name)
            self.update(perception_preview='/artifacts/perception_live.png',
                        perception_mask_preview='/artifacts/perception_mask.png')

    def load_model(self, name):
        self.ensure_env()
        path = self.checkpoint_path(name)
        key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        if key == self.loaded_key:
            return
        self.update(message=f'加载 EMA 权重：{path.name}')
        # Only local checkpoints explicitly selected from this project's runs.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if tuple(checkpoint.get('stream_order', ())) != self.stream_order:
            raise ValueError('checkpoint 输入路数与当前测试模式不匹配')
        if self.input_mode == 'masked_only' and checkpoint.get('input_mode') != 'single_rgb':
            raise ValueError('此模式只能加载原始 RGB 单路 ACT 权重')
        if tuple(tuple(x) for x in checkpoint.get('state_fields', ())) != STATE_FIELDS:
            raise ValueError('checkpoint state whitelist 不匹配')
        values = checkpoint['args']
        args_type = Args if self.input_mode == 'dual_rgb' else SingleArgs
        args = args_type(**{f.name: values[f.name] for f in fields(args_type) if f.name in values})
        if args.env_id not in ('TargetPickCube-v1', 'TargetPickEight-v1') or args.include_depth or args.control_mode != 'pd_joint_pos':
            raise ValueError('仅支持三方块或八物品任务、无 depth、pd_joint_pos 的 ACT')
        if args.max_episode_steps != 500:
            raise ValueError('此测试场景采用 500 步，请选择 max_episode_steps=500 的权重')
        demo_meta = (HERE / args.demo_path).resolve().with_suffix('.json')
        if not demo_meta.is_file():
            # A100 checkpoints may retain their absolute original demo path.
            source = Path(args.demo_path)
            if source.is_absolute() and 'demos' in source.parts:
                local = ROOT.joinpath(*source.parts[source.parts.index('demos'):]).with_suffix('.json')
                if local.is_file():
                    demo_meta = local
            if not demo_meta.is_file():
                raise FileNotFoundError(f'Checkpoint dataset metadata unavailable: {demo_meta}. Supply the matching local dataset metadata before using this checkpoint.')
        if not demo_meta.is_file():
            raise ValueError(f'无法验证训练相机配置，找不到 {demo_meta}')
        recorded = json.loads(demo_meta.read_text())['env_info']['env_kwargs']
        for k in ('sensor_configs', 'robot_uids', 'robot_init_qpos_noise'):
            if recorded.get(k) != self.recorded.get(k):
                raise ValueError(f'checkpoint 的训练场景配置不匹配: {k}')
        adapter = SimpleNamespace(single_observation_space=gym.spaces.Dict({
            'state': gym.spaces.Box(-np.inf, np.inf, (28,), dtype=np.float32),
            'rgb': gym.spaces.Box(0, 255, (len(self.stream_order), 3, 224, 224), dtype=np.uint8)}),
            single_action_space=gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        agent_type = Agent if self.input_mode == 'dual_rgb' else SingleAgent
        model = agent_type(adapter, args).to(device)
        model.load_state_dict(checkpoint['ema_agent'], strict=True)
        model.eval()
        stats = {k: torch.as_tensor(checkpoint['norm_stats'][k], device=device) for k in
                 ('state_mean', 'state_std', 'action_mean', 'action_std')}
        for k, v in stats.items():
            if not torch.isfinite(v).all() or ('std' in k and not (v > 0).all()):
                raise ValueError(f'无效归一化参数: {k}')
        self.model, self.args, self.stats, self.device = model, args, stats, device
        self.ckpt, self.ckpt_sha = path, file_digest(path)
        self.loaded_key = key
        self.update(loaded_checkpoint=str(path), weights='ema_agent',
                    visual_input='target_rgb_x_mask_only' if self.input_mode == 'masked_only' else 'dual_rgb')

    # Simulator tensors survive resets; keep inference_mode scoped to the policy.
    @torch.no_grad()
    def run_trial(self, target, group, perception=None):
        config = perception_config(perception, self.perception_defaults)
        obs, info = self.restore(target, require_visible=config['mask_source'] == 'oracle')
        kind = 'cube' if self.env_id == 'TargetPickCube-v1' else 'object'
        trial_id = f'{group}_{kind}{target}'
        folder = self.output / 'trials' / trial_id
        folder.mkdir(parents=True, exist_ok=False)
        if config['mask_source'] == 'perception':
            from perception_mask.client import PerceptionMaskClient
            self.update(message='VLM 正在定位目标，SAM2 初始化分割与 Cutie 跟踪…',
                        perception_config=config)
            client = PerceptionMaskClient(endpoint=config['endpoint'], timeout=180)
            try:
                raw = self.base.get_obs(info)
                obs = self.policy.start_perception(raw, config['target_text'],
                    dict(loss_mode=config['loss_mode'], intervention=config['intervention']), client)
                self.video._remember(obs, info)
            except Exception as exc:
                try:
                    self.policy.close_perception()
                finally:
                    write_json(folder / 'error.json', dict(mask_source='perception',
                        perception_config=config, error=str(exc), oracle_fallback=False))
                raise
        else:
            self.update(perception_config=config)
        original = obs['rgb'][:, 0] if self.input_mode == 'dual_rgb' else self.raw_camera_rgb()
        initial_hashes = dict(initial_rgb_sha256=digest(original),
                              initial_state_sha256=digest(obs['state']),
                              initial_masked_rgb_sha256=digest(obs['rgb'][:, self.mask_stream]))
        np.savez_compressed(folder / 'initial_policy_input.npz', rgb=array(obs['rgb']), state=array(obs['state']))
        if self.input_mode == 'dual_rgb':
            for stream, name in enumerate(('original', 'masked_rgb')):
                Image.fromarray(array(obs['rgb'][0, stream]).transpose(1, 2, 0)).save(folder / f'{name}.png')
        else:
            Image.fromarray(array(self.raw_camera_rgb())[0]).save(folder / 'original_for_display.png')
            Image.fromarray(array(obs['rgb'][0, 0]).transpose(1, 2, 0)).save(folder / 'masked_rgb_policy_input.png')
        q, horizon = self.args.num_queries, self.args.max_episode_steps
        history, prediction = [], None
        policy_clock = chunk_start = paused_frames = pause_streak = 0
        was_paused = False
        generation = self.policy.last_metrics.get('generation')
        generation_changes = 0
        actions, traces = [], []
        ever_held = set()
        target_grasped_ever = lifted_ever = grasped_lifted_ever = wrong_ever = success_once = False
        minimum_visible = 224 * 224
        start = time.monotonic()
        frame = self.video.render()[0]
        Image.fromarray(frame).save(folder / 'initial.png')
        self.publish(frame)
        writer = imageio.get_writer(folder / 'video.mp4', fps=self.base.control_freq,
                                   codec='libx264', quality=7, macro_block_size=16,
                                   ffmpeg_params=['-movflags', '+faststart'])
        stop_reason, failure, cleanup_error = 'horizon', None, None
        try:
            writer.append_data(frame)
            for t in range(horizon):
                if self.cancel.is_set():
                    stop_reason = 'cancelled'
                    break
                mask_input = copy.deepcopy(self.policy.last_metrics)
                current_generation = mask_input.get('generation')
                if current_generation != generation:
                    history, prediction = [], None
                    chunk_start, generation = policy_clock, current_generation
                    generation_changes += 1
                lost = (mask_input.get('policy_mask_pixels', 0) == 0 or
                        mask_input.get('state') in ('lost', 'reacquiring', 'failed'))
                paused = config['mask_source'] == 'perception' and config['loss_mode'] != 'observe' and lost
                # Keep the legacy metric in resized ACT policy-pixel coordinates.
                minimum_visible = min(minimum_visible,
                                      int(obs['rgb'][0, self.mask_stream].any(dim=0).sum().item()))
                if paused:
                    paused_frames += 1
                    pause_streak += 1
                    if pause_streak > config['max_pause_frames']:
                        stop_reason = 'mask_lost_timeout'
                        failure = 'Mask 持续丢失，已达到等待恢复步数上限并停止；未切换到真值 Mask'
                        break
                    action_np = hold_joint_action(obs)
                else:
                    if was_paused:
                        # No stale chunk or temporal votes survive a tracking pause.
                        history, prediction, chunk_start = [], None, policy_clock
                    pause_streak = 0
                    policy_obs = {k: v.detach().clone().to(self.device) for k, v in obs.items()}
                    policy_obs['state'] = (policy_obs['state'] - self.stats['state_mean']) / self.stats['state_std']
                    if self.args.temporal_agg or prediction is None or policy_clock - chunk_start >= q:
                        prediction = self.model.get_action(policy_obs)
                        chunk_start = policy_clock
                        if prediction.shape != (1, q, 8) or not torch.isfinite(prediction).all():
                            raise RuntimeError('模型输出 shape 错误或包含 NaN/Inf')
                    if self.args.temporal_agg:
                        history.append((policy_clock, prediction))
                        history = [(ts, pred) for ts, pred in history if policy_clock - ts < q]
                        current = torch.stack([pred[:, policy_clock-ts] for ts, pred in history], dim=1)
                        weights = torch.exp(-.01 * torch.arange(len(history), device=self.device))
                        raw = (current * (weights / weights.sum())[None, :, None]).sum(dim=1)
                    else:
                        raw = prediction[:, policy_clock - chunk_start]
                    action = raw * self.stats['action_std'] + self.stats['action_mean']
                    if not torch.isfinite(action).all():
                        raise RuntimeError('反归一化动作包含 NaN/Inf')
                    action_np = array(action)
                    policy_clock += 1
                was_paused = paused
                if self.cancel.is_set():
                    stop_reason = 'cancelled'
                    break
                obs, reward, terminated, truncated, info = self.video.step(action_np)
                actions.append(action_np[0])
                frame = self.video.render()[0]
                metadata = self.video.last_frame_metadata[0]
                held = [i + 1 for i in metadata['holding']]
                ever_held.update(held)
                flags = {k: bool(array(info[k]).item()) for k in
                         ('is_target_grasped', 'target_lifted', 'wrong_object_grasped', 'success')}
                target_grasped_ever |= flags['is_target_grasped']
                lifted_ever |= flags['target_lifted']
                grasped_lifted_ever |= flags['is_target_grasped'] and flags['target_lifted']
                wrong_ever |= flags['wrong_object_grasped']
                success_once |= flags['success']
                traces.append(dict(step=t+1, policy_step=policy_clock, holding=held,
                                   paused_for_mask=paused, perception=mask_input, **flags))
                writer.append_data(frame)
                if t % 5 == 0 or t == horizon - 1 or paused:
                    self.publish(frame)
                    self.update(progress=(t+1)/horizon,
                        message=f'目标 {self.target_labels[target - 1]} · {t+1}/{horizon} 步 · '
                                f'Mask {self.policy.last_metrics.get("state", "oracle")}' +
                                (' · 保持关节位置等待恢复' if paused else ''))
                # Match training evaluation: success does not terminate the trial.
                if bool(array(truncated).item()):
                    stop_reason = 'env_truncated'
                    break
        except Exception as exc:
            failure = str(exc)
            stop_reason = 'perception_error' if config['mask_source'] == 'perception' else 'inference_error'
            self.update(error=failure)
        finally:
            writer.close()
            try:
                self.policy.close_perception()
            except Exception as exc:
                # Cleanup transport must not change the measured grasp outcome.
                cleanup_error = f'感知会话关闭失败: {exc}'
        Image.fromarray(frame).save(folder / 'final.png')
        self.publish(frame)
        np.save(folder / 'actions.npy', np.asarray(actions, dtype=np.float32))
        write_json(folder / 'step_metrics.json', traces)
        cancelled = self.cancel.is_set()
        complete = not cancelled and not failure and stop_reason in ('horizon', 'env_truncated')
        mask_rows = [row['perception'] for row in traces]
        ious = [row['gt_iou'] for row in mask_rows if row.get('gt_iou') is not None]
        result = dict(id=trial_id, group=group, scene_id=self.scene['id'], seed=self.scene['seed'],
                      target=target, target_label=self.target_labels[target - 1], target_idx=target-1, checkpoint=str(self.ckpt),
                      checkpoint_sha256=self.ckpt_sha, weights='ema_agent',
                      steps=len(actions), cancelled=cancelled, complete=complete,
                      stop_reason=stop_reason, error=failure, cleanup_error=cleanup_error,
                      seconds=round(time.monotonic()-start, 2),
                      mask_source=config['mask_source'], perception_config=config, oracle_fallback=False,
                      mask_summary=dict(mean_gt_iou=float(np.mean(ious)) if ious else None,
                          empty_policy_mask_frames=sum(row.get('policy_mask_pixels', 0) == 0 for row in mask_rows),
                          lost_state_frames=sum(row.get('state') in ('lost', 'reacquiring', 'failed') for row in mask_rows),
                          paused_frames=paused_frames, policy_steps=policy_clock,
                          tracker_generation_changes=generation_changes),
                      grasped_object_ids=sorted(ever_held), grasped_cube_ids=sorted(ever_held),
                      target_grasped_ever=target_grasped_ever,
                      target_lifted_ever=lifted_ever, target_grasped_and_lifted_ever=grasped_lifted_ever,
                      wrong_object_grasped_ever=wrong_ever,
                      correct_pick=grasped_lifted_ever and not wrong_ever and complete,
                      success_once=success_once, success_at_end=bool(array(info['success']).item()),
                      minimum_visible_pixels=minimum_visible, **initial_hashes,
                      restored_initial_state=True, stream_order=self.stream_order,
                      checkpoint_stream_order=self.stream_order,
                      runtime_stream_order=('masked_rgb',) if self.input_mode == 'masked_only' else STREAM_ORDER,
                      visual_input='target_rgb_x_mask_only' if self.input_mode == 'masked_only' else 'dual_rgb',
                      state_fields=STATE_FIELDS,
                      inference=dict(num_queries=q, temporal_agg=self.args.temporal_agg,
                                     temporal_weight_k=0.01, max_episode_steps=horizon,
                                     control_mode=self.args.control_mode, device=str(self.device)),
                      video=self.url(folder / 'video.mp4'), initial=self.url(folder / 'initial.png'),
                      final=self.url(folder / 'final.png'), metrics=self.url(folder / 'result.json'),
                      folder=str(folder))
        write_json(folder / 'result.json', result)
        with self.lock:
            self.status['results'].append(result)
            self.status['results'] = self.status['results'][-100:]
        return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if not self.path.startswith(('/api/status', '/artifacts/live.png')):
            super().log_message(fmt, *args)

    def reply(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        origin = self.headers.get('Origin')
        if origin and urlparse(origin).netloc != self.headers.get('Host'):
            return self.reply({'error': 'Origin mismatch'}, 403)
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 16384:
                raise ValueError('Invalid request size')
            if self.headers.get_content_type() != 'application/json':
                raise ValueError('Expected application/json')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('Expected JSON object')
            route = urlparse(self.path).path
            if route == '/api/stop':
                self.server.lab.cancel.set()
            elif route == '/api/reacquire':
                state = self.server.lab.snapshot()
                if not state.get('busy') or state.get('perception_config', {}).get('mask_source') != 'perception':
                    raise ValueError('请在感知抓取测试运行时重新定位目标')
                self.server.lab.reacquire.set()
            elif route.startswith('/api/'):
                self.server.lab.submit(route[5:], data)
            else:
                raise ValueError('Unknown route')
            self.reply({'ok': True})
        except (ValueError, TypeError, KeyError) as exc:
            self.reply({'error': str(exc)}, 400)

    def do_GET(self):
        route = unquote(urlparse(self.path).path)
        if route == '/api/status':
            return self.reply(self.server.lab.snapshot())
        if route == '/api/checkpoints':
            return self.reply(self.server.lab.checkpoints())
        if route == '/':
            page = ('target_test_web_masked_only.html' if self.server.lab.input_mode == 'masked_only'
                    else 'target_test_web_eight.html' if self.server.lab.env_id == 'TargetPickEight-v1'
                    else 'target_test_web_mixed.html' if self.server.lab.env_id == 'TargetPickCubeYCB-v1'
                    else 'target_test_web.html')
            path = HERE / page
        elif route.startswith('/artifacts/'):
            path = (self.server.lab.output / route[len('/artifacts/'):]).resolve()
            if not path.is_relative_to(self.server.lab.output):
                return self.reply({'error': 'Forbidden'}, 403)
        else:
            return self.reply({'error': 'Not found'}, 404)
        if not path.is_file():
            return self.reply({'error': 'Not found'}, 404)
        if path.name == 'live.png':
            # Read one complete atomic image version before setting Content-Length.
            body = path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        range_header = self.headers.get('Range')
        if range_header:
            try:
                spec = range_header.removeprefix('bytes=').split('-')
                start = int(spec[0]) if spec[0] else max(0, size-int(spec[1]))
                end = min(int(spec[1]), size-1) if spec[0] and spec[1] else size-1
                if start < 0 or start > end:
                    raise ValueError()
                status = 206
            except (ValueError, IndexError):
                return self.reply({'error': 'Invalid range'}, 416)
        self.send_response(status)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(end-start+1))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-store')
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        try:
            with path.open('rb') as f:
                f.seek(start)
                remaining = end-start+1
                while remaining:
                    chunk = f.read(min(1024*1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18765)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/act_target_web')
    parser.add_argument('--dataset-json', type=Path, default=ROOT / 'demos/TargetPickCube-v1/motionplanning/target_pick_500_mask/dataset/train.json')
    parser.add_argument('--env-id', choices=('TargetPickCube-v1', 'TargetPickCubeYCB-v1', 'TargetPickEight-v1'),
                        default='TargetPickCube-v1')
    parser.add_argument('--input-mode', choices=('dual_rgb', 'masked_only'), default='dual_rgb')
    parser.add_argument('--mask-source', choices=('oracle', 'perception'), default='oracle')
    parser.add_argument('--perception-url', default='http://127.0.0.1:18830')
    parser.add_argument('--target-description', default='')
    parser.add_argument('--loss-mode', choices=('observe', 'pause', 'reacquire'), default='observe')
    parser.add_argument('--max-pause-frames', type=int, default=60)
    parser.add_argument('--mask-intervention', choices=('none', 'dropout', 'freeze', 'shift', 'occlusion'), default='none')
    parser.add_argument('--intervention-start', type=int, default=50)
    parser.add_argument('--intervention-duration', type=int, default=30)
    parser.add_argument('--intervention-dx', type=int, default=20)
    parser.add_argument('--intervention-dy', type=int, default=0)
    cli = parser.parse_args()
    if cli.input_mode == 'masked_only' and cli.env_id != 'TargetPickEight-v1':
        parser.error('masked_only mode currently supports TargetPickEight-v1')
    defaults = dict(mask_source=cli.mask_source, endpoint=cli.perception_url,
                    target_text=cli.target_description, loss_mode=cli.loss_mode,
                    max_pause_frames=cli.max_pause_frames,
                    intervention=dict(kind=cli.mask_intervention, start=cli.intervention_start,
                        duration=cli.intervention_duration, dx=cli.intervention_dx, dy=cli.intervention_dy))
    lab = Lab(cli.output_dir, cli.dataset_json, cli.env_id, cli.input_mode, defaults)
    server = ThreadingHTTPServer(('127.0.0.1', cli.port), Handler)
    server.lab = lab
    print(f'Target ACT Lab: http://127.0.0.1:{cli.port} | outputs: {lab.output}', flush=True)
    server.serve_forever()
