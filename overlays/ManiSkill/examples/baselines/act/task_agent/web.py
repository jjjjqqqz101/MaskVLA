"""Loopback console for natural-language, visually supervised collection tasks."""
from __future__ import annotations

import argparse
import http.client
import socket
from datetime import datetime
import json
import mimetypes
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse
from zoneinfo import ZoneInfo

from deployment_paths import task_runs_root, project_path
from .mask_dropout import normalize_mask_dropout
from .policies import POLICY_IDS, policy_options, policy_record
from .scene_reference import validate_scene_reference
from .scene_objects import (CLASSIC_OBJECT_IDS, catalog_public, normalize_object_selection,
                            normalize_scene_spec, resolve_scene_spec)


TERMINAL = {
    'completed', 'failed', 'stopped', 'error', 'aborted', 'needs_input',
    'retry_budget_exhausted', 'decision_budget_exhausted', 'subtask_budget_exhausted',
    'timeout', 'planner_aborted', 'visual_verification_disagreement',
    'final_verification_error', 'cleanup_error', 'partial', 'max_subtasks',
}
DEFAULTS = dict(instruction='把桌面上的球类收拾到绿色收纳区', policy='pi05', seed=17,
                max_retries=2, max_subtasks=4, monitor_hz=1.0,
                mask_dropout=normalize_mask_dropout())
PUBLIC_FILES = {
    'summary.json', 'status.json', 'plan.json', 'events.jsonl', 'vlm_calls.jsonl',
    'judgements.jsonl', 'monitor_results.jsonl', 'supervision.jsonl',
    'actions.jsonl', 'step_results.json', 'report.html', 'request.json',
    'mask_dropout.jsonl',
    'scene.json', 'scene_reuse.json',
    'scene_reference.json', 'scene_comparison.json',
    'scene_spec.json',
}


def read_json(path, default=None):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else default
    except (OSError, ValueError):
        return default


def read_lines(path, limit=100):
    """Bound repeated status reads even during a long task."""
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            offset = max(0, size - 256_000)
            stream.seek(offset)
            data = stream.read().decode('utf-8', errors='replace')
        lines = data.splitlines()
        if offset:
            lines = lines[1:]
    except OSError:
        return []
    result = []
    for line in lines[-limit:]:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
        except ValueError:
            pass
    return result


def public_value(value):
    """Do not expose accidental credential fields in status/model responses."""
    if isinstance(value, dict):
        return {str(k): ('[REDACTED]' if any(term in str(k).lower()
                for term in ('api_key', 'authorization', 'access_token', 'secret'))
                else public_value(v)) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [public_value(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'sk-[A-Za-z0-9_.-]+', '[REDACTED]', value)
    return value


def object_fields(spec):
    names = {item['id']:item['name'] for item in catalog_public()}
    ids = [item['id'] for item in spec['objects']]
    return dict(scene_spec=spec, selected_object_ids=ids, object_names=[names[item] for item in ids])


class Console:
    def __init__(self, runs_root, act_root=None, python_bin=None, process_factory=None):
        self.runs_root = project_path(runs_root, 'task runs root')
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.act_root = Path(act_root or Path(__file__).resolve().parents[1]).resolve()
        self.python_bin = str(python_bin or sys.executable)
        self.process_factory = process_factory or subprocess.Popen
        self.lock = threading.RLock()
        self.process = None
        self.active_run_id = None
        self.stop_requested = False
        self.token = secrets.token_urlsafe(32)

    def directory(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r'task_agent_[A-Za-z0-9_-]{1,100}', run_id):
            raise ValueError('无效的运行记录。')
        candidate = self.runs_root / run_id
        if candidate.is_symlink():
            raise ValueError('运行目录不可使用符号链接。')
        path = candidate.resolve()
        if path.parent != self.runs_root or not path.is_dir():
            raise ValueError('找不到这次任务。')
        return path

    def busy(self):
        return self.process is not None and self.process.poll() is None

    def snapshot(self, run_id):
        directory = self.directory(run_id)
        request = read_json(directory / 'web_request.json', {})
        progress = read_json(directory / 'status.json', {})
        summary = read_json(directory / 'summary.json', {})
        scene_manifest = read_json(directory/'scene.json', {})
        try:
            display_spec = normalize_scene_spec(scene_manifest.get('scene_spec', request.get('scene_spec')))
        except (ValueError, TypeError):
            display_spec = normalize_scene_spec()
        active = run_id == self.active_run_id and self.busy()
        data = {**request, **summary, **progress} if active else {**request, **progress, **summary}
        status = data.get('status', 'initializing' if active else 'error')
        if (active and request.get('mode') == 'scene_first' and (directory/'task_request.json').is_file()
                and status in ('preparing_scene','scene_ready')):
            status = 'starting_task'
            data.update(phase='starting_task', instruction=request.get('instruction',''))
        if active and self.stop_requested:
            status = 'stopping'
        elif not active and status not in TERMINAL:
            status = 'error'
            data['error'] = data.get('error') or '执行进程已退出，但没有保存完整结束状态。'
        if not active and status == 'completed' and data.get('success') is not True:
            status = 'failed'
            data['reason'] = data.get('reason') or '没有保存 VLM 确认任务完成的结果。'
        plan = data.get('plan', read_json(directory / 'plan.json', {}))
        steps = plan.get('steps', []) if isinstance(plan, dict) else plan if isinstance(plan, list) else []
        completed = data.get('completed_steps', 0)
        completed_count = len(completed) if isinstance(completed, list) else completed if type(completed) is int else 0
        judgements = data.get('judgements') if isinstance(data.get('judgements'), list) else []
        if not judgements:
            for name in ('judgements.jsonl', 'monitor_results.jsonl', 'supervision.jsonl'):
                judgements = read_lines(directory / name, 60)
                if judgements:
                    break
        artifacts = {}
        for key, name in [('video', 'execution.mp4'), ('summary', 'summary.json'),
                          ('plan', 'plan.json'), ('events', 'events.jsonl'),
                          ('mask_dropout', 'mask_dropout.jsonl'),
                          ('judgements', 'judgements.jsonl'), ('initial', 'initial.png'),
                          ('final', 'final.png'), ('report', 'report.html')]:
            p = directory / name
            if (p.is_file() and not p.is_symlink()
                    and not (key == 'video' and (active or p.stat().st_size == 0))):
                artifacts[key] = f'/artifacts/{run_id}/{name}'
                if key == 'video':
                    artifacts['video_download'] = artifacts[key]+'?download=1'
        created = request.get('created_at') or datetime.fromtimestamp(
            directory.stat().st_mtime, ZoneInfo('Asia/Shanghai')).isoformat()
        out = dict(id=run_id, status=status, busy=active, finished=not active,
                   success=data.get('success') is True, instruction=data.get('instruction', request.get('instruction', '')),
                   seed=data.get('seed', request.get('seed')),
                   created_at=created, phase=data.get('phase'), motion_phase=data.get('motion_phase'), plan=plan, steps=steps,
                   scene_id=request.get('scene_id'), scene_first=request.get('mode') == 'scene_first',
                   task_submitted=(directory/'task_request.json').is_file(),
                   initial_scene_available=self.initial_scene_available(run_id),
                   generation_mode=request.get('generation_mode', 'new'),
                   comparison_source_scene_id=request.get('comparison_source_scene_id'),
                   comparison_initial_layout_verified=self.comparison_layout_verified(directory, request, scene_manifest),
                   current_step=data.get('current_step', completed_count), completed_steps=completed,
                   completed_count=completed_count, total_steps=data.get('total_steps', len(steps)),
                   attempts=data.get('attempts', {}), step_results=data.get('step_results', []),
                   latest_judgement=data.get('latest_judgement'), judgements=judgements[-60:],
                   latest_goal_check=data.get('latest_goal_check'),
                   monitor_stats=data.get('monitor_stats', {}), events=read_lines(directory/'events.jsonl', 100),
                   vlm_calls=read_lines(directory/'vlm_calls.jsonl', 100),
                   error=data.get('error') or data.get('finalize_error'), reason=data.get('reason'),
                   target=data.get('target') or data.get('current_target'),
                   target_description=data.get('target_description'), active_bbox=data.get('active_bbox'),
                   plan_summary=data.get('plan_summary') or (plan.get('summary') if isinstance(plan, dict) else None),
                   tracking=data.get('tracking') or data.get('mask_metadata'),
                   mask_dropout=normalize_mask_dropout(data.get('mask_dropout')),
                   mask_dropout_state=data.get('mask_dropout_state', {}),
                   step_generation=data.get('step_generation', 0),
                   frame_id=data.get('frame_id'), elapsed_seconds=data.get('elapsed_seconds'),
                   stop_requested=active and self.stop_requested,
                   has_frame=any((directory/f).is_file() for f in ['live.png', 'final.png', 'initial.png']),
                   artifacts=artifacts, **object_fields(display_spec),
                   **policy_record(data.get('policy', request.get('policy', 'pi05')), data.get('model')))
        return public_value(out)

    def history(self):
        rows = []
        for path in self.runs_root.glob('task_agent_*'):
            if (not path.is_dir() or path.is_symlink() or
                    not any((path/name).is_file() for name in ('web_request.json', 'summary.json'))):
                continue
            try:
                value = self.snapshot(path.name)
                rows.append({key: value[key] for key in ('id','status','success','instruction','policy','policy_label','model',
                            'seed','created_at','completed_count','total_steps','finished',
                            'mask_dropout','mask_dropout_state','step_generation',
                            'scene_id','scene_first','task_submitted','initial_scene_available',
                            'generation_mode','comparison_source_scene_id',
                            'comparison_initial_layout_verified','scene_spec',
                            'selected_object_ids','object_names')})
            except ValueError:
                continue
        return sorted(rows, key=lambda x: x['created_at'], reverse=True)[:60]

    def status(self, run_id=None):
        with self.lock:
            selected = run_id or self.active_run_id
            if not selected:
                history = self.history()
                selected = history[0]['id'] if history else None
            return dict(ok=True, busy=self.busy(), active_run_id=self.active_run_id,
                        run=self.snapshot(selected) if selected else None,
                        scene=self.scene_state())

    def scene_state(self):
        if self.active_run_id is None:
            return None
        directory = self.directory(self.active_run_id)
        request = read_json(directory/'web_request.json', {})
        if request.get('mode') != 'scene_first':
            return None
        progress = read_json(directory/'status.json', request)
        summary = read_json(directory/'summary.json', {})
        manifest = read_json(directory/'scene.json', {})
        active = self.busy()
        data = progress if active else {**progress, **summary}
        submitted = (directory/'task_request.json').is_file()
        status = data.get('status', 'preparing_scene')
        if active and self.stop_requested:
            status = 'stopping'
        elif active and submitted and status in ('preparing_scene','scene_ready'):
            status = 'starting_task'
        elif not active and status not in TERMINAL:
            status = 'error'
        handshake = (manifest.get('scene_id') == self.active_run_id
                     and type(manifest.get('seed')) is int
                     and manifest['seed'] == request.get('seed'))
        comparison_verified = self.comparison_layout_verified(directory, request, manifest)
        try:
            scene_spec = self.scene_spec_for(directory, request, manifest)
        except (OSError, ValueError, TypeError):
            scene_spec = normalize_scene_spec(request.get('scene_spec'))
            handshake = False
        if request.get('generation_mode') == 'current':
            handshake = handshake and comparison_verified
        ready = bool(active and status == 'scene_ready' and progress.get('phase') == 'scene_ready'
                     and handshake and not submitted and not self.stop_requested
                     and not (directory/'stop.request').exists())
        return dict(id=self.active_run_id, seed=request.get('seed'), status=status,
                    ready=ready, submitted=submitted, busy=active,
                    generation_mode=request.get('generation_mode', 'new'),
                    comparison_source_scene_id=request.get('comparison_source_scene_id'),
                    comparison_initial_layout_verified=comparison_verified,
                    has_frame=any((directory/name).is_file() for name in ('live.png','initial.png')),
                    **object_fields(scene_spec))

    def scene_spec_for(self, directory, request=None, manifest=None):
        request = read_json(directory/'web_request.json', {}) if request is None else request
        manifest = read_json(directory/'scene.json', {}) if manifest is None else manifest
        requested = normalize_scene_spec(request.get('scene_spec'))
        actual = normalize_scene_spec(manifest.get('scene_spec'))
        if actual != requested:
            raise ValueError('场景实际物品组合与生成设置不一致。')
        path = directory/'scene_spec.json'
        if path.is_symlink():
            raise ValueError('物品配置不可使用符号链接。')
        if path.exists():
            if not path.is_file() or path.stat().st_size > 16000 or normalize_scene_spec(read_json(path, {})) != actual:
                raise ValueError('初始物品配置无效或不一致。')
        elif requested['preset'] != 'classic8':
            raise ValueError('未保存可复现的初始物品组合。')
        ids = [item['id'] for item in actual['objects']]
        if 'selected_object_ids' in manifest and manifest['selected_object_ids'] != ids:
            raise ValueError('场景物品清单与实际配置不一致。')
        return actual

    def scene_reference(self, source_scene_id):
        directory = self.directory(source_scene_id)
        paths = [directory/'web_request.json', directory/'scene.json']
        if any(not path.is_file() or path.is_symlink() or path.stat().st_size > 32000 for path in paths):
            raise ValueError('该记录没有可校验的初始场景，请先生成一个场景。')
        request, manifest = (read_json(path, {}) for path in paths)
        if (request.get('mode') != 'scene_first' or request.get('scene_id') != source_scene_id
                or manifest.get('scene_id') != source_scene_id
                or type(manifest.get('reset_count')) is not int or manifest['reset_count'] != 1
                or type(request.get('seed')) is not int or manifest.get('seed') != request['seed']):
            raise ValueError('初始场景信息不完整或不一致，无法用于对比。')
        if (request.get('generation_mode') == 'current'
                and not self.comparison_layout_verified(directory, request, manifest)):
            raise ValueError('该对比场景尚未通过初始布局校验，不能用作新的参考。')
        self.scene_spec_for(directory, request, manifest)
        return validate_scene_reference(dict(source_scene_id=source_scene_id, seed=manifest.get('seed'),
                                             rgb_sha256=manifest.get('rgb_sha256')))

    def comparison_layout_verified(self, directory, request, manifest):
        if (request.get('generation_mode') != 'current'
                or request.get('scene_id') != directory.name
                or type(request.get('seed')) is not int
                or manifest.get('comparison_initial_layout_verified') is not True
                or manifest.get('scene_id') != directory.name):
            return False
        reference_path = directory/'scene_reference.json'
        if not reference_path.is_file() or reference_path.is_symlink() or reference_path.stat().st_size > 32000:
            return False
        try:
            expected = validate_scene_reference(read_json(reference_path, {}), expected_seed=request.get('seed'))
            actual = validate_scene_reference(dict(source_scene_id=manifest.get('comparison_source_scene_id'),
                seed=manifest.get('seed'), rgb_sha256=manifest.get('rgb_sha256')), expected_seed=request.get('seed'))
            if expected != actual or actual['source_scene_id'] != request.get('comparison_source_scene_id'):
                return False
            source = self.directory(actual['source_scene_id'])
            return self.scene_spec_for(directory, request, manifest) == self.scene_spec_for(source)
        except (OSError, ValueError, TypeError):
            return False

    def initial_scene_available(self, run_id):
        try:
            self.scene_reference(run_id)
            return True
        except (OSError, ValueError, TypeError, KeyError):
            return False

    @staticmethod
    def _atomic_json(path, value):
        temporary = path.with_name(path.name+'.'+secrets.token_hex(8)+'.tmp')
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def prepare_scene(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('请选择随机、固定 seed 或使用当前场景。')
        reuse = payload.get('mode') == 'current'
        if reuse:
            if set(payload) != {'mode','source_scene_id'}:
                raise ValueError('使用当前场景时仅接受来源场景，不接受 seed 覆盖。')
        elif set(payload)-{'seed','objects'}:
            raise ValueError('生成场景仅接受随机、固定 seed 或使用当前场景。')
        with self.lock:
            if self.busy():
                raise RuntimeError('请先关闭待执行场景，或等待当前任务结束。')
            reference = self.scene_reference(payload['source_scene_id']) if reuse else None
            if reuse:
                source_directory = self.directory(reference['source_scene_id'])
                scene_spec = self.scene_spec_for(source_directory)
                selection = dict(mode='reference', source_scene_id=reference['source_scene_id'])
            else:
                selection = normalize_object_selection(payload.get('objects'))
                previous_spec = None
                previous_id = self.active_run_id
                if previous_id is None:
                    previous = self.history()
                    previous_id = previous[0]['id'] if previous else None
                if previous_id is not None:
                    try: previous_spec = self.scene_spec_for(self.directory(previous_id))
                    except (ValueError, OSError, TypeError): pass
                scene_spec = resolve_scene_spec(selection, previous_spec=previous_spec)
            seed = reference['seed'] if reuse else payload.get('seed')
            generation_mode = 'current' if reuse else 'random' if seed is None else 'fixed'
            if seed is None:
                seed = secrets.randbelow(2**31)
            if type(seed) is not int or not 0 <= seed < 2**31:
                raise ValueError('场景 seed 必须为 0–2147483647 的整数。')
            scene_id = 'task_agent_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3)
            directory = self.runs_root/scene_id
            directory.mkdir(exist_ok=False)
            request = dict(mode='scene_first', scene_id=scene_id, seed=seed, instruction='',
                generation_mode=generation_mode,
                scene_spec=scene_spec, object_selection=selection,
                status='preparing_scene', phase='preparing_scene', completed_steps=0, attempts={},
                created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                mask_dropout=normalize_mask_dropout())
            if reference is not None:
                request['comparison_source_scene_id'] = reference['source_scene_id']
                self._atomic_json(directory/'scene_reference.json', reference)
            self._atomic_json(directory/'scene_spec.json', scene_spec)
            self._atomic_json(directory/'web_request.json', request)
            command = [self.python_bin, '-B', '-m', 'task_agent.scene_worker',
                '--scene-id', scene_id, '--seed', str(seed), '--output', str(directory),
                '--stop-file', str(directory/'stop.request'), '--task-file', str(directory/'task_request.json')]
            with (directory/'stdout.log').open('w') as stream:
                try:
                    self.process = self.process_factory(command, cwd=self.act_root, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as error:
                    self._atomic_json(directory/'summary.json', dict(status='error', success=False,
                        error=f'无法生成场景：{error}', instruction='', seed=seed))
                    raise
            self.active_run_id = scene_id
            self.stop_requested = False
            return dict(ok=True, scene_id=scene_id, run_id=scene_id,
                        run=self.snapshot(scene_id), scene=self.scene_state())

    def submit_scene_task(self, payload):
        allowed = (set(DEFAULTS)-{'seed'}) | {'scene_id'}
        if not isinstance(payload, dict) or set(payload)-allowed:
            raise ValueError('任务设置包含不支持的字段；seed 沿用已生成的场景。')
        if 'instruction' not in payload:
            raise ValueError('查看场景后，请输入任务指令。')
        with self.lock:
            scene = self.scene_state()
            if not scene or payload.get('scene_id') != scene['id']:
                raise RuntimeError('场景已变化，请返回当前场景后再提交任务。')
            if not scene['ready']:
                raise RuntimeError('场景尚未就绪、已关闭或任务已经提交。')
            cfg = self._task_settings({**{k:v for k,v in payload.items() if k != 'scene_id'}, 'seed':scene['seed']})
            directory = self.directory(scene['id'])
            command_path = directory/'task_request.json'
            if command_path.exists():
                raise RuntimeError('这个场景已经提交过任务，请等待执行结束。')
            request = read_json(directory/'web_request.json', {})
            request.update(cfg, **policy_record(cfg['policy']),
                           task_submitted_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
            self._atomic_json(directory/'web_request.json', request)
            self._atomic_json(command_path, dict(scene_id=scene['id'], **cfg))
            return dict(ok=True, queued=True, scene_id=scene['id'], run_id=scene['id'],
                        scene=self.scene_state(), run=self.snapshot(scene['id']))

    def close_scene(self, payload):
        if not isinstance(payload, dict) or set(payload) != {'scene_id'}:
            raise ValueError('请指定要关闭的待执行场景。')
        with self.lock:
            scene = self.scene_state()
            if (not scene or payload['scene_id'] != scene['id'] or not scene['busy']
                    or scene['submitted']):
                raise RuntimeError('只能关闭当前尚未提交任务的场景。')
            directory = self.directory(scene['id'])
            progress = read_json(directory/'status.json', read_json(directory/'web_request.json', {}))
            if progress.get('status') not in ('preparing_scene','scene_ready'):
                raise RuntimeError('场景已进入执行阶段，请使用停止任务。')
            (directory/'stop.request').touch()
            self.stop_requested = True
            return dict(ok=True, requested=True, scene_id=scene['id'])

    def _task_settings(self, payload):
        if not isinstance(payload, dict) or set(payload) - set(DEFAULTS):
            raise ValueError('请求包含不支持的任务设置；物体由 VLM 选择。')
        cfg = dict(DEFAULTS, **payload)
        if not isinstance(cfg['instruction'], str) or not 1 <= len(cfg['instruction'].strip()) <= 2000:
            raise ValueError('请输入 1–2000 字的任务指令。')
        cfg['instruction'] = cfg['instruction'].strip()
        if not isinstance(cfg['policy'], str) or cfg['policy'] not in POLICY_IDS:
            raise ValueError('请选择 π0.5、双路 ACT 或三路 ACT 技能。')
        for key, low, high in [('seed', 0, 2**31-1), ('max_retries', 0, 3), ('max_subtasks', 1, 4)]:
            if type(cfg[key]) is not int or not low <= cfg[key] <= high:
                raise ValueError(f'{key} 超出有效整数范围 {low}–{high}。')
        if type(cfg['monitor_hz']) not in (int, float) or cfg['monitor_hz'] != 1:
            raise ValueError('当前任务以 1 Hz 请求场景判断。')
        cfg['mask_dropout'] = normalize_mask_dropout(cfg['mask_dropout'])
        return cfg

    def start(self, payload):
        # Preserve the original direct API; the webpage uses /api/scene/task.
        cfg = self._task_settings(payload)
        with self.lock:
            if self.busy():
                raise RuntimeError('当前任务仍在运行，请先等待完成或停止。')
            run_id = 'task_agent_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3)
            directory = self.runs_root / run_id
            directory.mkdir(exist_ok=False)
            request = dict(cfg, status='queued', created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                           completed_steps=0, attempts={}, phase='queued')
            request.update(policy_record(cfg['policy']))
            (directory/'web_request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2))
            command = [self.python_bin, '-B', '-m', 'task_agent.runner',
                       '--instruction', cfg['instruction'], '--policy', cfg['policy'],
                       '--seed', str(cfg['seed']), '--output', str(directory),
                       '--max-retries', str(cfg['max_retries']), '--max-subtasks', str(cfg['max_subtasks']),
                       '--monitor-hz', str(cfg['monitor_hz']), '--stop-file', str(directory/'stop.request'),
                       '--mask-dropout-config', json.dumps(cfg['mask_dropout'], allow_nan=False),
                       '--mask-dropout-control', str(directory/'mask_dropout_control.json')]
            with (directory/'stdout.log').open('w') as stream:
                try:
                    self.process = self.process_factory(command, cwd=self.act_root, stdout=stream,
                                                        stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as error:
                    (directory/'summary.json').write_text(json.dumps(dict(status='error', success=False,
                         error=f'无法启动执行进程：{error}', instruction=cfg['instruction']), ensure_ascii=False))
                    raise
            self.active_run_id = run_id
            self.stop_requested = False
            return dict(ok=True, run_id=run_id, run=self.snapshot(run_id))

    def stop(self):
        with self.lock:
            if not self.busy() or self.active_run_id is None:
                raise ValueError('没有正在执行的任务。')
            (self.directory(self.active_run_id)/'stop.request').touch()
            self.stop_requested = True
            return dict(ok=True, requested=True, message='已请求停止，将在当前调用或动作步结束后保存记录。')

    def mask_dropout(self, payload):
        if not isinstance(payload, dict) or set(payload)-{'run_id','action','duration_seconds'}:
            raise ValueError('人工干扰设置包含不支持的字段。')
        action = payload.get('action')
        if action not in ('trigger','restore'):
            raise ValueError('请选择触发或停止人工干扰。')
        duration = payload.get('duration_seconds')
        if action == 'trigger':
            if (type(duration) not in (int,float) or not math.isfinite(duration)
                    or not .1 <= duration <= 30):
                raise ValueError('干扰持续时间必须为 0.1–30 秒。')
        elif 'duration_seconds' in payload:
            raise ValueError('停止人工干扰不需要持续时间。')
        with self.lock:
            if not self.busy() or self.active_run_id is None:
                raise RuntimeError('没有正在执行的任务。')
            if payload.get('run_id') != self.active_run_id:
                raise RuntimeError('只能控制当前正在执行的任务，请返回当前任务。')
            run = self.snapshot(self.active_run_id)
            if run.get('scene_first') and not run.get('task_submitted'):
                raise RuntimeError('请先提交任务，人工干扰只用于任务执行。')
            generation = run.get('step_generation', 0)
            if type(generation) is not int or generation < 0:
                raise RuntimeError('执行器尚未发布有效的子任务状态。')
            if action == 'trigger' and (run.get('phase') != 'grasp' or generation <= 0
                    or run.get('mask_dropout_state', {}).get('ready') is False):
                raise RuntimeError('手动触发仅在模型实际执行抓取时可用。')
            command = dict(command_id=secrets.token_hex(16), action=action,
                           step_generation=generation, issued_at=time.time())
            if action == 'trigger':
                command['duration_seconds'] = float(duration)
            directory = self.directory(self.active_run_id)
            path = directory/'mask_dropout_control.json'
            tmp = directory/('mask_dropout_control.'+command['command_id']+'.tmp')
            try:
                tmp.write_text(json.dumps(command, ensure_ascii=False, allow_nan=False))
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
            return dict(ok=True, queued=True, command_id=command['command_id'],
                        message='已提交，等待执行器确认。')


ROBOCASA_PROXY_HOST = '127.0.0.1'
ROBOCASA_PROXY_PORT = 18850
ROBOCASA_GET_ROUTES = {'health', 'config', 'status', 'frame.png', 'episodes', 'catalog'}
ROBOCASA_POST_ROUTES = {'scene', 'record/start', 'record/stop', 'control',
                       'scripted_demo', 'stop', 'replay', 'target'}


def robocasa_proxy_path(path, method):
    """Allow only known relative routes to the fixed loopback simulator."""
    parsed = urlparse(path)
    prefix = '/api/robocasa/'
    if parsed.scheme or parsed.netloc or not parsed.path.startswith(prefix):
        raise ValueError('无效的 RoboCasa 接口。')
    suffix = parsed.path[len(prefix):]
    if '%' in suffix or '..' in suffix or '\\' in suffix:
        raise ValueError('无效的 RoboCasa 地址。')
    if method == 'GET':
        artifact = re.fullmatch(
            r'artifacts/[A-Za-z0-9_-]{1,100}/[A-Za-z0-9][A-Za-z0-9_.-]{0,120}\.(?:png|jpg|jpeg|mp4|h5|hdf5|json|jsonl|npz|zip)',
            suffix)
        if suffix not in ROBOCASA_GET_ROUTES and artifact is None:
            raise ValueError('未开放的 RoboCasa 接口。')
    elif method == 'POST':
        if suffix not in ROBOCASA_POST_ROUTES or parsed.query:
            raise ValueError('未开放的 RoboCasa 操作。')
    else:
        raise ValueError('未开放的 RoboCasa 方法。')
    if len(parsed.query) > 2000 or any(ord(c) < 32 for c in path):
        raise ValueError('无效的 RoboCasa 请求。')
    return '/' + suffix + ('?' + parsed.query if parsed.query else '')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if self.path.startswith(('/api/status','/api/frame','/api/runs','/api/robocasa/status','/api/robocasa/frame.png')):
            return
        super().log_message(fmt, *args)

    def valid_host(self):
        return urlparse('http://' + self.headers.get('Host', '')).hostname in ('127.0.0.1','localhost','::1')

    def reply(self, value, code=200):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def file(self, path, download_name=None):
        try:
            with path.open('rb') as stream:
                # The backend replaces live.png atomically; stat the same open
                # inode that is streamed so a new frame cannot change its size.
                size = os.fstat(stream.fileno()).st_size
                start, end, code = 0, size-1, 200
                header = self.headers.get('Range')
                if header:
                    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
                    if not match or not any(match.groups()):
                        return self.reply({'error':'Invalid range'}, 416)
                    if match[1]:
                        start = int(match[1]); end = int(match[2]) if match[2] else end
                    else:
                        start = max(0, size-int(match[2]))
                    if start >= size or start < 0 or end < start:
                        return self.reply({'error':'Range outside file'}, 416)
                    end = min(end, size-1); code = 206
                self.send_response(code)
                self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
                self.send_header('Content-Length', str(end-start+1))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('X-Content-Type-Options', 'nosniff')
                if download_name is not None:
                    self.send_header('Content-Disposition', 'attachment; filename="'+download_name+'"')
                if code == 206:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.end_headers()
                stream.seek(start)
                left = end-start+1
                while left:
                    chunk = stream.read(min(left, 65536))
                    if not chunk: break
                    self.wfile.write(chunk); left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def robocasa_proxy(self, method, path, payload=None):
        upstream_path = robocasa_proxy_path(path, method)
        connection = http.client.HTTPConnection(ROBOCASA_PROXY_HOST, ROBOCASA_PROXY_PORT,
                                                timeout=15)
        try:
            headers = {}
            data = None
            if payload is not None:
                data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
                headers['Content-Type'] = 'application/json'
            if self.headers.get('Range'):
                value = self.headers['Range']
                if not re.fullmatch(r'bytes=(?:[0-9]+-[0-9]*|-[0-9]+)', value):
                    return self.reply({'error':'Invalid range'}, 416)
                headers['Range'] = value
            connection.request(method, upstream_path, body=data, headers=headers)
            response = connection.getresponse()
            content_type = response.getheader('Content-Type', 'application/octet-stream')
            if 300 <= response.status < 400:
                return self.reply({'error':'RoboCasa 服务返回了未开放的跳转。'}, 502)
            if content_type.split(';', 1)[0].strip() == 'application/json':
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    return self.reply({'error':'RoboCasa 响应过大。'}, 502)
                try:
                    value = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    return self.reply({'error':'RoboCasa 返回了无效响应。'}, 502)
                return self.reply(public_value(value), response.status)
            allowed_types = {'image/png','image/jpeg','video/mp4','application/octet-stream',
                             'application/x-hdf5','application/zip'}
            if content_type.split(';', 1)[0].strip() not in allowed_types:
                return self.reply({'error':'RoboCasa 返回了未开放的文件类型。'}, 502)
            self.send_response(response.status)
            for key in ('Content-Type','Content-Length','Accept-Ranges','Content-Range',
                        'Content-Disposition'):
                value = response.getheader(key)
                if value is not None and len(value) < 500 and '\r' not in value and '\n' not in value:
                    self.send_header(key, value)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (TimeoutError, socket.timeout):
            return self.reply({'error':'RoboCasa 服务响应超时，请查看运行状态后重试。'}, 504)
        except OSError:
            return self.reply({'error':'RoboCasa 服务尚未就绪，请稍后刷新。'}, 503)
        finally:
            connection.close()

    def do_GET(self):
        if not self.valid_host():
            return self.reply({'error':'请通过本机 SSH 隧道访问。'}, 403)
        route = urlparse(self.path); query = parse_qs(route.query); console = self.server.console
        try:
            if route.path == '/robocasa':
                return self.file(Path(__file__).with_name('robocasa.html'))
            if route.path.startswith('/api/robocasa/'):
                return self.robocasa_proxy('GET', self.path)
            if route.path == '/':
                return self.file(Path(__file__).with_name('web.html'))
            if route.path == '/api/config':
                return self.reply(dict(ok=True, csrf_token=console.token, defaults=DEFAULTS,
                                  policies=list(POLICY_IDS), policy_options=policy_options(),
                                  scene_defaults=dict(random_seed=True, seed=17, objects=dict(mode='classic')),
                                  capabilities=dict(natural_task=True, visual_supervision=True, requested_monitor_hz=1.0,
                                                    mask_dropout=True, mask_dropout_manual=True, scene_first=True,
                                                    scene_initial_layout_reuse=True, scene_object_selection=True)))
            if route.path == '/api/object_catalog':
                objects = [dict(row, act_triple_trained=row['pi05_trained']) for row in catalog_public()]
                return self.reply(dict(ok=True, objects=objects, min_objects=1, max_objects=8,
                                       training_scope_note='训练范围表示本项目微调目标，不保证抓取成功。',
                                       classic_object_ids=list(CLASSIC_OBJECT_IDS)))
            if route.path == '/api/status':
                return self.reply(console.status(query.get('run_id',[None])[0]))
            if route.path == '/api/runs':
                return self.reply(dict(ok=True, runs=console.history()))
            if route.path == '/api/frame':
                run = query.get('run_id', [console.active_run_id])[0]
                directory = console.directory(run)
                for name in ('live.png', 'final.png', 'initial.png'):
                    path = directory/name
                    if path.is_file() and not path.is_symlink():
                        return self.file(path)
                return self.reply({'error':'正在准备第一帧。'}, 404)
            if route.path.startswith('/artifacts/'):
                parts = [unquote(x) for x in route.path.split('/')]
                if len(parts) != 4:
                    raise ValueError('无效的记录地址。')
                directory = console.directory(parts[2]); name = parts[3]
                if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
                    raise ValueError('无效的文件名。')
                path = directory/name
                allowed = name in PUBLIC_FILES or path.suffix.lower() in ('.png','.jpg','.jpeg','.mp4')
                if not allowed or path.is_symlink() or path.resolve().parent != directory or not path.is_file():
                    raise ValueError('文件不存在或未开放。')
                download_name = parts[2]+'.mp4' if name == 'execution.mp4' and query.get('download') == ['1'] else None
                return self.file(path, download_name=download_name)
            return self.reply({'error':'未找到页面。'}, 404)
        except (ValueError, KeyError) as error:
            return self.reply({'error':str(error)}, 400)

    def do_POST(self):
        if not self.valid_host():
            return self.reply({'error':'请通过本机 SSH 隧道访问。'}, 403)
        origin = self.headers.get('Origin')
        if origin and urlparse(origin).netloc != self.headers.get('Host'):
            return self.reply({'error':'Origin mismatch'}, 403)
        console = self.server.console
        if self.headers.get('X-CSRF-Token') != console.token:
            return self.reply({'error':'页面会话已过期，请刷新。'}, 403)
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 16000 or self.headers.get_content_type() != 'application/json':
                raise ValueError('请求格式不正确。')
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError('请求必须为对象。')
            route = urlparse(self.path).path
            if route.startswith('/api/robocasa/'):
                return self.robocasa_proxy('POST', self.path, payload)
            if route == '/api/start':
                return self.reply(console.start(payload), 202)
            if route == '/api/scene':
                return self.reply(console.prepare_scene(payload), 202)
            if route == '/api/scene/task':
                return self.reply(console.submit_scene_task(payload), 202)
            if route == '/api/scene/close':
                return self.reply(console.close_scene(payload), 202)
            if route == '/api/stop':
                return self.reply(console.stop())
            if route == '/api/mask_dropout':
                return self.reply(console.mask_dropout(payload), 202)
            return self.reply({'error':'未找到接口。'}, 404)
        except (ValueError, KeyError) as error:
            return self.reply({'error':str(error)}, 400)
        except RuntimeError as error:
            return self.reply({'error':str(error)}, 409)
        except OSError:
            return self.reply({'error':'无法启动执行进程，请查看服务器记录。'}, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18840)
    parser.add_argument('--runs-root', type=Path,
                        default=task_runs_root())
    parser.add_argument('--python', dest='python_bin', default=sys.executable)
    args = parser.parse_args()
    console = Console(args.runs_root, python_bin=args.python_bin)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.console = console
    print(f'Task agent console: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    finally:
        if console.busy():
            console.stop()
        server.server_close()


if __name__ == '__main__':
    main()
