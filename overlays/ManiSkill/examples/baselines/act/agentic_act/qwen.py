"""Small OpenAI-compatible Qwen client; credentials never enter artifacts."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from deployment_paths import qwen_config, project_path
from .contracts import parse_json_object


DEFAULT_CONFIG = qwen_config()


class QwenError(RuntimeError):
    pass


class QwenClient:
    def __init__(self, config=None, output=None, timeout=90):
        config = project_path(config or qwen_config(), 'Qwen config')
        stored = json.loads(config.read_text()) if config.is_file() else {}
        self.key = os.environ.get('QWEN_API_KEY') or os.environ.get('DASHSCOPE_API_KEY') or stored.get('api_key')
        self.base_url = (os.environ.get('QWEN_BASE_URL') or stored.get('base_url') or '').rstrip('/')
        self.model = os.environ.get('QWEN_MODEL') or stored.get('model') or 'qwen3.7-plus'
        self.timeout = timeout
        self.output = project_path(output, 'Qwen output') if output else None
        self.calls = 0
        if not self.key:
            raise QwenError(f'Qwen credential missing. Set QWEN_API_KEY or private config {config}.')
        parsed = urlparse(self.base_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise QwenError('Set QWEN_BASE_URL to your HTTPS OpenAI-compatible API Host ending /compatible-mode/v1.')

    def redact(self, message):
        return re.sub(r'sk-[A-Za-z0-9_.-]+', '[REDACTED]', str(message).replace(self.key, '[REDACTED]'))

    def request(self, purpose, system, prompt, images=()):
        content = [{'type': 'text', 'text': prompt}]
        image_hashes = []
        for image in images:
            buffer = io.BytesIO()
            image.convert('RGB').save(buffer, format='PNG')
            data = buffer.getvalue()
            image_hashes.append(hashlib.sha256(data).hexdigest())
            content.append({'type': 'image_url', 'image_url': {
                'url': 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')}})
        payload = dict(model=self.model, messages=[{'role': 'system', 'content': system},
                       {'role': 'user', 'content': content}], temperature=0,
                       max_tokens=1400, enable_thinking=False,
                       response_format={'type': 'json_object'})
        started = time.monotonic()
        result = None
        for attempt in range(3):
            try:
                req = Request(self.base_url + '/chat/completions',
                              data=json.dumps(payload, ensure_ascii=False).encode(),
                              headers={'Authorization': 'Bearer ' + self.key,
                                       'Content-Type': 'application/json'})
                with urlopen(req, timeout=self.timeout) as response:
                    result = json.load(response)
                break
            except HTTPError as exc:
                detail = self.redact(exc.read().decode(errors='replace')[:1200])
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise QwenError(f'Qwen HTTP {exc.code}: {detail}') from None
            except (URLError, TimeoutError) as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise QwenError(self.redact(f'Qwen connection failed: {exc}')) from None
        try:
            text = result['choices'][0]['message']['content']
            if not isinstance(text, str):
                raise ValueError('Expected textual JSON content')
            parsed = parse_json_object(text)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise QwenError(self.redact(f'Invalid JSON model response: {exc}')) from None
        self.calls += 1
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
            record = dict(call=self.calls, purpose=purpose, model=self.model,
                          endpoint=self.base_url, image_sha256=image_hashes,
                          elapsed_seconds=round(time.monotonic()-started, 3),
                          usage=result.get('usage'), response=parsed)
            with (self.output / 'vlm_calls.jsonl').open('a') as stream:
                stream.write(self.redact(json.dumps(record, ensure_ascii=False)) + '\n')
        return parsed


class QwenPlanner:
    def __init__(self, client):
        self.client = client
        self.step_count = 2

    def plan(self, instruction, observation, image, repair=None):
        prompt = json.dumps(dict(instruction=instruction, scene=observation,
             expected_steps=self.step_count,
             available_skill={'name':'pick_place', 'description':'Existing ACT grasps/lifts; motion planner transports, releases and retreats.',
                              'object_ids':['cube1','cube2','cube3'], 'region_ids':['zone_A','zone_B','zone_C']},
             previous_validation_error=repair), ensure_ascii=False)
        return self.client.request('plan',
            'You are a robot task planner. The image shows stable cube IDs and destination regions. '
            f'Decompose the user instruction into exactly {self.step_count} ordered pick_place calls. '
            'Use only listed objects/regions. Do not invent capabilities. Output JSON only: '
            '{"steps":[{"skill":"pick_place","object_id":"cube1","region_id":"zone_A"},...],"reason":"brief"}. '
            'The example IDs are schema examples; derive the actual order and destinations from the user instruction. '
            'Write the reason in concise Chinese; keep JSON field names and object/region IDs unchanged.',
            prompt, [image])

    def next_action(self, task, plan, observation, completed, attempts, last_result, image, repair=None):
        return self.client.request('next_action',
            'You supervise an executing robot. Follow the validated ordered plan. Never skip unfinished steps. '
            'Use execute for a first attempt, retry for a failed current step, abort if recovery is impossible, '
            'complete only when all verified steps are completed. Output JSON only '
            '{"decision":"execute|retry|abort|complete","step_index":0,"reason":"brief"}. '
            'step_index is zero-based, and equals the plan length when complete. A tagged injected timeout is a recoverable test failure. '
            'The current scene contact/geometry state is authoritative for held objects and gripper state. '
            'Visual-judge explanations may be mistaken; never repeat claims that contradict the current scene. '
            'For an injected no-motion timeout, report that no physical attempt was executed, not that a grasp failed. '
            'Write the reason in concise Chinese; keep JSON keys and IDs unchanged.',
            json.dumps(dict(instruction=task.instruction,plan=plan,scene=observation,
                            completed_steps=completed,attempts=attempts,last_result=last_result,
                            previous_validation_error=repair),ensure_ascii=False), [image])

    def judge(self, step, before_image, after_image):
        result = self.client.request('visual_judge',
            'Judge visible robot placement using the BEFORE and AFTER annotated images, in that order. '
            'Cube IDs are persistent. Destination rectangles are projected table regions. '
            'Check the requested cube is in the requested region and released on the table, not held. '
            'Do not infer success from the requested action. If occluded or uncertain return uncertain. '
            'First check whether the cube is inside the region. If not, report that simple visible fact. '
            'Do not claim the robot is holding a cube based only on proximity or overlap in the image. '
            'If BEFORE and AFTER are unchanged, do not invent a grasp, lift, or release event. '
            'Output JSON only {"status":"success|failure|uncertain","reason":"brief visual evidence"}. '
            'Write the reason in concise Chinese.',
            json.dumps(step.as_dict(),ensure_ascii=False), [before_image,after_image])
        if result.get('status') not in ('success','failure','uncertain') or not isinstance(result.get('reason'),str):
            raise QwenError('Visual judge returned an invalid status/reason schema.')
        return result
