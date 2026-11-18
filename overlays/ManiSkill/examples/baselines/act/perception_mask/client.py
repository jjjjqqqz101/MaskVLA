"""Lightweight client usable inside the existing Python 3.11 policy environment."""
from dataclasses import dataclass
import base64
import io
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


@dataclass
class MaskResult:
    mask: np.ndarray
    metadata: dict


def encode_rgb(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Perception expects uint8 H×W×3 RGB')
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def decode_png(encoded, mask=False):
    data = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(data)) as image:
        if max(image.size) > 2048 or min(image.size) < 1:
            raise ValueError('Image dimensions outside supported range')
        return np.array(image.convert('L' if mask else 'RGB')) > 0 if mask else np.array(image.convert('RGB'))


class PerceptionMaskClient:
    def __init__(self, endpoint='http://127.0.0.1:18830', timeout=180):
        p = urlparse(endpoint)
        if p.scheme != 'http' or p.hostname not in ('127.0.0.1', 'localhost', '::1') or p.username or p.password or p.query or p.fragment:
            raise ValueError('Perception endpoint must be a local HTTP service')
        self.endpoint = endpoint.rstrip('/')
        self.timeout = timeout
        self.session_id = None

    def _request(self, path, data=None, timeout=None):
        payload = None if data is None else json.dumps(data, allow_nan=False).encode()
        request = Request(self.endpoint + path, data=payload,
                          headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=self.timeout if timeout is None else timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read(4096)).get('error', 'request failed')
            except (ValueError, TypeError):
                detail = 'request failed'
            raise RuntimeError(f'Perception HTTP {exc.code}: {detail}') from None
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f'Perception service unavailable: {exc}') from None
        if not result.get('ok'):
            raise RuntimeError(result.get('error', 'Perception request failed'))
        return result

    def health(self):
        return self._request('/health')

    @staticmethod
    def _result(data, shape):
        mask = decode_png(data['mask_png'], mask=True)
        if mask.shape != shape:
            raise ValueError('Perception returned a mask with different image dimensions')
        return MaskResult(mask, data['metadata'])

    def start(self, rgb, target_text, options=None):
        self.close()
        data = self._request('/v1/start', dict(image_png=encode_rgb(rgb), target_text=target_text,
                                               options=options or {}))
        self.session_id = data['session_id']
        return self._result(data, np.asarray(rgb).shape[:2])

    def step(self, rgb, frame_index, force_reacquire=False):
        if self.session_id is None:
            raise RuntimeError('Start perception before tracking')
        data = self._request('/v1/step', dict(session_id=self.session_id, image_png=encode_rgb(rgb),
                         frame_index=int(frame_index), force_reacquire=bool(force_reacquire)))
        return self._result(data, np.asarray(rgb).shape[:2])

    def start_from_vlm(self,rgb,target_text,bbox_xyxy,options=None,request_id=None):
        """Use the task planner's actual VLM box instead of asking a second VLM."""
        self.close()
        data=self._request('/v1/start_from_vlm',dict(image_png=encode_rgb(rgb),target_text=target_text,
            bbox_xyxy=list(bbox_xyxy),options=options or {},request_id=request_id))
        self.session_id=data['session_id']
        return self._result(data,np.asarray(rgb).shape[:2])

    def close(self):
        if self.session_id is not None:
            self._request('/v1/close', dict(session_id=self.session_id),timeout=min(self.timeout,3))
            self.session_id = None
