"""Local RGB-only perception sidecar. Start with python -m perception_mask.server."""
import argparse
import base64
import io
import json
from pathlib import Path
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from deployment_paths import (sam2_checkpoint, cutie_checkpoint, perception_runs_root,
                              qwen_config, project_path)
from .adapters import CutieFactory, QwenBoxLocator, SAM2Segmenter
from .client import decode_png
from .core import SOURCE, TrackingSession


class Application:
    def __init__(self, locator, segmenter, tracker_factory, output=None, max_sessions=2):
        self.locator, self.segmenter, self.tracker_factory = locator,segmenter,tracker_factory
        self.output = project_path(output, 'perception output') if output else None
        self.max_sessions = max_sessions
        self.sessions = {}
        self.lock = threading.RLock()

    def ready(self):
        return dict(source=SOURCE,rgb_only=True,oracle_fallback=False,
                    supports_upstream_vlm_bbox=True,
                    supports_single_target_guard=True,
                    sam2_checkpoint_available=self.segmenter.checkpoint.is_file(),
                    cutie_checkpoint_available=self.tracker_factory.checkpoint.is_file(),
                    active_sessions=len(self.sessions))

    def _encode_result(self, sid, result):
        buffer = io.BytesIO()
        Image.fromarray(result.mask.astype(np.uint8)*255).save(buffer,format='PNG')
        response = dict(ok=True,session_id=sid,mask_png=base64.b64encode(buffer.getvalue()).decode(),
                        metadata=result.metadata)
        if self.output:
            folder = self.output/sid
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/'tracking.jsonl').open('a') as stream:
                stream.write(json.dumps(result.metadata,ensure_ascii=False,allow_nan=False)+'\n')
        return response

    def request(self, path, data):
        # Reject privileged inputs instead of accidentally allowing oracle masks
        # into a future implementation. Each endpoint has an explicit schema.
        allowed = {'/v1/start':{'image_png','target_text','options'},
                   '/v1/start_from_vlm':{'image_png','target_text','options','bbox_xyxy','request_id'},
                   '/v1/step':{'session_id','image_png','frame_index','force_reacquire'},
                   '/v1/close':{'session_id'}}
        if path not in allowed or not isinstance(data,dict) or set(data)-allowed[path]:
            raise ValueError('Unknown endpoint or privileged/unsupported request fields')
        with self.lock:
            now = time.monotonic()
            for sid,session in list(self.sessions.items()):
                if now-session.last_access > 1800:
                    session.close(); del self.sessions[sid]
            if path=='/v1/close':
                session = self.sessions.pop(data.get('session_id'),None)
                if session:session.close()
                return dict(ok=True)
            image = decode_png(data['image_png'])
            if path in ('/v1/start','/v1/start_from_vlm'):
                if len(self.sessions)>=self.max_sessions:
                    raise ValueError('Close another active experiment before starting a new one')
                sid = secrets.token_hex(16)
                initial=None
                if path=='/v1/start_from_vlm':
                    request_id=data.get('request_id')
                    if request_id is not None and (not isinstance(request_id,str) or len(request_id)>200):
                        raise ValueError('Invalid upstream VLM request ID')
                    initial=dict(found=True,bbox_xyxy=data['bbox_xyxy'],coordinate_space='pixels',
                                 origin='upstream_task_vlm',request_id=request_id,reason='Task VLM selected this target')
                session = TrackingSession(self.locator,self.segmenter,self.tracker_factory,
                                          data['target_text'],data.get('options'),initial_localization=initial)
                try:
                    result = session.process(image,0)
                except BaseException:
                    session.close(); raise
                self.sessions[sid] = session
                if self.output:
                    folder = self.output/sid; folder.mkdir(parents=True,exist_ok=True)
                    (folder/'session.json').write_text(json.dumps(dict(target_text=session.text,
                        options=session.options,source=SOURCE),ensure_ascii=False,indent=2))
            else:
                sid = data['session_id']
                if sid not in self.sessions:
                    raise ValueError('Tracking session expired; start a new trial')
                session = self.sessions[sid]
                result = session.process(image,data['frame_index'],data.get('force_reacquire',False))
            return self._encode_result(sid,result)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass

    def send_json(self,code,data):
        raw = json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(code); self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        if urlparse(self.path).path!='/health':
            return self.send_json(404,dict(ok=False,error='Not found'))
        self.send_json(200,dict(ok=True,**self.server.app.ready()))

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length',0))
            if not 0 < length <= 12_000_000:
                raise ValueError('Invalid request size')
            data = json.loads(self.rfile.read(length))
            result = self.server.app.request(urlparse(self.path).path,data)
            self.send_json(200,result)
        except (ValueError,KeyError,TypeError) as exc:
            self.send_json(400,dict(ok=False,error=str(exc)))
        except Exception as exc:
            # VLM client already removes credentials from remote error messages.
            self.send_json(503,dict(ok=False,error=f'{type(exc).__name__}: {exc}'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1',choices=['127.0.0.1','localhost','::1'])
    parser.add_argument('--port',type=int,default=18830)
    parser.add_argument('--qwen-config',type=Path,default=qwen_config())
    parser.add_argument('--sam2-checkpoint',type=Path,default=sam2_checkpoint())
    parser.add_argument('--cutie-checkpoint',type=Path,default=cutie_checkpoint())
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',type=Path,default=perception_runs_root())
    args = parser.parse_args()
    locator = QwenBoxLocator(args.qwen_config,args.output/'vlm')
    app = Application(locator,SAM2Segmenter(args.sam2_checkpoint,device=args.device),
                      CutieFactory(args.cutie_checkpoint,device=args.device),args.output)
    server = ThreadingHTTPServer((args.host,args.port),Handler)
    server.app = app
    print(f'RGB-only VLM → SAM2 → Cutie: http://{args.host}:{args.port}',flush=True)
    try:
        server.serve_forever()
    finally:
        for session in list(app.sessions.values()):session.close()
        server.server_close()


if __name__=='__main__':
    main()
