"""Loopback-only web console for the existing simulator agent."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import mimetypes
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse
from zoneinfo import ZoneInfo

from .contracts import Step, TaskSpec


REGIONS = [dict(id='zone_A', label='A 区 · 左侧', center=[0.0,-0.18]),
           dict(id='zone_B', label='B 区 · 右侧', center=[0.0,0.18]),
           dict(id='zone_C', label='C 区 · 前方', center=[0.18,0.0])]
TERMINAL = {'completed','stopped','error','plan_only','planner_aborted',
            'retry_budget_exhausted','decision_budget_exhausted',
            'visual_verification_disagreement','final_verification_error','cleanup_error'}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def read_lines(path, limit=200):
    try:
        lines = path.read_text().splitlines()[-limit:]
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except ValueError:
            pass
    return result


class Console:
    def __init__(self, runs_root, act_root):
        self.runs_root = Path(runs_root).resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.act_root = Path(act_root).resolve()
        self.lock = threading.RLock()
        self.process = None
        self.active_run_id = None
        self.stop_requested = False
        self.token = secrets.token_urlsafe(32)

    def directory(self, run_id):
        if not isinstance(run_id,str) or not re.fullmatch(r'agentic_act_[A-Za-z0-9_-]{1,120}',run_id):
            raise ValueError('无效的运行记录。')
        path = (self.runs_root / run_id).resolve()
        if path.parent != self.runs_root or not path.is_dir():
            raise ValueError('找不到这次运行。')
        return path

    def busy(self):
        return self.process is not None and self.process.poll() is None

    def snapshot(self, run_id):
        directory = self.directory(run_id)
        request = read_json(directory/'web_request.json',{})
        summary = read_json(directory/'summary.json',{})
        progress = read_json(directory/'status.json',{})
        state = read_json(directory/'backend_status.json',{})
        active = run_id == self.active_run_id and self.busy()
        # status.json is authoritative while running; summary records the terminal outcome.
        data = {**request, **summary, **progress} if active else {**request, **progress, **summary}
        status = data.get('status','initializing' if active else 'error')
        if active and self.stop_requested:
            status = 'stopping'
        elif not active and status not in TERMINAL:
            status = 'error'
            data.setdefault('error','执行进程已经退出，查看记录了解详情。')
        task = data.get('task', {'instruction':request.get('instruction',''),
                                 'steps':request.get('steps',[])})
        artifacts = {}
        for field, filename in [('report','report.html'),('video','execution.mp4'),
                                ('initial','initial.png'),('final','final.png'),
                                ('summary','summary.json')]:
            if (directory/filename).is_file():
                artifacts[field] = f'/artifacts/{run_id}/{filename}'
        final_verification = data.get('final_verification')
        events = read_lines(directory/'events.jsonl')
        if final_verification is None:
            for event in reversed(events):
                if event.get('event') == 'task_completed':
                    final_verification = event.get('verification'); break
        created_at=request.get('created_at') or datetime.fromtimestamp(directory.stat().st_mtime,ZoneInfo('Asia/Shanghai')).isoformat()
        return dict(id=run_id,status=status,success=data.get('success') is True,
                    task=task,created_at=created_at,
                    completed_steps=data.get('completed_steps',0),
                    total_steps=data.get('total_steps',len(task.get('steps',[]))),
                    current_step=data.get('current_step',data.get('completed_steps',0)),
                    attempts=data.get('attempts',{}),phase=data.get('phase'),reason=data.get('reason'),
                    events=events,vlm_calls=read_lines(directory/'vlm_calls.jsonl',100),
                    plan=data.get('plan',read_json(directory/'plan.json',{}).get('steps',[])),
                    final_verification=final_verification,error=data.get('error') or data.get('finalize_error') or data.get('cleanup_error'),
                    backend=state,has_frame=any((directory/f).is_file() for f in ['live.png','final.png','initial.png']),
                    finished=not active,artifacts=artifacts,elapsed_seconds=data.get('elapsed_seconds'),
                    injected_failure=data.get('injected_failure',data.get('inject_first_failure',False)),
                    stop_requested=active and self.stop_requested)

    def history(self):
        rows = []
        for path in self.runs_root.glob('agentic_act_*'):
            if not path.is_dir() or not any((path/n).exists() for n in ['summary.json','web_request.json']):
                continue
            try:
                row = self.snapshot(path.name)
            except ValueError:
                continue
            rows.append({key:row[key] for key in ['id','status','success','created_at','completed_steps','total_steps','finished']} |
                        {'instruction':row['task'].get('instruction','')})
        return sorted(rows,key=lambda x:x['id'],reverse=True)[:100]

    def status(self):
        with self.lock:
            run_id = self.active_run_id
            if not run_id:
                history = self.history()
                run_id = history[0]['id'] if history else None
            return dict(busy=self.busy(),active_run_id=self.active_run_id,
                        run=self.snapshot(run_id) if run_id else None)

    def start(self, payload):
        if not isinstance(payload,dict):raise ValueError('任务格式不正确。')
        if set(payload) - {'instruction','steps','seed','max_attempts','inject_first_failure','plan_only'}:
            raise ValueError('请求包含不支持的任务设置。')
        instruction = payload.get('instruction')
        if not isinstance(instruction,str) or not 1 <= len(instruction.strip()) <= 2000:
            raise ValueError('请输入 1–2000 字的任务指令。')
        rows = payload.get('steps')
        if not isinstance(rows,list) or not 1 <= len(rows) <= 3:
            raise ValueError('请安排 1–3 个目标步骤。')
        steps=[]
        for row in rows:
            if not isinstance(row,dict) or set(row) != {'object_id','region_id'}:
                raise ValueError('目标和区域格式不正确。')
            steps.append(Step(row['object_id'],row['region_id']))
        task = TaskSpec(instruction.strip(),tuple(steps))
        seed=payload.get('seed',2026092500);attempts=payload.get('max_attempts',3)
        if type(seed) is not int or not 0 <= seed < 2**31:raise ValueError('场景种子超出范围。')
        if type(attempts) is not int or not 1 <= attempts <= 5:raise ValueError('尝试次数应为 1–5。')
        for key in ['inject_first_failure','plan_only']:
            if key in payload and type(payload[key]) is not bool:raise ValueError('开关值必须为布尔值。')
        with self.lock:
            if self.busy():raise RuntimeError('当前任务仍在运行，请等待完成或先停止。')
            run_id = 'agentic_act_web_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3)
            directory=self.runs_root/run_id;directory.mkdir(exist_ok=False)
            created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
            request=dict(task=task.as_dict(),seed=seed,max_attempts=attempts,
                         inject_first_failure=payload.get('inject_first_failure',False),
                         plan_only=payload.get('plan_only',False),created_at=created,status='queued',
                         total_steps=len(steps),completed_steps=0,attempts={})
            (directory/'web_request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
            command=[sys.executable,'-B','-m','agentic_act','--instruction',task.instruction,
                     '--order',*[f'{s.object_id}:{s.region_id}' for s in steps],
                     '--seed',str(seed),'--max-attempts',str(attempts),
                     '--output',str(directory),'--stop-file',str(directory/'stop.request')]
            if request['inject_first_failure']:command.append('--inject-first-failure')
            if request['plan_only']:command.append('--plan-only')
            # Runner supports an existing empty web-owned bootstrap directory via --web-output.
            command.append('--web-output')
            log=(directory/'stdout.log').open('w')
            try:
                self.process=subprocess.Popen(command,cwd=self.act_root,stdout=log,stderr=subprocess.STDOUT)
            finally:
                log.close()
            self.active_run_id=run_id;self.stop_requested=False
            return self.snapshot(run_id)

    def stop(self):
        with self.lock:
            if not self.busy():raise ValueError('没有正在执行的任务。')
            directory=self.directory(self.active_run_id)
            (directory/'stop.request').touch();self.stop_requested=True
            return {'requested':True,'message':'已请求停止，将在当前调用或动作步结束后保存结果。'}


class Handler(BaseHTTPRequestHandler):
    def valid_host(self):
        return urlparse('http://' + self.headers.get('Host','')).hostname in {'localhost','127.0.0.1','::1'}

    def log_message(self, fmt, *args):
        # Keep high-frequency frame/status requests out of the console log.
        if self.path.startswith(('/api/frame','/api/status','/api/runs','/api/run?')):return
        super().log_message(fmt,*args)

    def reply(self, payload, status=200):
        data=json.dumps(payload,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(data)

    def file(self,path):
        size=path.stat().st_size;start=0;end=size-1;status=200
        ranged=self.headers.get('Range')
        if ranged:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',ranged)
            if not match or not any(match.groups()):return self.reply({'error':'Invalid byte range'},416)
            if match[1]:start=int(match[1]);end=int(match[2]) if match[2] else end
            else:start=max(0,size-int(match[2]))
            if start >= size or end < start:return self.reply({'error':'Range outside file'},416)
            end=min(end,size-1);status=206
        self.send_response(status);self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        if ranged:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        try:
            with path.open('rb') as stream:
                stream.seek(start);remaining=end-start+1
                while remaining:
                    chunk=stream.read(min(65536,remaining))
                    if not chunk:break
                    self.wfile.write(chunk);remaining-=len(chunk)
        except (BrokenPipeError,ConnectionResetError):pass

    def do_GET(self):
        if not self.valid_host():return self.reply({'error':'请通过本机 SSH 隧道访问。'},403)
        route=urlparse(self.path);query=parse_qs(route.query);console=self.server.console
        try:
            if route.path=='/':return self.file(Path(__file__).with_name('web.html'))
            if route.path=='/api/config':
                return self.reply(dict(objects=[dict(id=f'cube{i}',label=f'目标 {i} · cube{i}') for i in range(1,4)],
                     regions=REGIONS,csrf_token=console.token,defaults=dict(seed=2026092500,max_attempts=3,
                     instruction='先把 cube1 放到 zone_A 区域并松手，再把 cube2 放到 zone_B 区域并松手。',
                     steps=[dict(object_id='cube1',region_id='zone_A'),dict(object_id='cube2',region_id='zone_B')])))
            if route.path=='/api/status':return self.reply(console.status())
            if route.path=='/api/runs':return self.reply(console.history())
            if route.path=='/api/run':return self.reply(console.snapshot(query.get('id',[''])[0]))
            if route.path=='/api/frame':
                directory=console.directory(query.get('run',[''])[0])
                for name in ['live.png','final.png','initial.png']:
                    path=directory/name
                    if path.is_file() and not path.is_symlink() and path.resolve().parent==directory:
                        return self.file(path)
                return self.reply({'error':'画面正在准备中。'},404)
            if route.path.startswith('/artifacts/'):
                parts=[unquote(x) for x in route.path.split('/')]
                if len(parts)!=4:raise ValueError('无效的文件地址。')
                directory=console.directory(parts[2]);name=parts[3]
                if not re.fullmatch(r'[A-Za-z0-9_.-]+',name):raise ValueError('无效的文件名称。')
                path=(directory/name).resolve()
                allowed=name in {'report.html','execution.mp4','summary.json','plan.json','events.jsonl','vlm_calls.jsonl','stdout.log','final_state.json'} or name.endswith('.png')
                if not allowed or path.parent!=directory or not path.is_file():raise ValueError('文件不存在。')
                return self.file(path)
            if route.path=='/api/artifact':
                run=query.get('run',[''])[0];name=query.get('name',[''])[0]
                console.directory(run)
                if not re.fullmatch(r'[A-Za-z0-9_.-]+',name):raise ValueError('无效的文件名称。')
                self.send_response(302);self.send_header('Location',f'/artifacts/{run}/{name}');self.end_headers();return
            return self.reply({'error':'未找到页面。'},404)
        except (ValueError,KeyError) as error:return self.reply({'error':str(error)},400)
        except (BrokenPipeError,ConnectionResetError):return

    def do_POST(self):
        if not self.valid_host():return self.reply({'error':'请通过本机 SSH 隧道访问。'},403)
        console=self.server.console
        if self.headers.get('X-CSRF-Token')!=console.token:return self.reply({'error':'页面会话已过期，请刷新。'},403)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=20000:raise ValueError('请求大小不正确。')
            payload=json.loads(self.rfile.read(size))
            if self.path=='/api/start':return self.reply(console.start(payload),202)
            if self.path=='/api/stop':return self.reply(console.stop())
            return self.reply({'error':'未找到接口。'},404)
        except (ValueError,KeyError) as error:return self.reply({'error':str(error)},400)
        except RuntimeError as error:return self.reply({'error':str(error)},409)
        except OSError:return self.reply({'error':'无法启动执行进程，请查看服务器运行记录。'},500)
        except (BrokenPipeError,ConnectionResetError):return


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=18780)
    parser.add_argument('--runs-root',type=Path,default=Path(__file__).resolve().parents[4]/'outputs')
    args=parser.parse_args()
    console=Console(args.runs_root,Path(__file__).resolve().parents[1])
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.console=console
    print(f'Agentic ACT console: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
