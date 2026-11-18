"""Loopback-only bridge. All MuJoCo work belongs to exactly one engine thread."""
import argparse,copy,json,mimetypes,queue,re,threading,time,traceback,faulthandler,signal
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,unquote,parse_qs
from pathlib import Path
from .paths import ROOT,DATA,OUTPUT,episode_dir,inside
from .dataset import list_episodes
from .catalog import get_catalog
from .engine import Engine,Stopped

CONFIG=dict(ok=True,backend="robocasa",task="ordered_counter_to_tray_v1",
    objects=[dict(id="red_cube",name="红色方块"),dict(id="blue_cube",name="蓝色方块")],
    defaults=dict(seed=17,order=["red_cube","blue_cube"],layout_id=11,style_id=1),
    control=dict(translation_step_m=.01,rotation_step_rad=.05,max_steps=10,
                 translation_max_m=.05,rotation_max_rad=.5,frame="world",gripper_open=-1,gripper_close=1,
                 fixed_base=True,control_freq=20),
    translation_step_m=.01,rotation_step_rad=.05,max_steps=10,
    catalog_url="/catalog",min_objects=1,max_objects=6,object_selection_available=True,
    generic_manual_only=True,scripted_demo_available=True,policies_available=False,vlm_execution_available=False,
    mask_source="sim_gt_offline_annotation",
    modes=["manual_demonstration","scripted_expert_privileged_state","state_replay","action_replay"],
    notice="RoboCasa 数据采集与回放。尚未训练/接入该场景的 ACT、pi0.5 或 VLM 执行；绿色目标图为离线真值标注，不是 SAM2/Cutie。")

class Manager:
    def __init__(self,start_thread=True):
        self.lock=threading.RLock();self.queue=queue.Queue();self.stop_event=threading.Event()
        self.state=dict(ok=True,busy=False,phase="not_initialized",recording=False,scene=None,
            episode=None,frame_id=0,updated_at=time.time(),frame_url=None,error=None,
            video_url=None,policies_available=False,rendering={})
        self.frame=None;self.engine=None;self.busy=False
        self.thread=threading.Thread(target=self.run,name="robocasa-engine",daemon=True);self.thread.start() if start_thread else None

    def publish(self,state,frame):
        with self.lock:
            self.state.update(state,busy=self.busy,ok=True)
            if frame is not None:self.frame=frame

    def snapshot(self):
        with self.lock:return copy.deepcopy(self.state)

    def submit(self,name,params):
        with self.lock:
            if self.busy:raise ValueError("RoboCasa is busy; stop or wait for the current operation")
            self.busy=True;self.stop_event.clear()
            self.state.update(busy=True,error=None,phase=name+"_queued")
            job=f"job_{time.time_ns()}";self.state["job_id"]=job
            self.queue.put((name,params))
            return dict(ok=True,queued=True,job_id=job)

    def stop(self):
        self.stop_event.set()
        with self.lock:
            if self.busy:
                self.state["phase"]="stopping"
                return dict(ok=True,requested=True)
        return self.submit("stop",{})

    def run(self):
        self.engine=Engine(self.publish,self.stop_event)
        while True:
            name,params=self.queue.get()
            if name=="shutdown":
                self.engine.close()
                return
            error=None
            try:
                if name=="scene":self.engine.scene(params)
                elif name=="record_start":self.engine.start_record(params)
                elif name=="control":self.engine.control(params)
                elif name=="target":self.engine.select_target(params)
                elif name=="record_stop":
                    quality=params.get("quality","stopped")
                    if quality not in ("success","failure","stopped"):raise ValueError("quality must be success/failure/stopped")
                    self.engine.finish_record(quality,params.get("note",""))
                elif name=="scripted_demo":self.engine.scripted_demo(params)
                elif name=="replay":self.engine.replay(params)
                elif name=="stop":
                    if self.engine.record:self.engine.finish_record("stopped","User requested stop")
                else:raise ValueError("Unknown operation")
            except BaseException as exc:
                error=str(exc).replace(str(ROOT),"PROJECT")
                OUTPUT.mkdir(parents=True,exist_ok=True)
                with (OUTPUT/"service_errors.log").open("a") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S")+" "+name+"\n"+traceback.format_exc()+"\n")
                if self.engine.record:
                    try:self.engine.finish_record("stopped" if isinstance(exc,Stopped) else "failure",error,error)
                    except BaseException:pass
            finally:
                with self.lock:
                    self.busy=False
                    self.state["busy"]=False;self.state["error"]=error
                    self.state["updated_at"]=time.time()
                # Do not render again simply to mark a job idle.
                with self.lock:
                    self.state.update(self.engine.public_status(),busy=False,error=error,ok=True)
                self.queue.task_done()

class Handler(BaseHTTPRequestHandler):
    server_version="MaskPolicyRoboCasa/0.1"
    def log_message(self,fmt,*args):
        if self.path.split("?")[0] not in ("/status","/frame.png","/health"):super().log_message(fmt,*args)
    def reply(self,data,status=200):
        payload=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(payload)));self.send_header("Cache-Control","no-store")
        self.end_headers();self.wfile.write(payload)
    def file(self,path):
        size=path.stat().st_size;start=0;end=size-1;partial=False
        range_header=self.headers.get("Range")
        if range_header:
            m=re.fullmatch(r"bytes=(\d*)-(\d*)",range_header)
            if not m:return self.reply(dict(ok=False,error="Unsupported range"),416)
            if m[1]:
                start=int(m[1]);end=min(int(m[2]) if m[2] else size-1,size-1)
            else:start=max(0,size-int(m[2]))
            if start>end or start>=size:return self.reply(dict(ok=False,error="Range outside file"),416)
            partial=True
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type",mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges","bytes")
        self.send_header("Content-Length",str(end-start+1))
        if partial:self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
        if "download=1" in self.path:self.send_header("Content-Disposition",f'attachment; filename="{path.name}"')
        self.send_header("Cache-Control","no-store");self.end_headers()
        with path.open("rb") as f:
            f.seek(start);remain=end-start+1
            while remain:
                data=f.read(min(1024*256,remain))
                if not data:break
                self.wfile.write(data);remain-=len(data)
    def do_GET(self):
        try:
            path=unquote(urlparse(self.path).path)
            if path=="/health":
                return self.reply(dict(ok=True,backend="robocasa",engine_thread_alive=self.server.manager.thread.is_alive(),
                    policies_available=False,rendering=self.server.manager.snapshot().get("rendering",{})))
            if path=="/config":
                return self.reply({**CONFIG,"rendering":self.server.manager.snapshot().get("rendering",{})})
            if path=="/catalog":
                args=parse_qs(urlparse(self.path).query)
                return self.reply(get_catalog(query=args.get("query",[""])[0],source=args.get("source",[""])[0],
                    category_id=args.get("category_id",[""])[0],offset=args.get("offset",[0])[0],
                    limit=args.get("limit",[30])[0],refresh=args.get("refresh",["0"])[0]=="1"))
            if path=="/status":return self.reply(self.server.manager.snapshot())
            if path=="/episodes":return self.reply(dict(ok=True,episodes=list_episodes()))
            if path=="/frame.png":
                with self.server.manager.lock:data=self.server.manager.frame
                if not data:return self.reply(dict(ok=False,error="Generate a scene first"),404)
                self.send_response(200);self.send_header("Content-Type","image/png")
                self.send_header("Content-Length",str(len(data)));self.send_header("Cache-Control","no-store")
                self.end_headers();self.wfile.write(data);return
            parts=path.strip("/").split("/")
            if len(parts)==3 and parts[0]=="artifacts":
                eid,name=parts[1:]
                public={"execution.mp4","replay_states.mp4","replay_actions.mp4","summary.json",
                    "replay_states.json","replay_actions.json","metadata.json","initial_snapshot.json","model.xml",
                    "trajectory.h5","training_manifest.json","training_sample.npz","ui_preview.png"}
                if name not in public:raise ValueError("Artifact is not public")
                candidates=[episode_dir(eid,True)/name,episode_dir(eid)/name]
                p=next((x for x in candidates if x.is_file() and not x.is_symlink()),None)
                if p is None:raise ValueError("Artifact is not available")
                return self.file(inside(p))
            self.reply(dict(ok=False,error="Not found"),404)
        except (ValueError,KeyError) as exc:self.reply(dict(ok=False,error=str(exc)),400)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_POST(self):
        try:
            if self.headers.get_content_type()!="application/json":raise ValueError("JSON content required")
            size=int(self.headers.get("Content-Length",0))
            if not 0<size<16000:raise ValueError("Invalid request size")
            params=json.loads(self.rfile.read(size))
            if not isinstance(params,dict):raise ValueError("Object body required")
            path=urlparse(self.path).path
            if path=="/stop":return self.reply(self.server.manager.stop())
            jobs={"/scene":"scene","/record/start":"record_start","/control":"control","/target":"target",
                  "/record/stop":"record_stop","/scripted_demo":"scripted_demo","/replay":"replay"}
            if path not in jobs:return self.reply(dict(ok=False,error="Not found"),404)
            self.reply(self.server.manager.submit(jobs[path],params),202)
        except (ValueError,KeyError,TypeError) as exc:self.reply(dict(ok=False,error=str(exc)),400)
        except (BrokenPipeError,ConnectionResetError):pass

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=18850);args=parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
    server.manager=Manager(start_thread=False)
    # Keep EGL/MuJoCo on the process main thread. HTTP only handles queue I/O.
    server.manager.thread=threading.current_thread()
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1,all_threads=True)
    def shutdown(signum,frame):
        server.manager.stop_event.set()
        server.manager.queue.put(("shutdown",{}))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,shutdown)
    http=threading.Thread(target=server.serve_forever,name="robocasa-http",daemon=True)
    http.start()
    print(f"RoboCasa data console backend: http://127.0.0.1:{args.port}",flush=True)
    try:server.manager.run()
    finally:server.shutdown();server.server_close()
if __name__=="__main__":main()
