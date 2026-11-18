"""RoboCasa transport and UI contracts without simulator, model or GPU imports."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import task_agent.web as web


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.calls.append(("GET", self.path, dict(self.headers)))
        if self.server.mode == "redirect":
            self.send_response(302); self.send_header("Location", "http://outside.invalid/private")
            self.end_headers(); return
        if self.server.mode == "html":
            body=b"<html>not an allowed backend payload</html>"
            self.send_response(200); self.send_header("Content-Type","text/html")
            self.send_header("Content-Length",str(len(body))); self.end_headers();self.wfile.write(body);return
        if self.path.startswith("/artifacts/"):
            body=b"0123456789"; code=200
            if self.headers.get("Range")=="bytes=2-5":
                body=body[2:6];code=206
            self.send_response(code); self.send_header("Content-Type","video/mp4")
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Accept-Ranges","bytes")
            if code==206:self.send_header("Content-Range","bytes 2-5/10")
            if "download=1" in self.path:self.send_header("Content-Disposition",'attachment; filename="rc_test.mp4"')
            self.end_headers();self.wfile.write(body);return
        if self.path.startswith("/frame.png"):
            body=b"PNG-image-fixture";self.send_response(200)
            self.send_header("Content-Type","image/png");self.send_header("Content-Length",str(len(body)))
            self.end_headers();self.wfile.write(body);return
        self.json_reply(dict(ok=True, busy=False, api_key="fixture-must-redact", path=self.path))

    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.calls.append(("POST",self.path,body))
        self.json_reply(dict(ok=True,queued=True,received=body),202)

    def json_reply(self, value, code=200):
        body=json.dumps(value).encode()
        self.send_response(code);self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)


class ProxyContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.upstream=ThreadingHTTPServer(("127.0.0.1",0),Upstream)
        self.upstream.calls=[];self.upstream.mode="normal"
        self.up_thread=threading.Thread(target=self.upstream.serve_forever,daemon=True);self.up_thread.start()
        self.patch=patch.object(web,"ROBOCASA_PROXY_PORT",self.upstream.server_port);self.patch.start()
        self.server=ThreadingHTTPServer(("127.0.0.1",0),web.Handler)
        self.server.console=web.Console(Path(self.temp.name)/"runs")
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(2)
        self.patch.stop()
        self.upstream.shutdown();self.upstream.server_close();self.up_thread.join(2)
        self.temp.cleanup()

    def get(self,path,headers=None):
        return urlopen(Request(self.url+path,headers=headers or {}),timeout=3)

    def post(self,path,payload,token=True,origin=None):
        headers={"Content-Type":"application/json"}
        if token:headers["X-CSRF-Token"]=self.server.console.token
        if origin:headers["Origin"]=origin
        return urlopen(Request(self.url+path,data=json.dumps(payload).encode(),headers=headers),timeout=3)

    def test_only_known_routes_to_fixed_loopback_are_permitted(self):
        self.assertEqual(web.ROBOCASA_PROXY_HOST,"127.0.0.1")
        self.assertEqual(web.robocasa_proxy_path("/api/robocasa/status?v=1","GET"),"/status?v=1")
        for path in ("/api/robocasa/http://outside.invalid/","/api/robocasa/scene",
                     "/api/robocasa/artifacts/../private.json",
                     "/api/robocasa/artifacts/rc_test/%2e%2e.json",
                     "/api/robocasa/artifacts/rc_test/private.key",
                     "http://outside.invalid/api/robocasa/status",
                     "//outside.invalid/api/robocasa/status"):
            with self.subTest(path=path),self.assertRaises(ValueError):
                web.robocasa_proxy_path(path,"GET")
        for path in ("/api/robocasa/status","/api/robocasa/scene?redirect=evil"):
            with self.assertRaises(ValueError):web.robocasa_proxy_path(path,"POST")
        with self.assertRaises(ValueError):web.robocasa_proxy_path("/api/robocasa/status","DELETE")

    def test_config_and_navigation_preserve_original_models(self):
        with self.get("/api/config") as r:data=json.load(r)
        self.assertEqual(data["policies"],["act","act_triple","pi05"])
        self.assertEqual(data["defaults"]["monitor_hz"],1)
        self.assertTrue(data["capabilities"]["scene_initial_layout_reuse"])
        with self.get("/") as r:
            html=r.read().decode();self.assertIn('href="/robocasa"',html)
            self.assertIn('id="policy"',html);self.assertIn('id="maskTrigger"',html)
        with self.get("/robocasa") as r:
            html=r.read().decode();self.assertIn("权重尚未训练",html)
            self.assertIn('href="/"',html);self.assertIn("脚本专家",html)

    def test_json_is_redacted_and_query_is_preserved(self):
        with self.get("/api/robocasa/status?v=9") as r:data=json.load(r)
        self.assertEqual(data["api_key"],"[REDACTED]")
        self.assertEqual(data["path"],"/status?v=9")
        self.assertEqual(self.upstream.calls[0][0:2],("GET","/status?v=9"))

    def test_post_uses_existing_csrf_and_origin_guards(self):
        for token,origin in ((False,None),(True,"https://unrelated.invalid")):
            with self.assertRaises(HTTPError) as error:
                self.post("/api/robocasa/scene",{"seed":17},token=token,origin=origin)
            self.assertEqual(error.exception.code,403)
        self.assertFalse(self.upstream.calls)
        with self.post("/api/robocasa/control",{"translation":[.01,0,0],"gripper":1}) as r:
            self.assertEqual(r.status,202);data=json.load(r)
        self.assertEqual(data["received"]["translation"],[.01,0,0])
        self.assertEqual(self.upstream.calls[-1][0:2],("POST","/control"))

    def test_invalid_json_shape_host_and_paths_do_not_reach_backend(self):
        with self.assertRaises(HTTPError) as error:
            self.post("/api/robocasa/scene",[])
        self.assertEqual(error.exception.code,400)
        with self.assertRaises(HTTPError) as error:
            self.get("/api/robocasa/status",{"Host":"outside.invalid"})
        self.assertEqual(error.exception.code,403)
        with self.assertRaises(HTTPError) as error:
            self.get("/api/robocasa/artifacts/rc_test/%2e%2e.json")
        self.assertEqual(error.exception.code,400)
        self.assertFalse(self.upstream.calls)

    def test_binary_frames_ranges_and_download_headers_survive_proxy(self):
        with self.get("/api/robocasa/frame.png?v=12") as r:
            self.assertEqual(r.headers["Content-Type"],"image/png")
            self.assertEqual(r.read(),b"PNG-image-fixture")
        with self.get("/api/robocasa/artifacts/rc_test/execution.mp4",{"Range":"bytes=2-5"}) as r:
            self.assertEqual(r.status,206);self.assertEqual(r.read(),b"2345")
            self.assertEqual(r.headers["Content-Range"],"bytes 2-5/10")
            self.assertEqual(r.headers["Accept-Ranges"],"bytes")
        with self.get("/api/robocasa/artifacts/rc_test/execution.mp4?download=1") as r:
            self.assertIn("attachment",r.headers["Content-Disposition"])
        before=len(self.upstream.calls)
        with self.assertRaises(HTTPError) as error:
            self.get("/api/robocasa/frame.png",{"Range":"bytes=0-1,4-5"})
        self.assertEqual(error.exception.code,416);self.assertEqual(before,len(self.upstream.calls))

    def test_upstream_redirect_and_html_are_not_forwarded(self):
        for mode in ("redirect","html"):
            self.upstream.mode=mode
            with self.assertRaises(HTTPError) as error:self.get("/api/robocasa/status")
            self.assertEqual(error.exception.code,502)
            data=json.loads(error.exception.read());self.assertIn("error",data)

    def test_catalog_and_target_share_fixed_proxy_and_existing_guards(self):
        with self.get("/api/robocasa/catalog?query=%E8%8B%B9%E6%9E%9C&source=objaverse&offset=30&limit=30") as r:data=json.load(r)
        self.assertIn("query=%E8%8B%B9%E6%9E%9C",data["path"])
        self.assertIn("offset=30",data["path"])
        before=len(self.upstream.calls)
        with self.assertRaises(HTTPError) as error:
            self.post("/api/robocasa/target",{"target_id":"item_0"},token=False)
        self.assertEqual(error.exception.code,403);self.assertEqual(before,len(self.upstream.calls))
        with self.post("/api/robocasa/target",{"target_id":None}) as r:
            self.assertEqual(r.status,202);self.assertEqual(json.load(r)["received"],{"target_id":None})
        with self.assertRaises(ValueError):
            web.robocasa_proxy_path("/api/robocasa/catalog/http://outside.invalid","GET")


@unittest.skipUnless(shutil.which("node"),"Node is needed for UI contracts")
class RoboCasaUIContracts(unittest.TestCase):
    def run_js(self, checks):
        prefix=r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8'),all=[];
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.attrs={};this.dataset={};this.listeners={};this.hidden=false;this.disabled=false;this.value='';this.textContent='';this.paused=true;this.srcWrites=0;all.push(this)}
 append(...nodes){this.children.push(...nodes)}
 replaceChildren(...nodes){this.children=nodes;this.textContent=''}
 setAttribute(k,v){this.attrs[k]=v}
 getAttribute(k){return this.attrs[k]??null}
 addEventListener(k,v){(this.listeners[k]??=[]).push(v)}
 set src(v){this.attrs.src=v;this.srcWrites++}
 get src(){return this.attrs.src||''}
 pause(){this.paused=true}
 play(){this.paused=false;return Promise.resolve()}
 scrollIntoView(){}
 showModal(){this.open=true}
 close(){this.open=false}
}
const elements={},get=id=>elements[id]??(elements[id]=new Element());
get('order').value='red_cube,blue_cube';get('controlSteps').value='1';get('controlSteps').max='10';get('moveStep').value='.01';get('seed').value='17';
const move=new Element('button');move.dataset.move='1,0,0';
const rotation=new Element('button');rotation.dataset.rotate='0,0,1';
const document={getElementById:get,createElement:tag=>new Element(tag),querySelectorAll:selector=>{
 if(selector==='[data-move]')return [move];if(selector==='[data-rotate]')return [rotation];
 if(selector==='[data-episode-action]')return all.filter(e=>e.dataset.episodeAction);
 if(selector==='[data-selection-action]')return all.filter(e=>e.dataset.selectionAction);
 if(selector==='[data-catalog-add]')return all.filter(e=>e.dataset.catalogAdd);
 if(selector==='[data-move],[data-rotate],#hold,#open,#close')return [move,rotation,get('hold'),get('open'),get('close')];
 return [];
}};
const context={document,console,URLSearchParams,setTimeout,clearTimeout,setInterval:()=>{},fetch:async()=>({ok:true,json:async()=>({ok:true,episodes:[]})})};
vm.createContext(context);
const source=html.split('<script>')[1].split('</script>')[0];new vm.Script(source);
vm.runInContext(source.replace('init();setInterval(poll,1000);',''),context);
const evaluate=src=>vm.runInContext(src,context),call=(name,arg)=>{context.input=arg;return evaluate(name+'(input)')};
const flatten=e=>e.textContent+' '+e.children.map(flatten).join(' ');
const scene=(extra={})=>({ok:true,busy:false,phase:'ready',recording:false,scene:{seed:17,order:['red_cube','blue_cube'],robot:'PandaOmron'},frame_url:'/frame.png',frame_id:5,...extra});
const setup=()=>call('setupConfig',{objects:[{id:'red_cube',name:'红色方块'},{id:'blue_cube',name:'蓝色方块'}],scripted_demo_available:true,translation_step_m:.01,rotation_step_rad:.05,max_steps:10});
"""
        code=prefix+chr(10)+"(async()=>{"+chr(10)+checks+chr(10)+"})().catch(e=>{console.error(e);process.exitCode=1});"
        page=Path(__file__).parent/"task_agent"/"robocasa.html"
        result=subprocess.run([shutil.which("node"),"-",str(page)],input=code,text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_new_page_distinguishes_collection_from_trained_policies(self):
        self.run_js(r"""
assert(html.includes('RoboCasa 的 ACT / π0.5 权重尚未训练'));
assert(html.includes('VLM 规划与视觉完成判断尚未接入此后端'));
assert(html.includes('状态回放用于核对相机'));
assert(html.includes('质量标签不会替代仿真评分'));
assert(!html.includes('autoplay'));
setup();assert(!elements.scriptedInfo.textContent.includes('ACT'));
""")

    def test_ready_recording_busy_and_pending_controls_are_distinct(self):
        self.run_js(r"""
setup();call('renderStatus',scene({gripper:1}));assert.equal(evaluate('UI.gripper'),1);call('renderStatus',scene({gripper:-1}));assert.equal(evaluate('UI.gripper'),-1);assert(!elements.prepare.disabled);assert(!elements.record.disabled);assert(!move.disabled);assert(elements.save.disabled);assert(!elements.scripted.disabled);
call('renderStatus',scene({recording:true,phase:'recording'}));assert(elements.prepare.disabled);assert(elements.restore.disabled);assert(elements.record.disabled);assert(!move.disabled);assert(!elements.save.disabled);assert(!elements.stop.disabled);
call('renderStatus',scene({busy:true,phase:'scripted_demo'}));assert(move.disabled);assert(elements.prepare.disabled);assert(!elements.stop.disabled);
evaluate('UI.pending=true');call('setControls');assert(elements.stop.disabled);
""")

    def test_artifact_urls_stay_on_same_origin_and_frames_refresh_on_revision(self):
        self.run_js(r"""
for(const url of ['https://outside.invalid/x','//outside.invalid/x','/x','/artifacts/a/evil'+String.fromCharCode(92)+'file','/artifacts/a/evil'+String.fromCharCode(10)+'file'])assert.equal(call('assetUrl',url),null);
assert.equal(call('assetUrl','/artifacts/rc_a/execution.mp4'),'/api/robocasa/artifacts/rc_a/execution.mp4');
setup();call('renderStatus',scene());assert(elements.frame.src.startsWith('/api/robocasa/frame.png?'));const writes=elements.frame.srcWrites;
call('renderStatus',scene());assert.equal(elements.frame.srcWrites,writes);
call('renderStatus',scene({frame_id:6}));assert.equal(elements.frame.srcWrites,writes+1);
call('renderStatus',scene({frame_id:6,scene:{seed:99,order:['red_cube','blue_cube']}}));assert.equal(elements.frame.srcWrites,writes+2);
""")

    def test_quality_label_never_overrides_simulator_order_score(self):
        self.run_js(r"""
setup();call('renderStatus',scene({user_quality:'success',simulator_success:false,order_valid:false}));
assert.equal(elements.success.textContent,'顺序未通过');
call('renderStatus',scene({user_quality:'success',simulator_success:true,order_valid:false}));assert.equal(elements.success.textContent,'顺序未通过');
call('renderStatus',scene({simulator_success:true,order_valid:true,ordered_completed:['red_cube','blue_cube']}));assert.equal(elements.success.textContent,'已完成，顺序正确');
assert(elements.progress.children.every(e=>e.className==='done'));
call('renderStatus',scene({replay:{mode:'states',verified:true,initial_rgb_match:false,initial_state_verified:true,initial_visual_verified:true,max_abs_qpos_error:0}}));assert(elements.replayResult.textContent.includes('初态与图像容差核验通过'));assert(!elements.replayResult.textContent.includes('逐像素一致'));
""")

    def test_terminal_recording_shows_video_but_does_not_replace_selected_history(self):
        self.run_js(r"""
setup();call('renderStatus',scene({recording:true,phase:'recording',episode:{id:'rc_a'}}));
call('renderStatus',scene({recording:false,phase:'ready',episode:{id:'rc_a'},video_url:'/artifacts/rc_a/execution.mp4'}));
assert(!elements.video.hidden);assert(elements.video.src.includes('/api/robocasa/artifacts/rc_a/'));
assert.equal(elements.video.paused,true);assert(elements.downloadVideo.href.endsWith('?download=1'));
evaluate("UI.selectedEpisode='rc_history'");call('showVideo','bad-unused-url');
context.url='/artifacts/rc_history/execution.mp4';evaluate("showVideo(url,'rc_history','历史')");
call('renderStatus',scene({busy:true,phase:'replaying',episode:{id:'rc_b'}}));
call('renderStatus',scene({busy:false,episode:{id:'rc_b'},video_url:'/artifacts/rc_b/execution.mp4'}));
assert(elements.video.src.includes('/rc_history/'));
""")

    def test_config_retry_and_literal_control_payload(self):
        self.run_js(r"""
const calls=[];context.fetch=async(url,opts)=>{calls.push({url,opts});return {ok:true,json:async()=>url.endsWith('/config')?{ok:true,scripted_demo_available:true,objects:[]}:url.endsWith('/status')?scene():url.endsWith('/episodes')?{ok:true,episodes:[]}:{ok:true,queued:true}}};
evaluate("UI.token='fixture-token';UI.config=null;UI.connectionError=true");get('error').hidden=false;
await call('poll');assert(calls.some(c=>c.url==='/api/robocasa/config'));assert(elements.error.hidden);assert(!elements.scripted.disabled);
await evaluate("control([.01,0,0],[0,0,0],1)");
const cmd=calls.find(c=>c.url==='/api/robocasa/control');assert(cmd);assert.equal(cmd.opts.headers['X-CSRF-Token'],'fixture-token');
assert.deepEqual(JSON.parse(cmd.opts.body),{translation:[.01,0,0],rotation:[0,0,0],gripper:1,steps:1});
""")

    def test_history_sources_data_links_and_text_are_safe(self):
        self.run_js(r"""
assert(call('dateText',1793874550.683806).includes('2026'));assert(call('dateText',1793874550.683806).includes('北京时间'));assert(!call('dateText',1793874550.683806).includes('1793874550'));
setup();context.rows=[{id:'rc_a',label:'<img src=x onerror=alert(1)>',source:'scripted_expert_privileged_state',user_quality:'success',simulator_success:false,order_valid:false,steps:99,order:['red_cube','blue_cube'],video_url:'/artifacts/rc_a/execution.mp4',raw_url:'/artifacts/rc_a/demo.hdf5',manifest_url:'/artifacts/rc_a/manifest.json'}];
evaluate('UI.episodes=rows;UI.connected=true');call('renderEpisodes');
const row=elements.episodes.children[0];assert.equal(row.children[0].textContent,context.rows[0].label);
const display=flatten(row);assert(display.includes('仿真评分：未完成'));assert(display.includes('脚本专家（使用仿真状态）'));
const links=row.children.find(e=>e.className==='links');assert(links.children.some(e=>e.href==='/api/robocasa/artifacts/rc_a/demo.hdf5'));
""")

    def test_training_qc_is_independent_from_labels_and_unreplayable_rows_stay_disabled(self):
        self.run_js(r"""
setup();context.rows=[
 {id:'rc_pass',status:'completed',training_ready:true,simulator_success:true,order_valid:true,can_replay:true,manifest_url:'/artifacts/rc_pass/training_manifest.json'},
 {id:'rc_badmask',status:'completed',quality:'success',training_ready:false,simulator_success:true,order_valid:true,can_replay:true,data_quality_issues:['Original frontal base camera occludes blue target; all blue target masks zero'],manifest_url:'/artifacts/rc_badmask/training_manifest.json'},
 {id:'rc_pending',status:'recording',training_ready:false,can_replay:false},
 {id:'rc_legacy',status:'completed',training_ready:false,can_replay:false,raw_url:'/artifacts/rc_legacy/trajectory.h5',video_url:'/artifacts/rc_legacy/execution.mp4'}];
evaluate('UI.connected=true;UI.status=null;UI.episodes=rows');call('renderEpisodes');
const rendered=elements.episodes.children;
assert(flatten(rendered[0]).includes('训练数据校验通过'));
assert(flatten(rendered[1]).includes('训练数据校验未通过'));
assert(flatten(rendered[1]).includes('Mask 全为 0'));assert(flatten(rendered[1]).includes('仿真评分：完成'));
assert(flatten(rendered[2]).includes('训练数据待检查'));
assert(elements.dataSummary.textContent.includes('通过 1 条 / 未通过 2 条 / 待检查 1 条'));
const legacy=rendered[3],buttons=legacy.children.find(el=>el.className==='actions').children.filter(el=>el.dataset.episodeAction);
assert.equal(buttons.filter(b=>['states','actions'].includes(b.dataset.episodeAction)).length,2);
assert(buttons.every(b=>b.disabled));
call('setControls');assert(buttons.every(b=>b.disabled),'status refresh must never re-enable invalid replay');
assert(flatten(legacy).includes('仅可查看录像或下载原始数据'));
const links=rendered[1].children.find(el=>el.className==='links');assert(links.children.some(el=>el.href.endsWith('/rc_badmask/training_manifest.json')));
""")

    def test_missing_list_quality_details_are_read_from_summary_documents(self):
        self.run_js(r"""
setup();const calls=[];context.rows=[{id:'rc_qc',status:'completed',steps:100,training_ready:false,simulator_success:true,can_replay:true}];
context.fetch=async(url)=>{calls.push(url);return {ok:true,json:async()=>({episode_id:'rc_qc',training_ready:false,error:'original failed capture marker',quality_reason:'目标 Mask 没有覆盖当前目标',deterministic_replay_available:false,training_manifest_url:'/artifacts/rc_qc/training_manifest.json'})}};
evaluate('UI.episodeRevision=8;UI.episodes=rows;UI.connected=true');
context.generation=8;await evaluate('enrichEpisodeQuality(rows,generation)');
assert(calls.includes('/api/robocasa/artifacts/rc_qc/summary.json'));
assert.equal(evaluate('UI.episodes[0].deterministic_replay_available'),false);
assert(flatten(elements.episodes.children[0]).includes('目标 Mask 没有覆盖当前目标'));
const buttons=elements.episodes.children[0].children.find(el=>el.className==='actions').children;
assert(buttons.filter(b=>b.dataset.episodeAction).every(b=>b.disabled));
assert.equal(evaluate('UI.episodes[0].manifest_url'),'/artifacts/rc_qc/training_manifest.json');
""")

    def test_action_replay_completion_and_task_reproduction_do_not_hide_strict_failure(self):
        self.run_js(r"""
setup();call('renderStatus',scene({phase:'replay_finished',replay:{mode:'actions',replay_completed:true,task_reproduced:true,strict_trajectory_verified:false,verified:false,initial_rgb_match:true,max_abs_qpos_error:.000717335,max_eef_position_error_m:.0000279703,max_object_position_error_m:.0001692505,max_arm_joint_error_rad:.0000280243}}));
const message=elements.replayResult.textContent;
assert(message.includes('动作回放已完成'));assert(message.includes('任务已按指定顺序重现'));
assert(message.includes('严格动作轨迹核验未通过'));assert(message.includes('未放宽标准'));
assert(message.includes('0.028 毫米'));assert(message.includes('0.1693 毫米'));
assert(!message.includes('0.7173 毫米'),'mixed-unit qpos vector must never be treated as position');
assert(!message.includes('自主任务成功'));
""")

    def test_state_replay_is_not_autonomous_success_and_raw_error_keeps_mixed_units(self):
        self.run_js(r"""
setup();call('renderStatus',scene({phase:'replay_finished',simulator_success:true,order_valid:true,replay:{mode:'states',replay_completed:true,task_reproduced:null,strict_trajectory_verified:true,verified:true,initial_rgb_match:true,max_abs_qpos_error:0}}));
const message=elements.replayResult.textContent;
assert(message.includes('状态回放已完成'));assert(message.includes('不计作机器人自主任务成功'));
assert(message.includes('严格状态轨迹核验通过'));assert(!message.includes('任务已按指定顺序重现'));
assert(message.includes('混合单位'));assert(elements.success.textContent.includes('非自主执行'));
call('renderStatus',scene({phase:'replay_finished',replay:{mode:'actions',replay_completed:true,task_reproduced:false,strict_trajectory_verified:true,verified:true,initial_rgb_match:true}}));
assert(elements.replayResult.textContent.includes('本次未按指定顺序完成任务'));
assert(elements.replayResult.textContent.includes('严格动作轨迹核验通过'));
""")

    def test_restored_snapshot_phase_never_implies_an_active_recorder(self):
        self.run_js(r"""
setup();
for(const phase of ['manual_recording','recording','scripted_expert']){
 call('renderStatus',scene({phase,recording:false,busy:false,episode:null,frame_id:0}));
 assert.equal(call('recording'),false);
 assert.equal(elements.phase.textContent,'场景就绪');
 assert(!elements.record.disabled);assert(elements.save.disabled);assert(elements.stop.disabled);
}
call('renderStatus',scene({phase:'manual_recording',recording:true,busy:false}));
assert.equal(call('recording'),true);assert.equal(elements.phase.textContent,'手动采集中');
call('renderStatus',scene({phase:'replay_finished',recording:false,replay:{mode:'states',replay_completed:true,verified:true,strict_trajectory_verified:true,initial_rgb_match:true}}));
assert.equal(elements.phase.textContent,'回放已结束');assert(elements.replayResult.textContent.includes('不计作机器人自主任务成功'));
""")

    def test_object_selection_is_only_a_draft_and_payload_has_bounded_stable_ids(self):
        self.run_js(r"""
setup();call('renderStatus',scene());const before=JSON.stringify(evaluate('UI.status'));
assert.deepEqual(call('scenePayload'),{seed:17,order:['red_cube','blue_cube']});
context.apple={catalog_id:'objaverse:apple',name:'apple',name_zh:'苹果',source:'objaverse',graspable:true,selectable:true,available:true};
context.mug={catalog_id:'lightwheel:mug',name:'mug',name_zh:'马克杯',source:'lightwheel',graspable:true,selectable:true,available:true};
evaluate("UI.config.catalog_url='/catalog';UI.catalogById.set(apple.catalog_id,apple);UI.catalogById.set(mug.catalog_id,mug)");
get('objectMode').value='custom';call('setObjectMode');assert(elements.prepare.disabled);
call('addSelection',context.apple);call('addSelection',context.apple);
context.variant={instance_id:'variant_123',label:'杯子变体 123',selectable:true};
evaluate('addSelection(mug,variant)');evaluate("editSelection(1,'up')");
assert.deepEqual(call('scenePayload'),{seed:17,objects:[{catalog_id:'lightwheel:mug',instance_id:'variant_123',quantity:1},{catalog_id:'objaverse:apple',instance_id:null,quantity:2}]});
assert.equal(JSON.stringify(evaluate('UI.status')),before);assert(!elements.prepare.disabled);
""")

    def test_object_limits_and_graspable_availability_block_invalid_additions(self):
        self.run_js(r"""
setup();call('renderStatus',scene());evaluate("UI.config.catalog_url='/catalog'");get('objectMode').value='custom';call('setObjectMode');
context.ok={catalog_id:'objaverse:apple',name_zh:'苹果',source:'objaverse',graspable:true,selectable:true,available:true};
context.decor={catalog_id:'objaverse:plant',name_zh:'植物装饰',source:'objaverse',graspable:false,selectable:false,available:true,disabled_reason:'仅为装饰'};
call('addSelection',context.decor);assert.equal(call('selectionCount'),0);assert(elements.catalogError.textContent.includes('仅为装饰'));
for(let i=0;i<6;i++)call('addSelection',context.ok);
assert.equal(call('selectionCount'),6);call('addSelection',context.ok);assert.equal(call('selectionCount'),6);assert(elements.catalogError.textContent.includes('最多选择 6'));
evaluate("editSelection(0,'remove')");assert.equal(call('selectionCount'),0);assert(elements.prepare.disabled);
""")

    def test_catalog_and_variants_are_paged_and_filters_are_url_encoded(self):
        self.run_js(r"""
setup();evaluate("UI.config.catalog_url='/catalog'");get('catalogQuery').value='苹果';get('catalogSource').value='objaverse';
context.category={catalog_id:'objaverse:apple',name:'apple',name_zh:'苹果',source:'objaverse',instance_count:57,available:true,graspable:true,selectable:true};
const calls=[];context.fetch=async(url)=>{calls.push(url);const variant=url.includes('category_id=');return {ok:true,json:async()=>variant?{ok:true,category:context.category,instances:[{instance_id:'apple_model_51',label:'Apple 51',selectable:true,bbox_size_m:[.05,.06,.07]}],instance_total:57,total:300,offset:50}:{ok:true,categories:[context.category,{catalog_id:'aigen:decor',name:'decor',name_zh:'装饰',source:'aigen',instance_count:3,graspable:false,selectable:false,available:true,disabled_reason:'装饰不可抓取'}],total:100,offset:30,sources:['objaverse','aigen','lightwheel']}}};
await evaluate('loadCatalog(30)');assert.equal(elements.catalogResults.children.length,2);assert(elements.catalogPage.textContent.includes('31–32 / 100'));
let query=new URL('http://local'+calls[0]).searchParams;assert.equal(query.get('query'),'苹果');assert.equal(query.get('source'),'objaverse');assert.equal(query.get('limit'),'30');
assert(flatten(elements.catalogResults.children[1]).includes('装饰不可抓取'));
await evaluate('loadVariants(category,50)');query=new URL('http://local'+calls[1]).searchParams;
assert.equal(query.get('category_id'),'objaverse:apple');assert.equal(query.get('limit'),'50');assert.equal(evaluate('UI.variantTotal'),57);
assert(elements.variantPage.textContent.includes('51–51 / 57'));assert.equal(elements.catalogVariant.children.length,2);
""")

    def test_resolved_objects_names_and_scene_capability_override_default_expert(self):
        self.run_js(r"""
setup();evaluate("UI.config.catalog_url='/catalog'");
context.objects=[{id:'item_0',catalog_id:'objaverse:apple',name_zh:'苹果',source:'objaverse',instance_id:'fixed_a',instance_label:'Apple model A'},{id:'item_1',catalog_id:'lightwheel:mug',name_zh:'马克杯',source:'lightwheel',instance_id:'fixed_b',instance_label:'Mug model B'}];
call('renderStatus',scene({scene:{seed:17,order:['item_0','item_1'],selected_objects:context.objects,scripted_demo_available:false,scripted_demo_reason:'这些物品请使用手动采集'},progress:{current_target:'item_0'},target_override:null}));
assert(flatten(elements.progress).includes('苹果'));assert(flatten(elements.progress).includes('马克杯'));assert(!flatten(elements.progress).includes('红色方块'));
assert(flatten(elements.currentObjects).includes('Apple model A'));assert(flatten(elements.currentObjects).includes('Mug model B'));
assert(elements.scripted.disabled);assert(elements.scriptedInfo.textContent.includes('手动采集'));
context.historyRow={selected_objects:[{id:'item_0',name_zh:'历史杯子'}]};assert.equal(evaluate("historyObjectName(historyRow,'item_0')"),'历史杯子');
""")

    def test_current_target_switch_and_restore_do_not_replace_selected_objects(self):
        self.run_js(r"""
setup();evaluate("UI.config.catalog_url='/catalog';UI.token='fixture-token'");
context.current=scene({scene:{seed:17,order:['item_0'],selected_objects:[{id:'item_0',name_zh:'苹果',instance_id:'fixed_model'}],scripted_demo_available:false},recording:true,phase:'manual_recording',progress:{current_target:'item_0'},target_override:null});
call('renderStatus',context.current);assert(!elements.currentTarget.disabled);
const calls=[];context.fetch=async(url,opts)=>{calls.push({url,opts});return {ok:true,json:async()=>url.endsWith('/status')?context.current:url.endsWith('/episodes')?{ok:true,episodes:[]}:{ok:true,queued:true}}};
get('currentTarget').value='item_0';await elements.currentTarget.onchange();
const target=calls.find(row=>row.url==='/api/robocasa/target');assert.deepEqual(JSON.parse(target.opts.body),{target_id:'item_0'});
context.current={...context.current,recording:false,phase:'scene_preview'};call('renderStatus',context.current);
get('objectMode').value='custom';evaluate("UI.selection=[{catalog_id:'objaverse:banana',instance_id:null,quantity:1,name:'香蕉'}]");call('renderSelection');
await elements.restore.onclick();
const restore=calls.find(row=>row.url==='/api/robocasa/scene');assert.deepEqual(JSON.parse(restore.opts.body),{restore:true});
assert(flatten(elements.currentObjects).includes('fixed_model'));assert(!flatten(elements.currentObjects).includes('香蕉'));
""")
