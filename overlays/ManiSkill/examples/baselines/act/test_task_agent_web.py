"""Web transport and child lifecycle tests; no model, simulator or GPU calls."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from task_agent.web import Console, Handler


class FakeProcess:
    def __init__(self):
        self.returncode = None
    def poll(self):
        return self.returncode


class ProcessFactory:
    def __init__(self):
        self.calls = []
        self.processes = []
    def __call__(self, command, **kwargs):
        process = FakeProcess()
        self.calls.append((command, kwargs))
        self.processes.append(process)
        return process


class ConsoleContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.factory = ProcessFactory()
        self.console = Console(Path(self.temp.name)/'runs', act_root=Path(self.temp.name),
                               process_factory=self.factory)
    def tearDown(self):
        self.temp.cleanup()

    def start(self, **options):
        return self.console.start(dict(instruction='收拾球类', **options))

    def test_natural_task_launches_literal_argv_and_no_object_ids(self):
        instruction = '收拾球类; $(touch NOT_A_COMMAND)'
        result = self.console.start(dict(instruction=instruction, max_subtasks=4))
        command, options = self.factory.calls[0]
        self.assertEqual(command[command.index('--instruction')+1], instruction)
        self.assertEqual(command[command.index('-m')+1], 'task_agent.runner')
        self.assertNotIn('shell', options)
        self.assertNotIn('--order', command)
        self.assertEqual(result['run']['instruction'], instruction)
        self.assertEqual(result['run']['status'], 'queued')
        self.assertTrue(result['run']['busy'])
        self.assertTrue((self.console.directory(result['run_id'])/'web_request.json').is_file())

    def test_invalid_or_manually_selected_targets_are_rejected(self):
        for payload in ({'instruction':'球','object_id':'ball1'}, {'instruction':''},
                        {'instruction':'球','max_subtasks':5}, {'instruction':'球','seed':True},
                        {'instruction':'球','monitor_hz':2}, {'instruction':'球','policy':'shell'}):
            with self.assertRaises(ValueError):
                self.console.start(payload)
        self.assertFalse(self.factory.calls)

    def test_only_one_active_task_and_cooperative_stop(self):
        first = self.start()
        with self.assertRaises(RuntimeError):
            self.start()
        self.console.stop()
        self.assertTrue((self.console.directory(first['run_id'])/'stop.request').is_file())
        self.assertEqual(self.console.status()['run']['status'], 'stopping')
        self.assertIsNone(self.factory.processes[0].returncode)
        self.assertEqual(len(self.factory.calls), 1)

    def test_exited_process_cannot_leave_a_fake_running_or_completed_status(self):
        first = self.start(); folder = self.console.directory(first['run_id'])
        (folder/'status.json').write_text(json.dumps(dict(status='running', phase='grasp')))
        self.factory.processes[0].returncode = 0
        result = self.console.snapshot(first['run_id'])
        self.assertEqual(result['status'], 'error')
        self.assertFalse(result['success'])
        (folder/'summary.json').write_text(json.dumps(dict(status='completed', success=False)))
        self.assertEqual(self.console.snapshot(first['run_id'])['status'], 'failed')

    def test_visual_completion_and_monitor_history_are_exposed_without_secrets(self):
        first = self.start(); folder = self.console.directory(first['run_id'])
        (folder/'summary.json').write_text(json.dumps(dict(status='completed', success=True,
            plan={'steps':[{'description':'白色大球','reason':'球类'}]}, completed_steps=[{'description':'白色大球'}],
            latest_judgement={'assessment':'placed','decision':'complete','reason':'球已在绿色区',
                              'api_key':'do-not-expose'}, monitor_stats={'dispatch_hz':1.,'result_hz':.9})))
        (folder/'judgements.jsonl').write_text(json.dumps(dict(frame_id=50, accepted=True,
            assessment='placed', decision='complete', reason='球已在绿色区'))+'\n')
        (folder/'execution.mp4').write_bytes(b'not-a-real-video-unit-fixture')
        self.assertNotIn('video', self.console.snapshot(first['run_id'])['artifacts'])
        self.factory.processes[0].returncode = 0
        result = self.console.snapshot(first['run_id'])
        self.assertTrue(result['success']); self.assertEqual(result['completed_count'], 1)
        self.assertEqual(result['latest_judgement']['api_key'], '[REDACTED]')
        self.assertEqual(result['judgements'][0]['frame_id'], 50)
        self.assertIn('video', result['artifacts'])

    def test_run_traversal_and_directory_symlinks_are_rejected(self):
        for value in ('../private', 'task_agent_../private', '/tmp/task_agent_a'):
            with self.assertRaises(ValueError):self.console.directory(value)
        link = self.console.runs_root/'task_agent_symlink'
        link.symlink_to(Path(self.temp.name), target_is_directory=True)
        with self.assertRaises(ValueError):self.console.directory(link.name)

    def test_runner_localization_and_nested_failure_details_are_exposed(self):
        first = self.start(); folder = self.console.directory(first['run_id'])
        (folder/'status.json').write_text(json.dumps(dict(status='executing',
            target_description='机械臂右侧的黄绿色网球', active_bbox=[147.392,119.84,160.384,136.64],
            plan=['收拾球类'], plan_summary='只收拾绿色收纳区外的球类',
            diagnostic_gt={'actual_object':'must-not-expose'})))
        rows = [dict(event='target_selected', target=dict(description='网球',
                    reason='球在收纳区外', bbox_xyxy=[147.392,119.84,160.384,136.64])),
                dict(event='grasp_finished', result=dict(reason='mask_lost_timeout', steps=158)),
                dict(event='subtask_retry', reason='mask_lost_timeout', attempt=3, remaining_attempts=0)]
        (folder/'events.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
        calls = [dict(request_id='first_plan', purpose='task_plan', response={'summary':'计划'})]
        calls += [dict(request_id=str(i), purpose='visual_supervision_1hz', response={'reason':'观察'})
                  for i in range(50)]
        (folder/'vlm_calls.jsonl').write_text('\n'.join(json.dumps(row) for row in calls)+'\n')
        result = self.console.snapshot(first['run_id'])
        self.assertEqual(result['target_description'], '机械臂右侧的黄绿色网球')
        self.assertEqual(result['active_bbox'], [147.392,119.84,160.384,136.64])
        self.assertEqual(result['plan_summary'], '只收拾绿色收纳区外的球类')
        self.assertEqual(result['events'][0]['target']['reason'], '球在收纳区外')
        self.assertEqual(result['events'][1]['result']['reason'], 'mask_lost_timeout')
        self.assertEqual(result['events'][2]['remaining_attempts'], 0)
        self.assertEqual(result['vlm_calls'][0]['request_id'], 'first_plan')
        self.assertNotIn('diagnostic_gt', result)

    def test_completed_cli_runs_are_visible_but_unmanaged_running_cli_is_not(self):
        finished = self.console.runs_root/'task_agent_validation_balls'
        finished.mkdir()
        (finished/'summary.json').write_text(json.dumps(dict(status='retry_budget_exhausted',
            instruction='收拾球类', policy='pi05', seed=17, success=False)))
        running = self.console.runs_root/'task_agent_unmanaged_running'
        running.mkdir()
        (running/'status.json').write_text(json.dumps(dict(status='executing')))
        rows = self.console.history()
        self.assertEqual([row['id'] for row in rows], [finished.name])
        self.assertEqual(rows[0]['status'], 'retry_budget_exhausted')
        self.assertFalse(rows[0]['success'])
        self.assertTrue(rows[0]['created_at'])


class HTTPContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.factory = ProcessFactory()
        self.console = Console(Path(self.temp.name)/'runs', process_factory=self.factory)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.console = self.console
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
        self.temp.cleanup()
    def get(self, path):
        with urlopen(self.url+path) as response:return json.load(response)
    def post(self, path, data, token=True, origin=None):
        headers = {'Content-Type':'application/json'}
        if token:headers['X-CSRF-Token'] = self.console.token
        if origin:headers['Origin'] = origin
        request = Request(self.url+path, data=json.dumps(data).encode(), headers=headers)
        with urlopen(request) as response:return json.load(response)

    def test_origin_and_session_guards_then_start_stop(self):
        config = self.get('/api/config')
        self.assertEqual(config['defaults']['monitor_hz'], 1)
        self.assertEqual(config['defaults']['max_subtasks'], 4)
        for token,origin in [(False,None),(True,'https://unrelated.example')]:
            with self.assertRaises(HTTPError) as error:
                self.post('/api/start', {'instruction':'收拾球类'}, token=token, origin=origin)
            self.assertEqual(error.exception.code, 403)
        result = self.post('/api/start', {'instruction':'收拾球类'})
        self.assertEqual(self.get('/api/status')['run']['id'], result['run_id'])
        self.assertEqual(self.get('/api/runs')['runs'][0]['instruction'], '收拾球类')
        self.assertTrue(self.post('/api/stop', {})['requested'])

    def test_artifacts_support_video_ranges_but_do_not_expose_private_files(self):
        result = self.post('/api/start', {'instruction':'收拾球类'})
        folder = self.console.directory(result['run_id'])
        (folder/'execution.mp4').write_bytes(b'0123456789')
        (folder/'private.key').write_text('do-not-serve')
        url = self.url+'/artifacts/'+result['run_id']+'/execution.mp4'
        request = Request(url, headers={'Range':'bytes=2-5'})
        with urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b'2345')
        with self.assertRaises(HTTPError):
            urlopen(self.url+'/artifacts/'+result['run_id']+'/private.key')
        with self.assertRaises(HTTPError):
            urlopen(self.url+'/artifacts/'+result['run_id']+'/%2e%2e%2fprivate.key')


@unittest.skipUnless(shutil.which('node'), 'Node is needed for browser state contract checks')
class RecordingUIContracts(unittest.TestCase):
    """Execute the real UI functions with media/DOM doubles, without a browser or GPU."""
    def run_js(self, checks):
        prefix = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8');
class Element {
 constructor(){this.children=[];this.attrs={};this.listeners={};this.textContent='';this.hidden=false;this.value='planning';this.currentTime=0;this.paused=true;this.srcWrites=0;this.loads=0;this.pauses=0;this.plays=0}
 append(...nodes){this.children.push(...nodes)}
 replaceChildren(...nodes){this.children=nodes;this.textContent=''}
 addEventListener(name,fn){(this.listeners[name]??=[]).push(fn)}
 dispatch(name){for(const fn of this.listeners[name]||[])fn()}
 removeAttribute(name){delete this.attrs[name]}
 getAttribute(name){return this.attrs[name]??null}
 set src(value){this.attrs.src=value;this.srcWrites++}
 get src(){return this.attrs.src||''}
 load(){this.loads++;this.currentTime=0}
 pause(){this.paused=true;this.pauses++}
 play(){this.plays++;if(this.rejectPlay)return Promise.reject(this.rejectPlay);this.paused=false;return Promise.resolve()}
 scrollIntoView(){this.scrolled=true}
}
const elements={},document={getElementById:id=>elements[id]??(elements[id]=new Element()),createElement:()=>new Element(),querySelectorAll:()=>[]};
const context={document,console,setTimeout,clearTimeout,history:async()=>{}};vm.createContext(context);
const source=html.split('<script>')[1].split('</script>')[0];
vm.runInContext(source.split('async function history(){')[0],context);
const call=(name,arg)=>{context.input=arg;return vm.runInContext(name+'(input)',context)};
const run=(id,options={})=>({id,instruction:'收拾球类',policy:'act',seed:17,status:'failed',busy:false,finished:true,steps:[],...options});
const scene=(id,options={})=>({id,seed:17,status:'scene_ready',ready:true,submitted:false,busy:true,has_frame:true,...options});
const catalog=()=>({ok:true,min_objects:1,max_objects:8,classic_object_ids:Array.from({length:8},(_,i)=>'item_'+i),objects:Array.from({length:10},(_,i)=>({id:'item_'+i,name:['红球','蓝球','白杯','红方块'][i]||'物品'+i,category:i<2?'球类':i===2?'杯具':'其他',act_trained:i===0||i===2,pi05_trained:i<5,available:true})).concat([{id:'missing',name:'缺失资产',category:'其他',available:false,act_trained:false,pi05_trained:false}])});
const video=id=>'/artifacts/'+id+'/execution.mp4';
'''
        script = prefix + '\n(async()=>{\n' + checks + '\n})().catch(e=>{console.error(e);process.exitCode=1});\n'
        page = Path(__file__).parent/'task_agent'/'web.html'
        result = subprocess.run([shutil.which('node'), '-', str(page)], input=script, text=True,
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_video_is_shown_automatically_for_all_finished_outcomes(self):
        self.run_js(r'''
call('showRun',null);assert(elements.recordingEmpty.textContent.includes('任务结束后'));assert(elements.recordingVideo.hidden);
call('showRun',run('task_agent_live',{busy:true,finished:false,artifacts:{video:video('task_agent_live')}}));
assert(elements.recordingVideo.hidden);assert(elements.recordingActions.hidden);assert(elements.recordingEmpty.textContent.includes('准备任务'));
for(const status of ['completed','retry_budget_exhausted','stopped','error','timeout']){
 const id='task_agent_'+status;call('showRun',run(id,{status,success:status==='completed',artifacts:{video:video(id),video_download:video(id)+'?download=1'}}));
 assert(!elements.recordingVideo.hidden,status);assert(!elements.recordingActions.hidden,status);assert(elements.recordingEmpty.hidden,status);assert.equal(elements.recordingVideo.src,video(id));assert.equal(elements.downloadVideo.href,video(id)+'?download=1');assert.equal(elements.downloadVideo.download,id+'.mp4');assert.equal(elements.recordingVideo.plays,0,'Showing a saved video must not force playback');
}
call('showRun',run('task_agent_no_video',{status:'error',artifacts:{}}));assert(elements.recordingVideo.hidden);assert(elements.recordingEmpty.textContent.includes('没有生成可用录像'));assert.equal(elements.downloadVideo.getAttribute('href'),null);
const tag=html.match(/<video id="recordingVideo"[^>]*>/)[0];assert(tag.includes('controls'));assert(tag.includes('preload="metadata"'));assert(!tag.includes('autoplay'));
''')

    def test_polling_preserves_playback_and_switching_clears_the_previous_run(self):
        self.run_js(r'''
const id='task_agent_saved',saved=run(id,{artifacts:{video:video(id)}});call('showRun',saved);const player=elements.recordingVideo;
player.currentTime=12.5;player.paused=false;const counts=[player.srcWrites,player.loads,player.pauses];
for(let i=0;i<6;i++)call('showRun',saved);
assert.equal(player.currentTime,12.5);assert(!player.paused);assert.deepEqual([player.srcWrites,player.loads,player.pauses],counts);assert.equal(elements.downloadVideo.href,video(id)+'?download=1');
call('clearRecording',id);assert.equal(player.currentTime,12.5,'Selecting the same run must not reset playback');
call('showRun',run('task_agent_new',{busy:true,finished:false,status:'queued'}));assert(player.paused);assert.equal(player.getAttribute('src'),null);assert(player.hidden);assert(elements.recordingActions.hidden);assert.equal(elements.downloadVideo.getAttribute('href'),null);
call('showRun',run('task_agent_new',{status:'stopped',artifacts:{video:video('task_agent_new')}}));assert.equal(player.src,video('task_agent_new'));assert.equal(player.currentTime,0);
call('clearRecording','task_agent_history');assert(player.paused);assert.equal(player.getAttribute('src'),null);assert(player.hidden);assert(elements.recordingEmpty.textContent.includes('读取本次任务'));
''')

    def test_replay_and_playback_errors_keep_a_visible_download_fallback(self):
        self.run_js(r'''
vm.runInContext('setupRecording()',context);const id='task_agent_stopped',saved=run(id,{status:'stopped',artifacts:{video:video(id)}});call('showRun',saved);const player=elements.recordingVideo;
player.currentTime=19;await elements.replayVideo.onclick();assert.equal(player.currentTime,0);assert.equal(player.plays,1);assert(!player.paused);
player.error={code:4};player.dispatch('error');assert(!elements.recordingError.hidden);assert(elements.recordingError.textContent.includes('下载录像'));assert(!elements.recordingActions.hidden);assert(elements.downloadVideo.href.includes('download=1'));
call('showRun',saved);assert(!elements.recordingError.hidden,'Polling must not erase a playback error');player.dispatch('playing');assert(elements.recordingError.hidden);
player.rejectPlay={name:'NotAllowedError'};await elements.replayVideo.onclick();assert(!elements.recordingError.hidden);assert(elements.recordingError.textContent.includes('播放键'));
call('clearRecording','task_agent_next');player.dispatch('error');assert(elements.recordingError.hidden,'A cleared old video must not add errors to the new task');
''')

    def test_old_inflight_status_cannot_restore_a_previous_runs_video(self):
        self.run_js(r'''
let resolve;context.pendingResponse=new Promise(done=>{resolve=done});vm.runInContext("selectedRun='task_agent_old';api=()=>pendingResponse",context);const pendingPoll=call('poll');
vm.runInContext("selectedRun='task_agent_new'",context);const next=run('task_agent_new',{artifacts:{video:video('task_agent_new')}});call('showRun',next);const player=elements.recordingVideo;player.currentTime=7;const writes=player.srcWrites;
resolve({busy:false,run:run('task_agent_old',{artifacts:{video:video('task_agent_old')}})});await pendingPoll;
assert.equal(player.src,video('task_agent_new'));assert.equal(player.currentTime,7);assert.equal(player.srcWrites,writes);
''')


@unittest.skipUnless(shutil.which('node'), 'Node is needed for browser state contract checks')
class MaskDropoutUIContracts(unittest.TestCase):
    run_js = RecordingUIContracts.run_js

    def test_automatic_defaults_start_payload_ranges_and_busy_lock(self):
        self.run_js(r'''
call('applyDefaults',{instruction:'收拾球类',policy:'act',seed:17,max_retries:2,max_subtasks:4,mask_dropout:{enabled:false,start_seconds:2,duration_seconds:1}});
context.screen={busy:true,active_run_id:'task_agent_ready',run:run('task_agent_ready',{busy:true,finished:false,phase:'scene_ready'}),scene:scene('task_agent_ready')};vm.runInContext("state=screen;selectedRun='task_agent_ready'",context);elements.instruction.value='收拾球类';call('controls');
assert.equal(elements.maskAutoEnabled.checked,false);assert(elements.maskAutoStart.disabled);assert(elements.maskAutoDuration.disabled);
const defaults=call('taskPayload');assert.equal(JSON.stringify(defaults.mask_dropout),JSON.stringify({enabled:false,start_seconds:2,duration_seconds:1}));
elements.maskAutoEnabled.checked=true;elements.maskAutoStart.value='120';elements.maskAutoDuration.value='30';call('controls');assert(!elements.maskAutoStart.disabled);
const requests=[];context.fetch=async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>({ok:true})}};vm.runInContext("token='test-csrf'",context);context.payload=call('taskPayload');await vm.runInContext("api('scene/task',payload)",context);
assert.equal(requests[0].url,'/api/scene/task');assert.equal(requests[0].options.headers['X-CSRF-Token'],'test-csrf');const posted=JSON.parse(requests[0].options.body);assert.deepEqual(posted.mask_dropout,{enabled:true,start_seconds:120,duration_seconds:30});assert.equal(posted.scene_id,'task_agent_ready');assert(!('seed' in posted));
for(const [field,value] of [['maskAutoStart','121'],['maskAutoStart',''],['maskAutoDuration','0.09'],['maskAutoDuration','31']]){elements.maskAutoStart.value='2';elements.maskAutoDuration.value='1';elements[field].value=value;assert.throws(()=>call('taskPayload'))}
context.screen={busy:true,active_run_id:'task_agent_active',run:run('task_agent_active',{busy:true,finished:false,phase:'grasp'})};vm.runInContext("state=screen;selectedRun='task_agent_history'",context);call('controls');assert(elements.maskAutoEnabled.disabled);assert(elements.maskAutoStart.disabled);assert(elements.maskAutoDuration.disabled);
''')

    def test_manual_requests_are_bound_to_selected_active_run_and_grasp_phase(self):
        self.run_js(r'''
const requests=[];context.fetch=async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>({ok:true,queued:true,command_id:'mask_command_1'})}};
context.screen={busy:true,active_run_id:'task_agent_active',run:run('task_agent_active',{busy:true,finished:false,phase:'planning'})};vm.runInContext("state=screen;selectedRun='task_agent_active';token='csrf-mask';poll=async()=>{}",context);document.getElementById('maskManualDuration').value='1.5';call('controls');assert(elements.maskTrigger.disabled);assert(!elements.maskRestore.disabled);
await call('sendMaskCommand','trigger');assert.equal(requests.length,0);await call('sendMaskCommand','restore');assert.equal(requests.length,1);assert.deepEqual(JSON.parse(requests[0].options.body),{run_id:'task_agent_active',action:'restore'});assert.equal(requests[0].url,'/api/mask_dropout');assert.equal(requests[0].options.headers['X-CSRF-Token'],'csrf-mask');
context.screen.run.phase='grasp';context.screen.run.mask_dropout_state={disabled:true,auto_enabled:false,ready:false};call('controls');assert(elements.maskTrigger.disabled);await call('sendMaskCommand','trigger');assert.equal(requests.length,1,'Preflight must finish before a trigger is sent');context.screen.run.mask_dropout_state.ready=true;call('controls');assert(!elements.maskTrigger.disabled,'Restore must not prevent a later manual trigger');await call('sendMaskCommand','trigger');assert.deepEqual(JSON.parse(requests[1].options.body),{run_id:'task_agent_active',action:'trigger',duration_seconds:1.5});
elements.maskManualDuration.value='0';await call('sendMaskCommand','trigger');assert.equal(requests.length,2);assert(!elements.maskCommandFeedback.hidden);assert(elements.maskCommandFeedback.textContent.includes('0.1–30'));
vm.runInContext("selectedRun='task_agent_history'",context);call('controls');assert(elements.maskTrigger.disabled);assert(elements.maskRestore.disabled);assert(elements.maskManualDuration.disabled);await call('sendMaskCommand','trigger');await call('sendMaskCommand','restore');assert.equal(requests.length,2,'History must not send commands to another active run');
''')

    def test_artificial_and_natural_mask_loss_have_distinct_status_and_events(self):
        self.run_js(r'''
const current=run('task_agent_active',{busy:true,finished:false,phase:'grasp',mask_dropout:{enabled:true,start_seconds:2,duration_seconds:1},mask_dropout_state:{auto_enabled:true,active:false,natural_mask_pixels:144,effective_mask_pixels:144,elapsed_grasp_seconds:1}});
call('showMaskDropout',current);assert(elements.maskDropoutBadge.textContent.includes('已安排'));assert.equal(elements.maskPixelCounts.textContent,'144 / 144 px');
current.mask_dropout_state={auto_enabled:true,active:true,source:'manual',remaining_seconds:.8,natural_mask_pixels:144,effective_mask_pixels:0,elapsed_grasp_seconds:3};call('showMaskDropout',current);assert(elements.maskDropoutBadge.textContent.includes('人工 Mask 丢失生效'));assert(elements.maskDropoutRemaining.textContent.includes('手动触发'));assert(elements.maskDropoutRemaining.textContent.includes('0.8'));assert(elements.maskDropoutDescription.textContent.includes('相机原始 RGB 保持不变'));assert.equal(elements.maskPixelCounts.textContent,'144 / 0 px');
current.tracking={state:'lost'};current.mask_dropout_state={auto_enabled:false,disabled:true,active:false,natural_mask_pixels:0,effective_mask_pixels:0};call('showMaskDropout',current);assert(elements.maskDropoutBadge.textContent.includes('已停止人工干扰'));assert(elements.maskDropoutDescription.textContent.includes('自然 Mask 丢失'));assert(!elements.maskDropoutDescription.textContent.includes('恢复成功'));
current.mask_dropout_state={auto_enabled:false,active:false,natural_mask_pixels:144,effective_mask_pixels:144};current.events=[{event:'mask_dropout_disabled'},{event:'mask_dropout_ignored',reason:'expired'},{event:'mask_dropout_started',source:'automatic',duration_seconds:1},{event:'post_subtask_goal_check',result:{complete:false,reason:'仍有一个球在收纳区外'}}];call('showMaskDropout',current);assert(elements.maskDropoutBadge.textContent.includes('已停止人工干扰'));call('showEvents',current);const flatten=el=>el.textContent+' '+el.children.map(flatten).join(' ');const events=flatten(elements.events);assert(events.includes('命令已过期'));assert(events.includes('自动定时干扰'));assert(events.includes('持续 1.0 秒'));assert(events.includes('完成子任务后检查整个任务'));assert(events.includes('总任务尚未通过完成确认'));assert(events.includes('仍有一个球在收纳区外'));
''')

    def test_late_manual_command_response_cannot_change_a_history_view(self):
        self.run_js(r'''
let resolve;context.pendingCommand=new Promise(done=>{resolve=done});context.screen={busy:true,active_run_id:'task_agent_active',run:run('task_agent_active',{busy:true,finished:false,phase:'grasp'})};vm.runInContext("state=screen;selectedRun='task_agent_active';api=()=>pendingCommand",context);document.getElementById('maskManualDuration').value='1';const pending=call('sendMaskCommand','trigger');assert(elements.maskTrigger.disabled);
vm.runInContext("selectedRun='task_agent_history'",context);call('controls');resolve({ok:true,queued:true,command_id:'late'});await pending;assert(elements.maskCommandFeedback.hidden);assert(elements.maskTrigger.disabled);assert(elements.maskRestore.disabled);
''')

    def test_single_target_guard_is_natural_loss_and_artificial_loss_has_priority(self):
        self.run_js(r'''
const current=run('task_agent_guard',{tracking:{state:'lost',single_target_guard:{accepted:false,reason:'multiple_components_or_tracking_ambiguity'}},mask_dropout_state:{active:false}});
call('showRun',current);assert.equal(elements.trackingState.textContent,'单目标保护：检测到多物体或跟踪歧义，按自然丢失处理');assert(!elements.trackingGuardDetails.hidden);assert.equal(elements.trackingGuardReason.textContent,'multiple_components_or_tracking_ambiguity');assert(!elements.trackingState.textContent.includes('人工'));
current.mask_dropout_state.active=true;call('showRun',current);assert.equal(elements.trackingState.textContent,'人工 Mask 丢失生效');assert(elements.trackingGuardDetails.hidden);
current.mask_dropout_state.active=false;current.tracking.single_target_guard.accepted=true;call('showRun',current);assert.equal(elements.trackingState.textContent,'自然 Mask 丢失');assert(elements.trackingGuardDetails.hidden);
''')

    def test_empty_mask_only_reports_natural_loss_during_ready_lost_grasp(self):
        self.run_js(r'''
const current=run('task_agent_empty',{busy:false,finished:true,phase:'goal_verify',tracking:{state:'lost'},mask_dropout_state:{active:false,ready:true,natural_mask_pixels:0,effective_mask_pixels:0}});
call('showMaskDropout',current);assert(!elements.maskDropoutDescription.textContent.includes('自然 Mask 丢失'),'A finished task must not interpret a cleared mask as loss');
current.busy=true;current.finished=false;current.phase='idle';call('showMaskDropout',current);assert(!elements.maskDropoutDescription.textContent.includes('自然 Mask 丢失'),'An idle zero mask is expected');
current.phase='grasp';current.mask_dropout_state.ready=false;call('showMaskDropout',current);assert(!elements.maskDropoutDescription.textContent.includes('自然 Mask 丢失'),'Preflight has not established tracking yet');
current.mask_dropout_state.ready=true;current.tracking.state='tracking';call('showMaskDropout',current);assert(!elements.maskDropoutDescription.textContent.includes('自然 Mask 丢失'),'Zero pixels alone must not override tracking state');
current.tracking.state='lost';call('showMaskDropout',current);assert(elements.maskDropoutDescription.textContent.includes('自然 Mask 丢失'));
current.mask_dropout_state.active=true;current.mask_dropout_state.source='manual';call('showMaskDropout',current);assert(!elements.maskDropoutDescription.textContent.includes('自然 Mask 丢失'));assert(elements.maskDropoutBadge.textContent.includes('人工 Mask 丢失生效'));
''')


@unittest.skipUnless(shutil.which('node'), 'Node is needed for browser state contract checks')
class SceneUIContracts(unittest.TestCase):
    run_js = RecordingUIContracts.run_js

    def test_reuse_requires_validated_initial_layout_and_visible_idle_source(self):
        self.run_js(r'''
call('controls');assert(elements.reuseScene.disabled);assert(elements.reuseSceneSource.textContent.includes('先生成或打开'));
context.screen={busy:false,active_run_id:null,scene:null,run:run('task_agent_history',{seed:912,initial_scene_available:false})};vm.runInContext("state=screen;selectedRun='task_agent_history'",context);call('controls');assert(elements.reuseScene.disabled);assert(elements.reuseSceneSource.textContent.includes('缺少可校验'));
context.screen.run.initial_scene_available=true;call('controls');assert(!elements.reuseScene.disabled);assert(elements.reuseSceneSource.textContent.includes('seed 912'));assert(elements.reuseSceneSource.textContent.includes('不继续动作后的现场'));
context.screen.busy=true;context.screen.active_run_id='task_agent_hidden';context.screen.scene=scene('task_agent_hidden');call('controls');assert(elements.reuseScene.disabled);let posts=0;context.fetch=async()=>{posts++;return {ok:true,json:async()=>({ok:true})}};await call('prepareScene','current');assert.equal(posts,0,'A historical source must not close another waiting scene');
context.screen.scene=scene('task_agent_history',{submitted:true,ready:false});context.screen.active_run_id='task_agent_history';context.screen.run.busy=true;context.screen.run.finished=false;call('controls');assert(elements.reuseScene.disabled);await call('prepareScene','current');assert.equal(posts,0,'Executing tasks must not be interrupted by reuse');
''')

    def test_reuse_history_posts_source_without_seed_and_returns_to_blank_preview(self):
        self.run_js(r'''
call('applyDefaults',{policy:'act',max_retries:2,max_subtasks:4});context.screen={busy:false,active_run_id:null,scene:null,run:run('task_agent_source',{seed:813,initial_scene_available:true,instruction:'旧任务'})};vm.runInContext("state=screen;selectedRun='task_agent_source'",context);elements.randomSeed.checked=false;elements.seed.value='invalid seed must be ignored';elements.instruction.value='旧任务';const requests=[],id='task_agent_comparison';const comparison={busy:true,active_run_id:id,scene:scene(id,{seed:813,generation_mode:'current',comparison_source_scene_id:'task_agent_source',comparison_initial_layout_verified:true}),run:run(id,{seed:813,busy:true,finished:false,status:'scene_ready',phase:'scene_ready',instruction:'',generation_mode:'current',comparison_source_scene_id:'task_agent_source',comparison_initial_layout_verified:true,initial_scene_available:true})};
context.fetch=async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>url==='/api/scene'?{ok:true,scene_id:id,run_id:id,run:comparison.run,scene:comparison.scene}:comparison}};
await call('prepareScene','current');assert.equal(requests[0].url,'/api/scene');assert.deepEqual(JSON.parse(requests[0].options.body),{mode:'current',source_scene_id:'task_agent_source'});assert(!requests.some(row=>row.url==='/api/scene/task'||row.url==='/api/start'||row.url==='/api/stop'));assert.equal(elements.instruction.value,'');assert(!elements.instruction.disabled);assert(elements.start.disabled);assert(!elements.comparisonSceneStatus.hidden);assert.equal(elements.comparisonSceneStatus.textContent,'对比场景 · 初始布局已校验一致');
''')

    def test_reuse_waiting_scene_closes_it_before_generating_the_comparison(self):
        self.run_js(r'''
const old='task_agent_waiting_source',next='task_agent_compare';context.screen={busy:true,active_run_id:old,scene:scene(old),run:run(old,{seed:738,busy:true,finished:false,phase:'scene_ready',initial_scene_available:true})};vm.runInContext("state=screen;selectedRun='task_agent_waiting_source'",context);call('controls');assert(!elements.reuseScene.disabled);const requests=[];
context.fetch=async(url,options)=>{requests.push({url,options});let result;if(url==='/api/scene/close')result={ok:true,requested:true};else if(url==='/api/status?run_id='+old)result={busy:false,active_run_id:old,scene:scene(old,{ready:false,busy:false}),run:run(old,{seed:738,initial_scene_available:true})};else if(url==='/api/scene')result={ok:true,scene_id:next,run_id:next,scene:scene(next,{ready:false,status:'preparing_scene',generation_mode:'current',comparison_initial_layout_verified:false}),run:run(next,{busy:true,finished:false,status:'preparing_scene',phase:'preparing_scene',generation_mode:'current',comparison_initial_layout_verified:false})};else result={busy:true,active_run_id:next,scene:scene(next,{ready:false,status:'preparing_scene',generation_mode:'current',comparison_initial_layout_verified:false}),run:run(next,{busy:true,finished:false,status:'preparing_scene',phase:'preparing_scene',generation_mode:'current',comparison_initial_layout_verified:false})};return {ok:true,json:async()=>result}};
await call('prepareScene','current');assert.equal(requests[0].url,'/api/scene/close');assert.deepEqual(JSON.parse(requests[0].options.body),{scene_id:old});const clone=requests.find(row=>row.url==='/api/scene');assert(clone);assert.deepEqual(JSON.parse(clone.options.body),{mode:'current',source_scene_id:old});assert(!requests.some(row=>row.url==='/api/stop'));assert(elements.comparisonSceneStatus.textContent.includes('正在校验'));assert(!elements.comparisonSceneStatus.textContent.includes('已校验一致'));assert(elements.start.disabled);
''')

    def test_preview_history_has_a_scene_label_without_claiming_a_loaded_policy(self):
        self.run_js(r'''
const description=call('historyDescription',{instruction:'',scene_first:true,task_submitted:false,policy:'pi05',seed:938,status:'stopped',created_at:'2026-11-04T15:00:00+08:00'});assert.equal(description.title,'预览场景（未提交任务）');assert(description.detail.includes('场景预览 · seed 938'));assert(!description.detail.includes('PI05'));
const task=call('historyDescription',{instruction:'收拾球类',scene_first:true,task_submitted:true,policy:'act',seed:938,status:'completed',completed_count:2,total_steps:2,created_at:'2026-11-04T15:00:00+08:00'});assert.equal(task.title,'收拾球类');assert(task.detail.includes('ACT'));assert(!task.detail.includes('场景预览'));
''')

    def test_empty_initial_prompt_ready_gate_and_preview_controls(self):
        self.run_js(r'''
call('applyDefaults',{instruction:'A legacy default must not become the new task',policy:'pi05',max_retries:2,max_subtasks:4});call('applySceneDefaults',{random_seed:true,seed:17});assert.equal(elements.instruction.value,'');assert(elements.instruction.disabled);assert(elements.start.disabled);assert(elements.randomSeed.checked);assert(elements.seed.disabled);assert(!elements.prepareScene.disabled);
const initial=html.match(/<textarea id="instruction"[^>]*>([\s\S]*?)<\/textarea>/);assert.equal(initial[1],'');
context.screen={busy:true,active_run_id:'task_agent_scene',run:run('task_agent_scene',{busy:true,finished:false,phase:'preparing_scene'}),scene:scene('task_agent_scene',{ready:false,status:'preparing_scene'})};vm.runInContext("state=screen;selectedRun='task_agent_scene'",context);call('controls');assert(elements.instruction.disabled);assert(!elements.policy.disabled,'Waiting worker must allow task configuration');assert(elements.maskRestore.disabled);
context.screen.scene.ready=true;context.screen.scene.status='scene_ready';context.screen.run.phase='scene_ready';call('controls');assert(!elements.instruction.disabled);assert(elements.start.disabled);elements.instruction.value='收拾球类';call('controls');assert(!elements.start.disabled);assert.equal(elements.stop.textContent,'关闭场景');assert.equal(elements.prepareScene.textContent,'换一个场景');assert(!elements.randomSeed.disabled);elements.randomSeed.checked=false;call('controls');assert(!elements.seed.disabled,'Seed settings apply to the next scene and stay editable while waiting');
call('showRecording',context.screen.run);assert(elements.recordingEmpty.textContent.includes('预览场景不会执行抓取'));assert(elements.recordingVideo.hidden);assert(elements.maskTrigger.disabled);assert(elements.maskRestore.disabled);
''')

    def test_prepare_posts_seed_and_classic_objects_without_a_task(self):
        self.run_js(r'''
call('applyDefaults',{policy:'pi05',max_retries:2,max_subtasks:4});call('applySceneDefaults',{random_seed:true,seed:17});const requests=[],id='task_agent_random';const screen={ok:true,busy:true,active_run_id:id,scene:scene(id,{seed:938}),run:run(id,{seed:938,busy:true,finished:false,phase:'scene_ready',has_frame:true,instruction:''})};
context.fetch=async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>url==='/api/scene'?{ok:true,scene_id:id,run_id:id,scene:screen.scene,run:screen.run}:screen}};
elements.instruction.value='old draft';await call('prepareScene');assert.deepEqual(JSON.parse(requests[0].options.body),{seed:null,objects:{mode:'classic'}});assert.equal(requests[0].url,'/api/scene');assert(!requests.some(row=>row.url==='/api/start'||row.url==='/api/scene/task'));assert.equal(elements.instruction.value,'');assert(elements.actualSceneSeed.textContent.includes('938'));assert(!elements.instruction.disabled);
elements.randomSeed.checked=false;elements.seed.value='0';assert.equal(call('sceneSeed'),0);elements.seed.value='17';assert.equal(call('sceneSeed'),17);for(const value of ['','1.5','-1','2147483648']){elements.seed.value=value;assert.throws(()=>call('sceneSeed'))}
''')

    def test_submit_uses_the_existing_scene_without_seed_or_reset(self):
        self.run_js(r'''
call('applyDefaults',{policy:'act',max_retries:2,max_subtasks:4});const id='task_agent_same_scene';context.screen={busy:true,active_run_id:id,scene:scene(id,{seed:245}),run:run(id,{seed:245,busy:true,finished:false,phase:'scene_ready'})};vm.runInContext("state=screen;selectedRun='task_agent_same_scene'",context);elements.instruction.value='先收拾小球，再收拾大球';const requests=[];
const submitted={...context.screen,scene:scene(id,{seed:245,ready:false,submitted:true,status:'starting_task'}),run:run(id,{seed:245,busy:true,finished:false,status:'starting_task',phase:'starting_task',instruction:elements.instruction.value,policy:'act'})};
context.fetch=async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>url==='/api/scene/task'?{ok:true,run_id:id,run:submitted.run,scene:submitted.scene}:submitted}};
await call('submitSceneTask');assert.equal(requests[0].url,'/api/scene/task');const payload=JSON.parse(requests[0].options.body);assert.equal(payload.scene_id,id);assert.equal(payload.instruction,'先收拾小球，再收拾大球');assert(!('seed' in payload));assert(!requests.some(row=>row.url==='/api/scene'||row.url==='/api/start'));assert(elements.start.disabled);assert(elements.prepareScene.disabled);assert(elements.randomSeed.disabled);assert(elements.seed.disabled);assert.equal(elements.stop.textContent,'停止当前任务');const count=requests.length;await call('submitSceneTask');assert.equal(requests.length,count,'The same prepared scene may only receive one task');
submitted.busy=false;submitted.run.busy=false;submitted.run.finished=true;submitted.run.status='completed';context.screen=submitted;vm.runInContext('state=screen',context);call('controls');assert(elements.start.disabled);assert(!elements.prepareScene.disabled,'A finished task requires a new scene');
''')

    def test_history_cannot_submit_and_return_button_recovers_current_scene(self):
        self.run_js(r'''
const id='task_agent_current';context.screen={busy:true,active_run_id:id,scene:scene(id),run:run('task_agent_history')};vm.runInContext("state=screen;selectedRun='task_agent_history'",context);call('controls');assert(elements.instruction.disabled);assert(elements.start.disabled);assert(!elements.returnScene.hidden);assert(elements.maskRestore.disabled);let requests=0;context.fetch=async()=>{requests++;return {ok:true,json:async()=>({busy:true,active_run_id:id,scene:scene(id),run:run(id,{busy:true,finished:false,phase:'scene_ready'})})}};
await call('submitSceneTask');assert.equal(requests,0);await call('returnToScene');assert.equal(requests,1);assert(elements.returnScene.hidden);assert(!elements.instruction.disabled);assert(elements.sceneStatus.textContent.includes('沿用')||elements.sceneStatus.textContent.includes('不重新生成'));
''')

    def test_replacement_closes_only_waiting_scene_and_waits_before_preparing(self):
        self.run_js(r'''
call('applySceneDefaults',{random_seed:true});const old='task_agent_old_scene',next='task_agent_new_scene';context.screen={busy:true,active_run_id:old,scene:scene(old),run:run(old,{busy:true,finished:false,phase:'scene_ready'})};vm.runInContext("state=screen;selectedRun='task_agent_old_scene'",context);const requests=[];let polls=0;
context.fetch=async(url,options)=>{requests.push({url,options});let data;if(url==='/api/scene/close')data={ok:true,requested:true};else if(url==='/api/status?run_id='+old){polls++;data={busy:polls===1,active_run_id:old,scene:scene(old,{ready:false,busy:polls===1}),run:run(old,{busy:polls===1,finished:polls!==1})}}else if(url==='/api/scene')data={ok:true,scene_id:next,run_id:next,scene:scene(next),run:run(next,{busy:true,finished:false,phase:'scene_ready'})};else data={busy:true,active_run_id:next,scene:scene(next),run:run(next,{busy:true,finished:false,phase:'scene_ready'})};return {ok:true,json:async()=>data}};
await call('prepareScene');assert.equal(requests[0].url,'/api/scene/close');assert.deepEqual(JSON.parse(requests[0].options.body),{scene_id:old});const prepare=requests.findIndex(row=>row.url==='/api/scene');assert(prepare>=3,'The old worker must report not busy before prepare');assert(!requests.some(row=>row.url==='/api/stop'));assert.equal(elements.instruction.value,'');assert(elements.actualSceneSeed.textContent.includes('17'));
''')

    def test_replacement_aborts_if_another_active_run_appears(self):
        self.run_js(r'''
call('applySceneDefaults',{random_seed:true});const old='task_agent_old_scene';context.screen={busy:true,active_run_id:old,scene:scene(old),run:run(old,{busy:true,finished:false,phase:'scene_ready'})};vm.runInContext("state=screen;selectedRun='task_agent_old_scene'",context);const requests=[];
context.fetch=async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>url==='/api/scene/close'?{ok:true,requested:true}:{busy:true,active_run_id:'task_agent_other',scene:scene('task_agent_other',{submitted:true}),run:run('task_agent_other',{busy:true,finished:false})}}};await call('prepareScene');assert.equal(requests.filter(row=>row.url==='/api/scene/close').length,1);assert(!requests.some(row=>row.url==='/api/scene'||row.url==='/api/stop'));assert(elements.error.textContent.includes('运行已变化'));
''')

    def test_waiting_for_close_has_a_bounded_deadline(self):
        self.run_js(r'''
let requests=0;context.fetch=async()=>{requests++;return {ok:true,json:async()=>({busy:true,active_run_id:'task_agent_busy',scene:scene('task_agent_busy')})}};
await assert.rejects(vm.runInContext("waitForSceneClose('task_agent_busy',10,1)",context),/关闭|超时/);assert(requests>0);assert(requests<30);
''')

    def test_waiting_stop_closes_scene_but_submitted_task_uses_stop(self):
        self.run_js(r'''
const requests=[];context.fetch=async(url,options)=>{requests.push({url,options});if(options.method==='POST')return {ok:true,json:async()=>({ok:true,requested:true})};return {ok:true,json:async()=>context.screen}};
context.screen={busy:true,active_run_id:'task_agent_waiting',scene:scene('task_agent_waiting'),run:run('task_agent_waiting',{busy:true,finished:false,phase:'scene_ready'})};vm.runInContext("state=screen;selectedRun='task_agent_waiting'",context);await call('stopSceneOrTask');assert.equal(requests[0].url,'/api/scene/close');assert.deepEqual(JSON.parse(requests[0].options.body),{scene_id:'task_agent_waiting'});
context.screen={busy:true,active_run_id:'task_agent_running',scene:scene('task_agent_running',{ready:false,submitted:true}),run:run('task_agent_running',{busy:true,finished:false,phase:'grasp'})};vm.runInContext("state=screen;selectedRun='task_agent_running'",context);await call('stopSceneOrTask');assert(requests.some(row=>row.url==='/api/stop'));assert.equal(requests.filter(row=>row.url==='/api/scene/close').length,1);
''')

    def test_inflight_ready_poll_cannot_reopen_submission_after_task_acceptance(self):
        self.run_js(r'''
call('applyDefaults',{policy:'act',max_retries:2,max_subtasks:4});const id='task_agent_same';const preview={busy:true,active_run_id:id,scene:scene(id),run:run(id,{busy:true,finished:false,phase:'scene_ready'})};context.screen=preview;vm.runInContext("state=screen;selectedRun='task_agent_same'",context);elements.instruction.value='收拾球类';let resolve,first=true;const delayed=new Promise(done=>{resolve=done});const submitted={busy:true,active_run_id:id,scene:scene(id,{ready:false,submitted:true,status:'starting_task'}),run:run(id,{busy:true,finished:false,phase:'starting_task'})};
context.fetch=async(url,options)=>{if(url==='/api/scene/task')return {ok:true,json:async()=>({ok:true,run_id:id,scene:submitted.scene,run:submitted.run})};if(first){first=false;await delayed;return {ok:true,json:async()=>preview}}return {ok:true,json:async()=>submitted}};const oldPoll=call('poll');await call('submitSceneTask');resolve();await oldPoll;assert(elements.start.disabled);assert(elements.instruction.disabled);assert(elements.sceneStatus.textContent.includes('正在使用当前场景'));
''')


@unittest.skipUnless(shutil.which('node'), 'Node is needed for browser state contract checks')
class SceneObjectUIContracts(unittest.TestCase):
    run_js = RecordingUIContracts.run_js

    def test_catalog_loads_available_named_cards_and_microtuning_labels(self):
        self.run_js(r'''
call('applySceneDefaults',{objects:{mode:'classic'}});const requests=[];context.fetch=async(url)=>{requests.push(url);return {ok:true,json:async()=>catalog()}};await call('loadObjectCatalog');assert.deepEqual(requests,['/api/object_catalog']);assert(!elements.objectModeRandom.disabled);assert(!elements.objectModeManual.disabled);assert(elements.objectCatalogStatus.textContent.includes('10 类'));const flatten=el=>el.textContent+' '+el.children.map(flatten).join(' ');const choices=flatten(elements.objectChoices);assert(choices.includes('红球'));assert(choices.includes('ACT 训练目标'));assert(choices.includes('π0.5 微调目标'));assert(choices.includes('π0.5 微调留出'));assert(!choices.includes('缺失资产'));assert.equal(JSON.stringify(call('sceneObjectSelection')),JSON.stringify({mode:'classic'}));
''')

    def test_catalog_failure_keeps_classic_mode_available(self):
        self.run_js(r'''
call('applySceneDefaults',{objects:{mode:'classic'}});context.fetch=async()=>({ok:false,json:async()=>({error:'Catalog is unavailable'})});await call('loadObjectCatalog');assert(elements.objectModeRandom.disabled);assert(elements.objectModeManual.disabled);assert(elements.objectCatalogStatus.textContent.includes('仍可使用经典'));assert(!elements.retryObjectCatalog.hidden);assert.equal(JSON.stringify(call('sceneObjectSelection')),JSON.stringify({mode:'classic'}));elements.sceneObjectMode.value='manual';assert.throws(()=>call('sceneObjectSelection'),/目录/);
''')

    def test_manual_selection_is_bounded_searchable_and_retained_through_filters(self):
        self.run_js(r'''
call('applySceneDefaults',{objects:{mode:'manual'}});call('installObjectCatalog',catalog());vm.runInContext('setupSceneObjects()',context);elements.sceneObjectMode.onchange();assert(elements.manualObjectPicker.open);for(let i=0;i<8;i++)assert(call('chooseSceneObject',{id:'item_'+i,checked:true}));assert(!call('chooseSceneObject',{id:'item_8',checked:true}));assert(!call('chooseSceneObject',{id:'missing',checked:true}));assert(elements.objectSelectionError.textContent.includes('最多选择 8'));assert(elements.manualObjectCount.textContent.includes('8 / 8'));const selection=call('sceneObjectSelection');assert.equal(selection.mode,'manual');assert.equal(selection.object_ids.length,8);
elements.objectSearch.value='球';elements.objectCategory.value='球类';call('renderObjectChoices',true);const flatten=el=>el.textContent+' '+el.children.map(flatten).join(' ');const names=flatten(elements.objectChoices);assert(names.includes('红球')&&names.includes('蓝球'));assert(!names.includes('白杯'));assert.equal(call('sceneObjectSelection').object_ids.length,8,'Filtering must not remove selected objects');for(let i=0;i<8;i++)call('chooseSceneObject',{id:'item_'+i,checked:false});assert.throws(()=>call('sceneObjectSelection'),/选择/);
''')

    def test_random_refresh_posts_a_new_scene_even_with_the_same_layout_seed(self):
        self.run_js(r'''
call('applySceneDefaults',{random_seed:false,seed:17,objects:{mode:'random',count:3,pool:'all'}});call('installObjectCatalog',catalog());vm.runInContext('setupSceneObjects()',context);const requests=[];let current=null,closed=false,index=0;
context.fetch=async(url,options)=>{requests.push({url,options});let result;if(url==='/api/scene/close'){closed=true;result={ok:true,requested:true}}else if(url==='/api/scene'){const id='task_agent_objects_'+(++index);closed=false;current={busy:true,active_run_id:id,scene:scene(id,{selected_object_ids:['item_'+index],object_names:[index===1?'红球':'白杯']}),run:run(id,{busy:true,finished:false,phase:'scene_ready',has_frame:true,instruction:'',selected_object_ids:['item_'+index],object_names:[index===1?'红球':'白杯']})};result={ok:true,run_id:id,scene_id:id,run:current.run,scene:current.scene}}else result=closed?{...current,busy:false,scene:{...current.scene,ready:false,busy:false},run:{...current.run,busy:false,finished:true}}:current;return {ok:true,json:async()=>result}};
await elements.refreshObjects.onclick();await elements.refreshObjects.onclick();const posts=requests.filter(row=>row.url==='/api/scene');assert.equal(posts.length,2);for(const row of posts)assert.deepEqual(JSON.parse(row.options.body),{seed:17,objects:{mode:'random',count:3,pool:'all'}});assert.equal(requests.filter(row=>row.url==='/api/scene/close').length,1);assert(elements.actualSceneObjects.textContent.includes('白杯'));assert(!elements.actualSceneObjects.textContent.includes('红球'));assert(!requests.some(row=>row.url==='/api/scene/task'||row.url==='/api/start'));assert.equal(elements.instruction.value,'');
''')

    def test_manual_draft_stays_separate_from_actual_scene_and_task_payload(self):
        self.run_js(r'''
call('applyDefaults',{policy:'act',max_retries:2,max_subtasks:4});call('applySceneDefaults',{objects:{mode:'manual'}});call('installObjectCatalog',catalog());call('chooseSceneObject',{id:'item_2',checked:true});context.screen={busy:true,active_run_id:'task_agent_actual',scene:scene('task_agent_actual',{object_names:['红球','红方块']}),run:run('task_agent_actual',{busy:true,finished:false,phase:'scene_ready',object_names:['红球','红方块']})};vm.runInContext("state=screen;selectedRun='task_agent_actual'",context);elements.instruction.value='收拾球类';call('controls');assert(elements.objectDraftSummary.textContent.includes('白杯'));assert(elements.actualSceneObjects.textContent.includes('红球'));assert(!elements.actualSceneObjects.textContent.includes('白杯'));const payload=call('taskPayload');assert(!('objects' in payload));assert(!('object_ids' in payload));
context.screen.scene.submitted=true;context.screen.scene.ready=false;context.screen.run.phase='grasp';call('controls');assert(elements.sceneObjectMode.disabled);assert(elements.refreshObjects.disabled);assert(!call('chooseSceneObject',{id:'item_3',checked:true}));assert.equal(call('sceneObjectSelection').object_ids.length,1);
''')

    def test_manual_generation_posts_selected_ids_and_closes_the_picker(self):
        self.run_js(r'''
call('applySceneDefaults',{random_seed:true,objects:{mode:'manual'}});call('installObjectCatalog',catalog());let requests=[];context.fetch=async()=>{throw Error('Invalid selection must not send any request')};await call('prepareScene');assert(elements.error.textContent.includes('手动选择'));
call('chooseSceneObject',{id:'item_0',checked:true});call('chooseSceneObject',{id:'item_2',checked:true});document.getElementById('manualObjectPicker').open=true;const id='task_agent_manual';const ready={busy:true,active_run_id:id,scene:scene(id,{selected_object_ids:['item_0','item_2'],object_names:['红球','白杯']}),run:run(id,{busy:true,finished:false,phase:'scene_ready',object_names:['红球','白杯']})};context.fetch=async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>url==='/api/scene'?{ok:true,run_id:id,scene_id:id,scene:ready.scene,run:ready.run}:ready}};
await call('prepareScene');assert.deepEqual(JSON.parse(requests[0].options.body),{seed:null,objects:{mode:'manual',object_ids:['item_0','item_2']}});assert(!elements.manualObjectPicker.open);assert(elements.actualSceneObjects.textContent.includes('红球')&&elements.actualSceneObjects.textContent.includes('白杯'));assert.equal(elements.instruction.value,'');
''')

    def test_reuse_ignores_invalid_object_draft_and_restores_source_combination(self):
        self.run_js(r'''
call('applySceneDefaults',{random_seed:false,seed:17,objects:{mode:'manual'}});call('installObjectCatalog',catalog());elements.seed.value='invalid';document.getElementById('manualObjectPicker').open=true;context.screen={busy:false,active_run_id:null,scene:null,run:run('task_agent_source_objects',{initial_scene_available:true,object_names:['蓝球','白杯']})};vm.runInContext("state=screen;selectedRun='task_agent_source_objects'",context);const requests=[],id='task_agent_restored_objects';const restored={busy:true,active_run_id:id,scene:scene(id,{generation_mode:'current',comparison_initial_layout_verified:true,object_names:['蓝球','白杯']}),run:run(id,{busy:true,finished:false,phase:'scene_ready',instruction:'',generation_mode:'current',comparison_initial_layout_verified:true,initial_scene_available:true,object_names:['蓝球','白杯']})};
context.fetch=async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>url==='/api/scene'?{ok:true,run_id:id,scene_id:id,scene:restored.scene,run:restored.run}:restored}};await call('prepareScene','current');assert.deepEqual(JSON.parse(requests[0].options.body),{mode:'current',source_scene_id:'task_agent_source_objects'});assert(!elements.manualObjectPicker.open);assert(elements.actualSceneObjects.textContent.includes('蓝球')&&elements.actualSceneObjects.textContent.includes('白杯'));assert.equal(elements.instruction.value,'');assert(elements.reuseSceneSource.textContent.includes('忽略当前物品选择'));
''')


if __name__ == '__main__':
    unittest.main()
