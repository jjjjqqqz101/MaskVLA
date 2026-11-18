"""Three-policy browser contracts, using the real page and CPU-only DOM doubles."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from task_agent.policies import policy_options


@unittest.skipUnless(shutil.which('node'), 'Node is needed for browser state contracts')
class TripleModelUIContracts(unittest.TestCase):
    def run_js(self, checks):
        prefix = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8');
class Element {
 constructor(){this.children=[];this.attrs={};this.listeners={};this.textContent='';this.hidden=false;this.value='';this.currentTime=0;this.paused=true}
 append(...nodes){this.children.push(...nodes)}
 replaceChildren(...nodes){this.children=nodes;this.textContent=''}
 addEventListener(name,fn){(this.listeners[name]??=[]).push(fn)}
 removeAttribute(name){delete this.attrs[name]}
 getAttribute(name){return this.attrs[name]??null}
 set src(value){this.attrs.src=value}
 get src(){return this.attrs.src||''}
 load(){this.currentTime=0}
 pause(){this.paused=true}
 play(){this.paused=false;return Promise.resolve()}
 scrollIntoView(){}
}
const elements={},document={getElementById:id=>elements[id]??(elements[id]=new Element()),createElement:()=>new Element(),querySelectorAll:()=>[]};
const context={document,console,setTimeout,clearTimeout};vm.createContext(context);
const source=html.split('<script>')[1].split('</script>')[0];
new vm.Script(source); // Parse the entire shipped script, including startup.
vm.runInContext(source.split('async function history(){')[0],context);
const call=(name,arg)=>{context.input=arg;return vm.runInContext(name+'(input)',context)};
const evaluate=source=>vm.runInContext(source,context);
const flatten=element=>element.textContent+' '+element.children.map(flatten).join(' ');
const run=(id,options={})=>({id,instruction:'收拾球类',policy:'pi05',seed:17,status:'completed',busy:false,finished:true,completed_count:1,total_steps:1,steps:[],...options});
const scene=(id,options={})=>({id,seed:17,status:'scene_ready',ready:true,submitted:false,busy:true,...options});
'''
        script = prefix + '\nconst options=' + json.dumps(policy_options(), ensure_ascii=False)
        script += ';\n(async()=>{\n' + checks + '\n})().catch(e=>{console.error(e);process.exitCode=1});\n'
        page = Path(__file__).parent/'task_agent'/'web.html'
        result = subprocess.run([shutil.which('node'), '-', str(page)], input=script,
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_picker_metadata_inputs_and_default_stay_distinct(self):
        self.run_js(r'''
call('installPolicyOptions',options);assert.equal(elements.policy.value,'pi05');
assert.deepEqual(elements.policy.children.map(item=>item.value),['pi05','act','act_triple']);
assert(elements.policy.children.find(item=>item.value==='act_triple').textContent.includes('160000'));
assert(elements.policy.children.find(item=>item.value==='act_triple').textContent.includes('25目标'));
assert(elements.policySummary.textContent.includes('WristRGB'));assert(elements.policySummary.textContent.includes('8 维状态'));assert(elements.policySummary.textContent.includes('25 个微调目标 × 每目标 50 条'));
elements.policy.value='act';call('showPolicySettings');assert(elements.policySummary.textContent.includes('BaseRGB + BaseMaskedRGB'));assert(!elements.policySummary.textContent.includes('WristRGB'));assert(elements.policySummary.textContent.includes('8 个训练目标'));
elements.policy.value='act_triple';call('showPolicySettings');assert(elements.policySummary.textContent.includes('BaseRGB + WristRGB + BaseMaskedRGB'));assert(elements.policySummary.textContent.includes('25 个训练目标 × 每目标 50 条'));
const rows=elements.policyComparison.children;assert.equal(rows.length,3);assert(!flatten(rows[1]).includes('WristRGB'));assert(flatten(rows[2]).includes('WristRGB'));assert(flatten(rows[0]).includes('8 维状态'));
call('applyDefaults',{policy:'pi05',max_retries:2,max_subtasks:4});assert.equal(elements.policy.value,'pi05');assert(elements.policy.disabled);
assert(/<option value="pi05" selected>/.test(html));assert(html.includes('选择模型只修改待提交设置，不改变当前场景'));
''')

    def test_changing_model_only_changes_draft_then_submits_to_same_scene(self):
        self.run_js(r'''
call('installPolicyOptions',options);call('applyDefaults',{policy:'pi05',max_retries:2,max_subtasks:4});
const id='task_agent_model_draft',current={busy:true,active_run_id:id,scene:scene(id,{seed:813,object_names:['蓝球','白杯']}),run:run(id,{seed:813,object_names:['蓝球','白杯'],instruction:'',busy:true,finished:false,status:'scene_ready',initial_scene_available:true})};
context.current=current;evaluate("state=current;selectedRun=current.run.id;setupScene()");call('controls');
const before=JSON.stringify(evaluate('state')),requests=[];context.fetch=async(url,init)=>{requests.push({url,init});return {ok:true,json:async()=>current}};
elements.policy.value='act_triple';elements.policy.onchange();assert.equal(JSON.stringify(evaluate('state')),before);assert.equal(requests.length,0);assert(elements.actualSceneObjects.textContent.includes('蓝球'));assert(elements.actualSceneSeed.textContent.includes('813'));
elements.instruction.value='收拾球类';call('controls');assert(!elements.start.disabled);assert.equal(call('taskPayload').policy,'act_triple');assert.equal(call('taskPayload').scene_id,id);assert(!('objects' in call('taskPayload')));
const submitted={...current,scene:{...current.scene,submitted:true,ready:false},run:{...current.run,instruction:'收拾球类',policy:'act_triple',task_submitted:true,status:'starting_task'}};
context.fetch=async(url,init)=>{requests.push({url,init});return {ok:true,json:async()=>url==='/api/scene/task'?{ok:true,run_id:id,scene:submitted.scene,run:submitted.run}:submitted}};context.history=async()=>{};
await call('submitSceneTask');assert.equal(requests.filter(row=>row.url==='/api/scene/task').length,1);const payload=JSON.parse(requests.find(row=>row.url==='/api/scene/task').init.body);assert.equal(payload.policy,'act_triple');assert.equal(payload.scene_id,id);assert.equal(payload.monitor_hz,1);assert(!requests.some(row=>row.url==='/api/scene'||row.url==='/api/start'));assert(elements.policy.disabled);assert(elements.runLabel.textContent.includes('三路 ACT'));
''')

    def test_history_and_run_headers_use_recorded_model_names(self):
        self.run_js(r'''
call('installPolicyOptions',options);
for(const [policy,label,step] of [['act','双路 ACT','100000'],['act_triple','三路 ACT','160000'],['pi05','π0.5','49999']]){
 const record=run('task_agent_'+policy,{policy});call('showRun',record);assert(elements.runLabel.textContent.includes(label));assert(elements.runLabel.textContent.includes(step));assert(!elements.runLabel.textContent.includes('ACT_TRIPLE'));const description=call('historyDescription',record);assert(description.detail.includes(label));assert(description.detail.includes(step));
}
const saved=run('task_agent_saved',{policy:'act_triple',policy_label:'三路 ACT · 已保存版本 · 160000'});assert(call('historyDescription',saved).detail.includes('已保存版本'));call('showRun',saved);assert(elements.runLabel.textContent.includes('已保存版本'));
const fromModel=run('task_agent_model',{policy:'act',model:{policy:'act',policy_label:'双路 ACT · 历史版本 · 100000'}});assert(call('policyLabel',fromModel).includes('历史版本'));
const preview=call('historyDescription',run('task_agent_preview',{instruction:'',task_submitted:false}));assert(preview.detail.includes('场景预览'));assert(!preview.detail.includes('π0.5'));
''')

    def test_comparison_reuse_ignores_model_draft_and_requires_new_submission(self):
        self.run_js(r'''
call('installPolicyOptions',options);call('applyDefaults',{policy:'pi05',max_retries:2,max_subtasks:4});
const source='task_agent_old_model',id='task_agent_comparison',current={busy:false,active_run_id:null,scene:null,run:run(source,{policy:'act',initial_scene_available:true,seed:619,object_names:['红球']})};context.current=current;evaluate('state=current;selectedRun=current.run.id');elements.policy.value='act_triple';
const restored={busy:true,active_run_id:id,scene:scene(id,{seed:619,generation_mode:'current',comparison_initial_layout_verified:true,object_names:['红球']}),run:run(id,{seed:619,instruction:'',busy:true,finished:false,status:'scene_ready',phase:'scene_ready',generation_mode:'current',comparison_initial_layout_verified:true,initial_scene_available:true,object_names:['红球']})},requests=[];
context.fetch=async(url,init)=>{requests.push({url,init});return {ok:true,json:async()=>url==='/api/scene'?{ok:true,run_id:id,scene:restored.scene,run:restored.run}:restored}};context.history=async()=>{};
await call('prepareScene','current');const posts=requests.filter(row=>row.init?.method==='POST');assert.equal(posts.length,1);assert.equal(posts[0].url,'/api/scene');assert.deepEqual(JSON.parse(posts[0].init.body),{mode:'current',source_scene_id:source});assert.equal(elements.instruction.value,'');assert(elements.start.disabled);assert(!elements.policy.disabled);assert(elements.comparisonSceneStatus.textContent.includes('初始布局已校验一致'));assert(elements.reuseSceneSource.textContent.includes('物品组合与初始布局'));
elements.policy.value='act_triple';elements.instruction.value='收拾球类';call('controls');assert.equal(call('taskPayload').scene_id,id);assert.equal(call('taskPayload').policy,'act_triple');assert(!requests.some(row=>row.url==='/api/scene/task'));
''')

    def test_object_labels_separate_dual_and_triple_training_ranges(self):
        self.run_js(r'''
call('installObjectCatalog',{min_objects:1,max_objects:8,objects:[{id:'extra',name:'新增杯具',category:'杯具',available:true,act_trained:false,act_triple_trained:true,pi05_trained:true},{id:'legacy',name:'旧物品',category:'其他',available:true,act_trained:true,pi05_trained:false}]});
const choices=elements.objectChoices.children;assert(flatten(choices[0]).includes('双路 ACT 训练外'));assert(flatten(choices[0]).includes('三路 ACT 训练目标'));assert(flatten(choices[0]).includes('π0.5 微调目标'));assert(flatten(choices[1]).includes('双路 ACT 训练目标'));assert(!flatten(choices[1]).includes('三路 ACT'));
''')


if __name__ == '__main__':
    unittest.main()
