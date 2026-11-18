"""Execute a language goal with VLM-selected subtasks and live visual feedback."""
import argparse
import collections
import json
from pathlib import Path
import threading
import time

from .contracts import StopRequested,text
from .supervisor import VisualSupervisor
from .vlm import TaskVLM
from deployment_paths import project_path
from .mask_dropout import normalize_mask_dropout
from .policies import POLICY_IDS, policy_record


TERMINAL={'completed','failed','stopped','error','aborted','needs_input','retry_budget_exhausted','timeout'}


class AbortRequested(RuntimeError):
    pass


class StopToken:
    def __init__(self,path=None,deadline=None):
        self.path=Path(path) if path else None
        self.deadline=deadline
        self.event=threading.Event()
    def is_set(self):
        return self.event.is_set() or bool(self.path and self.path.exists()) or bool(self.deadline and time.monotonic()>=self.deadline)
    def set(self):self.event.set()


class VisualVotes:
    """Two fresh consecutive pixel-based opinions; never physical simulator facts."""
    def __init__(self,max_age=4):
        self.lock=threading.RLock();self.rows=collections.deque(maxlen=12);self.max_age=max_age
    def add(self,row):
        if row.get('accepted'):
            with self.lock:self.rows.append(dict(row))
    def clear(self):
        with self.lock:self.rows.clear()
    def matching(self,snapshot):
        now=time.monotonic()
        with self.lock:
            return [r for r in self.rows if r.get('step_generation')==snapshot['step_generation']
                and r.get('phase')==snapshot['phase'] and r.get('target_text')==snapshot['target_text']
                and 0<=now-r.get('captured_at',0)<=self.max_age]
    def confirmed(self,snapshot,assessment):
        rows=self.matching(snapshot)
        if len(rows)<2:return False
        a,b=rows[-2:]
        return (a.get('frame_id')!=b.get('frame_id') and b.get('captured_at',0)-a.get('captured_at',0)>=.35
                and all(r.get('assessment')==assessment and r.get('confidence',0)>=.6
                        and r.get('decision') not in ('retry','abort') for r in (a,b)))
    def retry(self,snapshot):
        rows=self.matching(snapshot)
        released=snapshot.get('phase')=='transport' and snapshot.get('motion_phase') in (
            'release','retreat_raise','retreat_home','retreat_joint_home','settle')
        def failed(row):
            # An empty hand after our commanded release is expected, and is
            # not itself evidence that placement failed. An explicit failed
            # placement judgement still requests recovery.
            if released and row.get('assessment')=='not_grasped':return False
            return (row.get('decision')=='retry' or row.get('assessment')=='failed') and row.get('confidence',0)>=.6
        return len(rows)>=2 and all(failed(r) for r in rows[-2:])
    def abort(self,snapshot):
        rows=self.matching(snapshot)
        return bool(rows and rows[-1].get('decision')=='abort' and rows[-1].get('confidence',0)>=.6)


class TaskRunner:
    def __init__(self,instruction,policy,seed,output,max_retries=2,max_subtasks=4,
                 monitor_hz=1,stop_file=None,max_seconds=300,backend_factory=None,vlm=None,
                 mask_dropout=None,mask_dropout_control=None):
        self.instruction=text(instruction,'instruction',2000)
        if not isinstance(policy,str) or policy not in POLICY_IDS:raise ValueError('Unknown skill policy')
        if not 0<=max_retries<=3 or not 1<=max_subtasks<=4:raise ValueError('Invalid task bounds')
        if float(monitor_hz)!=1:raise ValueError('This prototype uses a nominal 1 Hz visual observer')
        self.policy,self.seed=policy,int(seed)
        self.mask_dropout=normalize_mask_dropout(mask_dropout)
        self.mask_dropout_control=Path(mask_dropout_control) if mask_dropout_control else None
        self.output=project_path(output, 'task output');self.output.mkdir(parents=True,exist_ok=True)
        if (self.output/'summary.json').exists():raise ValueError('Refusing to overwrite an existing task run')
        self.lock=threading.RLock();self.log_lock=threading.RLock()
        self.started=time.monotonic();self.stop=StopToken(stop_file,self.started+max_seconds)
        self.max_retries,self.max_subtasks=max_retries,max_subtasks
        self.backend_factory=backend_factory;self.vlm=vlm
        self.backend=None;self.monitor=None;self.votes=VisualVotes(max_age=6)
        self.completed=[];self.generation=0;self.status=dict(status='initializing',phase='initializing',
            instruction=self.instruction,policy=policy,seed=seed,plan=[],completed_steps=0,current_step=0,
            attempts={},max_subtasks=max_subtasks,max_retries=max_retries,monitor_requested_hz=1,
            latest_judgement=None,monitor_stats={},visual_max_age_seconds=6,success=False,reason='',error=None,
            mask_dropout=self.mask_dropout,mask_dropout_state={},step_generation=0)
        self.status.update(policy_record(policy))
        self._write_status()

    def _write_status(self):
        with self.lock:
            data=dict(self.status)
            path=self.output/'status.json';tmp=self.output/'status.tmp.json'
            tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2));tmp.replace(path)

    def update(self,**values):
        with self.lock:self.status.update(values)
        self._write_status()

    def event(self,event,**values):
        row=dict(event=event,time=time.time(),elapsed_seconds=time.monotonic()-self.started,**values)
        with self.log_lock:
            with (self.output/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

    def _frame(self,snapshot):
        self.update(frame_id=snapshot['frame_id'],phase=snapshot['phase'],
                    motion_phase=snapshot.get('motion_phase'),mask_metadata=snapshot.get('mask_metadata'),
                    mask_dropout_state=snapshot.get('mask_dropout_state', {}),
                    step_generation=snapshot.get('step_generation', 0),
                    monitor_stats=self.monitor.stats() if self.monitor else {})

    def _judgement(self,row):
        with self.log_lock:
            with (self.output/'judgements.jsonl').open('a') as stream:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        self.votes.add(row)
        if row.get('accepted'):
            self.update(latest_judgement=row,monitor_stats=self.monitor.stats() if self.monitor else {})

    def _check(self):
        if self.stop.is_set():
            raise StopRequested('Task time budget reached' if self.stop.deadline and time.monotonic()>=self.stop.deadline else 'User requested stop')

    def _context(self,phase,target=''):
        self.votes.clear()
        self.backend.set_context(phase,target,self.generation)
        values=dict(phase=phase,target_description=target)
        if phase in ('idle','goal_verify') and not target:
            values['active_bbox']=None
        self.update(**values)

    def _control(self,expected=None,allow_retry=True):
        self._check()
        snapshot=self.backend.snapshot()
        if self.votes.abort(snapshot):return 'abort'
        if allow_retry and self.votes.retry(snapshot):return 'retry'
        if expected and self.votes.confirmed(snapshot,expected):
            return 'grasped' if expected=='grasped_lifted' else 'verified'
        return None

    def _verify_place(self,seconds=12):
        self._context('verify',self.status.get('target_description',''))
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            result=self.backend.hold(.25,lambda:self._control('placed'))
            if result['reason'] in ('verified','retry','abort'):return result['reason']
        return 'visual_placement_uncertain'

    def _validate_mask(self,snapshot):
        self._check()
        result=self.vlm.check_mask(snapshot)
        self._check()
        self.event('mask_visual_check',result=result,step_generation=self.generation,
                   mask_metadata=snapshot.get('mask_metadata'))
        return result

    def _goal_audit(self):
        self._check()
        self._context('goal_verify','')
        for check,settle_seconds in ((1,.1),(2,1.1)):
            # Advance real physics/cameras before each audit. A preceding
            # target-selection request can otherwise leave the first frame old.
            self._check()
            held=self.backend.hold(settle_seconds,lambda:self._control())
            self._check()
            if held['reason']=='abort':raise AbortRequested('VLM requested abort during final verification')
            if held['reason']=='retry':
                return dict(complete=False,reason='Live visual observer requested retry during final verification',
                            remaining_description='Reinspect the scene and unfinished targets')
            snapshot=self.backend.snapshot()
            self._check()
            result=self.vlm.finish(snapshot,self.completed)
            self._check()
            self.event('goal_visual_check',check=check,result=result)
            if not result['complete']:return result
        return result

    def _finish_status(self,status,reason='',**extra):
        self.update(status=status,success=status=='completed',reason=reason,**extra)

    def run(self):
        try:
            if self.backend_factory is None:
                from .backend import RobotBackend
                self.backend_factory=RobotBackend
            if self.vlm is None:self.vlm=TaskVLM(self.instruction,self.output,max_steps=self.max_subtasks)
            self.backend=self.backend_factory(policy=self.policy,seed=self.seed,output=self.output,
                frame_callback=self._frame,stop_event=self.stop,mask_dropout=self.mask_dropout,
                mask_dropout_control=self.mask_dropout_control,event_callback=self.event)
            self.update(status='planning',phase='planning',zone=self.backend.zone)
            self.event('scene_started',seed=self.seed,**policy_record(self.policy),verifier='VLM_visual_only',
                       collection_capacity=self.backend.zone['capacity'],mask_dropout=self.mask_dropout)
            plan=self.vlm.plan(self.backend.snapshot())
            (self.output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
            self.update(plan=plan['steps'],plan_summary=plan['summary'],selection_rule=plan['selection_rule'])
            self.event('task_planned',plan=plan)
            self._check()
            if plan['needs_clarification']:
                self._finish_status('needs_input',plan['summary']);return self.status
            self.monitor=VisualSupervisor(self.vlm.judge,self.backend.snapshot,self._judgement,
                                          hz=1,max_inflight=4,max_age=6)
            self.monitor.start()
            self.update(status='executing')
            planning_checks=0;uncertain=0;pending_hint=None
            while len(self.completed)<self.max_subtasks:
                self._check();planning_checks+=1
                if planning_checks>self.max_subtasks*3+3:
                    self._finish_status('failed','VLM repeatedly could not resolve the next target');break
                index=len(self.completed)
                self.update(current_step=index,completed_steps=index,phase='selecting_target')
                self._context('idle','')
                hint=pending_hint or (plan['steps'][index] if index<len(plan['steps']) else None)
                try:
                    target=self.vlm.target(self.backend.snapshot(),plan,self.completed,hint)
                except ValueError as error:
                    self.event('target_format_error',error=str(error));uncertain+=1
                    if uncertain>=3:
                        self._finish_status('needs_input','VLM repeatedly returned unusable target boxes');break
                    self.backend.hold(1);continue
                self.event('target_selected',step_index=index,target=target)
                self.update(active_bbox=target.get('bbox_xyxy'),target_description=target.get('description',''))
                if target['decision']=='uncertain':
                    uncertain+=1
                    if uncertain>=3:
                        self._finish_status('needs_input',target['reason']);break
                    self.backend.hold(1);continue
                if target['decision']=='complete':
                    result=self._goal_audit()
                    if result['complete']:
                        self._finish_status('completed',result['reason'],final_verification=result);break
                    uncertain+=1;pending_hint=result.get('remaining_description')
                    if uncertain>=3:
                        self._finish_status('failed',result['reason'],final_verification=result);break
                    continue
                uncertain=0;accomplished=False
                for attempt in range(1,self.max_retries+2):
                    self._check();self.generation+=1
                    attempts=dict(self.status['attempts']);attempts[str(index)]=attempt
                    self.update(attempts=attempts,active_bbox=target['bbox_xyxy'],target_description=target['description'])
                    self._context('grasp',target['description'])
                    self.event('skill_started',step_index=index,attempt=attempt,target=target,skill=self.policy)
                    result=self.backend.grasp(target,lambda:self._control('grasped_lifted'),validate_mask=self._validate_mask)
                    self.event('grasp_finished',step_index=index,attempt=attempt,result=result)
                    if result['reason']=='abort':
                        self._finish_status('aborted','VLM requested abort');break
                    if result['reason']=='grasped':
                        self._context('transport',target['description'])
                        self.event('visual_grasp_confirmed',step_index=index,attempt=attempt)
                        try:
                            placed=self.backend.transport_place(index,lambda:self._control())
                        except RuntimeError as error:
                            placed=dict(reason='motion_plan_error',error=str(error))
                        self.event('placement_motion_finished',step_index=index,attempt=attempt,result=placed)
                        if placed['reason']=='abort':
                            self._finish_status('aborted','VLM requested abort during transport');break
                        verdict=(self._verify_place() if placed['reason']=='placement_motion_completed' else placed['reason'])
                        if verdict=='abort':
                            self._finish_status('aborted','VLM requested abort during verification');break
                        if verdict=='verified':
                            record=dict(description=target['description'],slot=index,attempt=attempt,
                                        verification='two_fresh_visual_votes',step_generation=self.generation)
                            self.completed.append(record);self.event('subtask_completed',step_index=index,**record)
                            self.update(completed_steps=len(self.completed),completed=self.completed)
                            accomplished=True;pending_hint=None;break
                        result=dict(reason=verdict,grasp_result=result,placement_result=placed)
                    self.event('subtask_retry',step_index=index,attempt=attempt,
                               reason=result['reason'],remaining_attempts=self.max_retries+1-attempt)
                    self._context('recovery',target['description'])
                    recovery=self.backend.recover_open_home(lambda:self._control(allow_retry=False))
                    self.event('recovery_finished',result=recovery)
                    if recovery['reason']=='abort':
                        self._finish_status('aborted','VLM requested abort during recovery');break
                    if attempt<=self.max_retries:
                        self._context('idle','')
                        try:
                            retry_snapshot=self.backend.snapshot()
                            height,width=retry_snapshot['base_rgb'].shape[:2]
                            refreshed=self.vlm.target(retry_snapshot,plan,self.completed,
                                dict(description=target['description'],previous_bbox_xyxy=target['bbox_xyxy'],
                                     previous_bbox_coordinate_space='pixels',previous_image_size=[width,height],
                                     previous_bbox_normalized_1000=[value*1000/(width if axis%2==0 else height)
                                                                   for axis,value in enumerate(target['bbox_xyxy'])],
                                     failure_reason=result.get('reason'),evidence=result.get('evidence'),
                                     guidance=('Correct the failed localization; do not repeat a box that selected background.'
                                         if result.get('reason')=='mask_target_mismatch' else
                                         'Reinspect the SAME target. Do not switch to another object; a verification timeout does not prove a grasp or localization failure.')))
                        except ValueError:
                            refreshed=dict(decision='uncertain',reason='Invalid retry localization')
                        self.event('retry_localization',result=refreshed)
                        if refreshed['decision']=='pick':target=refreshed
                        else:
                            pending_hint=target['description']
                            self._finish_status('needs_input','Could not re-localize the failed target: '+refreshed['reason'])
                            break
                if self.status['status'] in TERMINAL:break
                if not accomplished:
                    self._finish_status('retry_budget_exhausted','The current subtask was not visually verified after bounded retries');break
                # A successful subtask is followed by a global visual audit
                # before any new object's bbox or tracking session is created.
                result=self._goal_audit()
                self.event('post_subtask_goal_check',completed_steps=len(self.completed),result=result)
                self.update(latest_goal_check=result)
                if result['complete']:
                    self._finish_status('completed',result['reason'],final_verification=result);break
                pending_hint=result.get('remaining_description') or None
                self._context('idle','')
                self.backend.hold(.5)
            else:
                result=self._goal_audit()
                self._finish_status('completed' if result['complete'] else 'needs_input',
                                    result['reason'],final_verification=result)
        except AbortRequested as error:
            self._finish_status('aborted',str(error))
        except StopRequested as error:
            status='timeout' if self.stop.deadline and time.monotonic()>=self.stop.deadline else 'stopped'
            self._finish_status(status,str(error))
        except Exception as error:
            from .backend import BackendStopped
            if isinstance(error,BackendStopped):
                status='timeout' if self.stop.deadline and time.monotonic()>=self.stop.deadline else 'stopped'
                self._finish_status(status,str(error))
            else:
                self.event('task_error',error=f'{type(error).__name__}: {error}')
                self._finish_status('error',str(error),error=f'{type(error).__name__}: {error}')
        finally:
            if self.monitor is not None:
                self.monitor.stop(wait=False)
                self.update(monitor_stats=self.monitor.stats())
            if self.backend is not None:
                try:self.backend.close()
                except Exception as error:self.update(cleanup_error=str(error))
            if self.monitor is not None:
                drain_deadline=time.monotonic()+4
                while self.monitor.stats().get('inflight_count',0) and time.monotonic()<drain_deadline:
                    time.sleep(.05)
                self.update(monitor_stats=self.monitor.stats())
            if self.status['status'] not in TERMINAL:
                self._finish_status('error','Run ended without a terminal state')
            summary=dict(self.status,completed=self.completed,elapsed_seconds=time.monotonic()-self.started,
                success_verifier='VLM_visual_only',diagnostic_gt_used_for_control=False,
                learned_skill_scope='grasp_and_lift',placement_skill='calibrated_mplib_transport_release_retreat')
            (self.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2))
            self.event('task_finished',status=summary['status'],success=summary['success'])
        return self.status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instruction',required=True)
    parser.add_argument('--policy',choices=POLICY_IDS,default='pi05')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-retries',type=int,default=2)
    parser.add_argument('--max-subtasks',type=int,default=4)
    parser.add_argument('--monitor-hz',type=float,default=1)
    parser.add_argument('--max-seconds',type=int,default=300)
    parser.add_argument('--stop-file',type=Path)
    parser.add_argument('--mask-dropout-config',dest='mask_dropout',type=json.loads)
    parser.add_argument('--mask-dropout-control',type=Path)
    args=parser.parse_args()
    result=TaskRunner(**vars(args)).run()
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
