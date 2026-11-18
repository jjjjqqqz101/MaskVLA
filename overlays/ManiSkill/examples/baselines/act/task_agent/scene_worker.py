"""Prepare one live scene, then bind a submitted language task without reset."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from deployment_paths import project_path
from .mask_dropout import normalize_mask_dropout
from .policies import POLICY_IDS, policy_record
from .scene_reference import validate_scene_reference
from .scene_objects import normalize_scene_spec


def validate_task_request(value,scene_id,seed):
    keys={'scene_id','instruction','policy','seed','max_retries','max_subtasks','monitor_hz','mask_dropout'}
    if not isinstance(value,dict) or set(value)!=keys:raise ValueError('Task request must contain exactly the scene task fields')
    if value['scene_id']!=scene_id:raise ValueError('Task request belongs to an expired or different scene')
    if type(value['seed']) is not int or value['seed']!=seed:raise ValueError('Task seed must match the prepared scene')
    instruction=value['instruction']
    if not isinstance(instruction,str) or not instruction.strip() or len(instruction)>2000:
        raise ValueError('Task instruction must be nonempty text of at most 2000 characters')
    if not isinstance(value['policy'],str) or value['policy'] not in POLICY_IDS:
        raise ValueError('Task policy must be act, act_triple or pi05')
    for key,low,high in (('max_retries',0,3),('max_subtasks',1,4)):
        if type(value[key]) is not int or not low<=value[key]<=high:raise ValueError('Invalid '+key)
    if isinstance(value['monitor_hz'],bool) or not isinstance(value['monitor_hz'],(int,float)) or value['monitor_hz']!=1:
        raise ValueError('Task monitor_hz must be 1')
    return dict(value,instruction=instruction.strip(),monitor_hz=1.,
                mask_dropout=normalize_mask_dropout(value['mask_dropout']))


def rgb_hashes(snapshot):
    return {name:hashlib.sha256(snapshot[name].tobytes()).hexdigest()
            for name in ('base_rgb','wrist_rgb','overview_rgb')}


def _write_json(path,value):
    path=Path(path);temporary=path.with_name(path.stem+'.tmp'+path.suffix)
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2))
    temporary.replace(path)


class FileStop:
    def __init__(self,path):self.path=Path(path)
    def is_set(self):return self.path.exists()


class SceneWorker:
    def __init__(self,seed,output,stop_file,task_file,scene_id,
                 backend_factory=None,runner_factory=None,poll_interval=.1,sleep=None):
        if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Scene seed must be an integer in 0..2^31-1')
        if not isinstance(scene_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',scene_id):
            raise ValueError('Invalid scene_id')
        self.seed=seed;self.scene_id=scene_id;self.output=project_path(output, 'scene output')
        self.output.mkdir(parents=True,exist_ok=True)
        self.stop_file=Path(stop_file).resolve();self.task_file=Path(task_file).resolve()
        if self.task_file.parent!=self.output:raise ValueError('Task request must be in the scene directory')
        self.backend_factory=backend_factory;self.runner_factory=runner_factory
        self.poll_interval=float(poll_interval);self.sleep=sleep or time.sleep
        if not 0<self.poll_interval<=1:raise ValueError('Invalid scene poll interval')
        self.backend=None;self.scene=None;self.started=time.monotonic();self.task_consumed_at=None
        self.reference=None
        self.scene_spec=None
        self.task_policy=None

    def _comparison_fields(self):
        return dict(comparison_source_scene_id=self.reference['source_scene_id'] if self.reference else None,
            initial_rgb_sha256=self.scene['rgb_sha256'] if self.scene else None,
            comparison_initial_layout_verified=bool(self.scene and self.scene.get('comparison_initial_layout_verified')),
            scene_spec=self.scene_spec,
            selected_object_ids=self.scene.get('selected_object_ids',[]) if self.scene else [])

    def _read_scene_spec(self):
        path=self.output/'scene_spec.json'
        if not path.exists() and not path.is_symlink():
            self.scene_spec=normalize_scene_spec(None);return
        if path.is_symlink() or path.stat().st_size>8192:raise ValueError('Invalid scene specification file')
        self.scene_spec=normalize_scene_spec(json.loads(path.read_text()))

    def _read_reference(self):
        path=self.output/'scene_reference.json'
        if not path.exists() and not path.is_symlink():return
        comparison=dict(scene_id=self.scene_id,seed=self.seed,source_scene_id=None,
                        expected_rgb_sha256=None,actual_rgb_sha256=None,verified=False)
        try:
            if path.is_symlink() or path.stat().st_size>4096:raise ValueError('Invalid reference file')
            self.reference=validate_scene_reference(json.loads(path.read_text()))
            comparison.update(source_scene_id=self.reference['source_scene_id'],
                expected_rgb_sha256=self.reference['rgb_sha256'],reference_seed=self.reference['seed'])
            validate_scene_reference(self.reference,expected_seed=self.seed)
        except (ValueError,TypeError) as error:
            comparison['reason']='invalid_reference: '+str(error)
            _write_json(self.output/'scene_comparison.json',comparison)
            raise ValueError('对比场景参考无效，未开始任务：'+str(error)) from None
        comparison['reason']='awaiting_initial_capture'
        _write_json(self.output/'scene_comparison.json',comparison)

    def _verify_reference(self):
        if self.reference is None:return
        actual=self.scene['rgb_sha256'];expected=self.reference['rgb_sha256']
        verified=actual==expected
        _write_json(self.output/'scene_comparison.json',dict(scene_id=self.scene_id,seed=self.seed,
            source_scene_id=self.reference['source_scene_id'],reference_seed=self.reference['seed'],
            expected_rgb_sha256=expected,actual_rgb_sha256=actual,verified=verified,
            reset_count=self.backend.reset_count,
            reason='initial_rgb_match' if verified else 'initial_rgb_mismatch'))
        if not verified:raise RuntimeError('对比场景初始布局与参考不一致，未开始任务')
        self.scene['comparison_initial_layout_verified']=True

    def _status(self,status,**values):
        row=dict(status=status,phase=status,scene_id=self.scene_id,seed=self.seed,instruction='',
                 success=False,completed_steps=0,task_started=self.task_consumed_at is not None)
        row.update(values)
        if self.task_policy is not None:row.update(policy_record(self.task_policy))
        _write_json(self.output/'status.json',row)
        return row

    def _event(self,event,**values):
        with (self.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=event,time=time.time(),scene_id=self.scene_id,
                scene_elapsed_seconds=time.monotonic()-self.started,**values),ensure_ascii=False)+'\n')

    def _bind(self,**kwargs):
        before=self.backend.refresh_observation()
        before_hashes=rgb_hashes(before)
        if (self.backend.reset_count!=self.scene['reset_count'] or before['frame_id']!=self.scene['frame_id']
                or before_hashes!=self.scene['rgb_sha256']):
            raise RuntimeError('Prepared scene changed before task binding')
        bound=self.backend.bind_task(**kwargs)
        if bound is not self.backend:raise RuntimeError('Task binding replaced the prepared backend')
        after=self.backend.refresh_observation();after_hashes=rgb_hashes(after)
        reused=dict(scene_id=self.scene_id,seed=self.seed,reset_count=self.backend.reset_count,
            reset_count_before_bind=self.scene['reset_count'],reset_count_after_bind=self.backend.reset_count,
            preview_frame_id=self.scene['frame_id'],before_bind_frame_id=before['frame_id'],
            after_bind_frame_id=after['frame_id'],preview_rgb_sha256=self.scene['rgb_sha256'],
            before_bind_rgb_sha256=before_hashes,after_bind_rgb_sha256=after_hashes,
            rgb_unchanged=before_hashes==after_hashes==self.scene['rgb_sha256'],
            reused_live_environment=True)
        _write_json(self.output/'scene_reuse.json',reused)
        if self.backend.reset_count!=1 or after['frame_id']!=before['frame_id'] or not reused['rgb_unchanged']:
            raise RuntimeError('Prepared scene was not preserved while binding the task')
        self._event('prepared_scene_bound',reset_count=self.backend.reset_count,frame_id=after['frame_id'],rgb_unchanged=True)
        return self.backend

    def run(self):
        result=None;runner_started=False
        if (self.output/'summary.json').exists():raise ValueError('Refusing to overwrite an existing scene run')
        self._status('preparing_scene')
        try:
            if self.task_file.exists():raise ValueError('Stale task request already exists before scene preparation')
            if self.stop_file.exists():raise InterruptedError('Scene preparation was stopped')
            self._read_scene_spec()
            self._read_reference()
            if self.backend_factory is None:
                from .backend import RobotBackend
                self.backend_factory=RobotBackend
            self.backend=self.backend_factory(policy='pi05',seed=self.seed,output=self.output,
                stop_event=FileStop(self.stop_file),defer_policy_loading=True,scene_spec=self.scene_spec)
            snapshot=self.backend.refresh_observation()
            if self.backend.reset_count!=1:raise RuntimeError('Scene must be initialized exactly once')
            actual=list(self.backend.selected_object_ids)
            expected=[item['id'] for item in self.scene_spec['objects']]
            if actual!=expected:raise RuntimeError('Prepared scene did not construct the requested object combination')
            self.scene=dict(scene_id=self.scene_id,seed=self.seed,reset_count=self.backend.reset_count,
                frame_id=snapshot['frame_id'],rgb_sha256=rgb_hashes(snapshot),created_at=time.time(),
                policy_loaded=False,comparison_source_scene_id=self.reference['source_scene_id'] if self.reference else None,
                comparison_initial_layout_verified=False,scene_spec=self.scene_spec,selected_object_ids=actual)
            _write_json(self.output/'scene.json',self.scene)
            self._verify_reference()
            _write_json(self.output/'scene.json',self.scene)
            self._status('scene_ready',reset_count=self.backend.reset_count,frame_id=snapshot['frame_id'],
                         rgb_sha256=self.scene['rgb_sha256'],policy_loaded=False,**self._comparison_fields())
            self._event('scene_ready',reset_count=self.backend.reset_count,frame_id=snapshot['frame_id'])
            while True:
                if self.stop_file.exists():raise InterruptedError('Prepared scene was stopped before task submission')
                if self.task_file.exists():
                    if self.task_file.is_symlink():raise ValueError('Task request may not be a symbolic link')
                    if self.task_file.stat().st_mtime+1<self.scene['created_at']:
                        raise ValueError('Task request predates the prepared scene')
                    if self.task_file.stat().st_size>16000:raise ValueError('Task request is too large')
                    request=validate_task_request(json.loads(self.task_file.read_text()),self.scene_id,self.seed)
                    self.task_policy=request['policy']
                    self.task_consumed_at=time.monotonic()
                    self._status('starting_task',instruction=request['instruction'],policy=request['policy'])
                    self._event('scene_task_submitted',**policy_record(request['policy']))
                    if self.runner_factory is None:
                        from .runner import TaskRunner
                        self.runner_factory=TaskRunner
                    runner=self.runner_factory(instruction=request['instruction'],policy=request['policy'],
                        seed=self.seed,output=self.output,max_retries=request['max_retries'],
                        max_subtasks=request['max_subtasks'],monitor_hz=request['monitor_hz'],
                        mask_dropout=request['mask_dropout'],mask_dropout_control=self.output/'mask_dropout_control.json',
                        stop_file=self.stop_file,backend_factory=self._bind)
                    runner_started=True;result=runner.run();break
                self.sleep(self.poll_interval)
        except (Exception,KeyboardInterrupt) as error:
            stopped=isinstance(error,(InterruptedError,KeyboardInterrupt)) or self.stop_file.exists()
            result=self._status('stopped' if stopped else 'error',reason=str(error),
                error=None if stopped else type(error).__name__+': '+str(error))
            self._event('scene_stopped' if stopped else 'scene_error',reason=str(error))
        finally:
            if self.backend is not None:
                try:self.backend.close()
                except Exception as error:
                    result=self._status('error',reason='Unable to close prepared scene',cleanup_error=str(error))
                    summary_path=self.output/'summary.json'
                    if summary_path.exists():
                        summary=json.loads(summary_path.read_text());summary.update(result)
                        _write_json(summary_path,summary)
            if not (self.output/'summary.json').exists():
                result=result or self._status('error',reason='Scene worker ended without a task result')
                summary=dict(result,elapsed_seconds=time.monotonic()-self.started,
                    reset_count=self.backend.reset_count if self.backend else 0,
                    task_started=self.task_consumed_at is not None,
                    vlm_calls=0 if not runner_started else None,scene_only=not runner_started)
                _write_json(self.output/'summary.json',summary)
            summary_path=self.output/'summary.json'
            summary=json.loads(summary_path.read_text());summary.update(self._comparison_fields())
            if self.task_policy is not None:summary.update(policy_record(self.task_policy,summary.get('model')))
            _write_json(summary_path,summary)
            if result is not None:result.update(self._comparison_fields())
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stop-file',type=Path,required=True);parser.add_argument('--task-file',type=Path,required=True)
    parser.add_argument('--scene-id',required=True)
    args=parser.parse_args()
    result=SceneWorker(**vars(args)).run()
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result.get('status') in ('completed','stopped') else 1


if __name__=='__main__':raise SystemExit(main())
