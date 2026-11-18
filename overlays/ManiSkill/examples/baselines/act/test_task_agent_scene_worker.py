"""Scene-first lifecycle contracts using fake RGB/physics and no policy models."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from task_agent.scene_worker import SceneWorker,FileStop,validate_task_request,rgb_hashes
from task_agent.mask_dropout import MaskDropout
from task_agent.scene_objects import normalize_scene_spec,resolve_scene_spec


def request(scene_id,seed=17):
    return dict(scene_id=scene_id,instruction='收拾球类',policy='pi05',seed=seed,
                max_retries=1,max_subtasks=4,monitor_hz=1,mask_dropout=None)


class FakeBackend:
    """Synthetic unit fixture, not a simulator or a recording demonstration."""
    def __init__(self,**kwargs):
        self.arguments=kwargs;self.output=Path(kwargs['output']);self.seed=kwargs['seed']
        self.scene_spec=normalize_scene_spec(kwargs.get('scene_spec'))
        self.selected_object_ids=[item['id'] for item in self.scene_spec['objects']]
        self.steps=5;self.reset_count=1;self.refreshes=0;self.binds=[];self.closed=False
        self.change_on_refresh=None;self.raise_on_close=False
        self.rgb={name:np.full((size,size,3),value,np.uint8) for name,size,value in (
            ('base_rgb',224,12),('wrist_rgb',224,23),('overview_rgb',448,34))}
        self.video=self.output/'execution.mp4';self.video.write_bytes(b'synthetic transport fixture')

    def refresh_observation(self):
        if self.arguments['stop_event'].is_set():raise InterruptedError('stopped fixture')
        self.refreshes+=1
        if self.refreshes==self.change_on_refresh:self.rgb['base_rgb'][0,0,0]+=1
        return dict(frame_id=self.steps,captured_at=float(self.refreshes),**self.rgb)

    def bind_task(self,**kwargs):self.binds.append(kwargs);return self
    def close(self):
        self.closed=True
        if self.raise_on_close:raise RuntimeError('fixture close failed')


class SceneWorkerLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.scene_id='task_agent_scene_fixture'
        self.output=Path(self.tmp.name)/self.scene_id;self.output.mkdir()
        self.stop=self.output/'stop.request';self.task=self.output/'task_request.json'
        self.backends=[];self.runner_calls=[]

    def tearDown(self):self.tmp.cleanup()

    def backend_factory(self,**kwargs):
        backend=FakeBackend(**kwargs);self.backends.append(backend);return backend

    def runner_factory(self,**kwargs):
        self.runner_calls.append(kwargs)
        def run():
            backend=kwargs['backend_factory'](policy=kwargs['policy'],seed=kwargs['seed'],output=kwargs['output'],
                frame_callback=lambda snapshot:None,stop_event=FileStop(self.stop),
                mask_dropout=kwargs['mask_dropout'],mask_dropout_control=kwargs['mask_dropout_control'],
                event_callback=lambda event,**values:None)
            self.assertIs(backend,self.backends[0])
            return dict(status='completed',phase='idle',success=True,completed_steps=1)
        return SimpleNamespace(run=run)

    def worker(self,sleep,**kwargs):
        return SceneWorker(17,self.output,self.stop,self.task,self.scene_id,
            backend_factory=self.backend_factory,runner_factory=self.runner_factory,sleep=sleep,**kwargs)

    def write_task(self,payload=None):
        temporary=self.task.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload or request(self.scene_id)))
        temporary.replace(self.task)

    def write_reference(self,seed=17,matching=True):
        images={name:np.full((size,size,3),value,np.uint8) for name,size,value in (
            ('base_rgb',224,12),('wrist_rgb',224,23),('overview_rgb',448,34))}
        value=dict(source_scene_id='task_agent_comparison_source',seed=seed,rgb_sha256=rgb_hashes(images))
        if not matching:value['rgb_sha256']['wrist_rgb']='0'*64
        (self.output/'scene_reference.json').write_text(json.dumps(value))
        return value

    def test_ready_wait_does_not_step_or_start_models_then_reuses_same_scene(self):
        polls=[]
        def wait(seconds):
            status=json.loads((self.output/'status.json').read_text())
            self.assertEqual(status['status'],'scene_ready');self.assertEqual(status['phase'],'scene_ready')
            self.assertEqual(status['scene_id'],self.scene_id);self.assertEqual(status['instruction'],'')
            self.assertEqual(self.backends[0].steps,5);self.assertFalse(self.backends[0].binds)
            self.assertFalse(self.runner_calls)
            polls.append(seconds)
            if len(polls)==3:self.write_task()
        worker=self.worker(wait);result=worker.run()
        self.assertEqual(result['status'],'completed');self.assertEqual(len(polls),3)
        backend=self.backends[0]
        self.assertTrue(backend.arguments['defer_policy_loading']);self.assertEqual(len(backend.binds),1)
        self.assertEqual(backend.reset_count,1);self.assertEqual(backend.steps,5);self.assertTrue(backend.closed)
        reused=json.loads((self.output/'scene_reuse.json').read_text())
        self.assertTrue(reused['rgb_unchanged']);self.assertEqual(reused['reset_count'],1)
        self.assertEqual(reused['preview_frame_id'],reused['after_bind_frame_id'])
        self.assertEqual(backend.binds[0]['mask_dropout_control'],self.output/'mask_dropout_control.json')
        self.assertIsNotNone(worker.task_consumed_at)

    def test_stop_before_task_closes_and_preserves_fixture_without_vlm(self):
        def wait(seconds):self.stop.touch()
        result=self.worker(wait).run();summary=json.loads((self.output/'summary.json').read_text())
        self.assertEqual(result['status'],'stopped');self.assertFalse(summary['task_started'])
        self.assertEqual(summary['vlm_calls'],0);self.assertEqual(summary['reset_count'],1)
        self.assertFalse(self.runner_calls);self.assertFalse(self.backends[0].binds)
        self.assertTrue(self.backends[0].closed)
        self.assertEqual(self.backends[0].video.read_bytes(),b'synthetic transport fixture')
        self.assertFalse((self.output/'vlm_calls.jsonl').exists())
        self.assertFalse((self.output/'scene_comparison.json').exists())
        self.assertIsNone(summary['comparison_source_scene_id'])
        self.assertFalse(summary['comparison_initial_layout_verified'])

    def test_reference_match_reaches_ready_and_persists_initial_comparison_in_task_summary(self):
        reference=self.write_reference()
        def wait(seconds):
            status=json.loads((self.output/'status.json').read_text())
            self.assertEqual(status['status'],'scene_ready');self.assertTrue(status['comparison_initial_layout_verified'])
            self.write_task()
        worker=self.worker(wait)
        original_factory=self.runner_factory
        def runner_factory(**kwargs):
            runner=original_factory(**kwargs)
            def run():
                result=runner.run()
                # Simulated task movement changes final RGB. The comparison
                # must continue to describe the prepared initial layout.
                self.backends[0].rgb['base_rgb'][:]=99
                (self.output/'summary.json').write_text(json.dumps(dict(result,elapsed_seconds=3.)))
                return result
            return SimpleNamespace(run=run)
        worker.runner_factory=runner_factory
        result=worker.run()
        comparison=json.loads((self.output/'scene_comparison.json').read_text())
        summary=json.loads((self.output/'summary.json').read_text())
        scene=json.loads((self.output/'scene.json').read_text())
        self.assertEqual(result['status'],'completed');self.assertTrue(comparison['verified'])
        self.assertEqual(comparison['expected_rgb_sha256'],comparison['actual_rgb_sha256'])
        self.assertTrue(scene['comparison_initial_layout_verified'])
        self.assertEqual(summary['comparison_source_scene_id'],reference['source_scene_id'])
        self.assertEqual(summary['initial_rgb_sha256'],reference['rgb_sha256'])
        self.assertTrue(summary['comparison_initial_layout_verified'])
        self.assertEqual(summary['elapsed_seconds'],3.)
        self.assertEqual(self.backends[0].reset_count,1)

    def test_reference_hash_mismatch_closes_without_ready_models_or_vlm(self):
        self.write_reference(matching=False)
        result=self.worker(lambda seconds:self.fail('Mismatch must not enter ready wait')).run()
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['reason'],'对比场景初始布局与参考不一致，未开始任务')
        comparison=json.loads((self.output/'scene_comparison.json').read_text())
        summary=json.loads((self.output/'summary.json').read_text())
        self.assertFalse(comparison['verified']);self.assertIsNotNone(comparison['actual_rgb_sha256'])
        self.assertFalse(summary['comparison_initial_layout_verified']);self.assertEqual(summary['vlm_calls'],0)
        self.assertFalse(self.runner_calls);self.assertFalse(self.backends[0].binds)
        self.assertTrue(self.backends[0].closed);self.assertTrue(self.backends[0].video.is_file())
        self.assertFalse((self.output/'vlm_calls.jsonl').exists())

    def test_reference_wrong_seed_is_rejected_before_creating_backend(self):
        self.write_reference(seed=18)
        result=self.worker(lambda seconds:self.fail('Wrong seed must not enter ready')).run()
        self.assertEqual(result['status'],'error');self.assertIn('参考无效',result['reason'])
        self.assertFalse(self.backends);self.assertFalse(self.runner_calls)
        self.assertFalse(json.loads((self.output/'scene_comparison.json').read_text())['verified'])

    def test_reference_preview_stop_retains_verified_comparison_and_video(self):
        reference=self.write_reference()
        result=self.worker(lambda seconds:self.stop.touch()).run()
        summary=json.loads((self.output/'summary.json').read_text())
        self.assertEqual(result['status'],'stopped');self.assertEqual(summary['vlm_calls'],0)
        self.assertTrue(summary['comparison_initial_layout_verified'])
        self.assertEqual(summary['comparison_source_scene_id'],reference['source_scene_id'])
        self.assertEqual(self.backends[0].video.read_bytes(),b'synthetic transport fixture')
        self.assertFalse(self.runner_calls)

    def test_custom_scene_spec_is_executed_and_recorded_in_manifest_and_summary(self):
        canonical=resolve_scene_spec({'mode':'manual','object_ids':['025_mug','013_apple']})
        (self.output/'scene_spec.json').write_text(json.dumps(canonical))
        def wait(seconds):self.write_task()
        result=self.worker(wait).run()
        scene=json.loads((self.output/'scene.json').read_text());summary=json.loads((self.output/'summary.json').read_text())
        self.assertEqual(result['status'],'completed');self.assertEqual(self.backends[0].arguments['scene_spec'],canonical)
        self.assertEqual(scene['scene_spec'],canonical);self.assertEqual(summary['scene_spec'],canonical)
        self.assertEqual(scene['selected_object_ids'],['025_mug','013_apple'])
        self.assertEqual(summary['selected_object_ids'],scene['selected_object_ids'])

    def test_backend_ignoring_requested_objects_is_rejected_before_ready(self):
        canonical=resolve_scene_spec({'mode':'manual','object_ids':['025_mug','013_apple']})
        (self.output/'scene_spec.json').write_text(json.dumps(canonical))
        factory=self.backend_factory
        def wrong_backend(**kwargs):
            backend=factory(**kwargs);backend.selected_object_ids=['025_mug'];return backend
        worker=self.worker(lambda seconds:self.fail('An ignored configuration cannot become ready'))
        worker.backend_factory=wrong_backend
        result=worker.run()
        self.assertEqual(result['status'],'error');self.assertIn('object combination',result['reason'])
        self.assertFalse(self.runner_calls);self.assertTrue(self.backends[0].closed)

    def test_stale_request_is_rejected_before_preparing_or_binding(self):
        self.write_task();result=self.worker(lambda seconds:None).run()
        self.assertEqual(result['status'],'error');self.assertIn('Stale',result['reason'])
        self.assertFalse(self.backends);self.assertFalse(self.runner_calls)

    def test_wrong_scene_request_is_explicit_error_without_binding(self):
        def wait(seconds):self.write_task(request('old_scene'))
        result=self.worker(wait).run()
        self.assertEqual(result['status'],'error');self.assertIn('expired or different scene',result['reason'])
        self.assertFalse(self.backends[0].binds);self.assertFalse(self.runner_calls)
        self.assertTrue(self.backends[0].closed)

    def test_changed_preview_hash_prevents_any_policy_binding(self):
        def wait(seconds):self.backends[0].change_on_refresh=2;self.write_task()
        result=self.worker(wait).run()
        self.assertEqual(result['status'],'error');self.assertIn('changed before task binding',result['reason'])
        self.assertFalse(self.backends[0].binds)

    def test_hash_change_during_binding_is_recorded_and_rejected(self):
        def wait(seconds):self.backends[0].change_on_refresh=3;self.write_task()
        result=self.worker(wait).run();reuse=json.loads((self.output/'scene_reuse.json').read_text())
        self.assertEqual(result['status'],'error');self.assertFalse(reuse['rgb_unchanged'])

    def test_cleanup_error_returns_and_saves_error_consistently(self):
        def wait(seconds):self.backends[0].raise_on_close=True;self.stop.touch()
        result=self.worker(wait).run()
        self.assertEqual(result['status'],'error')
        self.assertEqual(json.loads((self.output/'status.json').read_text())['status'],'error')
        self.assertEqual(json.loads((self.output/'summary.json').read_text())['status'],'error')

    def test_request_validation_strict_seed_bounds_and_supported_fields(self):
        self.assertEqual(validate_task_request(request(self.scene_id),self.scene_id,17)['mask_dropout']['enabled'],False)
        for key,value in (('seed',True),('seed',18),('policy','other'),('max_retries',True),
                          ('max_retries',4),('max_subtasks',0),('monitor_hz',True),
                          ('monitor_hz',float('nan')),('instruction','')):
            data=request(self.scene_id);data[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_task_request(data,self.scene_id,17)
        data=request(self.scene_id);data['output']='elsewhere'
        with self.assertRaises(ValueError):validate_task_request(data,self.scene_id,17)


class DeferredBackendContracts(unittest.TestCase):
    def fake(self,directory):
        from task_agent.backend import RobotBackend
        backend=RobotBackend.__new__(RobotBackend)
        backend.seed=17;backend.output=Path(directory).resolve();backend._closed=False
        backend._policy_bound=False;backend._binding_attempted=False;backend._decider=None
        backend.phase='idle';backend.step_generation=0;backend.steps=5;backend.reset_count=1
        backend._mask_dropout=MaskDropout();backend._dropout_event=Mock()
        backend.refresh_observation=Mock(return_value={'frame_id':5})
        def prepare(policy):backend._policy_bound=True
        backend._prepare_policy=Mock(side_effect=prepare)
        backend.raw=SimpleNamespace(reset=Mock(),step=Mock())
        (backend.output/'backend_config.json').write_text('{}')
        return backend

    def test_bind_preserves_scene_seed_output_and_initialization(self):
        with TemporaryDirectory() as directory:
            backend=self.fake(directory);callback=Mock()
            self.assertIs(backend.bind_task('pi05',17,directory,event_callback=callback),backend)
            backend._prepare_policy.assert_called_once_with('pi05');backend.refresh_observation.assert_called_once()
            backend.raw.reset.assert_not_called();backend.raw.step.assert_not_called()
            self.assertEqual((backend.steps,backend.reset_count),(5,1))
            # Closing preview bookkeeping is not a task dropout event.
            callback.assert_not_called()
            with self.assertRaisesRegex(RuntimeError,'only one task'):backend.bind_task('pi05',17,directory)

    def test_bind_rejects_seed_output_stop_and_repeated_loading(self):
        with TemporaryDirectory() as directory:
            backend=self.fake(directory)
            for seed,output in ((18,directory),(True,directory),(17,Path(directory)/'other')):
                with self.assertRaises(ValueError):backend.bind_task('pi05',seed,output)
            backend._prepare_policy.assert_not_called();backend.raw.reset.assert_not_called()
            stopped=threading.Event();stopped.set()
            with self.assertRaisesRegex(RuntimeError,'stop'):backend.bind_task('pi05',17,directory,stop_event=stopped)
            backend._prepare_policy.assert_not_called()
            with self.assertRaisesRegex(RuntimeError,'only one task'):backend.bind_task('pi05',17,directory)

    def test_refresh_captures_new_rgb_without_step_reset_or_task_evaluation(self):
        from task_agent.backend import RobotBackend
        with TemporaryDirectory() as directory:
            backend=self.fake(directory);del backend.refresh_observation
            fresh={'agent':{},'sensor_data':{}}
            backend.base=SimpleNamespace(get_obs=Mock(return_value=fresh))
            backend.stop_event=threading.Event();backend._publish=Mock();backend._save_image=Mock()
            backend.snapshot=Mock(return_value={'frame_id':5})
            self.assertEqual(RobotBackend.refresh_observation(backend),{'frame_id':5})
            backend.base.get_obs.assert_called_once_with(info={})
            backend.raw.reset.assert_not_called();backend.raw.step.assert_not_called()
            self.assertIs(backend.obs,fresh)

    def test_default_constructor_loads_policy_but_deferred_constructor_does_not(self):
        from task_agent.backend import RobotBackend
        observation={'agent':{'qpos':np.zeros((1,9),np.float32)}}
        for deferred in (False,True):
            with self.subTest(deferred=deferred),TemporaryDirectory() as directory:
                actors=[SimpleNamespace(name=name) for name in ('cube_0','cube_1','cube_2',
                    '003_cracker_box','025_mug','011_banana','054_softball','055_baseball')]
                raw=SimpleNamespace(unwrapped=SimpleNamespace(cubes=actors,agent=SimpleNamespace(tcp=SimpleNamespace(
                    pose=SimpleNamespace(sp=object())))),reset=Mock(return_value=(observation,{})),close=Mock())
                def prepare(backend,policy):backend._policy_bound=True;backend.device='cpu'
                raw.unwrapped.backend=SimpleNamespace(render_backend='sapien_cpu')
                with patch.dict('os.environ', {'MASK_RENDER_BACKEND':'cpu', 'MASK_SAPIEN_RENDERER':'lavapipe'}), \
                     patch('gymnasium.make',return_value=raw) as make,patch('task_agent.backend._register_scene'), \
                     patch('imageio.v2.get_writer',return_value=SimpleNamespace(close=Mock())), \
                     patch.object(RobotBackend,'_prepare_policy',autospec=True,side_effect=prepare) as load, \
                     patch.object(RobotBackend,'_publish'),patch.object(RobotBackend,'_step'), \
                     patch.object(RobotBackend,'_save_image'),patch('task_agent.backend.urlopen') as network:
                    backend=RobotBackend(policy='act',seed=17,output=directory,defer_policy_loading=deferred)
                    self.assertEqual(load.call_count,0 if deferred else 1)
                    self.assertEqual(backend._policy_bound,not deferred)
                    self.assertEqual(make.call_args.args[0],'TaskCollection-v1')
                    self.assertNotIn('scene_objects',make.call_args.kwargs)
                    self.assertEqual(make.call_args.kwargs['render_backend'],'cpu')
                    config=json.loads((Path(directory)/'backend_config.json').read_text())
                    self.assertEqual(config['render_backend'],'sapien_cpu');self.assertEqual(config['renderer'],'lavapipe')
                    self.assertEqual(backend.reset_count,1);raw.reset.assert_called_once_with(seed=17)
                    network.assert_not_called();backend.close()


if __name__=='__main__':unittest.main()
