"""CPU contracts for the target-free task backend; no simulator/model startup."""
import io
import json
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,Mock
from types import SimpleNamespace
import base64
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from task_agent.backend import (RobotBackend, BackendStopped, deposit_position,
                                pi_state_from_observation, hold_action_from_observation,
                                sampling_seed, sam_prompt_bbox, ZONE)
from task_agent.mask_dropout import MaskDropout


def observation(gap=.04):
    return {'agent':{'qpos':np.array([[.1,.2,.3,.4,.5,.6,.7,gap,gap]],np.float32)}}


class BackendContracts(unittest.TestCase):
    def test_joint_state_and_gripper_conventions(self):
        opened=pi_state_from_observation(observation(.04))
        closed=pi_state_from_observation(observation(0))
        self.assertEqual(opened.shape,(8,))
        self.assertAlmostEqual(float(opened[-1]),0)
        self.assertAlmostEqual(float(closed[-1]),1)
        self.assertAlmostEqual(float(hold_action_from_observation(observation(.04))[-1]),1,places=6)
        self.assertAlmostEqual(float(hold_action_from_observation(observation(0))[-1]),-1)

    def test_deposit_uses_fixed_calibration_and_bounded_capacity(self):
        positions=[deposit_position(i) for i in range(4)]
        for p in positions:
            self.assertAlmostEqual(float(p[0]),-.23)
            self.assertAlmostEqual(float(p[2]),.08)
        self.assertEqual(len({tuple(p) for p in positions}),4)
        for bad in (-1,4,True,1.5):
            with self.assertRaises(ValueError):deposit_position(bad)

    def test_collection_zone_has_release_margin_without_covering_spawn_area(self):
        center=np.asarray(ZONE['center']);half=np.asarray(ZONE['half_sizes'])
        self.assertAlmostEqual(float((center+half)[0]),-.13)
        self.assertLess(float((center+half)[0]),-.089)
        positions=np.array([deposit_position(i)[:2] for i in range(4)])
        clearances=half-np.abs(positions-center)
        self.assertTrue((clearances>=.10-1e-9).all())
        self.assertTrue(np.allclose(positions[:,1],[-.12,.12,-.04,.04]))

    def fake_placement_backend(self):
        b=self.fake_grasp_backend(np.ones((4,4),bool));b.trace=[]
        b.base=SimpleNamespace(agent=SimpleNamespace(tcp=SimpleNamespace(pose=SimpleNamespace(
            sp=SimpleNamespace(p=np.array([0.,0.,.2]),q=np.array([1.,0.,0.,0.]))))))
        b.initial_tcp=object()
        planner=SimpleNamespace(CLOSED=-1,OPEN=1,gripper_state=1,
            open_gripper=lambda t:b.trace.append(('open',t,b.steps)),
            close=lambda:b.trace.append(('close',)))
        b._planner=lambda:planner
        def move(planner,pose,phase):
            b.motion_phase=phase;b.trace.append(('move',phase))
        b._move=move
        def joint_home(planner,phase):b.trace.append(('joint_home',phase));return .001
        b._return_joint_home=joint_home
        step=b._step
        def record_step(action):
            b._check_stop();b.trace.append(('step',b.motion_phase,np.array(action)))
            step(action);b._check_stop()
        b._step=record_step
        return b

    def test_placement_settles_closed_for_fifteen_steps_before_opening(self):
        b=self.fake_placement_backend()
        result=b.transport_place(0)
        self.assertEqual(result['reason'],'placement_motion_completed')
        lowered=next(i for i,row in enumerate(b.trace) if row[:2]==('move','deposit_lower'))
        opened=next(i for i,row in enumerate(b.trace) if row[0]=='open')
        settle=b.trace[lowered+1:opened]
        self.assertEqual(len(settle),15)
        for kind,phase,action in settle:
            self.assertEqual((kind,phase),('step','deposit_settle_closed'))
            self.assertEqual(float(action[-1]),-1.)
            self.assertTrue(np.allclose(action[:7],b.obs['agent']['qpos'][0,:7]))
        self.assertIn(('move','retreat_raise'),b.trace[opened+1:])
        self.assertIn(('joint_home','retreat_joint_home'),b.trace[opened+1:])
        self.assertEqual(result['home_joint_max_error_rad'],.001)
        self.assertIsNone(b._decider)

    def test_visual_interrupt_during_closed_settle_prevents_release(self):
        b=self.fake_placement_backend()
        result=b.transport_place(0,lambda:'retry' if b.steps>=5 else None)
        self.assertEqual(result['reason'],'retry');self.assertEqual(result['steps'],5)
        self.assertFalse(any(row[0]=='open' for row in b.trace))
        self.assertEqual(b.trace[-1],('close',));self.assertIsNone(b._decider)

    def fake_joint_home(self,observed_error=.001,plan_status='Success'):
        b=self.fake_grasp_backend(np.ones((4,4),bool))
        b.initial_qpos=b.obs['agent']['qpos'][0].copy()
        b.obs['agent']['qpos'][0,:7]+=.3
        b.base=SimpleNamespace(control_timestep=.05)
        def follow(result,refine_steps):
            b.obs['agent']['qpos'][0,:7]=b.initial_qpos[:7]+observed_error
        planner=SimpleNamespace(OPEN=1,gripper_state=-1,use_point_cloud=False,
            planner=SimpleNamespace(plan_qpos_to_qpos=Mock(return_value={'status':plan_status})),
            follow_path=Mock(side_effect=follow))
        return b,planner

    def test_exact_joint_home_preserves_current_fingers_and_checks_observed_arm(self):
        b,planner=self.fake_joint_home()
        b.obs['agent']['qpos'][0,7:]=[.031,.029]
        before=b.obs['agent']['qpos'][0].copy()
        error=b._return_joint_home(planner,'retreat_joint_home')
        args=planner.planner.plan_qpos_to_qpos.call_args
        goal=args.args[0][0]
        self.assertEqual(goal.shape,(9,))
        self.assertTrue(np.allclose(goal[:7],b.initial_qpos[:7]))
        self.assertTrue(np.allclose(goal[7:],before[7:]))
        self.assertTrue(np.allclose(args.args[1],before))
        self.assertEqual(args.kwargs['time_step'],.05)
        planner.follow_path.assert_called_once_with({'status':'Success'},refine_steps=10)
        self.assertEqual(planner.gripper_state,1);self.assertAlmostEqual(error,.001,places=5)

    def test_joint_home_plan_failure_does_not_follow_path(self):
        b,planner=self.fake_joint_home(plan_status='RRT Failed')
        with self.assertRaisesRegex(RuntimeError,'initial_joint_home_plan_failed'):
            b._return_joint_home(planner,'retreat_joint_home')
        planner.follow_path.assert_not_called()

    def test_joint_home_stop_prevents_planning_or_following(self):
        b,planner=self.fake_joint_home();b.stop_event.set()
        with self.assertRaises(BackendStopped):b._return_joint_home(planner,'retreat_joint_home')
        planner.planner.plan_qpos_to_qpos.assert_not_called();planner.follow_path.assert_not_called()

    def test_joint_home_rejects_proprio_error_or_nonfinite_observation(self):
        for error in (.021,float('nan')):
            b,planner=self.fake_joint_home(observed_error=error)
            with self.assertRaisesRegex(RuntimeError,'initial_joint_home_not_reached'):
                b._return_joint_home(planner,'retreat_joint_home')

    def test_home_failure_prevents_post_placement_continuation(self):
        b=self.fake_placement_backend()
        b._return_joint_home=Mock(side_effect=RuntimeError('initial_joint_home_not_reached'))
        with self.assertRaisesRegex(RuntimeError,'initial_joint_home_not_reached'):
            b.transport_place(0)
        self.assertFalse(any(row[0]=='step' and row[1]=='settle' for row in b.trace))
        self.assertEqual(b.trace[-1],('close',));self.assertIsNone(b._decider)

    def test_recovery_uses_exact_joint_home(self):
        b=self.fake_placement_backend();b._close_perception=lambda:None;b._publish=lambda **kwargs:None
        result=b.recover_open_home()
        self.assertEqual(result['reason'],'recovery_motion_completed')
        self.assertIn(('joint_home','recovery_joint_home'),b.trace)
        self.assertEqual(result['home_joint_max_error_rad'],.001)

    def test_noise_seed_is_repeatable_per_scene_and_control_frame(self):
        self.assertEqual(sampling_seed(17,20),sampling_seed(17,20))
        self.assertNotEqual(sampling_seed(17,20),sampling_seed(17,21))

    def test_sam_prompt_padding_uses_floor_and_longest_side(self):
        original=[152.,161.,164.,174.]
        self.assertEqual(sam_prompt_bbox(original,224,224),[148.,157.,168.,178.])
        self.assertEqual(original,[152.,161.,164.,174.])
        self.assertEqual(sam_prompt_bbox([30,40,130,60],224,224),[15.,25.,145.,75.])

    def test_sam_prompt_padding_clips_all_image_boundaries(self):
        self.assertEqual(sam_prompt_bbox([1,2,223,224],224,224),[0.,0.,224.,224.])
        self.assertEqual(sam_prompt_bbox([0,0,2,2],224,224),[0.,0.,6.,6.])
        for invalid in ([10,10,5,20],[-1,0,4,4],[0,0,225,2],[0,0,1,2],[0,0,float('nan'),4]):
            with self.assertRaises(ValueError):sam_prompt_bbox(invalid,224,224)

    def test_perception_metadata_separates_vlm_box_from_padded_sam_prompt(self):
        b=RobotBackend.__new__(RobotBackend)
        b._last_target_bbox=[152.,161.,164.,174.]
        b._sam_prompt_bbox=[148.,157.,168.,178.]
        from types import SimpleNamespace
        result=SimpleNamespace(mask=np.ones((4,4),bool),metadata={'vlm_bbox_xyxy':b._sam_prompt_bbox.copy(),
            'events':[{'type':'vlm_bbox','origin':'upstream_task_vlm'}]})
        b._accept_perception(result,(4,4))
        self.assertEqual(b._mask_metadata['task_vlm_bbox_xyxy'],b._last_target_bbox)
        self.assertEqual(b._mask_metadata['vlm_bbox_xyxy'],b._last_target_bbox)
        self.assertEqual(b._mask_metadata['sam_prompt_bbox_xyxy'],b._sam_prompt_bbox)
        self.assertEqual(b._mask_metadata['events'][0]['origin'],'upstream_task_vlm_padded_by_task_backend')
        self.assertEqual(result.metadata['events'][0]['origin'],'upstream_task_vlm')

    def test_snapshot_is_independent_and_contains_no_simulator_state(self):
        backend=RobotBackend.__new__(RobotBackend)
        backend._snapshot_lock=threading.RLock()
        backend._snapshot=dict(frame_id=7,captured_at=4.,base_rgb=np.zeros((4,4,3),np.uint8),
            wrist_rgb=np.ones((4,4,3),np.uint8),mask=np.ones((4,4),bool),
            overview_rgb=np.full((8,8,3),2,np.uint8),
            mask_metadata={'state':'tracking','events':[]},phase='grasp',step_generation=2)
        a=backend.snapshot();a['base_rgb'][:]=255;a['overview_rgb'][:]=255
        a['mask'][:]=False;a['mask_metadata']['events'].append('edit')
        b=backend.snapshot()
        self.assertFalse(b['base_rgb'].any());self.assertTrue(b['mask'].all())
        self.assertTrue((b['overview_rgb']==2).all())
        self.assertEqual(b['mask_metadata']['events'],[])
        self.assertFalse({'objects','positions','segmentation','is_grasping','target_idx'} & set(b))

    def fake_tracking_lifecycle_backend(self):
        b=RobotBackend.__new__(RobotBackend)
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        b.output=Path(temporary.name)
        b.perception=SimpleNamespace(close=Mock())
        b._mask=np.ones((224,224),bool)
        b._mask_metadata={'state':'tracking','source':'vlm_sam2_cutie','generation':1,
            'vlm_bbox_xyxy':[10,20,30,40]}
        b._last_target_bbox=[10,20,30,40];b._sam_prompt_bbox=[6,16,34,44]
        b._tracking_frame=99;b.steps=1;b._captured_at=1.;b.motion_phase='settle';b.policy_name='pi05'
        b.frame_callback=None;b._snapshot=None;b._snapshot_lock=threading.RLock()
        b.obs=observation()
        b.obs['sensor_data']={
            'base_camera':{'rgb':np.full((1,224,224,3),51,np.uint8)},
            'hand_camera':{'rgb':np.full((1,224,224,3),73,np.uint8)},
            'overview_camera':{'rgb':np.full((1,448,448,3),95,np.uint8)}}
        return b

    def test_no_active_target_closes_tracking_and_clears_published_state(self):
        for phase in ('idle','goal_verify'):
            with self.subTest(phase=phase):
                b=self.fake_tracking_lifecycle_backend();old_session=b.perception
                b.set_context(phase,'',step_generation=7)
                snapshot=b.snapshot()
                old_session.close.assert_called_once();self.assertIsNone(b.perception)
                self.assertIsNone(snapshot['mask']);self.assertIsNone(snapshot['target_bbox_xyxy'])
                self.assertIsNone(snapshot['sam_prompt_bbox_xyxy'])
                self.assertEqual(snapshot['mask_metadata'],{'state':'idle','source':'none','generation':0})
                self.assertEqual(snapshot['phase'],phase);self.assertEqual(snapshot['step_generation'],7)
                self.assertEqual(b._tracking_frame,0)
                self.assertTrue((snapshot['base_rgb']==51).all());self.assertTrue((snapshot['wrist_rgb']==73).all())
                # A context-only update must replace live.png immediately,
                # even when step 1 is outside the usual five-step save cadence.
                from PIL import Image
                live=np.array(Image.open(b.output/'live.png'))
                np.testing.assert_array_equal(live[344:568,272:496],0)
                self.assertEqual(tuple(live[88+20,272+10]),(51,51,51))
                b.set_context(phase,'',step_generation=7)
                old_session.close.assert_called_once()

    def test_targeted_activity_phases_preserve_tracking_and_current_mask(self):
        for phase in ('grasp','transport','verify','recovery'):
            with self.subTest(phase=phase):
                b=self.fake_tracking_lifecycle_backend();session=b.perception
                b.set_context(phase,'blue cube',step_generation=8)
                snapshot=b.snapshot()
                self.assertIs(b.perception,session);session.close.assert_not_called()
                self.assertTrue(snapshot['mask'].all())
                self.assertEqual(snapshot['target_bbox_xyxy'],[10,20,30,40])
                self.assertEqual(snapshot['sam_prompt_bbox_xyxy'],[6,16,34,44])
                self.assertEqual(snapshot['mask_metadata']['state'],'tracking')
                self.assertEqual(snapshot['target_text'],'blue cube');self.assertEqual(b._tracking_frame,99)

    def test_overview_is_excluded_from_policy_images(self):
        backend=RobotBackend.__new__(RobotBackend)
        backend.obs=observation()
        backend.obs['sensor_data']={
            'base_camera':{'rgb':np.full((1,224,224,3),10,np.uint8)},
            'hand_camera':{'rgb':np.full((1,224,224,3),20,np.uint8)},
            'overview_camera':{'rgb':np.full((1,448,448,3),30,np.uint8)}}
        base,wrist=backend._raw_images()
        self.assertEqual(base.shape,(224,224,3));self.assertEqual(wrist.shape,(224,224,3))
        self.assertTrue((base==10).all());self.assertTrue((wrist==20).all())
        backend._mask=np.ones((224,224),bool);backend.seed=17;backend.steps=5
        backend.pi05_url='http://127.0.0.1:18833'
        sent={}
        def reply(request,timeout):
            sent.update(json.loads(request.data))
            return io.BytesIO(json.dumps({'noise_seed':sent['noise_seed'],'actions':[[0]*8]}).encode())
        with patch('task_agent.backend.urlopen',side_effect=reply):backend._pi_prediction()
        self.assertEqual(set(sent),{'base_rgb','wrist_rgb','target_rgb','state','noise_seed'})
        import base64
        for key in ('base_rgb','wrist_rgb','target_rgb'):
            self.assertEqual(len(base64.b64decode(sent[key])),224*224*3)

    def test_privileged_target_parameters_are_rejected(self):
        backend=RobotBackend.__new__(RobotBackend)
        for key in ['object_id','actor_id','target_idx','gt_mask','segmentation','target_pose','world_position']:
            with self.assertRaises(ValueError):
                backend._start_perception({'description':'red ball','bbox_xyxy':[2,3,8,9],key:1})

    def fake_grasp_backend(self,mask):
        b=RobotBackend.__new__(RobotBackend)
        b.stop_event=threading.Event();b._decider=None;b.steps=0;b.policy_name='pi05'
        b.max_grasp_steps=8;b.max_grasp_seconds=1;b.mask_lost_timeout=8;b.obs=observation()
        b._mask=mask;b._mask_metadata={'state':'tracking' if mask.any() else 'lost','generation':1}
        b._start_perception=lambda target:None
        b.executed=[]
        def step(action):b.executed.append(np.array(action));b.steps+=1
        b._step=step
        return b

    def test_lost_mask_holds_and_waits_for_external_visual_retry(self):
        b=self.fake_grasp_backend(np.zeros((4,4),bool))
        def forbidden():raise AssertionError('Lost target must not invoke policy')
        b._pi_prediction=forbidden
        result=b.grasp({'description':'red ball','bbox_xyxy':[1,1,3,3]},lambda:'retry' if b.steps>=2 else None)
        self.assertEqual(result['reason'],'retry');self.assertEqual(result['steps'],2)
        self.assertTrue(all(np.allclose(a,hold_action_from_observation(b.obs)) for a in b.executed))

    def test_mask_validation_can_reject_before_any_policy_action(self):
        b=self.fake_grasp_backend(np.ones((4,4),bool));b.snapshot=lambda:{'base_rgb':np.zeros((4,4,3),np.uint8),'mask':b._mask.copy()}
        def forbidden():raise AssertionError('Rejected visual target must not invoke policy')
        b._pi_prediction=forbidden
        result=b.grasp({'description':'ball','bbox_xyxy':[1,1,3,3]},
                      validate_mask=lambda snapshot:{'valid':False,'reason':'Mask shows wooden table'})
        self.assertEqual(result['reason'],'mask_target_mismatch')
        self.assertEqual(result['steps'],0);self.assertFalse(b.executed)
        self.assertIn('wooden table',result['evidence'])

    def test_stop_during_mask_validation_prevents_policy_action(self):
        b=self.fake_grasp_backend(np.ones((4,4),bool));b.snapshot=lambda:{}
        def stop_during_validation(snapshot):
            b.stop_event.set();return {'valid':True,'reason':'target matched'}
        with self.assertRaises(BackendStopped):
            b.grasp({'description':'ball','bbox_xyxy':[1,1,3,3]},validate_mask=stop_during_validation)
        self.assertFalse(b.executed)

    def test_mask_validation_rejects_nonboolean_valid_field(self):
        b=self.fake_grasp_backend(np.ones((4,4),bool));b.snapshot=lambda:{}
        for value in (1,'true',None):
            with self.assertRaises(ValueError):
                b.grasp({'description':'ball','bbox_xyxy':[1,1,3,3]},
                        validate_mask=lambda snapshot:{'valid':value})
        self.assertFalse(b.executed)

    def test_display_bbox_does_not_modify_raw_rgb(self):
        b=self.fake_grasp_backend(np.ones((4,4),bool))
        b.phase='grasp';b.motion_phase='grasp_lift'
        raw=np.full((224,224,3),42,np.uint8)
        snapshot=dict(base_rgb=raw,wrist_rgb=raw.copy(),mask=np.ones((224,224),bool),
            overview_rgb=np.zeros((448,448,3),np.uint8),target_bbox_xyxy=[10,20,30,40])
        frame=b._frame(snapshot)
        self.assertEqual(frame.shape,(576,512,3))
        self.assertTrue((raw==42).all())
        self.assertEqual(tuple(frame[88+20,272+10]),(50,230,255))
        # The displayed target is exactly the input's native RGB pixels;
        # its title lives in the gutter and cannot overwrite target pixels.
        np.testing.assert_array_equal(frame[344:568,272:496],raw)
        np.testing.assert_array_equal(frame[344:568,16:240],snapshot['wrist_rgb'])
        np.testing.assert_array_equal(frame[88:312,16:240],0)
        self.assertEqual(snapshot['overview_rgb'].shape,(448,448,3))
        for x,y in ((16,88),(272,88),(16,344),(272,344)):
            self.assertTrue((frame[y-16:y-1,x:x+224]>200).all(axis=2).any())
        self.assertFalse((frame[328:344,496:512]>200).all(axis=2).any())
        snapshot['mask']=np.zeros((224,224),bool)
        empty=b._frame(snapshot)
        np.testing.assert_array_equal(empty[344:568,272:496],0)

    def test_historical_metadata_bbox_is_not_drawn_without_current_target_box(self):
        b=self.fake_grasp_backend(np.ones((4,4),bool));b.phase='idle';b.motion_phase='settle'
        raw=np.full((224,224,3),42,np.uint8)
        snapshot=dict(base_rgb=raw,wrist_rgb=raw.copy(),overview_rgb=np.zeros((448,448,3),np.uint8),
            mask=None,target_bbox_xyxy=None,mask_metadata={'vlm_bbox_xyxy':[10,20,30,40]})
        frame=b._frame(snapshot)
        np.testing.assert_array_equal(frame[88:312,272:496],raw)
        np.testing.assert_array_equal(frame[344:568,272:496],0)

    def attach_dropout(self,b,config=None,control_path=None):
        clock=SimpleNamespace(now=100.,wall=1000.)
        b._mask_dropout=MaskDropout(config,control_path,
            clock=lambda:clock.now,wall_clock=lambda:clock.wall)
        b.phase='grasp';b.step_generation=3;b.target_text='current visual target'
        b._natural_mask=b._mask.copy()
        b._natural_mask_metadata=dict(b._mask_metadata,natural_lost=False,quality=.93,
                                      intervention='none',intervention_active=False)
        b._mask_dropout_state={}
        return clock

    def test_dropout_preserves_natural_quality_and_restores_only_natural_mask(self):
        b=self.fake_grasp_backend(np.ones((4,4),bool));clock=self.attach_dropout(b,
            {'enabled':True,'start_seconds':0,'duration_seconds':.1})
        b._mask_dropout.begin_attempt(3);b._refresh_mask_dropout()
        self.assertFalse(b._mask.any());self.assertTrue(b._natural_mask.all())
        self.assertEqual(b._mask_metadata['quality'],.93);self.assertFalse(b._mask_metadata['natural_lost'])
        self.assertEqual(b._mask_metadata['natural_state'],'tracking');self.assertEqual(b._mask_metadata['state'],'lost')
        self.assertTrue(b._mask_metadata['intervention_active']);self.assertEqual(b._mask_metadata['natural_mask_pixels'],16)
        clock.now+=.2;b._refresh_mask_dropout()
        self.assertTrue(b._mask.all());self.assertEqual(b._mask_metadata['state'],'tracking')
        self.assertFalse(b._mask_metadata['intervention_active'])
        # Manual/automatic restore is never a synthetic replacement for a
        # genuinely lost natural tracker output.
        b._natural_mask[:]=False;b._natural_mask_metadata.update(state='lost',natural_lost=True)
        b._refresh_mask_dropout();self.assertFalse(b._mask.any());self.assertTrue(b._mask_metadata['natural_lost'])

    def test_dropout_start_zero_reaches_policy_after_preflight_and_invalidates_chunk_on_return(self):
        b=self.fake_grasp_backend(np.ones((4,4),bool));clock=self.attach_dropout(b,
            {'enabled':True,'start_seconds':0,'duration_seconds':.1})
        b._publish=lambda **kwargs:None;b.snapshot=lambda:{'mask':b._mask.copy()}
        preflight=[];predictions=[]
        def validate(snapshot):
            preflight.append(snapshot['mask'].copy());clock.now+=20
            self.assertIsNone(b._mask_dropout.attempt_started)
            return {'valid':True,'reason':'actual visible target'}
        chunk=np.ones((10,8),np.float32);chunk[:,:7]=.2
        def predict():predictions.append(b._mask.copy());return chunk
        b._pi_prediction=predict;step=b._step
        def tick(action):step(action);clock.now+=.06
        b._step=tick
        result=b.grasp({'description':'ball','bbox_xyxy':[1,1,3,3]},
            lambda:'grasped' if b.steps>=4 else None,validate_mask=validate)
        self.assertTrue(preflight[0].all());self.assertFalse(predictions[0].any())
        self.assertTrue(predictions[-1].all());self.assertEqual(len(predictions),2)
        self.assertEqual(result['steps'],4)
        self.assertTrue(all(float(action[-1])==-1 for action in b.executed))
        self.assertTrue(b._mask.all());self.assertFalse(b._mask_dropout_state['active'])

    def test_dropout_pi_payload_zeroes_only_target_image(self):
        b=self.fake_tracking_lifecycle_backend();b.phase='grasp';b.step_generation=3
        b._mask=np.ones((224,224),bool);self.attach_dropout(b,{'enabled':True,'start_seconds':0})
        b.seed=17;b.pi05_url='http://127.0.0.1:18833';b._mask_dropout.begin_attempt(3);b._refresh_mask_dropout()
        sent={}
        def reply(request,timeout):
            sent.update(json.loads(request.data))
            return io.BytesIO(json.dumps({'noise_seed':sent['noise_seed'],'actions':[[0]*8]}).encode())
        with patch('task_agent.backend.urlopen',side_effect=reply):b._pi_prediction()
        self.assertEqual(set(sent),{'base_rgb','wrist_rgb','target_rgb','state','noise_seed'})
        self.assertTrue((np.frombuffer(base64.b64decode(sent['base_rgb']),np.uint8)==51).all())
        self.assertTrue((np.frombuffer(base64.b64decode(sent['wrist_rgb']),np.uint8)==73).all())
        self.assertFalse(np.frombuffer(base64.b64decode(sent['target_rgb']),np.uint8).any())
        self.assertTrue(b._natural_mask.all())

    def test_dropout_act_payload_zeroes_only_masked_stream_without_loading_model(self):
        import torch
        import sys
        b=self.fake_tracking_lifecycle_backend();b._mask=np.ones((224,224),bool)
        self.attach_dropout(b,{'enabled':True,'start_seconds':0})
        b._mask_dropout.begin_attempt(3);b._refresh_mask_dropout()
        b.device=torch.device('cpu');b._act_args=SimpleNamespace(num_queries=2)
        b._act_stats={'state_mean':torch.zeros(28),'state_std':torch.ones(28)}
        observed={}
        def dual(original,masked,resize):
            return torch.stack([torch.from_numpy(original).permute(0,3,1,2),
                                torch.from_numpy(masked).permute(0,3,1,2)],dim=1)
        def predict(observation):observed.update(observation);return torch.zeros((1,2,8))
        b._act_agent=SimpleNamespace(get_action=predict)
        module=SimpleNamespace(dual_rgb=dual,policy_state=lambda obs:torch.zeros((1,28)))
        with patch.dict(sys.modules,{'train_dual_rgb':module}):b._act_prediction()
        self.assertEqual(tuple(observed['rgb'].shape),(1,2,3,224,224))
        self.assertTrue((observed['rgb'][:,0]==51).all());self.assertFalse(observed['rgb'][:,1].any())
        self.assertTrue(b._natural_mask.all())

    def test_snapshot_reads_do_not_process_manual_control_file(self):
        with TemporaryDirectory() as temporary:
            path=Path(temporary)/'control.json';b=self.fake_tracking_lifecycle_backend()
            self.attach_dropout(b,control_path=path);b._mask_dropout.begin_attempt(3)
            b._publish(record_video=False)
            path.write_text(json.dumps(dict(command_id='manual1',action='trigger',duration_seconds=.5,
                step_generation=3,issued_at=1000.)))
            self.assertFalse(b.snapshot()['mask_dropout_state']['active'])
            self.assertFalse(b.snapshot()['mask_dropout_state']['active'])
            b._publish(record_video=False)
            state=b.snapshot()['mask_dropout_state'];self.assertTrue(state['active'])
            self.assertEqual(state['source'],'manual');self.assertEqual(state['last_command_id'],'manual1')

    def test_start_perception_uses_padded_sam_box_but_tight_single_target_reference(self):
        b=self.fake_tracking_lifecycle_backend();b.perception_url='http://127.0.0.1:18830'
        b._publish=lambda **kwargs:None
        client=Mock()
        client.start_from_vlm.return_value=SimpleNamespace(mask=np.ones((224,224),bool),
            metadata={'state':'tracking','generation':1,'events':[]})
        target={'description':'one blue cube','bbox_xyxy':[20,30,34,46],'request_id':'one'}
        with patch('perception_mask.client.PerceptionMaskClient',return_value=client):b._start_perception(target)
        call=client.start_from_vlm.call_args
        self.assertEqual(call.args[2],[16.,26.,38.,50.])
        self.assertEqual(call.kwargs['options'],{'loss_mode':'observe','single_target':True,
            'single_target_bbox':[20.,30.,34.,46.]})
        self.assertEqual(b._mask_metadata['task_vlm_bbox_xyxy'],target['bbox_xyxy'])

    def test_dropout_event_callback_does_not_override_root_event_timestamp(self):
        with TemporaryDirectory() as temporary:
            b=self.fake_grasp_backend(np.ones((4,4),bool));b.output=Path(temporary);b.phase='grasp'
            rows=[]
            def root_event(event,**values):rows.append(dict(event=event,time=1.,**values))
            b.event_callback=root_event
            b._dropout_event('mask_dropout_started',source='manual',step_generation=3)
            self.assertEqual(rows[0]['time'],1.);self.assertEqual(rows[0]['source'],'manual')
            self.assertTrue((b.output/'mask_dropout.jsonl').is_file())

    def test_close_publishes_dropout_end_before_saving_final_image(self):
        from PIL import Image
        with TemporaryDirectory() as temporary:
            b=self.fake_tracking_lifecycle_backend();b.output=Path(temporary);b._closed=False;b._decider=None
            self.attach_dropout(b,{'enabled':True,'start_seconds':0})
            b._mask_dropout.begin_attempt(3);b._publish(record_video=False)
            self.assertTrue(b.snapshot()['mask_dropout_state']['active'])
            b.close()
            self.assertFalse(b.snapshot()['mask_dropout_state']['active'])
            self.assertFalse(b.snapshot()['mask_dropout_state']['ready'])
            picture=np.asarray(Image.open(b.output/'final.png'))
            self.assertTrue((picture[344:568,272:496]==51).all())

    def test_observe_continues_policy_with_zero_mask_after_real_loss(self):
        b=self.fake_grasp_backend(np.ones((4,4),bool));seen=[]
        chunk=np.ones((10,8),np.float32);chunk[:,:7]=.2
        def predict():seen.append(b._mask.copy());return chunk
        b._pi_prediction=predict
        step=b._step
        def lose_after_step(action):
            step(action);b._mask=np.zeros((4,4),bool);b._mask_metadata['state']='lost'
        b._step=lose_after_step
        result=b.grasp({'description':'ball','bbox_xyxy':[1,1,3,3]},lambda:'retry' if b.steps>=3 else None)
        self.assertEqual(result['reason'],'retry');self.assertEqual(result['steps'],3)
        self.assertEqual(len(seen),2);self.assertTrue(seen[0].all());self.assertFalse(seen[1].any())
        self.assertTrue(all(float(a[-1])==-1 for a in b.executed))
        self.assertEqual(b.control_source,'pi05_zero_mask')

    def test_observe_loss_is_bounded_by_timeout(self):
        b=self.fake_grasp_backend(np.ones((4,4),bool));b.mask_lost_timeout=0
        b._pi_prediction=lambda:np.ones((10,8),np.float32)
        step=b._step
        def lose_after_step(action):
            step(action);b._mask=np.zeros((4,4),bool);b._mask_metadata['state']='lost'
        b._step=lose_after_step
        result=b.grasp({'description':'ball','bbox_xyxy':[1,1,3,3]})
        self.assertEqual(result['reason'],'mask_lost_timeout')
        self.assertEqual(result['steps'],1)

    def test_grasp_progress_requires_external_visual_decision(self):
        b=self.fake_grasp_backend(np.ones((4,4),bool))
        chunk=np.ones((10,8),np.float32);chunk[:,:7]=.2
        b._pi_prediction=lambda:chunk
        b.obs['diagnostic_gt']={'is_grasping':True,'success':True}
        result=b.grasp({'description':'ball','bbox_xyxy':[1,1,3,3]},lambda:'grasped' if b.steps>=3 else None)
        self.assertEqual(result['reason'],'grasped');self.assertEqual(result['steps'],3)
        self.assertTrue(all(float(a[-1])==-1 for a in b.executed))
        self.assertIsNone(b._decider)

    def test_simulator_success_field_does_not_end_grasp(self):
        b=self.fake_grasp_backend(np.ones((4,4),bool));b.max_grasp_steps=2
        b.obs['success']=True;b._pi_prediction=lambda:np.zeros((10,8),np.float32)
        result=b.grasp({'description':'ball','bbox_xyxy':[1,1,3,3]})
        self.assertEqual(result['reason'],'grasp_step_timeout')

    def test_cancel_prevents_another_action(self):
        b=self.fake_grasp_backend(np.ones((4,4),bool));b.stop_event.set()
        with self.assertRaises(BackendStopped):b.grasp({'description':'ball','bbox_xyxy':[1,1,3,3]})
        self.assertFalse(b.executed)


if __name__=='__main__':unittest.main()
