"""Three-stream task ACT contracts; CPU only, no simulator or model startup."""
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from task_agent.backend import RobotBackend, TRIPLE_CHECKPOINT, TRIPLE_OPTIMIZER_STEP


class TripleBackendContracts(unittest.TestCase):
    def fake_inputs(self):
        backend=RobotBackend.__new__(RobotBackend)
        backend.policy_name='act_triple'
        backend._policy_bound=True
        backend.device=torch.device('cpu')
        backend._act_args=SimpleNamespace(num_queries=30,temporal_agg=True)
        backend.obs=dict(
            agent=dict(qpos=np.arange(9,dtype=np.float32)[None],
                       qvel=np.arange(10,19,dtype=np.float32)[None]),
            extra=dict(tcp_pose=np.arange(20,27,dtype=np.float32)[None],
                       goal_pos=np.array([[8.,9.,10.]],np.float32),
                       target_pose=object(),target_idx=object()),
            sensor_data=dict(
                base_camera=dict(rgb=np.full((1,224,224,3),51,np.uint8)),
                hand_camera=dict(rgb=np.full((1,224,224,3),73,np.uint8)),
                overview_camera=dict(rgb=np.full((1,448,448,3),95,np.uint8))))
        backend._mask=np.zeros((224,224),bool)
        backend._mask[40:80,90:150]=True
        backend._natural_mask=np.ones((224,224),bool)
        return backend

    def capture_prediction(self,backend):
        captured={}
        normalized=torch.arange(240,dtype=torch.float32).reshape(1,30,8)
        def predict(observation,return_normalized=False):
            captured.update(observation)
            self.assertTrue(return_normalized)
            return normalized*100+1000,normalized
        backend._act_policy=SimpleNamespace(predict=predict)
        prediction=backend._act_prediction()
        self.assertIs(prediction,normalized)
        return captured

    def test_stream_order_uses_current_effective_mask_and_approved_state28(self):
        backend=self.fake_inputs()
        captured=self.capture_prediction(backend)
        rgb=captured['rgb']
        self.assertEqual(tuple(rgb.shape),(1,3,3,224,224))
        self.assertEqual(rgb.dtype,torch.uint8)
        self.assertTrue((rgb[:,0]==51).all())
        self.assertTrue((rgb[:,1]==73).all())
        expected=np.where(backend._mask[...,None],51,0)
        np.testing.assert_array_equal(rgb[0,2].numpy(),np.broadcast_to(expected,(224,224,3)).transpose(2,0,1))
        state=captured['state']
        self.assertEqual(tuple(state.shape),(1,28))
        np.testing.assert_array_equal(state[0,:25].numpy(),np.r_[np.arange(9),np.arange(10,19),np.arange(20,27)])
        np.testing.assert_allclose(state[0,25:].numpy(),[0,0,.2])
        self.assertTrue(backend._natural_mask.all())
        self.assertTrue((backend.obs['sensor_data']['base_camera']['rgb']==51).all())
        np.testing.assert_array_equal(backend.obs['extra']['goal_pos'],[[8,9,10]])

    def test_zero_effective_mask_keeps_raw_base_and_wrist_streams(self):
        for mask in (np.zeros((224,224),bool),None):
            with self.subTest(mask_absent=mask is None):
                backend=self.fake_inputs();backend._mask=mask
                captured=self.capture_prediction(backend)
                self.assertTrue((captured['rgb'][:,0]==51).all())
                self.assertTrue((captured['rgb'][:,1]==73).all())
                self.assertFalse(captured['rgb'][:,2].any())
                self.assertTrue(backend._natural_mask.all())

    def test_three_policy_ids_keep_distinct_loading_routes(self):
        backend=RobotBackend.__new__(RobotBackend)
        backend._act_checkpoint=None
        backend._load_act=Mock();backend._load_act_triple=Mock()
        backend.pi05_url='http://127.0.0.1:18833'
        backend._prepare_policy('act')
        legacy=backend._load_act.call_args.args[0]
        self.assertTrue(str(legacy).endswith('ckpt/maniskill/act_dual_rgb_8x100/100000.pt'))
        backend._load_act_triple.assert_not_called()
        backend._prepare_policy('act_triple')
        self.assertTrue(str(backend._load_act_triple.call_args.args[0]).endswith(TRIPLE_CHECKPOINT))
        with patch('task_agent.backend.urlopen',return_value=io.BytesIO(b'{"supports_noise_seed":true}')) as health:
            backend._prepare_policy('pi05')
        health.assert_called_once_with('http://127.0.0.1:18833/health',timeout=5)
        self.assertEqual(backend._load_act.call_count,1)
        self.assertEqual(backend._load_act_triple.call_count,1)

    def test_triple_loader_uses_strict_ema_facade_and_records_actual_provenance(self):
        backend=RobotBackend.__new__(RobotBackend)
        facade=SimpleNamespace(agent=object(),stats={'action_mean':torch.zeros((1,8))},
            metadata=dict(optimizer_step=160000,architecture={'num_queries':30},checkpoint_version=1))
        checkpoint=Path('/tmp/triple-task-contract/checkpoints/160000.pt')
        with patch('torch.cuda.is_available',return_value=False), \
             patch('evaluate_triple_rgb.CheckpointPolicy',return_value=facade) as loader:
            backend._load_act_triple(checkpoint)
        loader.assert_called_once_with(checkpoint,device=torch.device('cpu'))
        self.assertIs(backend._act_policy,facade)
        self.assertIs(backend._act_agent,facade.agent)
        self.assertIs(backend._act_stats,facade.stats)
        self.assertEqual(backend._act_args.num_queries,30)
        self.assertTrue(backend._act_args.temporal_agg)
        metadata=backend._act_metadata
        self.assertEqual(metadata['checkpoint_path'],str(checkpoint.resolve()))
        self.assertEqual(metadata['optimizer_step'],TRIPLE_OPTIMIZER_STEP)
        self.assertEqual(metadata['stream_order'],['base_rgb','wrist_rgb','base_masked_rgb'])
        self.assertEqual(metadata['state_fields'],['agent/qpos','agent/qvel','extra/tcp_pose','extra/goal_pos'])
        self.assertEqual(metadata['weights_source'],'ema_agent')
        self.assertEqual(metadata['policy_device_metadata']['device'],'cpu')
        json.dumps(metadata,allow_nan=False)

    def test_triple_loader_rejects_a_different_optimizer_step(self):
        for step in (None,100000,159999,200000):
            backend=RobotBackend.__new__(RobotBackend)
            facade=SimpleNamespace(metadata={'optimizer_step':step})
            with self.subTest(step=step),patch('torch.cuda.is_available',return_value=False), \
                 patch('evaluate_triple_rgb.CheckpointPolicy',return_value=facade), \
                 self.assertRaisesRegex(ValueError,'optimizer step 160000'):
                backend._load_act_triple(Path('/tmp/160000.pt'))

    def fake_grasp(self,policy='act_triple',temporal_agg=None,steps=3):
        backend=self.fake_inputs()
        backend.policy_name=policy
        backend.stop_event=threading.Event();backend._decider=None
        backend.steps=0;backend.max_grasp_steps=steps
        backend.max_grasp_seconds=10;backend.mask_lost_timeout=8
        backend._mask=np.ones((224,224),bool)
        backend._mask_metadata={'state':'tracking','generation':1}
        backend._natural_mask=backend._mask.copy()
        backend._natural_mask_metadata=dict(backend._mask_metadata)
        backend._act_args=SimpleNamespace(num_queries=30)
        if temporal_agg is not None:backend._act_args.temporal_agg=temporal_agg
        backend._act_stats=dict(action_mean=torch.arange(8,dtype=torch.float32)[None]+10,
                               action_std=torch.arange(8,dtype=torch.float32)[None]+2)
        backend._start_perception=Mock()
        backend.executed=[]
        def step(action):
            backend.executed.append(np.array(action));backend.steps+=1
        backend._step=step
        return backend

    def test_temporal_aggregation_defaults_to_original_act_and_denormalizes_once(self):
        traces=[]
        for policy,temporal in (('act',True),('act_triple',None)):
            backend=self.fake_grasp(policy,temporal)
            chunks=[]
            def predict():
                chunk=(torch.arange(30,dtype=torch.float32)[None,:,None]
                       +10*len(chunks)+1).expand(1,30,8).clone()
                chunks.append(chunk)
                return chunk
            backend._act_prediction=Mock(side_effect=predict)
            result=backend.grasp({'description':'visible target','bbox_xyxy':[1,1,3,3]})
            self.assertEqual(result['reason'],'grasp_step_timeout')
            self.assertEqual(backend._act_prediction.call_count,3)
            for tick,action in enumerate(backend.executed):
                weights=torch.exp(-.01*torch.arange(tick+1))
                raw=sum(chunks[t][:,tick-t]*(weights[t]/weights.sum()) for t in range(tick+1))
                expected=raw*backend._act_stats['action_std']+backend._act_stats['action_mean']
                np.testing.assert_allclose(action,expected[0].numpy(),rtol=1e-6)
            traces.append(backend.executed)
        np.testing.assert_allclose(traces[0],traces[1])

    def test_zero_mask_recovery_and_generation_changes_retire_cached_chunks(self):
        backend=self.fake_grasp(temporal_agg=False,steps=4)
        queries=[]
        def predict():
            queries.append((backend._mask.copy(),backend._mask_metadata['generation']))
            return torch.full((1,30,8),float(len(queries)))
        backend._act_prediction=Mock(side_effect=predict)
        original_step=backend._step
        def step(action):
            original_step(action)
            if backend.steps==1:
                backend._mask[:]=False;backend._mask_metadata['state']='lost'
            elif backend.steps==2:
                backend._mask[:]=True;backend._mask_metadata['state']='tracking'
            elif backend.steps==3:
                backend._mask_metadata['generation']=2
                backend._natural_mask_metadata['generation']=2
        backend._step=step
        backend.grasp({'description':'visible target','bbox_xyxy':[1,1,3,3]})
        self.assertEqual(len(queries),4)
        self.assertEqual([bool(mask.any()) for mask,_ in queries],[True,False,True,True])
        self.assertEqual([generation for _,generation in queries],[1,1,1,2])
        for tick,action in enumerate(backend.executed):
            expected=(tick+1)*backend._act_stats['action_std']+backend._act_stats['action_mean']
            np.testing.assert_allclose(action,expected[0].numpy())


if __name__=='__main__':unittest.main()
