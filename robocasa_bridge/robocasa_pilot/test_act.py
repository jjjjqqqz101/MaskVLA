"""Meaningful padding, gradient and checkpoint contract checks (no simulator)."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .act_train import ACTModel,CONTRACT,model_config,masked_l1,losses,atomic_save,ROOT
from .act_infer import ACTPolicy

class ACTContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(17)
        cls.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cls.model=ACTModel(model_config(horizon=4,hidden_dim=128,enc_layers=1,dec_layers=1,dropout=0.),pretrained=False).to(cls.device)
        cls.images=torch.randint(256,(1,3,3,224,224),dtype=torch.uint8,device=cls.device)
        cls.state=torch.randn(1,23,device=cls.device)
        cls.actions=torch.randn(1,4,7,device=cls.device)
        cls.pad=torch.tensor([[False,False,True,True]],device=cls.device)

    def test_padding_cannot_change_encoder_or_loss(self):
        model=self.model.eval(); changed=self.actions.clone();changed[:,2:]=10000
        torch.manual_seed(99);p1,z1=model(self.images,self.state,self.actions,self.pad)
        torch.manual_seed(99);p2,z2=model(self.images,self.state,changed,self.pad)
        torch.testing.assert_close(z1[0],z2[0],rtol=0,atol=0)
        torch.testing.assert_close(z1[1],z2[1],rtol=0,atol=0)
        torch.testing.assert_close(p1,p2,rtol=0,atol=0)
        torch.testing.assert_close(masked_l1(p1,self.actions,self.pad),masked_l1(p2,changed,self.pad))

    def test_gradients_exclude_padded_targets(self):
        prediction=torch.zeros((1,4,7),device=self.device,requires_grad=True)
        loss=masked_l1(prediction,self.actions,self.pad);loss.backward()
        self.assertEqual(float(prediction.grad[:,2:].abs().sum()),0.)
        self.assertGreater(float(prediction.grad[:,:2].abs().sum()),0.)
        model=self.model.train();model.zero_grad(set_to_none=True)
        loss,l1,kl=losses(model,self.images,self.state,self.actions,self.pad)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0.)

    def test_checkpoint_reloads_identical_inference(self):
        stats=dict(state_mean=np.zeros(23,np.float32),state_std=np.ones(23,np.float32),
                   action_mean=np.zeros(7,np.float32),action_std=np.ones(7,np.float32))
        directory=ROOT/'outputs/robocasa/pilot_v1/act_contracts';directory.mkdir(parents=True,exist_ok=True)
        checkpoint=directory/'synthetic_contract_checkpoint.pt'
        model=self.model.eval()
        atomic_save(checkpoint,dict(schema='robocasa.pilot.act.checkpoint.v1',contract=CONTRACT,
            model_config=model.config,model=model.state_dict(),ema_model=model.state_dict(),norm_stats=stats,step=0))
        policy=ACTPolicy(checkpoint,str(self.device))
        rgb=self.images[0].cpu().numpy().transpose(0,2,3,1)
        obs=dict(base_rgb=rgb[0],wrist_rgb=rgb[1],target_mask=np.ones((224,224),bool),robot_state=self.state[0].cpu().numpy())
        a=policy.predict(obs); b=policy.predict(obs)
        np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(4,7))
        original_images=torch.as_tensor(np.stack([rgb[0],rgb[1],rgb[0]]).transpose(0,3,1,2).copy(),device=self.device)[None]
        with torch.no_grad(): original,_=model(original_images,self.state)
        expected=np.clip(original[0].cpu().numpy(),-1.,1.)
        np.testing.assert_array_equal(a,expected)
        obs['target_mask']=np.zeros((224,224),bool)
        c=policy.predict(obs);self.assertTrue(np.isfinite(c).all())
        self.assertFalse(np.array_equal(a,c))

if __name__=='__main__':unittest.main(verbosity=2)
