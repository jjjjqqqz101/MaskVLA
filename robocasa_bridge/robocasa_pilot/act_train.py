"""RoboCasa pilot ACT: three RGB streams, 23D state, 7D world-frame delta action.

Independent entry point. Does not modify or resume ManiSkill trainers/checkpoints.
Time padding is excluded from both the CVAE encoder and reconstruction loss.
"""
# The DETR-derived forward pass extends code copyright (c) Facebook, Inc. and its affiliates.
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from robocasa_bridge.paths import ROOT
ACT_SOURCE = Path(os.environ.get("MASK_MANISKILL_ROOT", ROOT / "repo/ManiSkill")) / "examples/baselines/act"
if str(ACT_SOURCE) not in sys.path: sys.path.insert(0,str(ACT_SOURCE))
from act.detr.detr_vae import DETRVAE, build_encoder, reparametrize
from act.detr.transformer import build_transformer
from train_triple_rgb import offline_backbone

CONTRACT=dict(name='robocasa_pilot_act_v1',skill='fullpickplace',state_dim=23,action_dim=7,
    streams=['base_rgb','wrist_rgb','base_masked_rgb'],image_shape=[224,224,3],
    action_type='OSC_POSE_delta_world_normalized',control_hz=20,
    position_scale_m=0.05,rotation_scale_rad=0.5,gripper_open=-1,gripper_close=1,
    time_padding='masked_encoder_and_loss',mask_source_training='sim_gt_offline_annotation')


def inside(path):
    p=Path(path).resolve()
    if not p.is_relative_to(ROOT): raise ValueError(f'Outside authorized root: {p}')
    return p


def model_config(horizon=30,hidden_dim=256,enc_layers=2,dec_layers=4,dropout=0.1):
    return dict(horizon=horizon,hidden_dim=hidden_dim,enc_layers=enc_layers,
                dec_layers=dec_layers,dropout=dropout)


class PaddedDETRVAE(DETRVAE):
    def forward(self,obs,actions=None,is_pad=None):
        state=obs['state']; bs=state.shape[0]
        if actions is not None:
            if is_pad is None or is_pad.shape!=actions.shape[:2]:
                raise ValueError('Explicit [batch,horizon] padding is required')
            if is_pad.all(dim=1).any(): raise ValueError('All-padded samples are invalid')
            cls=self.cls_embed.weight[None].expand(bs,-1,-1)
            state_embed=self.encoder_state_proj(state)[:,None]
            clean_actions=actions.masked_fill(is_pad[...,None],0)
            action_embed=self.encoder_action_proj(clean_actions)
            enc_input=torch.cat([cls,state_embed,action_embed],dim=1).permute(1,0,2)
            padding=torch.cat([torch.zeros((bs,2),dtype=torch.bool,device=state.device),is_pad],dim=1)
            pos=self.pos_table[:, :enc_input.shape[0]].permute(1,0,2)
            enc=self.encoder(enc_input,pos=pos,src_key_padding_mask=padding)[0]
            info=self.latent_proj(enc); mu,logvar=info.chunk(2,dim=-1)
            latent=self.latent_out_proj(reparametrize(mu,logvar))
        else:
            mu=logvar=None
            latent=self.latent_out_proj(torch.zeros((bs,self.latent_dim),device=state.device,dtype=state.dtype))
        features=[]; positions=[]
        for camera in range(obs['rgb'].shape[1]):
            feat,pos=self.backbones[0](obs['rgb'][:,camera])
            features.append(self.input_proj(feat[0])); positions.append(pos[0])
        hs=self.transformer(torch.cat(features,dim=3),None,self.query_embed.weight,
            torch.cat(positions,dim=3),latent,self.input_proj_robot_state(state),self.additional_pos_embed.weight)[0]
        return self.action_head(hs),(mu,logvar)


class ACTModel(nn.Module):
    def __init__(self,config=None,pretrained=False):
        super().__init__(); self.config=config or model_config()
        c=self.config
        args=SimpleNamespace(hidden_dim=c['hidden_dim'],dropout=c['dropout'],nheads=8,
            dim_feedforward=512,enc_layers=c['enc_layers'],dec_layers=c['dec_layers'],
            pre_norm=False,num_queries=c['horizon'],include_depth=False,masks=False,
            dilation=False,position_embedding='sine',backbone='resnet18',lr_backbone=1e-5,
            pretrained_path=ROOT/'ckpt/maniskill/support/torch/hub/checkpoints/resnet18-f37072fd.pth')
        backbone=offline_backbone(args,pretrained=pretrained)
        self.model=PaddedDETRVAE([backbone],build_transformer(args),build_encoder(args),
            state_dim=23,action_dim=7,num_queries=c['horizon'])
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406]).view(1,1,3,1,1),persistent=False)
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225]).view(1,1,3,1,1),persistent=False)

    def forward(self,images,state,actions=None,is_pad=None):
        rgb=(images.float()/255-self.rgb_mean)/self.rgb_std
        return self.model({'rgb':rgb,'state':state},actions,is_pad)


def masked_l1(prediction,target,is_pad):
    valid=(~is_pad).unsqueeze(-1).expand_as(target)
    if not valid.any(): raise ValueError('No valid training actions')
    return torch.where(valid,(prediction-target).abs(),0).sum()/valid.sum()


def losses(model,images,state,actions,is_pad,kl_weight=10.):
    prediction,(mu,logvar)=model(images,state,actions,is_pad)
    l1=masked_l1(prediction.float(),actions.float(),is_pad)
    mu,logvar=mu.float(),logvar.float()
    kl=(-.5*(1+logvar-mu.square()-logvar.exp())).sum(-1).mean()
    return l1+kl_weight*kl,l1,kl


def collate(rows):
    keys=('images','state','actions','is_pad')
    return {k:torch.as_tensor(np.stack([np.asarray(row[k]) for row in rows])) for k in keys}


def normalize_batch(batch,stats,device):
    images=batch['images'].to(device)
    state=batch['state'].to(device,dtype=torch.float32)
    actions=batch['actions'].to(device,dtype=torch.float32)
    state=(state-stats['state_mean'])/stats['state_std']
    actions=(actions-stats['action_mean'])/stats['action_std']
    return images,state,actions,batch['is_pad'].to(device,dtype=torch.bool)


def atomic_save(path,value):
    path=inside(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp'); torch.save(value,temporary); os.replace(temporary,path)


def save_checkpoint(path,model,ema,optimizer,step,stats,config,manifest_hash,rng,best_val):
    atomic_save(path,dict(schema='robocasa.pilot.act.checkpoint.v1',contract=CONTRACT,
        model_config=config,model=model.state_dict(),ema_model=ema.state_dict(),
        optimizer=optimizer.state_dict(),step=step,norm_stats=stats,manifest_sha256=manifest_hash,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),loader_rng=rng.get_state(),
        best_validation_l1=best_val,created_at=time.time(),initialization='ImageNet_ResNet18_new_RoboCasa_ACT'))


@torch.no_grad()
def validation_l1(model,loader,stats,device,max_batches=None):
    model.eval(); total=0.; count=0
    for idx,batch in enumerate(loader):
        if max_batches is not None and idx>=max_batches: break
        images,state,actions,is_pad=normalize_batch(batch,stats,device)
        pred,_=model(images,state)
        n=int((~is_pad).sum())*7
        total+=float(masked_l1(pred,actions,is_pad))*n; count+=n
    return total/count if count else None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'ckpt/robocasa/pilot_v1/act')
    p.add_argument('--log-dir',type=Path,default=ROOT/'outputs/robocasa/pilot_v1/act')
    p.add_argument('--steps',type=int,default=3000); p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--horizon',type=int,default=30); p.add_argument('--seed',type=int,default=17)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--lr-backbone',type=float,default=1e-5)
    p.add_argument('--kl-weight',type=float,default=10.); p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--validate-every',type=int,default=250); p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=0); p.add_argument('--resume',type=Path)
    p.add_argument('--no-pretrained',action='store_true'); p.add_argument('--hidden-dim',type=int,default=256)
    p.add_argument('--dec-layers',type=int,default=4); p.add_argument('--torch-threads',type=int,default=4)
    a=p.parse_args(argv)
    for v in (a.steps,a.batch_size,a.horizon,a.save_every,a.validate_every):
        if v<1: p.error('steps, batch size, horizon and intervals must be positive')
    a.manifest=inside(a.manifest); a.output=inside(a.output); a.log_dir=inside(a.log_dir)
    if not a.resume and ((a.output/'latest.pt').exists() or (a.log_dir/'config.json').exists()):
        raise FileExistsError('Use a fresh output directory or explicit --resume')
    if a.workers!=0: raise ValueError('Exact-recovery sampler currently requires --workers 0')
    a.output.mkdir(parents=True,exist_ok=True); a.log_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.torch_threads); torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device=torch.device(a.device)
    from .data import TrajectoryDataset
    train=TrajectoryDataset(a.manifest,split='train',horizon=a.horizon)
    val=TrajectoryDataset(a.manifest,split='validation',horizon=a.horizon)
    rawstats=train.fit_stats()
    rawstats={k:np.asarray(v,np.float32) for k,v in rawstats.items()}
    for k in ('state_std','action_std'): rawstats[k]=np.maximum(rawstats[k],1e-4)
    stats={k:torch.tensor(v,device=device) for k,v in rawstats.items()}
    manifest_hash=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    config=model_config(a.horizon,hidden_dim=a.hidden_dim,dec_layers=a.dec_layers)
    model=ACTModel(config,pretrained=not a.no_pretrained and not a.resume).to(device)
    ema=deepcopy(model).eval()
    optimizer=torch.optim.AdamW([
        {'params':[v for k,v in model.named_parameters() if 'backbones' not in k],'lr':a.lr},
        {'params':[v for k,v in model.named_parameters() if 'backbones' in k],'lr':a.lr_backbone}],weight_decay=1e-4)
    generator=torch.Generator().manual_seed(a.seed)
    start_step=0;best_val=float('inf')
    if a.resume:
        ck=torch.load(inside(a.resume),map_location=device,weights_only=False)
        if ck['manifest_sha256']!=manifest_hash or ck['model_config']!=config: raise ValueError('Resume data/model mismatch')
        model.load_state_dict(ck['model']);ema.load_state_dict(ck['ema_model']);optimizer.load_state_dict(ck['optimizer'])
        start_step=ck['step']; best_val=ck['best_validation_l1']
        torch.set_rng_state(ck['torch_rng'].cpu()); generator.set_state(ck['loader_rng'].cpu())
        np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        if device.type=='cuda' and ck['cuda_rng']:torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
    val_loader=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=0,collate_fn=collate)
    # Independent replacement sampling permits exact recovery from the saved generator.
    log=(a.log_dir/'metrics.jsonl').open('a',buffering=1)
    description=dict(contract=CONTRACT,model_config=config,manifest=str(a.manifest),manifest_sha256=manifest_hash,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        train_samples=len(train),validation_samples=len(val),norm_stats={k:v.tolist() for k,v in rawstats.items()})
    (a.log_dir/'config.json').write_text(json.dumps(description,indent=2))
    begin=time.monotonic(); model.train()
    def checkpoint(name,step):
        save_checkpoint(a.output/name,model,ema,optimizer,step,rawstats,config,manifest_hash,generator,best_val)
    for step in range(start_step+1,a.steps+1):
        ids=torch.randint(len(train),(a.batch_size,),generator=generator).tolist()
        batch=collate([train[i] for i in ids]); images,state,actions,is_pad=normalize_batch(batch,stats,device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            loss,l1,kl=losses(model,images,state,actions,is_pad,a.kl_weight)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at step {step}')
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():
            for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,0.01)
            for eb,mb in zip(ema.buffers(),model.buffers()):eb.copy_(mb)
        if step==1 or step%50==0:
            event=dict(step=step,loss=float(loss),l1=float(l1),kl=float(kl),grad_norm=float(grad),
                       elapsed_seconds=time.monotonic()-begin,steps_per_second=(step-start_step)/(time.monotonic()-begin))
            log.write(json.dumps(event)+'\n');print(json.dumps(event),flush=True)
        if step%a.validate_every==0 or step==a.steps:
            value=validation_l1(ema,val_loader,stats,device)
            event=dict(step=step,validation_prior_l1=value);log.write(json.dumps(event)+'\n');print(json.dumps(event),flush=True)
            if value is not None and value<best_val:best_val=value;checkpoint('best_validation.pt',step)
        if step%a.save_every==0 or step==a.steps:checkpoint(f'{step}.pt',step);checkpoint('latest.pt',step)
    log.close()
    report=dict(completed=True,step=a.steps,best_validation_l1=best_val,
        latest=str(a.output/'latest.pt'),best_validation=str(a.output/'best_validation.pt'),
        elapsed_seconds=time.monotonic()-begin,manifest_sha256=manifest_hash,
        note='Validation L1 selects an offline candidate; autonomous held-out evaluation is required.')
    (a.log_dir/'training_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__': main()
