"""Stateless ACT chunk inference for the full RoboCasa pick-and-place pilot."""
from pathlib import Path
import numpy as np
import torch
from .act_train import ACTModel, CONTRACT, inside

class ACTPolicy:
    def __init__(self,checkpoint,device='cuda:0',use_ema=True):
        self.device=torch.device(device)
        self.checkpoint=inside(checkpoint)
        saved=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        if saved.get('schema')!='robocasa.pilot.act.checkpoint.v1':raise ValueError('Not a RoboCasa pilot ACT checkpoint')
        if saved.get('contract')!=CONTRACT:raise ValueError('Checkpoint action/state contract mismatch')
        self.model=ACTModel(saved['model_config'],pretrained=False).to(self.device).eval()
        self.model.load_state_dict(saved['ema_model' if use_ema else 'model'],strict=True)
        self.stats={k:torch.tensor(np.asarray(v,np.float32),device=self.device) for k,v in saved['norm_stats'].items()}
        self.step=saved['step'];self.horizon=saved['model_config']['horizon']
        self.contract=dict(CONTRACT)

    @torch.inference_mode()
    def predict(self,obs):
        base=np.asarray(obs['base_rgb']);wrist=np.asarray(obs['wrist_rgb'])
        mask=np.asarray(obs['target_mask']).astype(bool)
        state=np.asarray(obs['robot_state'],np.float32)
        if base.shape!=(224,224,3) or wrist.shape!=base.shape or mask.shape!=(224,224):
            raise ValueError('Expected two 224x224 RGB frames and aligned target mask')
        if base.dtype!=np.uint8 or wrist.dtype!=np.uint8 or state.shape!=(23,) or not np.isfinite(state).all():
            raise ValueError('Expected uint8 images and finite state[23]')
        target=np.where(mask[...,None],base,0).astype(np.uint8)
        images=np.stack([base,wrist,target]).transpose(0,3,1,2).copy()
        image=torch.from_numpy(images)[None].to(self.device)
        state=torch.from_numpy(state.copy())[None].to(self.device)
        state=(state-self.stats['state_mean'])/self.stats['state_std']
        prediction,_=self.model(image,state)
        prediction=prediction[0]*self.stats['action_std']+self.stats['action_mean']
        result=prediction.float().cpu().numpy()
        if result.shape!=(self.horizon,7) or not np.isfinite(result).all():raise FloatingPointError('Invalid ACT output')
        # No delta/absolute conversion: all seven channels already follow the raw OSC contract.
        return np.clip(result,-1.,1.).astype(np.float32)

    def reset(self):
        """No hidden state. The evaluator must clear its own queued actions on reset/target change."""
        return None


def main(argv=None):
    import argparse,base64,json,threading
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--serve',action='store_true');p.add_argument('--port',type=int,default=18861)
    p.add_argument('--raw-model',action='store_true',help='Use raw parameters instead of EMA')
    p.add_argument('--torch-threads',type=int,default=4)
    a=p.parse_args(argv)
    if not a.serve:p.error('Specify --serve for the loopback inference service')
    torch.set_num_threads(a.torch_threads)
    policy=ACTPolicy(a.checkpoint,a.device,use_ema=not a.raw_model);lock=threading.Lock()
    def decode(value,shape):
        if not isinstance(value,str):raise ValueError('Images and mask must be base64 strings of raw uint8 bytes')
        raw=base64.b64decode(value,validate=True)
        if len(raw)!=int(np.prod(shape)):raise ValueError(f'Raw byte count does not match {shape}')
        return np.frombuffer(raw,np.uint8).reshape(shape)
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,payload):
            raw=json.dumps(payload,allow_nan=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if self.path!='/health':self.send(404,dict(error='Unknown endpoint'));return
            self.send(200,dict(ok=True,policy='act',checkpoint=str(policy.checkpoint),step=policy.step,
                horizon=policy.horizon,contract=policy.contract))
        def do_POST(self):
            if self.path!='/infer':self.send(404,dict(error='Unknown endpoint'));return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=2_000_000:raise ValueError('Invalid request byte count')
                data=json.loads(self.rfile.read(length))
                obs=dict(base_rgb=decode(data['base_rgb'],(224,224,3)),wrist_rgb=decode(data['wrist_rgb'],(224,224,3)),
                    target_mask=decode(data['target_mask'],(224,224)),robot_state=np.asarray(data['robot_state'],np.float32))
                with lock:actions=policy.predict(obs)
                self.send(200,dict(actions=actions.tolist(),contract='OSC_POSE_delta_world_normalized',step=policy.step))
            except (KeyError,ValueError,TypeError) as exc:self.send(400,dict(error=str(exc)))
            except Exception as exc:self.send(500,dict(error=f'{type(exc).__name__}: {exc}'))
        def log_message(self,format,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    print(json.dumps(dict(ready=True,url=f'http://127.0.0.1:{a.port}',checkpoint=str(policy.checkpoint),step=policy.step)),flush=True)
    server.serve_forever()

if __name__=='__main__':main()
