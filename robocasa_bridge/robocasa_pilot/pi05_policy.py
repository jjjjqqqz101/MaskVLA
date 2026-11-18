"""Public RoboCasa pi0.5 predictor and a process-isolated loopback inference worker."""
import argparse
import base64
import contextlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .pi05_data import inside
from .protocol import OUTPUT_ROOT

class Pi05Policy:
    def __init__(self, checkpoint, device=None):
        import jax
        from openpi.training.config import get_config
        import jax.numpy as jnp
        from openpi.models.model import restore_params
        from openpi.policies.policy import Policy
        from openpi.training.checkpoints import load_norm_stats
        from openpi import transforms
        checkpoint = inside(checkpoint)
        if not (checkpoint/"params").is_dir():
            steps = sorted((p for p in checkpoint.iterdir() if p.is_dir() and p.name.isdigit() and (p/"params").is_dir()), key=lambda p:int(p.name))
            if not steps:
                raise FileNotFoundError(f"No completed pi05 checkpoint under {checkpoint}")
            checkpoint = steps[-1]
        config = get_config("pi05_robocasa_pilot_v1")
        if device is None:
            devices = jax.devices()
            self.device = next((d for d in devices if d.platform == "gpu"), devices[0])
        else:
            value = str(device)
            if value == "cpu":
                self.device = jax.devices("cpu")[0]
            else:
                local_index = int(value.split(":")[-1]) if ":" in value or value.isdigit() else 0
                self.device = jax.devices("gpu")[local_index]
        self.checkpoint = str(checkpoint)
        self.horizon = config.model.action_horizon
        self.last_timing = {}
        with jax.default_device(self.device):
            params = restore_params(checkpoint/"params", dtype=jnp.bfloat16,
                                    sharding=jax.sharding.SingleDeviceSharding(self.device))
            model = config.model.load(params)
            data_config = config.data.create(config.assets_dirs,config.model)
            stats = load_norm_stats(checkpoint/"assets",data_config.asset_id)
            self.policy = Policy(model,
                transforms=[*data_config.data_transforms.inputs,
                            transforms.Normalize(stats,use_quantiles=data_config.use_quantile_norm),
                            *data_config.model_transforms.inputs],
                output_transforms=[*data_config.model_transforms.outputs,
                                   transforms.Unnormalize(stats,use_quantiles=data_config.use_quantile_norm),
                                   *data_config.data_transforms.outputs],
                sample_kwargs={"num_steps":10}, metadata=config.policy_metadata)

    def predict(self, obs):
        import jax
        base = np.asarray(obs["base_rgb"])
        wrist = np.asarray(obs["wrist_rgb"])
        mask = np.asarray(obs["target_mask"])
        state = np.asarray(obs["robot_state"], np.float32)
        if base.shape != (224,224,3) or wrist.shape != (224,224,3) or base.dtype != np.uint8 or wrist.dtype != np.uint8:
            raise ValueError("Expected uint8 base/wrist RGB 224x224x3")
        if mask.shape != (224,224) or state.shape != (23,) or not np.isfinite(state).all():
            raise ValueError("Expected target_mask224x224 and finite 23D robot_state")
        data = {"observation/image":base, "observation/wrist_image":wrist,
                "observation/target_image":np.where(mask.astype(bool)[...,None],base,0).astype(np.uint8),
                "observation/state":state}
        tick = time.monotonic()
        with jax.default_device(self.device):
            result = self.policy.infer(data)
        actions = np.asarray(result["actions"],np.float32)
        if actions.shape != (self.horizon,7) or not np.isfinite(actions).all():
            raise RuntimeError("pi05 produced invalid Hx7 actions")
        # Robot controller consumes normalized OSC deltas and a discrete bipolar gripper.
        actions[:,:6] = np.clip(actions[:,:6], -1., 1.)
        actions[:,6] = np.where(actions[:,6] >= 0., 1., -1.)
        self.last_timing = {"wall_seconds":time.monotonic()-tick, **result.get("policy_timing",{})}
        return actions

def decode_array(value, shape):
    payload = base64.b64decode(value, validate=True)
    expected = int(np.prod(shape))
    if len(payload) != expected:
        raise ValueError(f"Expected {expected} uint8 bytes")
    return np.frombuffer(payload,np.uint8).reshape(shape)

def serve(policy, port):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass
        def reply(self, value, status=200):
            data=json.dumps(value,allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            if self.path != "/health":
                return self.reply({"ok":False,"error":"not found"},404)
            self.reply({"ok":True,"model":"pi05_robocasa_pilot_v1","checkpoint":policy.checkpoint,
                        "device":str(policy.device),"action_horizon":policy.horizon,"action_dim":7,
                        "state_dim":23,"same_frame_target_image":True,"temporal_padding":False,"pid":os.getpid()})
        def do_POST(self):
            if self.path != "/infer":
                return self.reply({"ok":False,"error":"not found"},404)
            try:
                size = int(self.headers.get("Content-Length",0))
                if not 0 < size < 1024*1024:
                    raise ValueError("Invalid JSON request size")
                data=json.loads(self.rfile.read(size))
                obs={"base_rgb":decode_array(data["base_rgb"],(224,224,3)),
                     "wrist_rgb":decode_array(data["wrist_rgb"],(224,224,3)),
                     "target_mask":decode_array(data["target_mask"],(224,224)),
                     "robot_state":np.asarray(data["robot_state"],np.float32)}
                actions=policy.predict(obs)
                self.reply({"ok":True,"actions":actions.tolist(),"timing":policy.last_timing})
            except (ValueError,KeyError,TypeError) as exc:
                self.reply({"ok":False,"error":str(exc)},400)
            except Exception as exc:
                self.reply({"ok":False,"error":str(exc)},500)
    server=HTTPServer(("127.0.0.1",int(port)),Handler)
    pidfile=inside(OUTPUT_ROOT / f"pi05_worker_{port}.pid")
    pidfile.parent.mkdir(parents=True,exist_ok=True)
    pidfile.write_text(str(os.getpid()))
    print(json.dumps({"ready":True,"url":f"http://127.0.0.1:{port}","checkpoint":policy.checkpoint}),flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if pidfile.exists() and pidfile.read_text().strip()==str(os.getpid()):
            pidfile.unlink()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--device",default=None,help="JAX local visible device, e.g. cuda:0; physical selection via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--serve",action="store_true")
    parser.add_argument("--port",type=int,default=18862)
    parser.add_argument("--manifest")
    parser.add_argument("--split",default="validation",choices=["train","validation","test"])
    parser.add_argument("--output",default=str(OUTPUT_ROOT / "pi05/inference_smoke.json"))
    args=parser.parse_args()
    policy=Pi05Policy(args.checkpoint,args.device)
    if args.serve:
        return serve(policy,args.port)
    if not args.manifest:
        parser.error("Use --serve or provide --manifest for a real-data inference smoke")
    from .pi05_data import RawH5Dataset,read_manifest
    _,splits,_=read_manifest(args.manifest)
    dataset=RawH5Dataset(splits[args.split],policy.horizon)
    try:
        p,frame=dataset.index[0]
        row=dataset[0]
        f=dataset.files[p]
        obs={"base_rgb":row["observation/image"],"wrist_rgb":row["observation/wrist_image"],
             "target_mask":f["observations/gt_target_mask_offline"][frame],"robot_state":row["observation/state"]}
        actions=policy.predict(obs)
        report={"ok":True,"checkpoint":policy.checkpoint,"shape":list(actions.shape),
                "finite":bool(np.isfinite(actions).all()),"actions":actions.tolist(),
                "timing":policy.last_timing,"source_episode":str(p),"source_frame":frame,
                "simulation_success_evaluated":False}
        output=inside(args.output)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    finally:
        dataset.close()

if __name__ == "__main__":
    main()
