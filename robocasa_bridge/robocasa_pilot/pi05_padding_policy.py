"""Independent inference-only ablation for masked7 training / internal32 sampling.

Adapted from Physical Intelligence OpenPI Pi0.sample_actions (Apache-2.0). Prefix KV, suffix attention, AdaRMS,
noise RNG, step count, and the seven real action Euler updates are unchanged.
After each Euler update, padding follows max(next_time, 0) * initial_noise.
This module does not change model weights, training, native policy, or scoring.
"""
import argparse
import base64
import json
import os
import time
import types
import einops
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from .pi05_data import inside
from .protocol import OUTPUT_ROOT
from .pi05_policy import Pi05Policy, decode_array

DECODER = "masked7_padding_train_match_v1"

def sample_actions_padding_match(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    num_steps: int | at.Int[at.Array, ""] = 10,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
) -> _model.Actions:
    observation = _model.preprocess_observation(None, observation, train=False)
    # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
    # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    # first fill KV cache with a forward pass of the prefix
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
        # other
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
        # prefix tokens
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
        # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (
            batch_size,
            suffix_tokens.shape[1],
            prefix_tokens.shape[1] + suffix_tokens.shape[1],
        )
        # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # Match the zero-padded action distribution used by masked7 training.
        # Only padding is constrained. The real seven channels keep native Euler updates.
        next_time = time + dt
        next_x = x_t + dt * v_t
        next_x = next_x.at[..., 7:].set(jnp.maximum(next_time, 0.0) * noise[..., 7:])
        return next_x, next_time

    def cond(carry):
        x_t, time = carry
        # robust to floating-point error
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0.at[..., 7:].set(0.0)

class PaddingPi05Policy(Pi05Policy):
    def __init__(self, checkpoint, device=None):
        super().__init__(checkpoint, device)
        bound = types.MethodType(sample_actions_padding_match, self.policy._model)
        self.policy._sample_actions = nnx_utils.module_jit(bound)
        self.decoder = DECODER


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
                "internal_action_dim":32,"decoder":DECODER,"num_steps":10,
                "padding_path":"max(t,0)*initial_noise; final zero",
                "real7_update":"native Euler","state_dim":23,"same_frame_target_image":True,
                "temporal_padding":False,"pid":os.getpid()})
        def do_POST(self):
            if self.path != "/infer":
                return self.reply({"ok":False,"error":"not found"},404)
            try:
                size=int(self.headers.get("Content-Length",0))
                if not 0<size<1024*1024:
                    raise ValueError("Invalid JSON request size")
                data=json.loads(self.rfile.read(size))
                obs={"base_rgb":decode_array(data["base_rgb"],(224,224,3)),
                    "wrist_rgb":decode_array(data["wrist_rgb"],(224,224,3)),
                    "target_mask":decode_array(data["target_mask"],(224,224)),
                    "robot_state":np.asarray(data["robot_state"],np.float32)}
                actions=policy.predict(obs)
                self.reply({"ok":True,"actions":actions.tolist(),"timing":policy.last_timing,
                            "decoder":DECODER})
            except (ValueError,KeyError,TypeError) as exc:
                self.reply({"ok":False,"error":str(exc)},400)
            except Exception as exc:
                self.reply({"ok":False,"error":str(exc)},500)
    server=HTTPServer(("127.0.0.1",int(port)),Handler)
    pidfile=inside(OUTPUT_ROOT / f"pi05_padding_ablation/worker_{port}.pid")
    pidfile.parent.mkdir(parents=True,exist_ok=True)
    pidfile.write_text(str(os.getpid()))
    print(json.dumps({"ready":True,"url":f"http://127.0.0.1:{port}","checkpoint":policy.checkpoint,
                      "decoder":DECODER,"pid":os.getpid()}),flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if pidfile.exists() and pidfile.read_text().strip()==str(os.getpid()):
            pidfile.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--device",default=None)
    parser.add_argument("--port",type=int,default=18862)
    args=parser.parse_args()
    return serve(PaddingPi05Policy(args.checkpoint,args.device),args.port)


if __name__=="__main__":
    main()
