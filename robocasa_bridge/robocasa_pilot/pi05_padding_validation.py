"""Independent, fixed-noise padding decoder validation, optionally serving after it."""
import argparse
import hashlib
import inspect
import json
import types
from pathlib import Path
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as model_module
from openpi.models.pi0 import Pi0
from openpi.shared import nnx_utils
from .pi05_data import RawH5Dataset, read_manifest, inside
from .pi05_padding_policy import PaddingPi05Policy, sample_actions_padding_match, DECODER, serve


class FakeFlow:
    """Coupled velocity makes an incorrect real7 update or padding path observable."""
    action_horizon=2
    action_dim=32
    def __init__(self):
        self.trace=[]
        self.PaliGemma=types.SimpleNamespace(llm=self.llm)
    def embed_prefix(self, observation):
        return jnp.zeros((1,2,32)),jnp.ones((1,2),bool),jnp.zeros((2,),bool)
    def embed_suffix(self, observation, x_t, time):
        self.trace.append((np.asarray(x_t).copy(),float(time[0])))
        return x_t,jnp.ones((1,2),bool),jnp.ones((2,),bool),time
    def llm(self, tokens, **kwargs):
        if tokens[1] is None:
            assert kwargs["mask"].shape==(1,2,2)
            return (None,None),"fixed-prefix-kv"
        assert kwargs["kv_cache"]=="fixed-prefix-kv"
        assert kwargs["mask"].shape==(1,2,4)
        assert kwargs["adarms_cond"][0] is None
        assert kwargs["adarms_cond"][1].shape==(1,)
        return (None,tokens[1]),None
    @staticmethod
    def action_out_proj(x):
        return .3*x+.2*jnp.mean(x[...,7:],axis=-1,keepdims=True)+jnp.arange(32,dtype=x.dtype)*.01


def unit_validation():
    noise=jnp.asarray(np.random.default_rng(771).standard_normal((1,2,32)),dtype=jnp.float32)
    observation=types.SimpleNamespace(state=jnp.zeros((1,23)))
    fake=FakeFlow()
    with patch.object(model_module,"preprocess_observation",lambda _,obs,train:obs),jax.disable_jit():
        actual=np.asarray(sample_actions_padding_match(fake,jax.random.key(7),observation,num_steps=10,noise=noise))
        trace=fake.trace
        native_fake=FakeFlow()
        native=np.asarray(Pi0.sample_actions(native_fake,jax.random.key(7),observation,num_steps=10,noise=noise))
    assert len(trace)==10
    pad_error=max(float(np.max(np.abs(x[...,7:]-max(time,0)*np.asarray(noise)[...,7:]))) for x,time in trace)
    real_errors=[]
    for i,(x,time) in enumerate(trace):
        expected=x-.1*np.asarray(FakeFlow.action_out_proj(jnp.asarray(x)))
        next_x=trace[i+1][0] if i+1<len(trace) else actual
        real_errors.append(float(np.max(np.abs(next_x[...,:7]-expected[...,:7]))))
    first_step_error=float(np.max(np.abs(trace[1][0][...,:7]-native_fake.trace[1][0][...,:7])))
    assert pad_error<1e-6 and max(real_errors)<1e-6 and first_step_error<1e-6
    assert float(np.max(np.abs(actual[...,7:])))==0
    assert float(np.max(np.abs(actual[...,:7]-native[...,:7])))>1e-5
    assert float(np.std(actual[...,:7]))>1e-3
    return dict(ok=True,steps=len(trace),padding_trajectory_max_abs_error=pad_error,
        real7_native_euler_update_max_abs_error=max(real_errors),first_step_real7_native_difference=first_step_error,
        native_final_padding_abs_max=float(np.max(np.abs(native[...,7:]))),
        matched_final_padding_abs_max=float(np.max(np.abs(actual[...,7:]))),
        final_real7_native_difference_abs_max=float(np.max(np.abs(actual[...,:7]-native[...,:7]))),
        times=[time for _,time in trace],
        prefix_kv_suffix_attention_adarms_assertions=True)


def source_validation(output):
    previous=json.loads((output/"original_source_hashes.json").read_text())
    current={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in previous}
    assert current==previous,"A protected source changed during independent ablation"
    import textwrap
    native=textwrap.dedent(inspect.getsource(Pi0.sample_actions)).replace("@override\n","",1)
    custom=textwrap.dedent(inspect.getsource(sample_actions_padding_match))
    start=custom.index("        # Match the zero-padded")
    end=custom.index("\n\n    def cond(carry):",start)
    reverted=custom[:start]+"        return x_t + dt * v_t, time + dt"+custom[end:]
    reverted=reverted.replace("def sample_actions_padding_match(","def sample_actions(",1)
    reverted=reverted.replace("    return x_0.at[..., 7:].set(0.0)\n","    return x_0\n")
    assert reverted==native,"Unexpected source difference from native sampler"
    return dict(protected_source_hashes_unchanged=True,exact_native_copy_except_padding_update=True,hashes=current)


def checkpoint_validation(policy, manifest, output):
    _,splits,_=read_manifest(manifest)
    dataset=RawH5Dataset(splits["validation"],policy.horizon)
    try:
        row=dataset[0]
        transformed=policy.policy._input_transform({k:v for k,v in row.items() if k!="actions"})
        inputs=jax.tree.map(lambda x:jnp.asarray(x)[None,...],transformed)
        observation=model_module.Observation.from_dict(inputs)
        noise=jax.random.normal(jax.random.key(773),(1,policy.horizon,32))
        native_sampler=nnx_utils.module_jit(policy.policy._model.sample_actions)
        rng=jax.random.key(774)
        native=np.asarray(native_sampler(rng,observation,num_steps=10,noise=noise))
        matched=np.asarray(policy.policy._sample_actions(rng,observation,num_steps=10,noise=noise))
        repeat=np.asarray(policy.policy._sample_actions(rng,observation,num_steps=10,noise=noise))
        noise2=noise.at[...,:7].add(.1)
        perturb=np.asarray(policy.policy._sample_actions(rng,observation,num_steps=10,noise=noise2))
        assert np.isfinite(native).all() and np.isfinite(matched).all()
        assert matched.shape==(1,policy.horizon,32)
        assert np.array_equal(matched,repeat)
        assert np.max(np.abs(matched[...,7:]))==0
        assert np.max(np.abs(matched[...,:7]-native[...,:7]))>1e-6
        assert np.max(np.abs(matched[...,:7]-perturb[...,:7]))>1e-6
        np.savez_compressed(output/"fixed_noise_native_vs_padding500.npz",noise=np.asarray(noise),native=native,
                            padding_match=matched,perturbed_real7_noise=np.asarray(noise2),perturbed_output=perturb)
        source,frame=dataset.index[0]
        return dict(ok=True,checkpoint=policy.checkpoint,source_episode=str(source),source_frame=frame,
            same_observation_and_noise=True,repeat_exact=True,shape=list(matched.shape),
            native_final_padding_abs_mean=float(np.mean(np.abs(native[...,7:]))),
            native_final_padding_abs_max=float(np.max(np.abs(native[...,7:]))),
            matched_final_padding_abs_max=float(np.max(np.abs(matched[...,7:]))),
            real7_native_difference_abs_mean=float(np.mean(np.abs(matched[...,:7]-native[...,:7]))),
            real7_native_difference_abs_max=float(np.max(np.abs(matched[...,:7]-native[...,:7]))),
            real7_noise_perturbation_output_difference_abs_max=float(np.max(np.abs(matched[...,:7]-perturb[...,:7]))),
            raw_internal32_before_output_transforms=True,simulation_success_evaluated=False)
    finally:
        dataset.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--serve",action="store_true")
    ap.add_argument("--port",type=int,default=18862)
    args=ap.parse_args()
    output=inside(args.output);output.mkdir(exist_ok=True,parents=True)
    report=dict(schema="robocasa.pi05.padding_ablation.validation.v1",decoder=DECODER,
                source=source_validation(output),unit=unit_validation())
    policy=PaddingPi05Policy(args.checkpoint,"cuda:0")
    with jax.default_device(policy.device):
        report["checkpoint_test"]=checkpoint_validation(policy,args.manifest,output)
    (output/"validation500.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    if args.serve:
        serve(policy,args.port)

if __name__=="__main__":
    main()
