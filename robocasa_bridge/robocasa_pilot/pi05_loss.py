"""Pilot-only flow matching over the seven real robot control dimensions.

Adapted from Physical Intelligence OpenPI (Apache-2.0).
The Pi0.5 network and diffusion state remain 32D. No old OpenPI/ManiSkill
model or loss is modified; only this explicit RoboCasa training entry uses it.
"""
import jax
import jax.numpy as jnp
from openpi.models import model as model_module
from openpi.models.pi0 import make_attn_mask

ROBOT_ACTION_DIM = 7
INTERNAL_ACTION_DIM = 32

def robot_action_mse(prediction, target):
    if prediction.shape != target.shape or prediction.shape[-1] != INTERNAL_ACTION_DIM:
        raise ValueError("RoboCasa pi05 flow tensors must have matching internal32 shapes")
    return jnp.mean(jnp.square(prediction[..., :ROBOT_ACTION_DIM] - target[..., :ROBOT_ACTION_DIM]), axis=-1)

def robot_flow_loss(model, rng, observation, actions, *, train=False):
    """Native Pi0.5 flow objective with loss restricted to actual OSC/gripper7.

    The raw-H5 loader supplies complete physical windows only, so all H time
    positions are valid and there is no temporal is_pad to ignore accidentally.
    """
    if not model.pi05 or actions.shape[-1] != INTERNAL_ACTION_DIM:
        raise ValueError("This objective is only for internal32 pi0.5 RoboCasa models")
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng,3)
    observation = model_module.preprocess_observation(preprocess_rng,observation,train=train)
    noise = jax.random.normal(noise_rng,actions.shape)
    time = jax.random.beta(time_rng,1.5,1,actions.shape[:-2]) * .999 + .001
    expanded = time[...,None,None]
    noisy_actions = expanded*noise+(1-expanded)*actions
    target_velocity = noise-actions
    prefix_tokens,prefix_mask,prefix_ar = model.embed_prefix(observation)
    suffix_tokens,suffix_mask,suffix_ar,adarms = model.embed_suffix(observation,noisy_actions,time)
    input_mask = jnp.concatenate([prefix_mask,suffix_mask],axis=1)
    ar_mask = jnp.concatenate([prefix_ar,suffix_ar],axis=0)
    attention_mask = make_attn_mask(input_mask,ar_mask)
    positions = jnp.cumsum(input_mask,axis=1)-1
    (_,suffix_out),_ = model.PaliGemma.llm([prefix_tokens,suffix_tokens],mask=attention_mask,
                                         positions=positions,adarms_cond=[None,adarms])
    prediction = model.action_out_proj(suffix_out[:,-model.action_horizon:])
    return robot_action_mse(prediction,target_velocity)

def train_step(config,rng,state,batch):
    """Same optimizer/state updates as OpenPI, with the explicit robot7 loss."""
    import dataclasses
    import flax.nnx as nnx
    import optax
    model = nnx.merge(state.model_def,state.params)
    model.train()
    train_rng = jax.random.fold_in(rng,state.step)
    observation,actions = batch
    def loss_fn(model):
        return jnp.mean(robot_flow_loss(model,train_rng,observation,actions,train=True))
    loss,grads = nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,config.trainable_filter))(model)
    params = state.params.filter(config.trainable_filter)
    updates,opt_state = state.tx.update(grads,state.opt_state,params)
    nnx.update(model,optax.apply_updates(params,updates))
    params = nnx.state(model)
    result = dataclasses.replace(state,step=state.step+1,params=params,opt_state=opt_state)
    if state.ema_decay is not None:
        result = dataclasses.replace(result,ema_params=jax.tree.map(
            lambda old,new:state.ema_decay*old+(1-state.ema_decay)*new,state.ema_params,params))
    return result,{"loss":loss,"grad_norm":optax.global_norm(grads)}
