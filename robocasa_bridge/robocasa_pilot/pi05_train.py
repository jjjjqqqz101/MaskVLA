"""Train real pi0.5 on RoboCasa raw H5, without a LeRobot export or time padding.

Run in the project OpenPI environment with the requested number of GPU devices.
This is a new domain-transfer experiment, never a resume of ManiSkill training.
"""
import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import time

from .pi05_data import ASSET_ID, RawBatchLoader, RawH5Dataset, fit_norm_stats, inside, read_manifest

import os
from robocasa_bridge.paths import ROOT

def write_json(path, value):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))

def restore_sampler_rng(loader, completed_steps):
    """Restore the seeded NumPy index stream without replaying H5 reads."""
    if completed_steps < 0:
        raise ValueError("completed_steps must be nonnegative")
    for _ in range(completed_steps):
        loader.rng.integers(len(loader.dataset), size=loader.batch_size)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT/"data/robocasa/pilot_v1/manifest.json"))
    parser.add_argument("--checkpoint-dir", default=str(ROOT/"ckpt/robocasa/pilot_v1/pi05"))
    parser.add_argument("--output-dir", default=str(ROOT/"outputs/robocasa/pilot_v1/pi05"))
    parser.add_argument("--init-checkpoint", default=str(ROOT/"ckpt/maniskill/pi05_25x50/49999"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--fsdp-devices", type=int, default=3)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true", help="Only resume this RoboCasa run with an identical manifest")
    parser.add_argument("--freeze-vlm", action="store_true", help="Pilot option: freeze vision/language, train action expert and action/time projections")
    parser.add_argument("--inspect", action="store_true", help="Validate real raw data, stats, views and state tokens without loading model weights")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import numpy as np
    from openpi.training import config as cfg
    from openpi.training import checkpoints, sharding, weight_loaders
    from openpi.shared import normalize, nnx_utils
    from openpi.policies.robocasa_policy import PROMPT

    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("Positive steps and batch size required")
    manifest, splits, digest = read_manifest(args.manifest)
    output = inside(args.output_dir)
    checkpoint_dir = inside(args.checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = dataclasses.replace(cfg.get_config("pi05_robocasa_pilot_v1"),
        seed=args.seed, batch_size=args.batch_size, fsdp_devices=args.fsdp_devices,
        num_train_steps=args.steps, save_interval=args.save_interval,
        resume=args.resume, exp_name="raw_h5_domain_transfer")
    init_checkpoint = inside(args.init_checkpoint)
    init_params = init_checkpoint if init_checkpoint.name == "params" else init_checkpoint/"params"
    if not init_params.is_dir():
        raise FileNotFoundError(init_params)
    config = dataclasses.replace(config, weight_loader=weight_loaders.CheckpointWeightLoader(str(init_params)))
    if args.freeze_vlm:
        config = dataclasses.replace(config, freeze_filter=nnx.All(
            nnx_utils.PathRegex("PaliGemma/.*"), nnx.Not(nnx_utils.PathRegex(".*_1/.*"))))
    train = RawH5Dataset(splits["train"], config.model.action_horizon)
    validation = RawH5Dataset(splits["validation"], config.model.action_horizon) if splits["validation"] else None
    stats = fit_norm_stats(splits["train"])
    data_config = config.data.create(config.assets_dirs, config.model)
    metadata = dict(schema="robocasa.pi05.raw_h5_training.v1", manifest=str(inside(args.manifest)),
        manifest_sha256=digest, splits={k:[str(p) for p in v] for k,v in splits.items()},
        initialization="domain_transfer_warm_start", initialization_params=str(init_params),
        resume_maniskill_training=False, optimizer_initial_step=0,
        prompt=PROMPT, state_dim=23, policy_action_dim=7, internal_action_dim=32,
        control_hz=20, action_horizon=config.model.action_horizon, discrete_state_input=True,
        action_semantics="world OSC delta normalized to +/-1; 0.05m and 0.5rad per step; gripper -1 open +1 close",
        action_delta_transform=False, target_image="current frame base_rgb * same-frame gt_target_mask_offline",
        temporal_padding=False, window_limit="Only complete context-stable horizon10 windows; final9 start positions excluded, recorded real settle tail preserves release supervision",
        normalization_scope="training split only", frozen_vlm=args.freeze_vlm,
        flow_loss_action_dimensions=7, loss_normalization="mean over seven real OSC/gripper dimensions only; internal25 padded channels excluded",
        train_windows=len(train), validation_windows=len(validation) if validation else 0,
        train_episodes=train.episode_reports, devices=[str(d) for d in jax.devices()],
        requested_steps=args.steps, batch_size=args.batch_size, seed=args.seed, resume_requested=args.resume)
    # Contract check: the 23D state must influence the pi05 language tokens.
    probe = RawBatchLoader(train, data_config, stats, 1, seed=args.seed)
    row = train[0]
    transformed = probe.transform(row)
    changed = dict(row)
    changed["observation/state"] = row["observation/state"].copy()
    changed["observation/state"][0] += 100.
    changed_tokens = probe.transform(changed)["tokenized_prompt"]
    tokens_differ = not np.array_equal(transformed["tokenized_prompt"], changed_tokens)
    if not tokens_differ:
        # Try the other bound when the actual first state value already saturates.
        changed["observation/state"][0] -= 200.
        tokens_differ = not np.array_equal(transformed["tokenized_prompt"], probe.transform(changed)["tokenized_prompt"])
    if not tokens_differ:
        raise RuntimeError("Robot state did not change pi05 discrete tokens")
    metadata["state_token_influence_verified"] = True
    metadata["first_batch_shapes"] = {"state":list(transformed["state"].shape), "actions":list(transformed["actions"].shape),
                                       "tokenized_prompt":list(transformed["tokenized_prompt"].shape)}
    write_json(output/"data_contract.json", metadata)
    normalize.save(output/"train_norm_stats", stats)
    if args.inspect:
        print(json.dumps({"ok":True,"train_windows":len(train),"state_tokens_differ":tokens_differ,"report":str(output/"data_contract.json")}), flush=True)
        train.close()
        if validation: validation.close()
        return
    if args.batch_size % jax.device_count():
        raise ValueError("Batch size must be divisible by visible device count")
    previous_meta = checkpoint_dir/"run_metadata.json"
    if args.resume and previous_meta.exists():
        previous = json.loads(previous_meta.read_text())
        if previous["manifest_sha256"] != digest:
            raise ValueError("Resume manifest differs; normalization/data changed. Start a new explicitly named experiment")
        if previous.get("flow_loss_action_dimensions") != 7:
            raise ValueError("Resume flow objective differs; start a new robot7 experiment")
        if previous.get("frozen_vlm", False) != args.freeze_vlm:
            raise ValueError("Resume freeze/optimizer scope differs; start a new experiment")
        if previous.get("batch_size") != args.batch_size or previous.get("seed", 17) != args.seed:
            raise ValueError("Resume batch size/seed differs; deterministic sampler continuation requires both unchanged")
    manager, resuming = checkpoints.initialize_checkpoint_dir(checkpoint_dir, keep_period=500, overwrite=False, resume=args.resume)
    # Use the maintained OpenPI model initialization, optimizer, gradient and checkpoint implementation.
    entry = Path(os.environ.get("MASK_OPENPI_ROOT", ROOT / "repo/openpi")) / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("robocasa_openpi_train", entry)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = RawBatchLoader(train, data_config, stats, args.batch_size, seed=args.seed, sharding=data_sharding)
    val_loader = RawBatchLoader(validation, data_config, stats, args.batch_size, seed=1017, sharding=data_sharding) if validation else None
    fixed_val = next(val_loader) if val_loader else None
    init_rng, train_rng = jax.random.split(jax.random.key(args.seed))
    start = time.monotonic()
    state, state_sharding = trainer.init_train_state(config, init_rng, mesh, resume=resuming)
    if resuming:
        state = checkpoints.restore_state(manager, state, loader)
    jax.block_until_ready(state)
    resume_step = int(state.step)
    if resuming and resume_step:
        # Upstream restore_state restores model/optimizer, not our NumPy sampler.
        # Replay RNG index draws without loading any H5 images to continue exactly.
        restore_sampler_rng(loader, resume_step)
    metadata["resume_from_step"] = resume_step if resuming else None
    metadata["experiment_initialization"] = "domain_transfer_warm_start"
    metadata["this_run_initialization"] = "resume_robocasa_checkpoint" if resuming else "domain_transfer_warm_start"
    metadata["resume_from_checkpoint"] = str(checkpoint_dir / str(resume_step)) if resuming else None
    metadata["training_source_sha256"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("pi05_train.py", "pi05_data.py", "pi05_loss.py")}
    metadata["run_start_optimizer_step"] = resume_step
    metadata["optimizer_initial_step_this_run"] = resume_step
    metadata["requested_new_optimizer_updates"] = max(0, args.steps - resume_step)
    metadata["sampler_continuation"] = "seed+completed-step index draws replayed without I/O" if resuming else "fresh seeded sampler"
    metadata["initialization_seconds"] = time.monotonic()-start
    metadata["trainable_parameters"] = int(sum(np.prod(x.shape) for x in jax.tree.leaves(state.params.filter(config.trainable_filter))))
    metadata["total_parameters"] = int(sum(np.prod(x.shape) for x in jax.tree.leaves(state.params)))
    write_json(previous_meta, metadata)
    write_json(output/"run_metadata.json", metadata)
    from .pi05_loss import robot_flow_loss, train_step as robot_train_step
    step_fn = jax.jit(functools.partial(robot_train_step, config),
        in_shardings=(replicated,state_sharding,data_sharding),
        out_shardings=(state_sharding,replicated), donate_argnums=(1,))
    def validation_loss(params, batch):
        model = nnx.merge(state.model_def, params)
        model.eval()
        return jnp.mean(robot_flow_loss(model,jax.random.key(9917),batch[0],batch[1],train=False))
    val_fn = jax.jit(validation_loss, in_shardings=(state_sharding.params,data_sharding), out_shardings=replicated)
    metrics = output/"metrics.jsonl"
    initial_val = None
    if fixed_val is not None:
        with sharding.set_mesh(mesh):
            initial_val = float(val_fn(state.params, fixed_val))
        print(f"INITIAL validation_loss={initial_val:.6f}", flush=True)
    start_step = int(state.step)
    last_val = initial_val
    try:
        for step in range(start_step, args.steps):
            batch = next(loader)
            tick = time.monotonic()
            with sharding.set_mesh(mesh):
                state, info = step_fn(train_rng, state, batch)
            info = jax.device_get(info)
            seconds = time.monotonic()-tick
            if not np.isfinite(float(info["loss"])) or not np.isfinite(float(info["grad_norm"])):
                write_json(output/"training_failure.json", {"step":int(state.step), "reason":"Nonfinite loss or gradient; checkpoint not saved"})
                raise RuntimeError("Nonfinite pi05 loss/gradient")
            record = {"step":int(state.step), "loss":float(info["loss"]), "grad_norm":float(info["grad_norm"]),
                      "step_seconds":seconds, "samples_per_second":args.batch_size/seconds}
            if fixed_val is not None and ((step+1)%args.validation_interval == 0 or step+1 == args.steps):
                with sharding.set_mesh(mesh):
                    last_val = float(val_fn(state.params, fixed_val))
                record["validation_loss"] = last_val
            with metrics.open("a") as stream:
                stream.write(json.dumps(record)+"\n")
            if step == start_step or (step+1)%10 == 0 or step+1 == args.steps:
                print(json.dumps(record), flush=True)
            if (step+1)%args.save_interval == 0 or step+1 == args.steps:
                checkpoints.save_state(manager, state, loader, step+1)
        manager.wait_until_finished()
        summary = dict(ok=True, completed_optimizer_steps=int(state.step), latest_checkpoint=str(checkpoint_dir/str(int(state.step))),
                       initial_validation_loss=initial_val, final_validation_loss=last_val,
                       elapsed_seconds=time.monotonic()-start, initialization="domain_transfer_warm_start",
                       resume_requested=args.resume, resume_from_step=resume_step if resuming else None,
                       optimizer_initial_step_this_run=resume_step, completed_optimizer_steps_this_run=int(state.step)-resume_step,
                       experiment_optimizer_initial_step=0,
                       this_run_initialization="resume_robocasa_checkpoint" if resuming else "domain_transfer_warm_start",
                       resume_from_checkpoint=str(checkpoint_dir / str(resume_step)) if resuming else None,
                       flow_loss_action_dimensions=7, batch_size=args.batch_size,
                       simulation_success_evaluated=False, note="Flow loss/optimizer/checkpoint verification; closed-loop success must be evaluated separately")
        write_json(output/"training_summary.json", summary)
        print(json.dumps(summary), flush=True)
    finally:
        train.close()
        if validation: validation.close()
        manager.wait_until_finished()
        manager.close()

if __name__ == "__main__":
    main()
