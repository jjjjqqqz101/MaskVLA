"""Offline three-stream ACT training, with optimizer-step based recovery.

Run from the ManiSkill ACT baseline (or put it on PYTHONPATH).  This entry point
does not create a simulator or run online evaluation.  Dataset preparation and
the deterministic memmap loader live in the adjacent, independent modules.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler


ALGO_NAME = "BC_ACT_triple_rgb"
STREAM_ORDER = ("base_rgb", "wrist_rgb", "base_masked_rgb")
STATE_FIELDS = (("agent", "qpos"), ("agent", "qvel"),
                ("extra", "tcp_pose"), ("extra", "goal_pos"))
CHECKPOINT_VERSION = 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path, help="Full recovery checkpoint, e.g. checkpoints/latest.pt")
    p.add_argument("--total-steps", type=int, default=200_000, help="Successful optimizer updates, never epochs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--expected-demos", type=int, default=1250, help="0 disables the count assertion for development")
    p.add_argument("--num-demos", type=int, default=None, help="None uses the entire prepared dataset")
    p.add_argument("--control-mode", default="pd_joint_pos")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=False,
                   help="Use CUDA fused AdamW; same ACT optimizer and hyperparameters")
    p.add_argument("--ema-foreach", action=argparse.BooleanOptionalAction, default=False,
                   help="Batch EMA tensor operations; same EMA decay and update count")
    p.add_argument("--compile-mode", choices=("off", "default", "reduce-overhead"), default="off",
                   help="Compile fixed-shape training forward/loss; model checkpoint keys stay unchanged")
    p.add_argument("--lr-drop", type=int, default=None, help="Default: floor(2/3 total optimizer steps)")
    p.add_argument("--kl-weight", type=float, default=10.0)
    p.add_argument("--max-grad-norm", type=float, default=float("inf"), help="inf retains the existing ACT behavior")
    p.add_argument("--save-every", type=int, default=40_000, help="Numbered full checkpoints; 0 disables interval saves")
    p.add_argument("--recovery-every", type=int, default=1_000, help="Overwrite latest.pt; 0 disables interval recovery saves")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--diagnostic-every", type=int, default=5_000, help="Fixed training-example loss diagnostic, not validation")
    p.add_argument("--diagnostic-batch-size", type=int, default=16)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pretrained-path", type=Path, default=None, help="ResNet18 IMAGENET1K_V1 local .pth; never downloads")
    p.add_argument("--position-embedding", default="sine", choices=("sine", "learned"))
    p.add_argument("--backbone", default="resnet18", choices=("resnet18",))
    p.add_argument("--enc-layers", type=int, default=2)
    p.add_argument("--dec-layers", type=int, default=4)
    p.add_argument("--dim-feedforward", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--num-queries", type=int, default=30)
    p.add_argument("--pre-norm", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args(argv)
    # These are part of the original DETR builders' argument contract.
    args.include_depth, args.masks, args.dilation = False, False, False
    if args.lr_drop is None:
        args.lr_drop = max(1, (2 * args.total_steps) // 3)
    for key in ("total_steps", "batch_size", "torch_threads", "num_queries", "lr_drop", "log_every", "diagnostic_batch_size"):
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("num_workers", "save_every", "recovery_every", "diagnostic_every", "expected_demos"):
        if getattr(args, key) < 0:
            p.error(f"--{key.replace('_', '-')} must be nonnegative")
    if args.prefetch_factor <= 0:
        p.error("--prefetch-factor must be positive")
    if args.num_demos is not None and args.num_demos <= 0:
        p.error("--num-demos must be positive")
    return args


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def atomic_json(path: Path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as handle:
        json.dump(jsonable(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temp, path)


def atomic_torch_save(path: Path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


class ResumableBatchSampler(Sampler):
    """A shuffled, drop-last stream indexed by committed optimizer updates.

    Worker prefetch may request future batches.  It never changes the committed
    cursor saved by the training loop.  Each epoch permutation uses its own RNG,
    so reconstructing the next batch does not consume the policy's RNG.
    """

    def __init__(self, dataset_size, batch_size, total_batches, start_batch=0, seed=1):
        self.dataset_size, self.batch_size = int(dataset_size), int(batch_size)
        self.total_batches, self.start_batch, self.seed = int(total_batches), int(start_batch), int(seed)
        self.batches_per_epoch = self.dataset_size // self.batch_size
        if self.batches_per_epoch < 1:
            raise ValueError("batch_size exceeds dataset length with drop_last=True")
        if not 0 <= self.start_batch <= self.total_batches:
            raise ValueError("Invalid committed dataloader cursor")

    def __iter__(self) -> Iterator[list[int]]:
        cursor, epoch = self.start_batch, self.start_batch // self.batches_per_epoch
        while cursor < self.total_batches:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            indices = torch.randperm(self.dataset_size, generator=generator)
            first = cursor % self.batches_per_epoch
            for batch_in_epoch in range(first, self.batches_per_epoch):
                if cursor >= self.total_batches:
                    return
                offset = batch_in_epoch * self.batch_size
                yield indices[offset:offset + self.batch_size].tolist()
                cursor += 1
            epoch += 1

    def __len__(self):
        return self.total_batches - self.start_batch

    def state_dict(self, completed_batches):
        return dict(completed_batches=completed_batches, seed=self.seed,
                    dataset_size=self.dataset_size, batch_size=self.batch_size,
                    batches_per_epoch=self.batches_per_epoch, drop_last=True,
                    next_epoch=completed_batches // self.batches_per_epoch,
                    next_batch_in_epoch=completed_batches % self.batches_per_epoch,
                    sampler="epoch_randperm_v1")


def offline_backbone(args, pretrained=True):
    """Same ACT ResNet18 modules and keys, with explicit local-only weights."""
    import torchvision
    from act.detr.backbone import BackboneBase, FrozenBatchNorm2d, Joiner
    from act.detr.position_encoding import build_position_encoding

    resnet = torchvision.models.resnet18(weights=None,
        replace_stride_with_dilation=[False, False, args.dilation], norm_layer=FrozenBatchNorm2d)
    if pretrained:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        filename = weights.url.rsplit("/", 1)[-1]
        path = args.pretrained_path or Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained ResNet18 cache missing: {path}. Supply --pretrained-path; automatic downloads are disabled.")
        # Validate the known torchvision cache before expensive model startup.
        if path.name == filename:
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = filename.rsplit("-", 1)[-1].split(".", 1)[0]
            if not checksum.startswith(expected):
                raise ValueError(f"ResNet18 cache checksum mismatch: {path}")
        resnet.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        print("PRETRAINED_LOCAL", str(path), flush=True)
    backbone = BackboneBase(resnet, args.lr_backbone > 0, 512, args.masks)
    model = Joiner(backbone, build_position_encoding(args))
    model.num_channels = 512
    return model


def static_spaces(state_dim=28, action_dim=8):
    """The Agent only needs shape metadata; no simulator is initialized."""
    return SimpleNamespace(
        single_observation_space={"state": SimpleNamespace(shape=(state_dim,)),
                                  "rgb": SimpleNamespace(shape=(3, 3, 224, 224))},
        single_action_space=SimpleNamespace(shape=(action_dim,)))


class Agent(nn.Module):
    def __init__(self, env, args, pretrained=None):
        super().__init__()
        from act.detr.transformer import build_transformer
        from act.detr.detr_vae import build_encoder, DETRVAE

        state_shape = env.single_observation_space["state"].shape
        rgb_shape = env.single_observation_space["rgb"].shape
        action_shape = env.single_action_space.shape
        if state_shape != (28,) or rgb_shape != (3, 3, 224, 224) or len(action_shape) != 1:
            raise ValueError(f"Invalid triple RGB contract: {state_shape}, {rgb_shape}, {action_shape}")
        self.state_dim, self.act_dim, self.kl_weight = 28, action_shape[0], args.kl_weight
        self.include_depth = False
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1), persistent=False)
        backbone = offline_backbone(args, args.pretrained if pretrained is None else pretrained)
        self.model = DETRVAE([backbone], build_transformer(args), build_encoder(args),
                            state_dim=28, action_dim=self.act_dim, num_queries=args.num_queries)

    def normalized_observations(self, obs):
        # Dataset holds uint8 resized frames.  Convert once, on the policy device;
        # do not mutate caller batches, including the fixed diagnostic batch.
        rgb = obs["rgb"].float().div(255.0)
        return {"state": obs["state"], "rgb": (rgb - self.rgb_mean) / self.rgb_std}

    def compute_loss(self, obs, action_seq):
        prediction, (mu, logvar) = self.model(self.normalized_observations(obs), action_seq)
        # Keep KL exp/reduction in float32 under AMP for numerical stability.
        mu, logvar = mu.float(), logvar.float()
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(1).mean()
        l1 = F.l1_loss(prediction.float(), action_seq.float(), reduction="mean")
        return {"l1": l1, "kl": kl, "loss": l1 + kl * self.kl_weight}

    def get_action(self, obs):
        return self.model(self.normalized_observations(obs))[0]


def architecture(args, action_dim):
    fields = ("backbone", "position_embedding", "enc_layers", "dec_layers", "dim_feedforward",
              "hidden_dim", "dropout", "nheads", "num_queries", "pre_norm", "include_depth", "masks", "dilation")
    return dict(name="ACT_ResNet18_DETRVAE", shared_backbone=True, latent_dim=32,
                state_dim=28, action_dim=action_dim, rgb_shape=[3, 3, 224, 224],
                stream_order=list(STREAM_ORDER), **{key: getattr(args, key) for key in fields})


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def move_batch(batch, device):
    return ({key: value.to(device, non_blocking=True) for key, value in batch["observations"].items()},
            batch["actions"].to(device, non_blocking=True))


def autocast(args, device):
    if args.amp == "off":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16 if args.amp == "bf16" else torch.float16)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def fixed_diagnostic(agent, dataset, args, device, indices):
    from torch.utils.data import default_collate

    batch = default_collate([dataset[index] for index in indices])
    obs, actions = move_batch(batch, device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    was_training = agent.training
    agent.eval()
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad(), autocast(args, device):
            torch.manual_seed(args.seed + 10_000_019)
            losses = agent.compute_loss(obs, actions)
        return {key: float(value) for key, value in losses.items()}
    finally:
        agent.train(was_training)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.amp == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA")
    if args.amp == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 AMP requested but unsupported by this GPU")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)

    from triple_rgb_dataset import TripleRGBMemmapDataset
    from diffusers.training_utils import EMAModel

    dataset = TripleRGBMemmapDataset(args.data_path, num_queries=args.num_queries,
        num_traj=args.num_demos, include_depth=False, control_mode=args.control_mode)
    if tuple(dataset.stream_order) != STREAM_ORDER:
        raise ValueError("Dataset stream order does not match the three-stream policy")
    if args.expected_demos and dataset.num_traj != args.expected_demos:
        raise ValueError(f"Expected {args.expected_demos} demonstrations, got {dataset.num_traj}")
    if not dataset.manifest.get("complete") or dataset.manifest.get("schema") != "act_triple_rgb_memmap_v1":
        raise ValueError("Dataset is incomplete or has an unexpected schema")
    sample = dataset[0]
    rgb, state, actions = sample["observations"]["rgb"], sample["observations"]["state"], sample["actions"]
    if rgb.shape != (3, 3, 224, 224) or rgb.dtype != torch.uint8 or state.shape != (28,):
        raise ValueError(f"Unexpected observation shapes/dtype: {rgb.shape}, {rgb.dtype}, {state.shape}")
    if actions.shape != (args.num_queries, 8) or actions.dtype != torch.float32 or state.dtype != torch.float32:
        raise ValueError(f"Expected float32 state and action chunk [{args.num_queries}, 8]")
    arch = architecture(args, action_dim=actions.shape[-1])
    fingerprint = dataset.dataset_fingerprint
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = None
    completed_steps = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION or not checkpoint.get("resume_safe", False):
            raise ValueError("Checkpoint is incompatible or was interrupted during a partial optimizer update")
        if checkpoint["architecture"] != arch or tuple(checkpoint["stream_order"]) != STREAM_ORDER:
            raise ValueError("Checkpoint architecture or camera order differs")
        if (checkpoint["dataset"]["fingerprint"] != fingerprint or
                checkpoint["dataset"]["norm_stats_fingerprint"] != dataset.norm_stats_fingerprint):
            raise ValueError("Dataset fingerprint differs from the recovery checkpoint")
        critical = ("batch_size", "num_queries", "seed", "lr", "lr_backbone", "weight_decay", "lr_drop",
                    "kl_weight", "amp", "control_mode", "num_demos", "deterministic", "tf32", "max_grad_norm",
                    "fused_adamw", "ema_foreach", "compile_mode")
        resume_defaults = {"fused_adamw": False, "ema_foreach": False, "compile_mode": "off"}
        mismatches = [key for key in critical if checkpoint["args"].get(key, resume_defaults.get(key)) != jsonable(getattr(args, key))]
        if mismatches:
            raise ValueError(f"Resume configuration changed: {mismatches}")
        completed_steps = checkpoint["optimizer_step"]
        if checkpoint["dataloader"]["completed_batches"] != completed_steps:
            raise ValueError("One optimizer update per consumed batch is required")
        if args.total_steps < completed_steps:
            raise ValueError("total_steps is below the recovered optimizer step")

    sampler = ResumableBatchSampler(len(dataset), args.batch_size, args.total_steps, completed_steps, args.seed)
    if checkpoint and checkpoint["dataloader"] != sampler.state_dict(completed_steps):
        raise ValueError("Checkpoint dataloader cursor or sampling contract differs")
    loader_rng = torch.Generator().manual_seed(args.seed + 1_000_003)
    if checkpoint:
        # Recreate the same worker base seed as the original single iterator.
        # The saved committed sampler cursor selects the remaining batches.
        loader_rng.set_state(checkpoint["dataloader_worker_generator_start_state"])
    loader_rng_start_state = loader_rng.get_state().clone()
    loader_kwargs = dict(dataset=dataset, batch_sampler=sampler, num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda", worker_init_fn=seed_worker, generator=loader_rng)
    if args.num_workers:
        loader_kwargs.update(prefetch_factor=args.prefetch_factor, persistent_workers=args.persistent_workers)
    loader = DataLoader(**loader_kwargs)

    agent = Agent(static_spaces(action_dim=8), args, pretrained=args.pretrained and checkpoint is None).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in agent.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in agent.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
    ], lr=args.lr, weight_decay=args.weight_decay, fused=True if args.fused_adamw else None)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and device.type == "cuda")
    ema = EMAModel(parameters=agent.parameters(), power=0.75, foreach=args.ema_foreach)
    ema_agent = deepcopy(agent).eval().requires_grad_(False)
    if checkpoint:
        agent.load_state_dict(checkpoint["agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        ema.load_state_dict(checkpoint["ema"])
        ema.to(device=device)
        ema_agent.load_state_dict(checkpoint["ema_agent"])
        restore_rng(checkpoint["rng"])
        print("RESUMED", args.resume, "optimizer_step", completed_steps, flush=True)
    # Keep only the active models/optimizer; free CPU checkpoint storage.
    checkpoint = None
    agent.train()
    # Compile the callable only; Agent/DETRVAE retain their original state_dict
    # hierarchy for online inference, EMA and full recovery checkpoints.
    train_loss_fn = agent.compute_loss
    if args.compile_mode != "off":
        train_loss_fn = torch.compile(train_loss_fn, mode=args.compile_mode, dynamic=False)

    diagnostic_indices = np.linspace(0, len(dataset) - 1, min(args.diagnostic_batch_size, len(dataset)), dtype=int).tolist()
    config = dict(algo=ALGO_NAME, args=jsonable(vars(args)), architecture=arch, stream_order=STREAM_ORDER,
        state_fields=STATE_FIELDS, dataset=dict(path=str(args.data_path.resolve()), manifest=str((args.data_path / "manifest.json").resolve()),
        fingerprint=fingerprint, norm_stats_fingerprint=dataset.norm_stats_fingerprint,
        demos_loaded=dataset.num_traj, samples=len(dataset)),
        normalization="legacy_ACT_chunk_weighted_actions_torch_unbiased_std_clip_1e-2",
        diagnostic=dict(source="fixed_training_examples", is_validation=False, policy="EMA", indices=diagnostic_indices,
                        seed=args.seed + 10_000_019), online_evaluation=False)
    config_path = args.run_dir / "config.json"
    if not config_path.exists():
        atomic_json(config_path, config)
        atomic_torch_save(args.run_dir / "norm_stats.pt", dataset.norm_stats)
        atomic_json(args.run_dir / "norm_stats.json", dataset.norm_stats)
        atomic_json(args.run_dir / "dataset_reference.json", config["dataset"])
    elif not args.resume:
        raise FileExistsError(f"Run directory already contains a config; use --resume or another --run-dir: {args.run_dir}")
    else:
        existing_config = json.loads(config_path.read_text())
        if existing_config["architecture"] != arch or existing_config["dataset"] != config["dataset"]:
            raise ValueError("Existing run directory describes another architecture or dataset")
    atomic_json(args.run_dir / "last_invocation.json", dict(args=jsonable(vars(args)), resumed_step=completed_steps))
    print("INPUT_CONTRACT", json.dumps(jsonable(config)), flush=True)

    stop_signal = [None]
    def on_signal(signum, _frame):
        stop_signal[0] = signum
    old_handlers = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    in_update = False
    uncommitted_rng = None

    def save_checkpoint(name, reason, resume_safe=True, exception=None, rng_override=None):
        ema.copy_to(ema_agent.parameters())
        value = dict(checkpoint_version=CHECKPOINT_VERSION, algo=ALGO_NAME,
            optimizer_step=completed_steps, completed_batches=completed_steps, target_optimizer_steps=args.total_steps,
            architecture=arch, stream_order=STREAM_ORDER, state_fields=STATE_FIELDS,
            args=jsonable(vars(args)), norm_stats=dataset.norm_stats, dataset=config["dataset"],
            agent=agent.state_dict(), ema_agent=ema_agent.state_dict(), ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            rng=capture_rng() if rng_override is None else rng_override, dataloader=sampler.state_dict(completed_steps),
            dataloader_generator_state=loader_rng.get_state(), resume_safe=resume_safe,
            dataloader_worker_generator_start_state=loader_rng_start_state,
            checkpoint_reason=reason, exception=exception, timestamp=time.time())
        path = checkpoint_dir / name
        atomic_torch_save(path, value)
        print("CHECKPOINT_SAVED", path, "optimizer_step", completed_steps, "resume_safe", resume_safe, flush=True)

    metrics = (args.run_dir / "metrics.jsonl").open("a", buffering=1)
    def emit(record):
        metrics.write(json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False), flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    tick, run_start = time.perf_counter(), time.perf_counter()
    interval_count, wait_total = 0, 0.0
    totals = torch.zeros(4, device=device)
    previous_end = time.perf_counter()
    exit_code = 0
    try:
        emit(dict(event="training_start", optimizer_step=completed_steps, target_optimizer_steps=args.total_steps,
                  batch_size=args.batch_size, amp=args.amp, tf32=args.tf32, dataset_fingerprint=fingerprint))
        # Loader RNG is independent of the policy RNG.  Dataset has no stochastic
        # transforms; recreating workers on restart preserves every sampled item.
        for batch in loader:
            wait_total += time.perf_counter() - previous_end
            # Preserve the start-of-batch RNG for an exception before any
            # optimizer mutation.  Retrying that batch then uses the same latent
            # and dropout samples rather than silently advancing randomness.
            uncommitted_rng = capture_rng()
            obs, action_seq = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(args, device):
                loss_dict = train_loss_fn(obs, action_seq)
            if not torch.isfinite(torch.stack(tuple(loss_dict.values()))).all():
                raise FloatingPointError(f"Nonfinite loss before optimizer step {completed_steps + 1}")
            scaler.scale(loss_dict["loss"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            in_update = True
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.step(agent.parameters())
            completed_steps += 1
            in_update = False
            uncommitted_rng = None
            totals += torch.stack([loss_dict["loss"].detach(), loss_dict["l1"].detach(), loss_dict["kl"].detach(), grad_norm.detach()])
            interval_count += 1

            if completed_steps == 1 or completed_steps % args.log_every == 0 or completed_steps == args.total_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                now = time.perf_counter()
                elapsed = now - tick
                values = (totals / interval_count).cpu().tolist()
                emit(dict(event="train", optimizer_step=completed_steps, loss=values[0], l1=values[1], kl=values[2], grad_norm=values[3],
                    lr=optimizer.param_groups[0]["lr"], lr_backbone=optimizer.param_groups[1]["lr"], ips=interval_count / elapsed,
                    samples_per_sec=interval_count * args.batch_size / elapsed, interval_steps=interval_count,
                    data_wait_sec_mean=wait_total / interval_count, elapsed_sec=now - run_start,
                    max_memory=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                    reserved=torch.cuda.memory_reserved(device) if device.type == "cuda" else 0))
                totals.zero_()
                interval_count, wait_total = 0, 0.0
                tick = time.perf_counter()

            if args.diagnostic_every and (completed_steps % args.diagnostic_every == 0 or completed_steps == args.total_steps):
                ema.copy_to(ema_agent.parameters())
                values = fixed_diagnostic(ema_agent, dataset, args, device, diagnostic_indices)
                emit(dict(event="fixed_training_loss_diagnostic", optimizer_step=completed_steps, policy="EMA",
                          is_validation=False, examples=len(diagnostic_indices), **values))
            numbered = (args.save_every and completed_steps % args.save_every == 0) or completed_steps == args.total_steps
            if numbered:
                save_checkpoint(f"{completed_steps}.pt", "optimizer_step_milestone")
            if (args.recovery_every and completed_steps % args.recovery_every == 0) or numbered:
                save_checkpoint("latest.pt", "recovery")
            if stop_signal[0] is not None:
                save_checkpoint("latest.pt", f"signal_{stop_signal[0]}")
                emit(dict(event="interrupted", optimizer_step=completed_steps, signal=stop_signal[0]))
                exit_code = 128 + stop_signal[0]
                break
            # Release the finished GPU inputs and pinned host batch before the
            # next H2D copy, keeping only model/optimizer/EMA state resident.
            del obs, action_seq, loss_dict, grad_norm, batch
            previous_end = time.perf_counter()
        else:
            emit(dict(event="training_complete", optimizer_step=completed_steps, target_optimizer_steps=args.total_steps))
    except BaseException as exc:
        # A partial AdamW/EMA update cannot be asserted to be resumable.  Preserve
        # an emergency artifact, while retaining the previous safe latest.pt.
        try:
            save_checkpoint("exception.pt", "exception", resume_safe=not in_update,
                            exception=f"{type(exc).__name__}: {exc}", rng_override=uncommitted_rng)
        except BaseException as save_exc:
            print(f"EMERGENCY_CHECKPOINT_FAILED: {type(save_exc).__name__}: {save_exc}", file=sys.stderr, flush=True)
        raise
    finally:
        metrics.close()
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
