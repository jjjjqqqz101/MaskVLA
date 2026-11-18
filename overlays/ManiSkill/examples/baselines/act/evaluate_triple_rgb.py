#!/usr/bin/env python3
"""Load a three-stream ACT EMA checkpoint and check its offline input contract.

Default execution uses CPU and synthetic inputs; it never creates a simulator,
trains a model, downloads weights, or reports a success rate.  --data-path uses
one raw training frame for an input-contract check, not independent validation.
TriRGBObservationWrapper is a reusable adapter for explicitly requested future
simulation evaluation, and is not instantiated by this command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


STREAM_ORDER = ("base_rgb", "wrist_rgb", "base_masked_rgb")
STATE_FIELDS = (("agent", "qpos"), ("agent", "qvel"),
                ("extra", "tcp_pose"), ("extra", "goal_pos"))
STATE_DIMS = (9, 9, 7, 3)
RGB_SHAPE = (3, 3, 224, 224)


def policy_state(observation):
    """Read only qpos9, qvel9, tcp_pose7 and goal_pos3, in that order."""
    parts = []
    for (group, key), dim in zip(STATE_FIELDS, STATE_DIMS):
        value = torch.as_tensor(observation[group][key], dtype=torch.float32)
        if value.ndim != 2 or value.shape[-1] != dim:
            raise ValueError(f"Expected batched {group}/{key} [N,{dim}], got {tuple(value.shape)}")
        parts.append(value)
    if any(p.shape[0] != parts[0].shape[0] for p in parts):
        raise ValueError("Approved state fields have inconsistent batch sizes")
    state = torch.cat([p.to(parts[0].device) for p in parts], dim=-1)
    if state.shape[-1] != 28 or not torch.isfinite(state).all():
        raise ValueError("The approved 28-dimensional state must be finite")
    return state


def triple_rgb(base_rgb, wrist_rgb, target_mask):
    """Stack raw 224x224 RGB, wrist RGB and black-background target RGB.

    No resize, pixel scaling or ImageNet normalization happens here.  Target
    identity/segmentation is used only to produce the third image, never state.
    """
    base, wrist = torch.as_tensor(base_rgb), torch.as_tensor(wrist_rgb)
    if base.ndim != 4 or tuple(base.shape[1:]) != (224, 224, 3):
        raise ValueError(f"Base RGB must be uint8 [N,224,224,3], got {tuple(base.shape)}")
    if wrist.shape != base.shape or base.dtype != torch.uint8 or wrist.dtype != torch.uint8:
        raise ValueError("Base and wrist RGB must have matching uint8 NHWC shapes")
    mask = torch.as_tensor(target_mask, device=base.device)
    if mask.shape != base.shape[:-1] + (1,):
        raise ValueError("Target mask must have shape [N,224,224,1]")
    if mask.dtype != torch.bool:
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Target mask must be Boolean or contain only zero and one")
        mask = mask.bool()
    masked = torch.where(mask, base, 0)
    return torch.stack([base.permute(0, 3, 1, 2),
                        wrist.to(base.device).permute(0, 3, 1, 2),
                        masked.permute(0, 3, 1, 2)], dim=1).contiguous()


# gymnasium already belongs to the simulator environment.  Keeping it optional
# permits offline checkpoint inspection in an otherwise minimal PyTorch setup.
try:
    from gymnasium import ObservationWrapper as _ObservationWrapper
except ImportError:
    class _ObservationWrapper:
        def __init__(self, env):
            raise RuntimeError("TriRGBObservationWrapper requires the existing gymnasium simulator dependency")


class TriRGBObservationWrapper(_ObservationWrapper):
    """Explicit three-stream simulation adapter; returns UNNORMALIZED state.

    With mask_provider=None, the simulator's target segmentation gives an oracle
    mask, matching demonstration collection.  A caller may supply
    mask_provider(raw_observation) -> [N,224,224,1] for another mask source.
    CheckpointPolicy.predict performs state normalization and unnormalizes the
    action chunk; the Agent itself handles RGB pixel/ImageNet normalization.
    """
    def __init__(self, env, mask_provider=None):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.mask_provider = mask_provider
        initial = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(initial)
        self.observation_space = env.observation_space

    def _oracle_mask(self, observation):
        camera = observation["sensor_data"]["base_camera"]
        segmentation = torch.as_tensor(camera["segmentation"])
        if segmentation.ndim != 4 or tuple(segmentation.shape[1:]) != (224, 224, 1):
            raise ValueError("Base segmentation must have shape [N,224,224,1]")
        ids = torch.stack([obj.per_scene_id for obj in self.base_env.cubes], dim=1)
        target_index = self.base_env.target_idx.to(device=ids.device, dtype=torch.long)
        target_ids = ids.gather(1, target_index[:, None]).to(segmentation.device)
        return segmentation == target_ids[:, None, None, :]

    def observation(self, observation):
        mask = self._oracle_mask(observation) if self.mask_provider is None else self.mask_provider(observation)
        cameras = observation["sensor_data"]
        state = policy_state(observation)
        rgb = triple_rgb(cameras["base_camera"]["rgb"], cameras["hand_camera"]["rgb"], mask)
        if state.shape[0] != rgb.shape[0]:
            raise ValueError("State and RGB have different batch sizes")
        return {"state": state, "rgb": rgb}


def checked_stats(value):
    result = {}
    for key, shape in (("state_mean", (1, 28)), ("state_std", (1, 28)),
                       ("action_mean", (1, 8)), ("action_std", (1, 8))):
        tensor = torch.as_tensor(value[key], dtype=torch.float32, device="cpu")
        if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid checkpoint normalization: {key} must be finite {shape}")
        if key.endswith("std") and not (tensor >= 1e-2 - 1e-8).all():
            raise ValueError(f"Invalid checkpoint normalization: {key} must be clipped to at least .01")
        result[key] = tensor
    return result


class CheckpointPolicy:
    """Strict EMA-only inference facade. Input: raw state and uint8 images."""
    def __init__(self, checkpoint_path, device="cpu"):
        from train_triple_rgb import Agent, architecture, static_spaces

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        # Training recovery checkpoints contain Python/NumPy RNG states; load
        # only locally produced/trusted checkpoints with this full-state API.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_version") != 1 or checkpoint.get("algo") != "BC_ACT_triple_rgb":
            raise ValueError("Unsupported three-stream ACT checkpoint")
        if tuple(checkpoint.get("stream_order", ())) != STREAM_ORDER:
            raise ValueError("Checkpoint image stream order differs")
        if tuple(tuple(x) for x in checkpoint.get("state_fields", ())) != STATE_FIELDS:
            raise ValueError("Checkpoint state whitelist differs from the approved 28 values")
        args = SimpleNamespace(**checkpoint["args"])
        expected_arch = architecture(args, action_dim=8)
        if checkpoint["architecture"] != expected_arch or args.num_queries != 30:
            raise ValueError("Checkpoint architecture differs from ACT three RGB / state28 / actions30x8")
        self.stats = {k: v.to(self.device) for k, v in checked_stats(checkpoint["norm_stats"]).items()}
        self.agent = Agent(static_spaces(action_dim=8), args, pretrained=False).to(self.device)
        self.agent.load_state_dict(checkpoint["ema_agent"], strict=True)
        self.agent.eval().requires_grad_(False)
        if any(not torch.isfinite(p).all() for p in self.agent.parameters()):
            raise ValueError("Checkpoint EMA parameters contain nonfinite values")
        self.metadata = {key: checkpoint.get(key) for key in (
            "checkpoint_version", "optimizer_step", "target_optimizer_steps", "architecture",
            "dataset", "resume_safe", "checkpoint_reason")}

    @torch.inference_mode()
    def predict(self, observation, return_normalized=False):
        """Return physical pd_joint_pos action chunks [N,30,8]."""
        state = torch.as_tensor(observation["state"], dtype=torch.float32, device=self.device)
        rgb = torch.as_tensor(observation["rgb"], device=self.device)
        if state.ndim != 2 or state.shape[1] != 28 or not torch.isfinite(state).all():
            raise ValueError("Policy state must be finite raw float32 [N,28]")
        if tuple(rgb.shape) != (state.shape[0],) + RGB_SHAPE or rgb.dtype != torch.uint8:
            raise ValueError("Policy RGB must be uint8 [N,3,3,224,224], in the checkpoint stream order")
        state = (state - self.stats["state_mean"]) / self.stats["state_std"]
        normalized = self.agent.get_action({"state": state, "rgb": rgb})
        if tuple(normalized.shape) != (state.shape[0], 30, 8) or not torch.isfinite(normalized).all():
            raise ValueError("EMA policy did not produce finite normalized actions [N,30,8]")
        action = normalized.float() * self.stats["action_std"] + self.stats["action_mean"]
        if not torch.isfinite(action).all():
            raise ValueError("Action denormalization produced nonfinite values")
        return (action, normalized) if return_normalized else action


def smoke_observation(policy, data_path=None, sample_index=0):
    if data_path is None:
        # Distinct, deterministic streams exercise stream and channel dimensions.
        rgb = torch.arange(3 * 3 * 224 * 224, dtype=torch.int64).remainder(251).to(torch.uint8)
        return {"state": policy.stats["state_mean"].cpu().clone(),
                "rgb": rgb.reshape(1, *RGB_SHAPE)}, "synthetic"
    root = Path(data_path)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "act_triple_rgb_memmap_v1" or not manifest.get("complete"):
        raise ValueError("Smoke sample dataset is incomplete or has another schema")
    expected = policy.metadata["dataset"]["fingerprint"]
    if manifest["dataset_fingerprint"] != expected:
        raise ValueError("Smoke sample dataset differs from the checkpoint")
    if tuple(manifest["stream_order"]) != STREAM_ORDER or manifest["state_fields"] != ["/".join(x) for x in STATE_FIELDS]:
        raise ValueError("Smoke sample observation contract differs")
    if not 0 <= sample_index < manifest["num_frames"]:
        raise IndexError(sample_index)
    rgb = np.load(root / "rgb.npy", mmap_mode="r", allow_pickle=False)
    state = np.load(root / "state.npy", mmap_mode="r", allow_pickle=False)
    if rgb.dtype != np.uint8 or rgb.shape[1:] != (3, 224, 224, 3) or state.shape[1:] != (28,):
        raise ValueError("Smoke sample arrays have invalid shape or dtype")
    return {"state": torch.from_numpy(np.array(state[sample_index:sample_index + 1], dtype=np.float32, copy=True)),
            "rgb": torch.from_numpy(rgb[sample_index:sample_index + 1].transpose(0, 1, 4, 2, 3).copy())}, "training_frame"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="CPU by default; GPU requires explicit choice")
    parser.add_argument("--data-path", type=Path, help="Optional prepared memmap training frame")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    policy = CheckpointPolicy(args.checkpoint, args.device)
    observation, source = smoke_observation(policy, args.data_path, args.sample_index)
    action, normalized = policy.predict(observation, return_normalized=True)
    repeat = policy.predict(observation)
    if not torch.equal(action, repeat):
        raise ValueError("EMA eval inference was not deterministic for identical inputs")
    report = dict(event="offline_contract_smoke", passed=True, policy="EMA", source=source,
                  is_validation=False, online_evaluation=False, device=str(policy.device),
                  optimizer_step=policy.metadata["optimizer_step"],
                  checkpoint_reason=policy.metadata["checkpoint_reason"],
                  stream_order=list(STREAM_ORDER), state_fields=["/".join(x) for x in STATE_FIELDS],
                  input_rgb_shape=list(observation["rgb"].shape), input_rgb_dtype="uint8",
                  raw_state_shape=list(observation["state"].shape),
                  action_shape=list(action.shape), finite_normalized_actions=bool(torch.isfinite(normalized).all()),
                  finite_actions=bool(torch.isfinite(action).all()), deterministic=True,
                  action_min=float(action.min()), action_max=float(action.max()),
                  normalization="checkpoint state z-score; Agent RGB /255 then ImageNet; checkpoint action inverse z-score",
                  resume_safe=policy.metadata["resume_safe"])
    output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output)
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
