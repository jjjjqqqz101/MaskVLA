"""Original-RGB-only ACT ablation; same actions, state and optimization as dual RGB."""
ALGO_NAME = 'BC_ACT_single_rgb'

import argparse
import json
import h5py
import os
import random
from distutils.util import strtobool
from functools import partial
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from evaluate_dual_rgb import evaluate
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.registration import REGISTERED_ENVS

from collections import defaultdict

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from act.utils import IterationBasedBatchSampler, worker_init_fn
from act.make_env import make_eval_envs
from act.single_rgb_video import SingleRGBVideoWrapper
from diffusers.training_utils import EMAModel
from act.detr.backbone import build_backbone
from act.detr.transformer import build_transformer
from act.detr.detr_vae import build_encoder, DETRVAE
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import tyro

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "TargetPickCube-v1"
    """the id of the environment"""
    demo_path: str = '../../../demos/TargetPickCube-v1/motionplanning/target_pick_500_mask/dataset/train.h5'
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = 450
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""

    # ACT specific arguments
    lr: float = 1e-4
    """the learning rate of the Action Chunking with Transformers"""
    kl_weight: float = 10
    """weight for the kl loss term"""
    temporal_agg: bool = True
    """if toggled, temporal ensembling will be performed"""

    # Backbone
    position_embedding: str = 'sine'
    backbone: str = 'resnet18'
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = False

    # Transformer
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = 500
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 100
    """the frequency of logging the training metrics"""
    eval_freq: int = 20000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = 29200
    """the frequency of saving the model checkpoints. The default is 29200 updates, plus best and final checkpoints."""
    num_eval_episodes: int = 50
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 1
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = 'pd_joint_pos'
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    eval_only_checkpoint: Optional[str] = None
    """Evaluate an existing single-RGB checkpoint without training; architecture arguments must match."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


# This whitelist is shared by the offline dataset and online evaluation.
# Target identity and object poses must never be flattened into the policy state.
STATE_FIELDS = (("agent", "qpos"), ("agent", "qvel"),
                ("extra", "tcp_pose"), ("extra", "goal_pos"))
STREAM_ORDER = ("original_rgb",)


def policy_state(obs):
    parts = [torch.as_tensor(obs[group][key]).float() for group, key in STATE_FIELDS]
    state = torch.cat(parts, dim=-1)
    if state.shape[-1] != 28:
        raise ValueError(f"Expected Panda state dimension 28, got {state.shape}")
    return state


def single_rgb(raw_rgb, transforms):
    raw_rgb = torch.as_tensor(raw_rgb)
    if raw_rgb.ndim != 4 or raw_rgb.shape[-1] != 3 or raw_rgb.dtype != torch.uint8:
        raise ValueError("Original RGB must be uint8 NHWC")
    # Keep exactly one camera stream. No segmentation, masked image or dummy stream.
    return transforms(raw_rgb.permute(0, 3, 1, 2)).unsqueeze(1)


class SingleRGBObservationWrapper(gym.ObservationWrapper):
    """Evaluation adapter returning only RGB and the approved 28-D state."""

    def __init__(self, env):
        super().__init__(env)
        self.base_env: BaseEnv = env.unwrapped
        self.transforms = T.Resize((224, 224), antialias=True)
        self.base_env.update_obs_space(self.observation(self.base_env._init_raw_obs))

    def observation(self, observation):
        camera = observation["sensor_data"]["base_camera"]
        return dict(state=policy_state(observation),
                    rgb=single_rgb(camera["rgb"], self.transforms))


class SmallDemoDataset_ACTPolicy(Dataset): # Load everything into memory
    def __init__(self, data_path, num_queries, num_traj, include_depth=False, control_mode="pd_joint_pos"):
        self.control_mode = control_mode
        if include_depth:
            raise ValueError("Single-stream ACT uses one RGB stream, without depth")
        self.include_depth = include_depth
        self.transforms = T.Resize((224, 224), antialias=True)
        # Read only original RGB, approved state fields and expert actions.
        # Do not open derived/, segmentation, target metadata or object poses.
        trajectories = dict(observations=[], actions=[])
        with h5py.File(data_path, "r") as source:
            names = sorted(source.keys(), key=lambda name: int(name.split("_")[-1]))
            if num_traj is not None:
                if not 0 < num_traj <= len(names):
                    raise ValueError("Requested demo count is unavailable")
                names = names[:num_traj]
            for name in names:
                trajectory = source[name]
                observation = {
                    "agent": {key: trajectory[f"obs/agent/{key}"][:] for key in ("qpos", "qvel")},
                    "extra": {key: trajectory[f"obs/extra/{key}"][:] for key in ("tcp_pose", "goal_pos")},
                    "sensor_data": {"base_camera": {"rgb": trajectory["obs/sensor_data/base_camera/rgb"][:]}},
                }
                actions = trajectory["actions"][:]
                frames = len(actions) + 1
                if any(len(observation[group][key]) != frames for group, key in STATE_FIELDS):
                    raise ValueError(f"State/action time alignment error in {name}")
                if len(observation["sensor_data"]["base_camera"]["rgb"]) != frames:
                    raise ValueError(f"RGB/action time alignment error in {name}")
                # Process one episode at a time: do not retain a second full
                # copy of the raw RGB dataset when loading hundreds of demos.
                trajectories["observations"].append(self.process_obs(observation))
                trajectories["actions"].append(actions)
        self.obs_keys = list(trajectories['observations'][0].keys())

        # Pre-process the actions
        for i in range(len(trajectories['actions'])):
            trajectories['actions'][i] = torch.Tensor(trajectories['actions'][i])
        print('Obs/action pre-processing is done.')

        # When the robot reaches the goal state, its joints and gripper fingers need to remain stationary
        if 'delta_pos' in self.control_mode or self.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,))
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        # else:
        #     raise NotImplementedError(f'Control Mode {self.control_mode} not supported')

        self.slices = []
        self.num_traj = len(trajectories['actions'])
        for traj_idx in range(self.num_traj):
            episode_len = trajectories['actions'][traj_idx].shape[0]
            self.slices += [
                (traj_idx, ts) for ts in range(episode_len)
            ]

        print(f"Length of Dataset: {len(self.slices)}")

        self.num_queries = num_queries
        self.trajectories = trajectories
        self.delta_control = 'delta' in self.control_mode
        self.norm_stats = self.get_norm_stats() if not self.delta_control else None

    def __getitem__(self, index):
        traj_idx, ts = self.slices[index]

        # get state at start_ts only
        state = self.trajectories['observations'][traj_idx]['state'][ts]
        # get num_queries actions
        act_seq = self.trajectories['actions'][traj_idx][ts:ts+self.num_queries]
        action_len = act_seq.shape[0]

        # Pad after the trajectory, so all the observations are utilized in training
        if action_len < self.num_queries:
            if 'delta_pos' in self.control_mode or self.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
                act_seq = torch.cat([act_seq, pad_action.repeat(self.num_queries-action_len, 1)], dim=0)
                # making the robot (arm and gripper) stay still
            elif not self.delta_control:
                target = act_seq[-1]
                act_seq = torch.cat([act_seq, target.repeat(self.num_queries-action_len, 1)], dim=0)

        # normalize state and act_seq
        if not self.delta_control:
            state = (state - self.norm_stats["state_mean"][0]) / self.norm_stats["state_std"][0]
            act_seq = (act_seq - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # get rgb or rgbd data at start_ts and combine with state to form obs
        if self.include_depth:
            rgb = self.trajectories['observations'][traj_idx]['rgb'][ts]
            depth = self.trajectories['observations'][traj_idx]['depth'][ts]
            obs = dict(state=state, rgb=rgb, depth=depth)
        else:
            rgb = self.trajectories['observations'][traj_idx]['rgb'][ts]
            obs = dict(state=state, rgb=rgb)

        return {
            'observations': obs,
            'actions': act_seq,
        }

    def __len__(self):
        return len(self.slices)

    def process_obs(self, obs_dict):
        rgb = single_rgb(obs_dict["sensor_data"]["base_camera"]["rgb"], self.transforms)
        return dict(state=policy_state(obs_dict), rgb=rgb)

    def get_norm_stats(self):
        all_state_data = []
        all_action_data = []
        for traj_idx, ts in self.slices:
            state = self.trajectories['observations'][traj_idx]['state'][ts]
            act_seq = self.trajectories['actions'][traj_idx][ts:ts+self.num_queries]
            action_len = act_seq.shape[0]
            if action_len < self.num_queries:
                target_pos = act_seq[-1]
                act_seq = torch.cat([act_seq, target_pos.repeat(self.num_queries-action_len, 1)], dim=0)
            all_state_data.append(state)
            all_action_data.append(act_seq)

        all_state_data = torch.stack(all_state_data)
        all_action_data = torch.concatenate(all_action_data)

        # normalize obs (state) data
        state_mean = all_state_data.mean(dim=0, keepdim=True)
        state_std = all_state_data.std(dim=0, keepdim=True)
        state_std = torch.clip(state_std, 1e-2, np.inf) # clipping

        # normalize action data
        action_mean = all_action_data.mean(dim=0, keepdim=True)
        action_std = all_action_data.std(dim=0, keepdim=True)
        action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

        stats = {"action_mean": action_mean, "action_std": action_std,
                 "state_mean": state_mean, "state_std": state_std,
                 "example_state": state}

        return stats


class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        assert len(env.single_observation_space['state'].shape) == 1 # (obs_dim,)
        assert len(env.single_observation_space['rgb'].shape) == 4 # (num_cams, C, H, W)
        assert len(env.single_action_space.shape) == 1 # (act_dim,)
        #assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()

        self.state_dim = env.single_observation_space['state'].shape[0]
        self.act_dim = env.single_action_space.shape[0]
        self.kl_weight = args.kl_weight
        self.include_depth = args.include_depth
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        # CNN backbone
        backbones = []
        backbone = build_backbone(args)
        backbones.append(backbone)

        # CVAE decoder
        transformer = build_transformer(args)

        # CVAE encoder
        encoder = build_encoder(args)

        # ACT ( CVAE encoder + (CNN backbones + CVAE decoder) )
        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=self.state_dim,
            action_dim=self.act_dim,
            num_queries=args.num_queries,
        )

    def compute_loss(self, obs, action_seq):
        # normalize rgb data
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])

        # depth data
        if self.include_depth:
            obs['depth'] = obs['depth'].float()

        # forward pass
        a_hat, (mu, logvar) = self.model(obs, action_seq)

        # compute l1 loss and kl loss
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        all_l1 = F.l1_loss(action_seq, a_hat, reduction='none')
        l1 = all_l1.mean()

        # store all loss
        loss_dict = dict()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
        return loss_dict

    def get_action(self, obs):
        # normalize rgb data
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])

        # depth data
        if self.include_depth:
            obs['depth'] = obs['depth'].float()

        # forward pass
        a_hat, (_, _) = self.model(obs) # no action, sample from prior

        return a_hat


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def save_ckpt(run_name, tag):
    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        'norm_stats': dataset.norm_stats,
        'args': vars(args),
        'stream_order': STREAM_ORDER,
        'state_fields': STATE_FIELDS,
        'input_mode': 'single_rgb',
        'iteration': step,
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
    }, f'runs/{run_name}/checkpoints/{tag}.pt')

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.include_depth:
        raise ValueError("Use --no-include-depth for the single RGB stream")
    if args.total_iters < 2:
        raise ValueError("total_iters must be at least 2")
    torch.set_num_threads(4)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # Each invocation owns its output directory; never overwrite a prior run.
    os.makedirs(f"runs/{run_name}", exist_ok=False)

    if args.demo_path.endswith('.h5'):
        import json
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="rgb", render_mode="rgb_array")
    if args.demo_path.endswith('.h5'):
        # Keep the evaluation camera identical to the camera used to collect data.
        recorded_kwargs = demo_info['env_info']['env_kwargs']
        for key in ('sensor_configs', 'robot_uids', 'robot_init_qpos_noise'):
            if key in recorded_kwargs:
                env_kwargs[key] = recorded_kwargs[key]
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = None
    wrappers = [SingleRGBObservationWrapper]
    if args.capture_video:
        wrappers.append(SingleRGBVideoWrapper)
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, other_kwargs, video_dir=f'runs/{run_name}/videos' if args.capture_video else None, wrappers=wrappers, video_info=False)

    # dataloader setup
    dataset = SmallDemoDataset_ACTPolicy(args.demo_path, args.num_queries, num_traj=args.num_demos, include_depth=args.include_depth, control_mode=args.control_mode)
    if args.batch_size > len(dataset):
        raise ValueError("batch_size exceeds dataset length with drop_last=True")
    sample = dataset[0]
    assert sample["observations"]["rgb"].shape == (1, 3, 224, 224)
    assert sample["observations"]["state"].shape == (28,)
    print("INPUT_CONTRACT", dict(sample_rgb=[1, 3, 224, 224], state_dim=28,
                                 stream_order=STREAM_ORDER, state_fields=STATE_FIELDS), flush=True)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )
    if args.num_demos is None:
        args.num_demos = dataset.num_traj

    obs_mode = "rgb+depth" if args.include_depth else "rgb"

    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="ACT",
            tags=["act"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    with open(f"runs/{run_name}/config.json", "w") as config_file:
        json.dump(dict(args=vars(args), stream_order=STREAM_ORDER, state_fields=STATE_FIELDS,
                       state_dim=28, sample_rgb_shape=[1, 3, 224, 224],
                       demos_loaded=dataset.num_traj, dataset_samples=len(dataset)), config_file, indent=2)
    metrics_file = open(f"runs/{run_name}/metrics.jsonl", "w", buffering=1)

    # agent setup
    agent = Agent(envs, args).to(device)

    # optimizer setup
    param_dicts = [
        {"params": [p for n, p in agent.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in agent.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)

    # LR drop by a factor of 10 after lr_drop iters
    lr_drop = int((2/3)*args.total_iters)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)

    # Keep normalization tensors on the device used by the shared evaluator.
    # CPU evaluation explicitly converts state stats to NumPy; action stats need CUDA.
    eval_norm_stats = None if dataset.norm_stats is None else {
        k: v.to(device) for k, v in dataset.norm_stats.items()
    }

    # Evaluation
    #eval_kwargs = dict(
    #    stats=eval_norm_stats, num_queries=args.num_queries, temporal_agg=args.temporal_agg,
    #    max_timesteps=gym_utils.find_max_episode_steps_value(envs), device=device, sim_backend=args.sim_backend
    #)
    eval_kwargs = dict(
        stats=eval_norm_stats, num_queries=args.num_queries, temporal_agg=args.temporal_agg,
        max_timesteps=gym_utils.find_max_episode_steps_value(envs), device=device, sim_backend=args.sim_backend
    )

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    if args.eval_only_checkpoint:
        checkpoint = torch.load(args.eval_only_checkpoint, map_location=device, weights_only=False)
        if tuple(checkpoint["stream_order"]) != STREAM_ORDER or tuple(checkpoint["state_fields"]) != STATE_FIELDS:
            raise ValueError("Checkpoint observation contract does not match single RGB")
        ema_agent.load_state_dict(checkpoint["ema_agent"])
        eval_kwargs["stats"] = None if checkpoint["norm_stats"] is None else {
            key: value.to(device) for key, value in checkpoint["norm_stats"].items()
        }
        metrics = evaluate(args.num_eval_episodes, ema_agent, envs, eval_kwargs)
        result = {key: float(np.mean(value)) for key, value in metrics.items()}
        metrics_file.write(json.dumps(dict(checkpoint=args.eval_only_checkpoint, evaluation=result)) + "\n")
        print("EVALUATION_COMPLETE", json.dumps(result), flush=True)
        metrics_file.close()
        envs.close()
        writer.close()
        raise SystemExit(0)

    agent.train()

    # Save a best checkpoint even if the first measured success rate is zero.
    best_eval_metrics = defaultdict(lambda: -float("inf"))
    timings = defaultdict(float)

    for cur_iter, data_batch in enumerate(train_dataloader):
        last_tick = time.time()
        # copy data from cpu to gpu
        obs_batch_dict = data_batch['observations']
        obs_batch_dict = {k: v.to(device, non_blocking=True) for k, v in obs_batch_dict.items()}
        act_batch = data_batch['actions'].to(device, non_blocking=True)

        if cur_iter == 0:
            assert obs_batch_dict["rgb"].shape == (args.batch_size, 1, 3, 224, 224)
            print("BATCH_SHAPE", list(obs_batch_dict["rgb"].shape), flush=True)

        # forward and compute loss
        loss_dict = agent.compute_loss(
            obs=obs_batch_dict, # obs_batch_dict['state'] is (B, obs_dim)
            action_seq=act_batch, # (B, num_queries, act_dim)
        )
        total_loss = loss_dict['loss']  # total_loss = l1 + kl * self.kl_weight

        if not all(torch.isfinite(value).all() for value in loss_dict.values()):
            raise FloatingPointError(f"Nonfinite loss at iteration {cur_iter + 1}")

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        lr_scheduler.step() # step lr scheduler every batch, this is different from standard pytorch behavior

        # update Exponential Moving Average of the model weights
        ema.step(agent.parameters())
        timings["update"] += time.time() - last_tick

        step = cur_iter + 1
        metrics_file.write(json.dumps(dict(iteration=step, **{k: float(v.detach()) for k, v in loss_dict.items()},
                                          grad_norm=float(grad_norm))) + "\n")

        if step == args.total_iters:
            save_ckpt(run_name, "final")

        # Evaluate at the requested interval and after the final update.
        if step % args.eval_freq == 0 or step == args.total_iters:
            last_tick = time.time()

            ema.copy_to(ema_agent.parameters())

            eval_metrics = evaluate(args.num_eval_episodes, ema_agent, envs, eval_kwargs)
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = float(np.mean(eval_metrics[k]))
                writer.add_scalar(f"eval/{k}", eval_metrics[k], cur_iter)
                print(f"{k}: {eval_metrics[k]:.4f}")

            metrics_file.write(json.dumps(dict(iteration=step, evaluation=eval_metrics)) + "\n")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(f'New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint.')

        if step == 1 or step % args.log_freq == 0:
            print(f"Iteration {step}, loss: {total_loss.item()}, l1: {loss_dict['l1'].item()}, kl: {loss_dict['kl'].item()}", flush=True)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], cur_iter)
            writer.add_scalar("charts/backbone_learning_rate", optimizer.param_groups[1]["lr"], cur_iter)
            writer.add_scalar("losses/total_loss", total_loss.item(), cur_iter)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, cur_iter)

        # Checkpoint
        if args.save_freq is not None and step % args.save_freq == 0:
            save_ckpt(run_name, str(step))

    save_ckpt(run_name, "final")
    metrics_file.close()
    envs.close()
    writer.close()
