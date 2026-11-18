# Run the console and prepare the research environment

The console can be previewed on a Mac with Python 3.11+ and Git. Simulated rollouts use a Linux NVIDIA host, a working Vulkan renderer, simulation assets, a policy checkpoint and a configured vision-language model.

## Preview the interface

From the project root, install the ManiSkill source tree, start the local console and open [localhost:18840](http://127.0.0.1:18840).

```bash
python3 setup/prepare.py ManiSkill
cd repo/ManiSkill/examples/baselines/act
python3 -m task_agent.web --port 18840
```

The console serves the task and RoboCasa pages immediately. Creating a scene or starting a rollout requires the corresponding backend. The console listens on the loopback interface.

## Reconstruct the source tree

`python3 setup/prepare.py` fetches each dependency at the revision below. If you already prepared ManiSkill, run `python3 setup/prepare.py openpi lerobot sam2 Cutie` to add the others. The script applies project changes in `setup/patches/`, copies the additional modules from `overlays/` and places the results under `repo/`. It preserves existing component directories. A checkout of the upstream project may contain its own examples and assets.

| Dependency | Revision | Local additions |
| --- | --- | --- |
| [ManiSkill](https://github.com/haosulab/ManiSkill) | `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` | Target-picking tasks, demonstration generation, ACT training, attention tools, perception service and task console |
| [openpi](https://github.com/Physical-Intelligence/openpi) | `15a9616a00943ada6c20a0f158e3adb39df2ccac` | ManiSkill and RoboCasa observations, policy transforms and training configurations |
| [LeRobot](https://github.com/huggingface/lerobot) | `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` | Used by openpi to load training data |
| [SAM 2](https://github.com/facebookresearch/sam2) | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Used to initialize the target mask from a box |
| [Cutie](https://github.com/hkchengrex/Cutie) | `ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a` | Propagates the target mask across frames |

Upstream code retains its original license. Notices for the ManiSkill and openpi source changes are collected in [licenses](licenses/). The openpi revision was selected by comparing the project copy against upstream history. Its unchanged source and configuration files match this revision.

## Linux environments

The original setup used Python 3.11, CUDA 12.6 PyTorch wheels and separate environments for simulation, segmentation and JAX inference. The following commands assume a CUDA-compatible NVIDIA driver and Python 3.11 with `venv` installed. Run them from the project root in Bash.

```bash
source setup/environment.sh
python3.11 -m venv "$MASK_VENV_ROOT/motion-planning"
"$MASK_ACT_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1 torchvision==0.22.1
"$MASK_ACT_PYTHON" -m pip install -r setup/motion-requirements.txt
"$MASK_ACT_PYTHON" -m pip install --no-deps -e repo/ManiSkill -e repo/ManiSkill/examples/baselines/act
"$MASK_ACT_PYTHON" -m mani_skill.utils.download_asset ycb

python3.11 -m venv "$MASK_VENV_ROOT/perception-tracking"
"$PERCEPTION_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1 torchvision==0.22.1
"$PERCEPTION_PYTHON" -m pip install --no-build-isolation -e repo/third_party/sam2
"$PERCEPTION_PYTHON" -m pip install -e repo/third_party/Cutie
```

For the optional π0.5 service, install its pinned CUDA packages and runtime in a dedicated environment. Keep checkpoints, tokenizer and normalization statistics associated with the policy that produced them.

```bash
python3.11 -m venv "$MASK_VENV_ROOT/openpi-inference"
"$OPENPI_PYTHON" -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu126 -r setup/openpi-cu126.txt
"$OPENPI_PYTHON" -m pip install --no-deps -r setup/openpi-runtime.txt
"$OPENPI_PYTHON" -m pip install --no-deps -e repo/third_party/lerobot -e repo/openpi/packages/openpi-client -e repo/openpi
```

## Local inputs and services

Place inputs inside the project root. The paths below are defaults relative to that root. Environment variables allow the selected inputs to be changed.

| Input | Default path | Override |
| --- | --- | --- |
| Dual-stream ACT | `ckpt/maniskill/act_dual_rgb_8x100/100000.pt` | `MASK_ACT_CHECKPOINT` |
| Triple-stream ACT | `ckpt/maniskill/act_triple_rgb_25x50/160000.pt` | `MASK_ACT_TRIPLE_CHECKPOINT` |
| π0.5 JAX checkpoint | `ckpt/maniskill/pi05_25x50/49999/` | `MASK_PI05_CHECKPOINT` |
| SAM 2.1 Hiera Small | `ckpt/perception/sam2.1_hiera_small.pt` | `SAM2_CHECKPOINT` |
| Cutie Base Mega | `ckpt/perception/cutie-base-mega.pth` | `CUTIE_CHECKPOINT` |
| ManiSkill assets root | `assets/maniskill/` | `MS_ASSET_DIR` |

Configure the VLM in the shell that launches the ManiSkill task console. The [agent guide](../docs/vlm-agent.md#run-a-live-task) shows how to enter your own key with terminal echo disabled and select your HTTPS endpoint and vision model. For a persistent local setup the client also accepts `base_url`, `model` and `api_key` in the Git-ignored `config/private/qwen.json` file.

Use a separate shell for each service. From the project root, run `source setup/environment.sh`, then `cd repo/ManiSkill/examples/baselines/act` in each shell.

```bash
# Perception service, port 18830
./run_perception_mask.sh

# π0.5 inference, port 18833, when using that policy
"$OPENPI_PYTHON" pi05_inference_worker.py --port 18833 \
  --checkpoint "${MASK_PI05_CHECKPOINT:-$MASK_POLICY_ROOT/ckpt/maniskill/pi05_25x50/49999}"

# Task console, port 18840
./run_task_agent_web.sh
```

For an ACT task, choose a scene, generate it, inspect its cameras, select a policy and enter a task instruction. The scene reference records the original camera images so different policies can be compared on the same initial layout. The RoboCasa page routes its controls to the separate [RoboCasa bridge](../robocasa_bridge/).

## Run the VLM task agent

The [task runner](../overlays/ManiSkill/examples/baselines/act/task_agent/runner.py) connects the VLM's visual decisions to the low-level skill. The [prompts and image assembly](../overlays/ManiSkill/examples/baselines/act/task_agent/vlm.py) cover planning, grounding one target at a time, checking its initial mask and judging task progress. The learned policy grasps and lifts the object. A motion planner then carries it to the collection area, releases it and retreats. The [supervisor](../overlays/ManiSkill/examples/baselines/act/task_agent/supervisor.py) samples visual feedback at a nominal 1 Hz and filters stale responses; the runner uses that evidence to continue, retry or stop.

With the services and inputs above ready, you can also run a task directly from the ACT directory. Choose a fresh output path for each run. The policy IDs are `act` (two RGB streams), `act_triple` (three RGB streams) and `pi05`. For example:

```bash
"$MASK_ACT_PYTHON" -m task_agent.runner \
  --instruction 'Move a visible object into the green collection area.' \
  --policy act --seed 17 --output "$MASK_POLICY_ROOT/outputs/task_agent/cli-demo-01"
```

The runner records VLM decisions and the final status locally. Its completion decision is a visual assessment. To exercise the decision loop **offline**, use Python 3.11+, NumPy and Pillow in the same ACT directory:

```bash
python -m unittest test_task_agent_grounding test_task_agent_vlm_observation \
  test_task_agent_supervisor test_task_agent_single_target_cycle
```

These tests supply synthetic images and scripted responses, so no provider account, API key, GPU or checkpoint is required.

## Training and checks

Run `prepare_triple_rgb_dataset.py --help` from the ACT directory using the simulation environment to see how to prepare raw demonstrations. The data include base and wrist RGB frames and masked base RGB. The three-stream ACT trainer accepts the prepared directory with `--data-path` and writes checkpoints below `--run-dir`. For the π0.5 recipe, see `repo/openpi/scripts/train_maniskill_pi05.sh` and the `pi05_maniskill_25x50` configuration. The supplied full training recipe expects three GPUs with at least 76,000 MiB free on each; the console and ACT have different requirements.

For CPU checks, source `setup/environment.sh` and run the following from the ACT directory with Python 3.11+, NumPy, Pillow and Node.js available.

```bash
python3 -m unittest test_deployment_paths test_task_agent_contracts \
  test_scene_objects test_mask_dropout test_task_agent_mask_dropout_api test_robocasa_web
```
