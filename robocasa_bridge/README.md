# RoboCasa kitchen bridge

Build a counter-top pick-and-place scene, record synchronized camera images and robot actions, and replay the result. The classic scene asks a PandaOmron to place a red and a blue cube in a green tray in the requested order. A separate five-object pilot trains and evaluates ACT and π0.5 on individual objects.

## Run the CPU checks

From the repository root, use Python 3.11 or newer. The tests create tiny synthetic fixtures in a disposable workspace and exercise the action, image, selection and replay contracts.

```bash
python -m venv .venv-bridge
source .venv-bridge/bin/activate
python -m pip install -e ./robocasa_bridge
MASK_POLICY_ROOT="$(mktemp -d)" python -m unittest discover -s robocasa_bridge/tests -v
```

## Build a simulation environment

The original simulator used Linux, NVIDIA EGL, Python 3.11 and MuJoCo 3.3.1. Use a separate Python environment for the GPU simulator. From the repository root, set the project workspace and clone the exact source revisions:

```bash
export MASK_POLICY_ROOT="$PWD"
export ROBOCASA_ROOT="$MASK_POLICY_ROOT/repo/robocasa"
export ROBOSUITE_ROOT="$MASK_POLICY_ROOT/repo/robosuite"
mkdir -p "$MASK_POLICY_ROOT/repo"

git clone https://github.com/robocasa/robocasa.git "$ROBOCASA_ROOT"
git -C "$ROBOCASA_ROOT" checkout 3443cdf3dd20f9ae8ed41adba442059b506891d0
git clone https://github.com/ARISE-Initiative/robosuite.git "$ROBOSUITE_ROOT"
git -C "$ROBOSUITE_ROOT" checkout 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
git -C "$ROBOSUITE_ROOT" apply "$MASK_POLICY_ROOT/robocasa_bridge/upstream/robosuite-egl.patch"

python3.11 -m venv .venv-robocasa
source .venv-robocasa/bin/activate
python -m pip install -e ./robocasa_bridge -e "$ROBOSUITE_ROOT" \
  "numpy==2.2.5" "numba==0.61.2" "scipy==1.15.3" "mujoco==3.3.1" \
  "opencv-python==4.11.0.86" pygame pyyaml h5py lxml hidapi gymnasium
python -m pip install --no-deps -e "$ROBOCASA_ROOT"
```

These commands install the collection and simulation dependencies used by the bridge. The optional upstream training and dataset wrappers have separate dependencies.

Install the kitchen layouts, fixtures, textures and your selected object collections with the [RoboCasa asset installer](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/scripts/download_kitchen_assets.py). Its default destination is the simulator's own asset tree. The catalog reads this tree as `ROBOCASA_ASSET_ROOT`. Installing the simulator package alone does not install the meshes. The reference object collections are in the [RoboCasa assets](https://huggingface.co/datasets/robocasa/robocasa-assets/tree/1b92c3d02ca4354984fec961357db0bff7b32166) and [NVIDIA kitchen MJCF](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF/tree/420a04af939c34873e6839a586b70844baf28aab) repositories. Each distribution has its own terms.

```bash
export ROBOCASA_ASSET_ROOT="$ROBOCASA_ROOT/robocasa/models/assets"
python -m robocasa.scripts.download_kitchen_assets --type all

# Aigen XML files refer to a legacy ceramic texture. The bridge verifies its SHA-256.
mkdir -p "$ROBOCASA_ASSET_ROOT/shared/textures"
cp "$ROBOSUITE_ROOT/robosuite/models/assets/textures/ceramic.png" "$ROBOCASA_ASSET_ROOT/shared/textures/"
```

Select a valid EGL device for the machine. The service binds to loopback and initializes the kitchen when it receives a scene request.

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
python -m robocasa_bridge.server --port 18850
```

From a second terminal, request the scene and poll its status. When initialization finishes, submit a scripted demonstration:

```bash
curl -fsS http://127.0.0.1:18850/health
curl -fsS -H 'Content-Type: application/json' -d '{"seed":17,"order":["red_cube","blue_cube"]}' http://127.0.0.1:18850/scene
curl -fsS http://127.0.0.1:18850/status
curl -fsS -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:18850/scripted_demo
```

## Recording and replay

Each transition retains the native 12D controller action and its 7D arm/gripper subspace at 20 Hz. A trajectory with `T` actions stores `T+1` observations. The policy receives two RGB views, the current target cutout and a 23D robot state. Privileged object poses, contacts and simulator states support expert control, replay and scoring. Training chunks stop at target boundaries.

Raw episodes are written to `$MASK_POLICY_ROOT/data/robocasa/episodes`, shared compiled MuJoCo models to `data/robocasa/models`, and videos to `outputs/robocasa/episodes`. State replay restores recorded simulator states. Action replay executes the stored actions and reports deviations from the original trajectory. The custom tray geometry is packaged with the Python module.

- [HTTP API, object selection and replay modes](API.md)
- [Five-object pilot, training and evaluation](robocasa_pilot/README.md)
- [Upstream component notices](upstream/README.md)
