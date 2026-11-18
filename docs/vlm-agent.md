# Run the ManiSkill VLM agent

Give the agent a request such as “Put the balls in the green collection area.” It examines the simulated tabletop, chooses one visible target at a time, and checks the outcome before moving on. In the ManiSkill console, you choose which policy performs each grasp: dual-stream ACT (`act`), triple-stream ACT (`act_triple`), or π0.5 (`pi05`).

## What makes the decisions?

| Stage | What happens |
|---|---|
| Plan | The VLM reads the instruction and the overview, base and wrist RGB views. It proposes up to four pick-and-place subtasks, or requests clarification if the available skill cannot accomplish the instruction. |
| Locate one target | The VLM draws a coarse box on the base-camera image, then refines it on an enlarged crop. SAM 2 segments the target and the VLM checks that the segmented pixels show the requested object. Cutie tracks the mask as the arm moves. |
| Grasp and place | The chosen learned policy commands the grasp and lift. Once the visual supervisor confirms a lift, the MPLib motion planner takes the arm to the collection area. The backend opens the gripper and retreats. |
| Verify or recover | A visual observer inspects new frames on a nominal one-request-per-second schedule. Two fresh, agreeing votes from distinct frames confirm a lift or placement. A failed attempt can trigger recovery and another localization of the same target. Whole-task completion requires two visual checks of the scene. |

The task runner records its success criterion as `VLM_visual_only`. Occlusion or a mistaken visual judgement can affect the result. The [RoboCasa pilot](evaluation.md) uses a separate protocol: its policies receive simulator target masks and its scoring uses the target's collision geometry.

The [VLM prompts and image assembly](../overlays/ManiSkill/examples/baselines/act/task_agent/vlm.py), [task runner](../overlays/ManiSkill/examples/baselines/act/task_agent/runner.py), [visual observer](../overlays/ManiSkill/examples/baselines/act/task_agent/supervisor.py), and [skill backend](../overlays/ManiSkill/examples/baselines/act/task_agent/backend.py) implement this loop. The repository also includes the [ordered-cube agent](../overlays/ManiSkill/examples/baselines/act/agentic_act/__main__.py), an earlier experiment with labeled cube and zone IDs, ACT grasping, and a separate physical check of the final state.

## Try it without an API key

From the repository root, run the grounding, observation and supervision tests with Python 3.11+. The tests use synthetic camera frames and scripted model responses. They exercise crop coordinates, the images sent for a decision, and rejection of stale visual votes.

```bash
python3.11 -m venv .venv-agent
source .venv-agent/bin/activate
python -m pip install numpy pillow
(cd overlays/ManiSkill/examples/baselines/act && \
  python -m unittest test_task_agent_grounding test_task_agent_vlm_observation test_task_agent_supervisor)
```

To preview the local interface, follow [Preview the interface](../setup/README.md#preview-the-interface). Opening the console does not call a model.

## Run a live task

Prepare the [Linux simulator, perception service, assets and policy checkpoint](../setup/README.md#linux-environments). Start the perception service, and also the π0.5 worker if using `pi05`. In a separate Bash terminal on the simulation host, run the following from the repository root. Enter your own HTTPS API base and vision model ID when prompted. The client posts camera views, the task instruction and relevant task history to that endpoint's `/chat/completions` route, using Qwen-compatible image input and JSON responses.

```bash
source setup/environment.sh
read -r -p 'Vision API base URL: ' QWEN_BASE_URL
read -r -p 'Vision model ID: ' QWEN_MODEL
read -r -s -p 'Vision API key: ' QWEN_API_KEY
printf '\n'
export QWEN_BASE_URL QWEN_MODEL QWEN_API_KEY
cd repo/ManiSkill/examples/baselines/act
./run_task_agent_web.sh
```

Open [the task console](http://127.0.0.1:18840/) on the host. Generate a scene, inspect the cameras, choose a policy and enter an instruction that matches visible objects. The console starts the task in that scene. `plan.json`, `vlm_calls.jsonl`, `events.jsonl` and `summary.json` under the local run directory show the plan, model judgements, executed steps and outcome. The API key is entered with terminal echo disabled and supplied to the [provider client](../overlays/ManiSkill/examples/baselines/act/agentic_act/qwen.py) through the process environment. After stopping the console, run `unset QWEN_API_KEY` in that terminal.
