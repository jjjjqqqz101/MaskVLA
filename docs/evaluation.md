# Five-object kitchen pilot

A single robot arm picks one object from the counter, places it in the green tray, opens its gripper and retreats. The kitchen and tray are fixed. Target placement varies by up to ±3.5 cm in the table plane and ±0.65 radians in yaw. There are no distractors in this experiment.

## Inputs and training

Both policies receive base RGB, wrist RGB, base RGB multiplied by the current target mask, and a 23-dimensional proprioceptive state. The mask is ground truth from the simulator. Actions have seven dimensions: three translation deltas, three rotation deltas and a gripper command, executed at 20 Hz.

The demonstration collection comprised 70 expert trajectories (28,142 action steps). Whole episodes were split into 50 training, 10 validation and 10 test demonstrations. The closed-loop evaluation below uses a separate set of ten preselected scene seeds, two per object. No expert or planner intervened during these evaluations.

| Policy | Checkpoint selection | Actions before a new observation |
|---|---|---:|
| ACT | Trained for 10,000 updates. Validation selected the EMA weights at update 9,250. | 30 (1.5 s) |
| π0.5 | Initialized from a ManiSkill policy, then trained for 2,000 updates using the seven action dimensions. Used the native openpi decoder. | 10 (0.5 s) |

## What counts as success?

The gripper must lift the target more than 6 cm with two-finger contact. The target's collision geometry must fit within the tray. The gripper must open and retreat more than 7 cm from the target, with object speed below 0.035 m/s for at least ten consecutive control steps. Rechecking the recorded states and contacts reproduced the online scores for both policies.

| Object | ACT | π0.5 |
|---|---:|---:|
| Apple | 1 / 2 | 1 / 2 |
| Banana | 2 / 2 | 2 / 2 |
| Can | 2 / 2 | 2 / 2 |
| Bar of soap | 1 / 2 | 1 / 2 |
| Mug | 1 / 2 | 1 / 2 |
| **All five** | **7 / 10** | **7 / 10** |

ACT lost its grip on the apple and mug during transport in two of its failures. In another, the side of a finger moved the soap without a valid grasp. π0.5 failed to lift one apple securely, dropped one bar of soap and moved one mug without establishing the required grasp. Each object was tested in only two scenes. Generalization to new assets, scenes with distractors or masks estimated from RGB remains unmeasured here.

## Videos

The README preview plays at 2× speed. Each linked MP4 retains the full-speed, four-view recording of one held-out rollout.

| Seed | Policy | Recording |
|---:|---|---|
| 951000 | ACT | [Banana](media/act-banana.mp4) |
| 952000 | π0.5 | [Can](media/pi05-can.mp4) |
| 954001 | π0.5 | [Mug](media/pi05-mug.mp4) |

See the [pilot instructions](../robocasa_bridge/robocasa_pilot/README.md) to collect trajectories, train and evaluate with your own assets and checkpoints.
