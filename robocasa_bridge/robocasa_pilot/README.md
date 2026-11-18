# Five-object pick-and-place pilot

A fixed kitchen contains a single apple, banana, can, bar of soap or mug. Each rollout perturbs the object's initial horizontal position by up to 3.5 cm per axis and yaw by up to 0.65 radians. Success requires bilateral contact during a lift of more than 6 cm, the collision geometry inside the tray, an open gripper, end-effector clearance above 7 cm and at least ten settled control steps. The simulated target mask comes from the renderer.

## Collect trajectories

First complete the [simulator and asset setup](../README.md). Run in the simulator environment with `MASK_POLICY_ROOT` set. The collector checks five concrete mesh identities against the source XML hashes in `selection.py`.

```bash
python -m robocasa_pilot.collect --train 10 --validation 2 --test 2
```

The quotas are per object. The collector writes raw HDF5 episodes and a manifest under `data/robocasa/pilot_v1/`. Each row contains a 23D state, a 7D arm/gripper action, a base camera image, a wrist camera image and a same-frame target mask. Splits operate on complete episodes. Freeze a copy of the finished manifest before training.

```bash
cp "$MASK_POLICY_ROOT/data/robocasa/pilot_v1/manifest.json" "$MASK_POLICY_ROOT/data/robocasa/pilot_v1/manifest-frozen.json"
export MANIFEST="$MASK_POLICY_ROOT/data/robocasa/pilot_v1/manifest-frozen.json"
```

## Train policies

Install this bridge into the corresponding ACT or OpenPI environment as well. The ACT trainer imports the project's patched ManiSkill ACT baseline from `$MASK_POLICY_ROOT/repo/ManiSkill`. The π0.5 trainer uses the project's patched OpenPI source at `$MASK_POLICY_ROOT/repo/openpi`, including the `pi05_robocasa_pilot_v1` configuration. Override the two locations with `MASK_MANISKILL_ROOT` and `MASK_OPENPI_ROOT` if needed.

```bash
# In the ACT environment, initialize without pretrained weights:
python -m robocasa_pilot.act_train --manifest "$MANIFEST" --steps 3000 --no-pretrained

# In the OpenPI environment, supply a compatible initialization checkpoint:
python -m robocasa_pilot.pi05_train --manifest "$MANIFEST" \
  --init-checkpoint "$MASK_POLICY_ROOT/ckpt/initializer" \
  --fsdp-devices 1 --batch-size 1 --steps 200
```

The original ACT experiment used ImageNet ResNet-18 initialization. To use that mode, supply the torchvision `resnet18-f37072fd.pth` file at `ckpt/maniskill/support/torch/hub/checkpoints/` and omit `--no-pretrained`. The original π0.5 experiment warm-started from a prior ManiSkill checkpoint. Both trajectories and initialization weights are supplied by the user. The ACT objective masks padding in the encoder and reconstruction loss. π0.5 samples full ten-action windows and computes the flow loss on the seven active controls.

## Evaluate

Run a local inference worker in the matching policy environment. Set `ACT_CKPT` to a trained ACT checkpoint file or `PI05_CKPT` to a trained π0.5 step directory under the workspace:

```bash
python -m robocasa_pilot.act_infer --checkpoint "$ACT_CKPT" --serve --port 18861
# Alternatively, serve the OpenPI policy in its environment:
python -m robocasa_pilot.pi05_infer --checkpoint "$PI05_CKPT" --serve --port 18862
```

From the simulation environment, evaluate the selected policy. The worker predicts action chunks. The evaluator executes 30 steps per ACT query or 10 per π0.5 query in these examples.

```bash
python -m robocasa_pilot.evaluate --policy act --trials 2 --execute 30 --seed-base 950000 --tag heldout_act
python -m robocasa_pilot.evaluate --policy pi05 --trials 2 --execute 10 --seed-base 950000 --tag heldout_pi05
```

The recorded experiment used 50 training demonstrations, 10 validation demonstrations and 10 test demonstrations. A separate closed-loop evaluation used ten held-out initial scenes with two trials per object. ACT and π0.5 each completed 7 of those 10 trials. These measurements used a fixed kitchen, known object meshes, a single target per scene and simulator-provided masks.
