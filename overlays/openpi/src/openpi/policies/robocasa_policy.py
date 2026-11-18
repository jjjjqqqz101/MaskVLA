"""RoboCasa PandaOmron OSC-delta contract; independent of ManiSkill adapters."""
import dataclasses
import numpy as np
from openpi import transforms

PROMPT = "Pick up the object shown in the target image and place it into the green tray."

def rgb(value):
    value = np.asarray(value)
    if value.shape != (224, 224, 3) or value.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB (224,224,3), got {value.shape} {value.dtype}")
    return value

@dataclasses.dataclass(frozen=True)
class RoboCasaInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], np.float32)
        if state.shape != (23,) or not np.isfinite(state).all():
            raise ValueError("RoboCasa robot_state must be finite and exactly 23D")
        result = {"state": state,
                  "image": {"base_0_rgb": rgb(data["observation/image"]),
                            "left_wrist_0_rgb": rgb(data["observation/wrist_image"]),
                            "right_wrist_0_rgb": rgb(data["observation/target_image"])},
                  "image_mask": {key: np.True_ for key in
                                 ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
                  "prompt": PROMPT}
        if "actions" in data:
            actions = np.asarray(data["actions"], np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 7 or not np.isfinite(actions).all():
                raise ValueError("RoboCasa actions must be finite Hx7 OSC deltas")
            result["actions"] = actions.copy()
        return result

@dataclasses.dataclass(frozen=True)
class ClipNormalizedState(transforms.DataTransformFn):
    """Bound train-stat normalized state before pi0.5 discrete tokenization."""
    def __call__(self, data: dict) -> dict:
        data["state"] = np.clip(data["state"], -1.0, 1.0)
        return data

@dataclasses.dataclass(frozen=True)
class RoboCasaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Native OSC controls already are delta commands. Never AbsoluteActions.
        return {"actions": np.asarray(data["actions"], np.float32)[..., :7]}
