"""Three-camera observation adapter for the ManiSkill target-pick dataset."""

import dataclasses

import einops
import numpy as np

from openpi import transforms


def _rgb(value):
    image = np.asarray(value)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 224x224x3 RGB, got {image.shape} {image.dtype}")
    return image


@dataclasses.dataclass(frozen=True)
class ManiSkillInputs(transforms.DataTransformFn):
    """Map base, wrist and target-only RGB to openpi's three image slots."""

    def __call__(self, data: dict) -> dict:
        base = _rgb(data["observation/image"])
        wrist = _rgb(data["observation/wrist_image"])
        target = _rgb(data["observation/target_image"])
        result = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": target,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32).copy()
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class ManiSkillOutputs(transforms.DataTransformFn):
    """Return eight robot action dimensions after normalization is reversed."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :8]}
