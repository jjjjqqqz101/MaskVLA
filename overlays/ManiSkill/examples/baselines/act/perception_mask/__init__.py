"""RGB-only VLM → SAM2 → Cutie perception for existing robot policies."""

from .client import MaskResult, PerceptionMaskClient

__all__ = ['MaskResult', 'PerceptionMaskClient']
