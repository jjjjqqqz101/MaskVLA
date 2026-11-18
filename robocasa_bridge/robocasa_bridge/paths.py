import os
from pathlib import Path
import re

# Runtime files are separate from the installed Python modules.
ROOT = Path(os.environ.get("MASK_POLICY_ROOT", Path.cwd())).expanduser().resolve()
ROBOCASA_ROOT = Path(os.environ.get("ROBOCASA_ROOT", ROOT / "repo/robocasa")).expanduser().resolve()
ROBOSUITE_ROOT = Path(os.environ.get("ROBOSUITE_ROOT", ROOT / "repo/robosuite")).expanduser().resolve()
ASSETS = Path(os.environ.get("ROBOCASA_ASSET_ROOT", ROBOCASA_ROOT / "robocasa/models/assets")).expanduser().resolve()
DATA = ROOT / "data/robocasa"
OUTPUT = ROOT / "outputs/robocasa"
def inside(value):
    p = Path(value).resolve()
    if p != ROOT and ROOT not in p.parents:
        raise ValueError("Path must remain inside mask_policy")
    return p
def episode_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"rc_[A-Za-z0-9_-]{1,100}", value):
        raise ValueError("Invalid RoboCasa episode id")
    return value
def episode_dir(value, output=False):
    p = (OUTPUT if output else DATA) / "episodes" / episode_id(value)
    if p.is_symlink():
        raise ValueError("Episode symlinks are forbidden")
    return inside(p)
