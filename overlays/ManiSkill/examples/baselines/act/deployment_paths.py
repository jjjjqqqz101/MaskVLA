"""Deployment paths shared by the task console and local model services.

The H20 layout is ``MASK_POLICY_ROOT/{repo,data,ckpt,.venvs,outputs}``.
Overrides are read when a path is requested, before any model is loaded.  This
module deliberately has no simulator, tensor-library or credential imports.
"""
from pathlib import Path
import os


ACT_ROOT = Path(__file__).resolve().parent
MANISKILL_ROOT = ACT_ROOT.parents[2]
_REPOSITORY_PARENT = MANISKILL_ROOT.parent
DEFAULT_ROOT = (_REPOSITORY_PARENT.parent if _REPOSITORY_PARENT.name == 'repo'
                else _REPOSITORY_PARENT)
ACT_CHECKPOINT_RELATIVE = Path('ckpt/maniskill/act_dual_rgb_8x100/100000.pt')
TRIPLE_CHECKPOINT_RELATIVE = Path('ckpt/maniskill/act_triple_rgb_25x50/160000.pt')


def _resolved(value):
    return Path(value).expanduser().resolve()


def policy_root():
    root = _resolved(os.environ.get('MASK_POLICY_ROOT') or DEFAULT_ROOT)
    if not root.is_relative_to(DEFAULT_ROOT):
        raise ValueError(f'MASK_POLICY_ROOT must stay inside {DEFAULT_ROOT}')
    return root


def project_path(value, label='project path'):
    """Validate a project input/output before opening or creating it."""
    path = _resolved(value)
    root = policy_root()
    if not path.is_relative_to(root):
        raise ValueError(f'{label} must stay inside {root}')
    return path


def _path(variable, default):
    return project_path(os.environ.get(variable) or default, variable)


def act_checkpoint():
    return _path('MASK_ACT_CHECKPOINT', policy_root() / ACT_CHECKPOINT_RELATIVE)


def act_triple_checkpoint():
    return _path('MASK_ACT_TRIPLE_CHECKPOINT', policy_root() / TRIPLE_CHECKPOINT_RELATIVE)


def pi05_checkpoint():
    return _path('MASK_PI05_CHECKPOINT', policy_root() / 'ckpt/maniskill/pi05_25x50/49999')


def data_root():
    return _path('MASK_DATA_ROOT', policy_root() / 'data/maniskill/pi05_train_25x50')


def task_runs_root():
    return _path('MASK_TASK_RUNS_ROOT', policy_root() / 'outputs/task_agent')


def perception_runs_root():
    return _path('PERCEPTION_RUNS_ROOT', policy_root() / 'outputs/perception_tracking')


def sam2_checkpoint():
    return _path('SAM2_CHECKPOINT', policy_root() / 'ckpt/perception/sam2.1_hiera_small.pt')


def cutie_checkpoint():
    return _path('CUTIE_CHECKPOINT', policy_root() / 'ckpt/perception/cutie-base-mega.pth')


def sam2_repo():
    return _path('SAM2_REPO', policy_root() / 'repo/third_party/sam2')


def cutie_repo():
    return _path('CUTIE_REPO', policy_root() / 'repo/third_party/Cutie')


def torch_cache():
    return _path('TORCH_HOME', policy_root() / 'ckpt/maniskill/support/torch')


def openpi_cache():
    return _path('OPENPI_DATA_HOME', policy_root() / 'ckpt/maniskill/support/openpi')


def assets_root():
    # ManiSkill appends /data itself; this must be the parent of that directory.
    return _path('MS_ASSET_DIR', policy_root() / 'assets/maniskill')


def qwen_config():
    return _path('QWEN_CONFIG', policy_root() / 'config/private/qwen.json')
