#!/usr/bin/env python3
"""Fetch pinned upstream sources and apply Mask Policy's source changes."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "ManiSkill": ("haosulab/ManiSkill", "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3", "ManiSkill"),
    "openpi": ("Physical-Intelligence/openpi", "15a9616a00943ada6c20a0f158e3adb39df2ccac", "openpi"),
    "lerobot": ("huggingface/lerobot", "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5", "third_party/lerobot"),
    "sam2": ("facebookresearch/sam2", "2b90b9f5ceec907a1c18123530e92e794ad901a4", "third_party/sam2"),
    "Cutie": ("hkchengrex/Cutie", "ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a", "third_party/Cutie"),
}


def prepare(name):
    upstream, revision, subdir = SOURCES[name]
    target = ROOT / "repo" / subdir
    if target.exists() or target.is_symlink():
        raise SystemExit(f"{target} already exists. Select only missing components.")
    target.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
        source = Path(scratch) / name
        source.mkdir()
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")

        def git(*args):
            subprocess.run(["git", *args], cwd=source, env=env, check=True)

        git("init", "--quiet")
        git("remote", "add", "origin", f"https://github.com/{upstream}.git")
        git("fetch", "--depth", "1", "origin", revision)
        git("checkout", "--quiet", "--detach", "FETCH_HEAD")
        patch = ROOT / "setup" / "patches" / f"{name}.patch"
        if patch.is_file():
            git("apply", "--check", str(patch))
            git("apply", str(patch))
        overlay = ROOT / "overlays" / name
        if overlay.is_dir():
            shutil.copytree(overlay, source, dirs_exist_ok=True)
        source.rename(target)
    print(f"{name} ready at {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("components", nargs="*", metavar="COMPONENT",
                        help="Any of: " + ", ".join(SOURCES) + ". Defaults to all.")
    selection = parser.parse_args().components or list(SOURCES)
    unknown = set(selection) - SOURCES.keys()
    if unknown:
        parser.error("Unknown components: " + ", ".join(sorted(unknown)))
    for component in dict.fromkeys(selection):
        prepare(component)
