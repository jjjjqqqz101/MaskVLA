"""CPU integration check: ACT checkpoint restart equals an uninterrupted run."""
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from test_triple_rgb_dataset import fixture
from train_triple_rgb import main, ResumableBatchSampler


def assert_identical(left, right, key="root"):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), key
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right), key
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), key
        for child in left:
            assert_identical(left[child], right[child], f"{key}.{child}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), key
        for i, (a, b) in enumerate(zip(left, right)):
            assert_identical(a, b, f"{key}.{i}")
    else:
        assert left == right, f"{key}: {left!r} != {right!r}"


def run_test():
    assert not torch.cuda.is_available(), "Run with CUDA_VISIBLE_DEVICES='' to keep this check CPU-only"
    stream = list(ResumableBatchSampler(29, 4, 33, seed=7))
    for start in (0, 1, 6, 7, 8, 32, 33):
        assert list(ResumableBatchSampler(29, 4, 33, start_batch=start, seed=7)) == stream[start:]
    with tempfile.TemporaryDirectory(prefix="act-triple-train-test-") as name:
        root = Path(name)
        raw, converted = fixture(root), root / "converted"
        subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_triple_rgb_dataset.py")),
            "--raw-root", str(raw), "--output", str(converted), "--workers", "1", "--block-frames", "2",
            "--expected-targets", "1", "--expected-per-target", "2", "--expected-shards", "1", "--expected-frames", "8"], check=True)
        common = ["--data-path", str(converted), "--device", "cpu", "--amp", "off", "--deterministic",
            "--total-steps", "2", "--lr-drop", "1", "--batch-size", "2", "--num-workers", "0", "--torch-threads", "2",
            "--expected-demos", "2", "--num-queries", "2", "--hidden-dim", "32", "--dim-feedforward", "64",
            "--nheads", "4", "--enc-layers", "1", "--dec-layers", "1", "--save-every", "1",
            "--recovery-every", "0", "--diagnostic-every", "1", "--diagnostic-batch-size", "2", "--log-every", "1"]
        assert main(common + ["--run-dir", str(root / "full")]) == 0
        assert main(common + ["--run-dir", str(root / "resumed"), "--resume", str(root / "full/checkpoints/1.pt")]) == 0
        full = torch.load(root / "full/checkpoints/2.pt", map_location="cpu", weights_only=False)
        resumed = torch.load(root / "resumed/checkpoints/2.pt", map_location="cpu", weights_only=False)
        for key in ("optimizer_step", "agent", "ema_agent", "ema", "optimizer", "scheduler", "scaler", "rng",
                    "norm_stats", "architecture", "stream_order", "dataloader", "dataloader_generator_state",
                    "dataloader_worker_generator_start_state"):
            assert_identical(full[key], resumed[key], key)
        assert full["optimizer_step"] == full["ema"]["optimization_step"] == 2
    print("CPU_TRAIN_RESUME_EXACT_OK: model, EMA/shadow/step, AdamW, scheduler, scaler, RNG, sampler, loader RNG")


if __name__ == "__main__":
    run_test()
