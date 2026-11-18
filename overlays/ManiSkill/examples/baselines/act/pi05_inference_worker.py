"""Serve one official JAX openpi checkpoint to the local ManiSkill web evaluator."""

import argparse
import base64
import ctypes
import json
import logging
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np



PROMPT = "Pick up and lift the object shown in the target image."
IMAGE_SHAPE = (224, 224, 3)


def decode_image(encoded):
    raw = base64.b64decode(encoded, validate=True)
    image = np.frombuffer(raw, dtype=np.uint8)
    if image.size != np.prod(IMAGE_SHAPE):
        raise ValueError(f"Expected {np.prod(IMAGE_SHAPE)} image bytes, got {image.size}")
    return image.reshape(IMAGE_SHAPE).copy()


def noise_from_seed(seed, shape):
    """Use public infer(noise=...) instead of mutating the policy RNG."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("noise_seed must be an unsigned 32-bit integer")
    return np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format_string, *args):
        if self.path != "/health":
            super().log_message(format_string, *args)

    def reply(self, data, status=200):
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self.reply({"ok": True, "checkpoint": self.server.checkpoint,
                               "supports_noise_seed": True, "noise_shape": list(self.server.noise_shape)})
        self.reply({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path != "/infer":
            return self.reply({"error": "Not found"}, 404)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size < 1_000_000:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(size))
            state = np.asarray(data["state"], dtype=np.float32)
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError("Expected finite eight-dimensional state")
            observation = {
                "observation/image": decode_image(data["base_rgb"]),
                "observation/wrist_image": decode_image(data["wrist_rgb"]),
                "observation/target_image": decode_image(data["target_rgb"]),
                "observation/state": state,
                "prompt": PROMPT,
            }
            noise_seed = data.get("noise_seed")
            noise = noise_from_seed(noise_seed, self.server.noise_shape) if "noise_seed" in data else None
            with self.server.policy_lock:
                started = time.monotonic()
                result = (self.server.policy.infer(observation, noise=noise) if noise is not None
                          else self.server.policy.infer(observation))
                elapsed = time.monotonic() - started
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
                raise RuntimeError(f"Invalid policy output: {actions.shape}")
            self.reply({"actions": actions.tolist(), "inference_seconds": elapsed, "noise_seed": noise_seed})
        except Exception as error:
            logging.exception("Inference failed")
            self.reply({"error": str(error)}, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18773)
    args = parser.parse_args()
    if not (args.checkpoint / "_CHECKPOINT_METADATA").is_file():
        parser.error(f"Checkpoint is missing or incomplete: {args.checkpoint}; weights were not copied with the code")
    try:
        from openpi.policies import policy_config
        from openpi.training import config as training_config
    except ImportError as error:
        parser.error(f"Optional openpi environment is unavailable: {error}. Run with the configured OPENPI_PYTHON environment.")
    # The web service owns this worker. Release GPU memory if that parent exits.
    parent_pid = os.getppid()
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0 or os.getppid() != parent_pid:
        raise RuntimeError("Could not attach inference worker to its parent process")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = training_config.get_config("pi05_maniskill_25x50")
    policy = policy_config.create_trained_policy(
        config, args.checkpoint,
        default_prompt=PROMPT,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.policy = policy
    server.noise_shape = (config.model.action_horizon, config.model.action_dim)
    server.policy_lock = threading.Lock()
    server.checkpoint = str(args.checkpoint)
    logging.info("READY checkpoint=%s port=%s", args.checkpoint, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
