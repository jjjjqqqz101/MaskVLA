#!/usr/bin/env python3
"""Read-only visual inspection of the 25x50 pi0.5 ManiSkill collection."""

import argparse
import io
import json
import os
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import h5py
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("TARGET_PICK_FORTY_DATA", ROOT / "demos/TargetPickForty-v1/pi05_train_25x50")).expanduser()
LEROBOT = Path(os.environ.get("TARGET_PICK_FORTY_LEROBOT", ROOT / "lerobot/maniskill/target_pick_forty_pi05_25x50")).expanduser()
ROWS = []


def initialize():
    for path in (DATA / "progress.json", LEROBOT / "episode_manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing optional forty-object dataset: {path}. Set TARGET_PICK_FORTY_DATA / TARGET_PICK_FORTY_LEROBOT to existing data; datasets were not copied with the code.")
    progress = json.loads((DATA / "progress.json").read_text())
    converted = json.loads((LEROBOT / "episode_manifest.json").read_text())
    if progress["total"] != 1250 or len(converted) != 1250:
        raise RuntimeError("Collection or LeRobot conversion is incomplete")
    for i, row in enumerate(converted):
        if row["episode_index"] != i:
            raise RuntimeError("Episode manifest index mismatch")
        ROWS.append({key: row[key] for key in ("episode_index", "target_name", "selected_names", "seed", "steps", "prompt")})


def source_shard_path(recorded):
    """Relocate a read-only A100 manifest entry without changing its provenance."""
    source = Path(recorded).expanduser()
    if source.exists():
        return source
    # Collection manifests store absolute source_shard paths. Keep their original
    # values intact and resolve the path beneath the configured local raw root.
    marker = ("TargetPickForty-v1", "pi05_train_25x50")
    for index in range(len(source.parts) - 1):
        if source.parts[index:index + 2] == marker:
            relocated = DATA.joinpath(*source.parts[index + 2:])
            if relocated.exists():
                return relocated
    raise FileNotFoundError(f"Raw source shard is unavailable: {recorded}. Set TARGET_PICK_FORTY_DATA to its local collection root.")


@lru_cache(maxsize=2)
def load(episode):
    row = json.loads((LEROBOT / "episode_manifest.json").read_text())[episode]
    with h5py.File(source_shard_path(row["source_shard"]) / "trajectory.h5", "r") as file:
        group = file[row["trajectory"]]
        rgb = group["obs/sensor_data/base_camera/rgb"][:]
        wrist = group["obs/sensor_data/hand_camera/rgb"][:]
        mask = group["derived/base_camera/target_mask"][:]
        masked = group["derived/base_camera/masked_rgb"][:]
        seg = group["obs/sensor_data/base_camera/segmentation"][:]
        ids = group["alignment/object_seg_ids"][:]
        target_id = group["alignment/target_seg_id"][:]
        target_idx = int(group.attrs["target_idx"])
        success = group["success"][:]
        actions = len(group["actions"])
    if not (len(rgb) == len(wrist) == len(mask) == len(masked) == actions + 1):
        raise RuntimeError("Frame and action counts differ")
    if not np.array_equal(mask, seg == target_id.reshape(-1, 1, 1, 1)):
        raise RuntimeError("Target Mask does not match segmentation")
    if not np.array_equal(masked, rgb * mask):
        raise RuntimeError("Masked RGB does not match RGB × Mask")
    if not np.all(ids[:, target_idx] == target_id.reshape(-1)):
        raise RuntimeError("Target identity changed during the trajectory")
    return rgb, wrist, mask, masked, seg, ids, target_idx, success


def render(episode, t):
    rgb, wrist, mask, masked, seg, ids, target_idx, success = load(episode)
    if not 0 <= t < len(rgb):
        raise ValueError("Frame out of range")
    names = ROWS[episode]["selected_names"]
    base = Image.fromarray(rgb[t])
    annotated = base.resize((448, 448), Image.Resampling.NEAREST)
    pen = ImageDraw.Draw(annotated)
    for i, object_id in enumerate(ids[t]):
        y, x = np.where(seg[t, :, :, 0] == object_id)
        if len(x) == 0:
            continue
        color = "#39f098" if i == target_idx else "#ffffff"
        box = (int(x.min() * 2), int(y.min() * 2), int(x.max() * 2 + 1), int(y.max() * 2 + 1))
        pen.rectangle(box, outline=color, width=2)
        label = names[i] + (" TARGET" if i == target_idx else "")
        lx, ly = max(0, min(box[0], 448 - 6 * len(label) - 8)), max(0, box[1] - 15)
        pen.rectangle((lx, ly, lx + 6 * len(label) + 5, ly + 14), fill="black")
        pen.text((lx + 2, ly + 1), label, fill=color)
    panels = [annotated, base, Image.fromarray(wrist[t]),
              Image.fromarray(np.repeat(mask[t].astype(np.uint8) * 255, 3, axis=2)),
              Image.fromarray(masked[t])]
    canvas = Image.new("RGB", (2240, 520), "#101926")
    draw = ImageDraw.Draw(canvas)
    titles = ["Target labels", "Base RGB", "Wrist RGB", "Binary target mask", "RGB x Mask"]
    for i, (panel, title) in enumerate(zip(panels, titles)):
        canvas.paste(panel.resize((448, 448), Image.Resampling.NEAREST), (i * 448, 32))
        draw.text((i * 448 + 12, 10), title, fill="white")
    state = "initial frame" if t == 0 else f"success after action={bool(success[t - 1])}"
    draw.text((12, 492), f"episode {episode}  target={ROWS[episode]['target_name']}  "
              f"frame={t}/{len(rgb) - 1}  seed={ROWS[episode]['seed']}  {state}", fill="white")
    return canvas


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, content, mime="application/json", status=200):
        if not isinstance(content, bytes):
            content = json.dumps(content).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        try:
            url = urlparse(self.path)
            if url.path == "/":
                return self.send(Path(__file__).with_suffix(".html").read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/index":
                return self.send(ROWS)
            if url.path == "/api/frame":
                query = parse_qs(url.query)
                episode = int(query.get("episode", ["0"])[0])
                t = int(query.get("t", ["0"])[0])
                if not 0 <= episode < len(ROWS):
                    raise ValueError("Episode out of range")
                buffer = io.BytesIO()
                render(episode, t).save(buffer, format="PNG")
                return self.send(buffer.getvalue(), "image/png")
            return self.send({"error": "Not found"}, status=404)
        except (KeyError, ValueError, IndexError) as exc:
            return self.send({"error": str(exc)}, status=400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            return self.send({"error": str(exc)}, status=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18771)
    arguments = parser.parse_args()
    initialize()
    print(f"Viewer ready: {len(ROWS)} episodes on port {arguments.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", arguments.port), Handler).serve_forever()
