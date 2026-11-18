#!/usr/bin/env python3
"""Read-only H5 trajectory viewer. No torch, policy, or simulator required."""
import argparse
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get('TARGET_PICK_EIGHT_DATA', ROOT / 'demos/TargetPickEight-v1/train_800')).expanduser()
OUT = ROOT / 'outputs/act_dataset_viewer_eight'
LABELS = ['red_cube_1', 'red_cube_2', 'blue_cube', 'cracker_box', 'mug', 'banana', 'softball', 'baseball']
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=1)
JOBS = {}
INDEX = {}


def initialize():
    for name in ('train.json', 'train.h5'):
        if not (DATA / name).is_file():
            raise FileNotFoundError(f'Missing optional eight-object dataset: {DATA / name}. Set TARGET_PICK_EIGHT_DATA to an existing collection; datasets were not copied with the code.')
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ('train',):
        meta = json.loads((DATA / f'{split}.json').read_text())
        rows = []
        with h5py.File(DATA / f'{split}.h5', 'r') as f:
            assert len(meta['episodes']) == len(f)
            for ep in meta['episodes']:
                g = f[f"traj_{ep['episode_id']}"]
                assert int(g.attrs['target_idx']) == ep['target_idx']
                rows.append(dict(ep, frames=len(g['obs/sensor_data/base_camera/rgb']),
                                 actions=len(g['actions']), split=split))
        INDEX[split] = rows
    (OUT / 'manifest.json').write_text(json.dumps(INDEX, indent=2))


@lru_cache(maxsize=2)
def load(split, episode):
    if split not in INDEX or not 0 <= episode < len(INDEX[split]):
        raise ValueError('Unknown split or trajectory')
    paths = dict(rgb='obs/sensor_data/base_camera/rgb',
                 masked='derived/base_camera/masked_rgb',
                 mask='derived/base_camera/target_mask',
                 seg='obs/sensor_data/base_camera/segmentation',
                 ids='alignment/object_seg_ids', tid='alignment/target_seg_id',
                 target='env_states/target_idx', success='success')
    with h5py.File(DATA / f'{split}.h5', 'r') as f:
        g = f[f'traj_{episode}']
        d = {k: g[p][:] for k, p in paths.items()}
    expected = d['seg'] == d['tid'].reshape(-1, 1, 1, 1)
    if not np.array_equal(expected, d['mask']):
        raise ValueError('Stored target mask disagrees with target segmentation ID')
    if not np.array_equal(d['masked'], np.where(d['mask'], d['rgb'], 0)):
        raise ValueError('Stored masked RGB disagrees with original RGB and mask')
    if not np.array_equal(d['tid'].reshape(-1), d['ids'][np.arange(len(d['rgb'])), d['target'].reshape(-1)]):
        raise ValueError('Target segmentation ID disagrees with object identity')
    return d


def frame(d, split, episode, t):
    if not 0 <= t < len(d['rgb']):
        raise ValueError('Frame out of bounds')
    target = int(d['target'][t, 0])
    # Only this first panel is annotated. The other three panels retain H5 pixels.
    ann = Image.fromarray(d['rgb'][t]).resize((448, 448), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(ann)
    for k, sid in enumerate(d['ids'][t]):
        y, x = np.where(d['seg'][t, ..., 0] == sid)
        if len(x):
            color = '#54ff99' if k == target else '#ffffff'
            box = (int(x.min()*2), int(y.min()*2), int(x.max()*2+1), int(y.max()*2+1))
            draw.rectangle(box, outline=color, width=2)
            label = LABELS[k] + (' TARGET' if k == target else '')
            xx, yy = max(0, min(box[0], 448-len(label)*6-6)), max(0, box[1]-15)
            draw.rectangle((xx, yy, xx+len(label)*6+4, yy+14), fill='black')
            draw.text((xx+2, yy+1), label, fill=color)
    canvas = Image.new('RGB', (1792, 520), '#101926')
    panels = [ann, Image.fromarray(d['rgb'][t]),
              Image.fromarray(np.repeat(d['mask'][t].astype(np.uint8)*255, 3, axis=-1)),
              Image.fromarray(d['masked'][t])]
    pen = ImageDraw.Draw(canvas)
    for i, (im, title) in enumerate(zip(panels, ['Object labels (visible pixels)', 'Original RGB', 'Binary target mask', 'Masked RGB'])):
        canvas.paste(im.resize((448, 448), Image.Resampling.NEAREST), (i*448, 32))
        pen.text((i*448+12, 10), title, fill='white')
    if t:
        status = f'task_success_after_action={bool(d["success"][t-1])}'
    else:
        status = 'initial observation; no action executed yet'
    pen.text((12, 493), f'{split} / traj_{episode} | target={LABELS[target]} (target_idx={target}) | frame {t}/{len(d["rgb"])-1} | {status}', fill='white')
    return canvas


def export(split, episode):
    import imageio.v2 as imageio
    with LOCK:
        d = load(split, episode)
    dest = OUT / split / f'traj_{episode}'
    dest.mkdir(parents=True, exist_ok=True)
    temporary = dest / 'video.partial.mp4'
    try:
        with imageio.get_writer(temporary, fps=20, codec='libx264', quality=8,
                               macro_block_size=1, ffmpeg_params=['-threads', '1', '-movflags', '+faststart']) as writer:
            for t in range(len(d['rgb'])):
                writer.append_data(np.asarray(frame(d, split, episode, t)))
        for key, name in [('rgb','original'), ('masked','masked_rgb')]:
            Image.fromarray(d[key][0]).save(dest / f'{name}.png')
        Image.fromarray(d['mask'][0,...,0].astype(np.uint8)*255).save(dest / 'mask.png')
        frame(d, split, episode, 0).save(dest / 'overview.png')
        (dest / 'episode.json').write_text(json.dumps(INDEX[split][episode], indent=2))
        temporary.replace(dest / 'video.mp4')
    finally:
        temporary.unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, kind='application/json', status=200):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            u = urlparse(self.path)
            if u.path == '/':
                return self.send(Path(__file__).with_suffix('.html').read_bytes(), 'text/html; charset=utf-8')
            if u.path == '/api/index':
                return self.send(INDEX)
            if u.path.startswith('/files/'):
                path = (OUT / u.path[len('/files/'):]).resolve()
                if not path.is_relative_to(OUT.resolve()) or not path.is_file():
                    return self.send({'error':'Not found'}, status=404)
                kind = 'video/mp4' if path.suffix == '.mp4' else 'image/png' if path.suffix == '.png' else 'application/json'
                if path.suffix == '.mp4':
                    size = path.stat().st_size
                    start, end = 0, size-1
                    byte_range = self.headers.get('Range')
                    if byte_range:
                        import re
                        match = re.fullmatch(r'bytes=(\d+)-(\d*)', byte_range)
                        if not match:
                            return self.send({'error':'Invalid range'}, status=416)
                        start = int(match[1])
                        end = min(int(match[2]) if match[2] else end, end)
                        if start > end:
                            return self.send({'error':'Invalid range'}, status=416)
                    self.send_response(206 if byte_range else 200)
                    self.send_header('Content-Type', kind)
                    self.send_header('Accept-Ranges', 'bytes')
                    self.send_header('Content-Length', str(end-start+1))
                    if byte_range:
                        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                    self.end_headers()
                    with path.open('rb') as stream:
                        stream.seek(start)
                        left = end-start+1
                        while left:
                            block = stream.read(min(left, 1024*1024))
                            self.wfile.write(block)
                            left -= len(block)
                    return
                return self.send(path.read_bytes(), kind)
            q = parse_qs(u.query)
            split, episode = q.get('split',['train'])[0], int(q.get('episode',['0'])[0])
            if split not in INDEX or not 0 <= episode < len(INDEX[split]):
                raise ValueError('Invalid trajectory')
            if u.path == '/api/frame':
                with LOCK:
                    d = load(split, episode)
                buf = io.BytesIO()
                frame(d, split, episode, int(q.get('t',['0'])[0])).save(buf, format='PNG')
                return self.send(buf.getvalue(), 'image/png')
            if u.path == '/api/export':
                key = (split, episode)
                path = OUT / split / f'traj_{episode}' / 'video.mp4'
                with LOCK:
                    if path.exists():
                        return self.send({'status':'ready', 'url':f'/files/{split}/traj_{episode}/video.mp4', 'directory':str(path.parent)})
                    job = JOBS.get(key)
                    if job is None:
                        JOBS[key] = POOL.submit(export, split, episode)
                    elif job.done() and job.exception():
                        return self.send({'status':'error','error':str(job.exception())})
                return self.send({'status':'working'})
            return self.send({'error':'Not found'}, status=404)
        except (ValueError, KeyError) as e:
            self.send({'error':str(e)}, status=400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            self.send({'error':str(e)}, status=500)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=18770)
    args = parser.parse_args()
    initialize()
    print(f'Dataset viewer: http://127.0.0.1:{args.port}; train={len(INDEX["train"])}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
