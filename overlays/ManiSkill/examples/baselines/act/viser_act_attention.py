#!/usr/bin/env python3
"""Interactive Viser viewer for recorded ACT inputs and attention weights."""

import argparse
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import re
import threading
import time
import traceback

from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
import viser

from act.attention_maps import compute_attention_maps


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COLORS = colormaps['inferno']
PANEL_WIDTH, PANEL_HEIGHT = 480, 600
POSITIONS = ((-0.54, 0.68, 0), (0.54, 0.68, 0), (-0.54, -0.68, 0), (0.54, -0.68, 0))
HEAT_MAP_KEYS = ('action_original', 'action_masked', 'mask_to_original', 'decoder_original')
FOURTH_PANEL_MODES = ('Mask relation', 'Action -> masked RGB')


@dataclass
class Recording:
    folder: Path
    metadata: dict
    arrays: dict
    feature_shape: tuple
    label: str

    @classmethod
    def load(cls, folder):
        metadata = json.loads((folder / 'metadata.json').read_text())
        archive = folder / metadata.get('viser_frames_file', 'viser_frames.npz')
        if archive.is_file():
            with np.load(archive, allow_pickle=False) as source:
                arrays = {key: source[key] for key in source.files}
        else:
            # Old exports retained exact inputs at only three snapshot frames.
            snapshots = sorted(folder.glob('frame_*_attention.npz'))
            if not snapshots:
                raise ValueError(f'No raw attention frames in {folder}')
            rows = []
            for path in snapshots:
                with np.load(path, allow_pickle=False) as source:
                    rows.append(dict(step=int(path.name.split('_')[1]), rgb=source['rgb'][0],
                                     target_mask=source['target_mask'],
                                     encoder_attention=source['encoder_attention'][:, 0],
                                     decoder_attention=source['decoder_attention'][0]))
            arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0] if key != 'step'}
            arrays['steps'] = np.asarray([row['step'] for row in rows], dtype=np.int64)
        feature_shape = tuple(metadata.get('feature_shape', (7, 7)))
        steps, rgb, mask = arrays['steps'], arrays['rgb'], arrays['target_mask']
        if steps.ndim != 1 or not len(steps) or (np.diff(steps) <= 0).any():
            raise ValueError(f'Expected increasing frame numbers: {folder}')
        if rgb.ndim != 5 or rgb.shape[0] != len(steps) or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f'Invalid RGB archive: {folder}')
        if mask.shape != (len(steps), *rgb.shape[-2:]):
            raise ValueError(f'Mask and RGB shapes differ: {folder}')
        streams = len(metadata['stream_order'])
        if rgb.shape[1] != streams:
            raise ValueError(f'Input stream count differs from metadata: {folder}')
        tokens = 2 + streams * math.prod(feature_shape)
        encoder, decoder = arrays['encoder_attention'], arrays['decoder_attention']
        if encoder.ndim != 4 or encoder.shape[0] != len(steps) or encoder.shape[-2:] != (tokens, tokens):
            raise ValueError(f'Invalid encoder attention archive: {folder}')
        if decoder.ndim != 3 or decoder.shape[0] != len(steps) or decoder.shape[-1] != tokens:
            raise ValueError(f'Invalid decoder attention archive: {folder}')
        if 'heat_maxima' in arrays:
            maxima = arrays['heat_maxima']
            if maxima.shape != (len(steps), len(HEAT_MAP_KEYS)) or not np.isfinite(maxima).all() or (maxima < 0).any():
                raise ValueError(f'Invalid shared heat maxima: {folder}')
        shared = metadata.get('shared_heat_maxima', {})
        if any(not np.isfinite(float(value)) or float(value) < 0 for value in shared.values()):
            raise ValueError(f'Invalid shared heat maxima in metadata: {folder}')
        label = metadata.get('display_label') or ('With mask' if streams == 2 else 'Without mask')
        return cls(folder, metadata, arrays, feature_shape, label)

    @property
    def steps(self):
        return self.arrays['steps']

    @property
    def streams(self):
        return self.arrays['rgb'].shape[1]

    @property
    def counterfactual(self):
        return self.metadata.get('source') == 'mask_counterfactual'

    @property
    def condition_label(self):
        condition = self.metadata.get('counterfactual_condition')
        if condition == 'empty':
            return 'Empty mask'
        if condition == 'full':
            return 'Full mask'
        target = self.metadata.get('target')
        return f'Target cube{target}' if target is not None else 'Target unspecified'

    @property
    def default_fourth_panel(self):
        return FOURTH_PANEL_MODES[1] if self.counterfactual and self.streams == 2 else FOURTH_PANEL_MODES[0]

    def heat_maxima(self, index):
        if not self.counterfactual:
            return {}
        if 'heat_maxima' in self.arrays:
            return dict(zip(HEAT_MAP_KEYS, self.arrays['heat_maxima'][index].tolist()))
        return self.metadata.get('shared_heat_maxima', {})

    def nearest_index(self, step):
        return int(np.argmin(np.abs(self.steps - step)))

    def maps(self, index, query, encoder_layer):
        return compute_attention_maps(
            self.arrays['encoder_attention'][index], self.arrays['decoder_attention'][index],
            self.feature_shape, self.streams, self.arrays['target_mask'][index],
            query_index=query, encoder_layer=encoder_layer,
            decoder_layer_index=self.metadata.get('decoder_layer', 0))


def load_recordings(path):
    folders = [path] if (path / 'metadata.json').is_file() else sorted(path.glob('*/metadata.json'))
    folders = [item.parent if item.is_file() else item for item in folders]
    recordings = [Recording.load(folder) for folder in folders]
    if not recordings:
        raise ValueError(f'No ACT attention recordings under {path}')
    labels = set()
    for recording in recordings:
        if recording.label in labels:
            recording.label += f' ({recording.folder.name})'
        labels.add(recording.label)
    return recordings


def latest_recording():
    candidates = sorted((ROOT / 'outputs').glob('act_attention*'), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if (path / 'summary.json').is_file() and list(path.glob('*/metadata.json')):
            return path
    raise ValueError('No attention export found. Run visualize_act_attention.py first.')


def draw_outline(rgb, mask):
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    edge = np.asarray(image.filter(ImageFilter.MaxFilter(3))) != np.asarray(image.filter(ImageFilter.MinFilter(3)))
    result = rgb.copy()
    result[edge] = (50, 245, 130)
    return result


def overlay(rgb, weights, opacity, smooth, maximum=None):
    maximum = float(weights.max()) if maximum is None else float(maximum)
    values = weights / maximum if maximum > 0 else np.zeros_like(weights)
    method = Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST
    values = np.asarray(Image.fromarray(values.astype(np.float32)).resize((rgb.shape[1], rgb.shape[0]), method))
    values = np.clip(values, 0, 1)
    alpha = (opacity * values)[..., None]
    result = rgb * (1 - alpha) + COLORS(values)[..., :3] * 255 * alpha
    return np.round(result).astype(np.uint8), maximum


def panel(image, title, subtitle, footer='', maximum=None):
    canvas = Image.new('RGB', (PANEL_WIDTH, PANEL_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), title, font=ImageFont.truetype('DejaVuSans.ttf', 23), fill='#202729')
    draw.text((12, 39), subtitle, font=ImageFont.truetype('DejaVuSans.ttf', 15), fill='#596569')
    canvas.paste(Image.fromarray(image).resize((480, 480), Image.Resampling.NEAREST), (0, 64))
    if maximum is not None:
        ramp = (COLORS(np.linspace(0, 1, 456))[None, :, :3] * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(np.repeat(ramp, 10, axis=0)), (12, 552))
        footer = f'0 to {maximum:.5f} weight / feature cell'
    draw.text((12, 574), footer, font=ImageFont.truetype('DejaVuSans.ttf', 15), fill='#596569')
    return np.asarray(canvas)


def build_panels(recording, index, maps, query, opacity, smooth, show_outline, binary_mask,
                 fourth_mode=None):
    images = recording.arrays['rgb'][index].transpose(0, 2, 3, 1)
    rgb, mask = images[0], recording.arrays['target_mask'][index]
    mark = (lambda image: draw_outline(image, mask)) if show_outline else (lambda image: image)
    maxima = recording.heat_maxima(index)
    fourth_mode = recording.default_fourth_panel if fourth_mode is None else fourth_mode
    condition = recording.condition_label
    query_label = 'All action queries' if query == -1 else f'Action query {query}'
    grid = f'{recording.feature_shape[0]}x{recording.feature_shape[1]}'
    original_subtitle = (f'Frozen input / {condition}'
                         if recording.counterfactual else condition)
    original = panel(mark(rgb), 'Original RGB', original_subtitle, 'Policy input / RGB stream 0')
    if recording.streams == 2 and not binary_mask:
        second = panel(images[1], 'Masked RGB', condition, 'Policy input / RGB stream 1')
    else:
        binary = np.repeat(mask[..., None], 3, axis=-1).astype(np.uint8) * 255
        second = panel(binary, 'Binary mask' if recording.counterfactual else 'Target mask', condition,
                       'Binary target segmentation' if recording.streams == 2 else 'Reference only / not a policy input')
    action, action_max = overlay(rgb, maps['action_original'], opacity, smooth, maxima.get('action_original'))
    third = panel(mark(action), 'Action -> RGB', f'{query_label} / attention rollout / {grid}', maximum=action_max)
    background, empty_message = rgb, 'Target not visible: no mask relation'
    if recording.streams == 2 and fourth_mode == FOURTH_PANEL_MODES[1]:
        title = 'Action -> masked RGB'
        subtitle = f'{query_label} / attention rollout / {grid}'
        background = images[1]
        key, valid = 'action_masked', True
    elif recording.streams == 2:
        title = f'{condition} -> RGB' if condition in ('Empty mask', 'Full mask') else 'Mask target -> RGB'
        subtitle = f'Encoder {maps["selected_encoder_layer"] + 1} / mask query cells / {grid}'
        key = 'mask_to_original'
        valid = maps['mask_relation_valid']
        if recording.metadata.get('counterfactual_condition') == 'empty':
            empty_message = 'Empty mask: no mask relation'
    else:
        title = 'Decoder -> RGB'
        subtitle = f'{query_label} / direct attention / {grid}'
        key, valid = 'decoder_original', True
    heat, maximum = overlay(background, maps[key], opacity, smooth, maxima.get(key))
    fourth = panel(mark(heat if valid else background), title, subtitle,
                   footer=empty_message if not valid else '',
                   maximum=maximum if valid else None)
    return [original, second, third, fourth]


class ViewerSession:
    def __init__(self, client, recordings):
        self.client, self.recordings = client, recordings
        self.by_label = {recording.label: recording for recording in recordings}
        self.stop_event, self.dirty = threading.Event(), threading.Event()
        self.lock = threading.RLock()
        self.playing = False
        self.next_tick = 0.0
        self.last_panels = None
        self.last_maps = None
        self.last_step = 0
        self.last_label = recordings[0].label
        self.planes = []
        gui = client.gui
        gui.configure_theme(control_width='medium' if any(run.counterfactual for run in recordings) else 'small',
                            brand_color=(0, 137, 105), show_logo=False, show_share_button=False)
        gui.main_panel.dock_left()
        compact = client.camera.image_width < 650 or client.camera.aspect < 0.85
        # Viser's mobile main panel ignores minimize(), but folders can collapse.
        with gui.add_folder('ACT attention', expand_by_default=not compact):
            self.add_controls(gui, recordings[0])
        self.model.on_update(self.model_changed)
        self.frame.on_update(self.frame_changed)
        self.average.on_update(self.average_changed)
        self.fourth.on_update(self.fourth_changed)
        for handle in (self.query, self.encoder, self.opacity, self.smooth, self.outline, self.binary):
            handle.on_update(lambda _: self.dirty.set())
        blank = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 255, dtype=np.uint8)
        for index, position in enumerate(POSITIONS):
            self.planes.append(client.scene.add_image(f'/attention/panel_{index}', blank,
                               render_width=1.0, render_height=1.25, position=position,
                               wxyz=(0.0, 1.0, 0.0, 0.0),
                               format='jpeg', jpeg_quality=95, cast_shadow=False, receive_shadow=False))
        self.fit_camera()
        self.dirty.set()
        self.thread = threading.Thread(target=self.worker, name=f'viser-attention-{client.client_id}', daemon=True)
        self.thread.start()

    def add_controls(self, gui, first):
        self.model = gui.add_dropdown('Recording', tuple(self.by_label))
        self.recording_kind = (first.streams, first.counterfactual)
        self.status = gui.add_markdown('')
        self.frame = gui.add_slider('Frame', min=int(first.steps[0]), max=max(int(first.steps[-1]), 1),
                                    step=self.frame_stride(first), initial_value=int(first.steps[0]),
                                    disabled=len(first.steps) == 1)
        self.play = gui.add_button('Play', icon=viser.Icon.PLAYER_PLAY, hint='Play or pause the recorded trajectory')
        with gui.add_folder('Playback', expand_by_default=False):
            self.speed = gui.add_dropdown('Speed', ('0.25x', '0.5x', '1x', '2x', '4x'), initial_value='1x')
            self.loop = gui.add_checkbox('Loop', True)
            previous = gui.add_button('Previous frame', icon=viser.Icon.CHEVRON_LEFT)
            following = gui.add_button('Next frame', icon=viser.Icon.CHEVRON_RIGHT)
        with gui.add_folder('Attention'):
            self.fourth = gui.add_dropdown('Fourth panel', FOURTH_PANEL_MODES,
                                           initial_value=first.default_fourth_panel,
                                           visible=first.streams == 2)
            self.average = gui.add_checkbox('Average queries', False)
            self.query = gui.add_slider('Action query', min=0, max=first.arrays['decoder_attention'].shape[1] - 1,
                                        step=1, initial_value=0)
            self.encoder = gui.add_slider('Encoder layer', min=1, max=first.arrays['encoder_attention'].shape[1],
                                          step=1, initial_value=1,
                                          disabled=first.streams == 1 or self.fourth.value != FOURTH_PANEL_MODES[0],
                                          hint='Layer used for the mask-to-RGB relation')
        with gui.add_folder('Display'):
            self.opacity = gui.add_slider('Heat opacity', min=0.0, max=1.0, step=0.05, initial_value=0.7)
            self.smooth = gui.add_checkbox('Smooth heatmap', False)
            self.outline = gui.add_checkbox('Target outline', True)
            self.binary = gui.add_checkbox('Binary mask', first.streams == 1, disabled=first.streams == 1)
            reset = gui.add_button('Reset view', icon=viser.Icon.HOME, hint='Fit all four images in the viewport')
        self.mass = gui.add_markdown('')
        download = gui.add_button('Download image', icon=viser.Icon.DOWNLOAD)
        download_maps = gui.add_button('Download maps', icon=viser.Icon.FILE_DOWNLOAD)
        with gui.add_folder('Recording', expand_by_default=False):
            self.details = gui.add_markdown('')
        self.error = gui.add_markdown('', visible=False)
        self.play.on_click(lambda _: self.set_playing(not self.playing))
        previous.on_click(lambda _: self.move(-1))
        following.on_click(lambda _: self.move(1))
        reset.on_click(lambda _: self.fit_camera())
        download.on_click(lambda _: self.download_image())
        download_maps.on_click(lambda _: self.download_maps())

    @staticmethod
    def frame_stride(recording):
        diffs = np.diff(recording.steps)
        return int(diffs[0]) if len(diffs) and np.all(diffs == diffs[0]) else 1

    def fit_camera(self):
        aspect = max(self.client.camera.aspect, 0.15)
        fov = math.radians(45)
        distance = max(2.78 / (2 * math.tan(fov / 2)), 2.30 / (2 * math.tan(fov / 2) * aspect))
        with self.client.atomic():
            self.client.camera.fov = fov
            self.client.camera.position = (0.0, 0.0, distance)
            self.client.camera.look_at = (0.0, 0.0, 0.0)
            self.client.camera.up_direction = (0.0, 1.0, 0.0)

    def set_playing(self, playing):
        with self.lock:
            self.playing = playing
            self.next_tick = time.monotonic()
            self.play.label = 'Pause' if playing else 'Play'
            self.play.icon = viser.Icon.PLAYER_PAUSE if playing else viser.Icon.PLAYER_PLAY

    def model_changed(self, _):
        with self.lock:
            self.set_playing(False)
            recording = self.by_label[self.model.value]
            step = int(recording.steps[recording.nearest_index(self.frame.value)])
            self.frame.min = int(recording.steps[0])
            self.frame.max = max(int(recording.steps[-1]), 1)
            self.frame.step = self.frame_stride(recording)
            self.frame.disabled = len(recording.steps) == 1
            self.frame.value = step
            self.query.max = recording.arrays['decoder_attention'].shape[1] - 1
            self.query.value = min(self.query.value, self.query.max)
            self.encoder.max = recording.arrays['encoder_attention'].shape[1]
            self.encoder.value = min(self.encoder.value, self.encoder.max)
            kind = (recording.streams, recording.counterfactual)
            if kind != self.recording_kind:
                self.fourth.value = recording.default_fourth_panel
            self.recording_kind = kind
            self.fourth.visible = recording.streams == 2
            self.encoder.disabled = recording.streams == 1 or self.fourth.value != FOURTH_PANEL_MODES[0]
            self.binary.value = recording.streams == 1
            self.binary.disabled = recording.streams == 1
            self.dirty.set()

    def frame_changed(self, event):
        if event.client is not None:
            self.set_playing(False)
        self.dirty.set()

    def average_changed(self, _):
        self.query.disabled = self.average.value
        self.dirty.set()

    def fourth_changed(self, _):
        recording = self.by_label[self.model.value]
        self.encoder.disabled = recording.streams == 1 or self.fourth.value != FOURTH_PANEL_MODES[0]
        self.dirty.set()

    def move(self, delta):
        with self.lock:
            self.set_playing(False)
            recording = self.by_label[self.model.value]
            index = np.clip(recording.nearest_index(self.frame.value) + delta, 0, len(recording.steps) - 1)
            self.frame.value = int(recording.steps[index])
            self.dirty.set()

    def render(self):
        with self.lock:
            recording = self.by_label[self.model.value]
            index = recording.nearest_index(self.frame.value)
            query = -1 if self.average.value else self.query.value
            layer = self.encoder.value - 1
            display = (self.opacity.value, self.smooth.value, self.outline.value, self.binary.value,
                       self.fourth.value)
        maps = recording.maps(index, query, layer)
        panels = build_panels(recording, index, maps, query, *display)
        step = int(recording.steps[index])
        fps = recording.metadata.get('fps', 20)
        if recording.counterfactual:
            status = (f'Frozen-input test / {recording.condition_label}\n\n'
                      f'**Reference frame {step} / {recording.steps[-1]}** / {step / fps:.2f} s')
        else:
            source = 'Policy rollout' if recording.metadata.get('closed_loop') else 'Recorded demonstration'
            status = (f'{source} / {recording.condition_label}\n\n'
                      f'**Frame {step} / {recording.steps[-1]}** / {step / fps:.2f} s')
        scale_note = ('Colorbars share a fixed scale across mask conditions at this reference frame.'
                      if recording.heat_maxima(index) else "Colorbars use each map's maximum.")
        pixels_label = 'Mask pixels' if recording.counterfactual else 'Target pixels'
        reference_target = recording.metadata.get('reference_target')
        reference_label = f'cube{reference_target}' if reference_target is not None else 'unknown'
        reference_note = (f'Reference trajectory target: {reference_label}\n\n'
                          if recording.counterfactual else '')
        with self.client.atomic():
            for plane, image in zip(self.planes, panels):
                plane.image = image
            self.status.content = status
            self.mass.content = (f'RGB mass **{maps["action_mass_original"]:.3f}** / '
                                 f'masked **{maps["action_mass_masked"]:.3f}**\n\n'
                                 f'{pixels_label} **{int(recording.arrays["target_mask"][index].sum())}**')
            checkpoint = Path(recording.metadata['checkpoint'])
            self.details.content = (f'Checkpoint: `{checkpoint.name}`\n\n'
                                    f'Run: `{checkpoint.parent.parent.name}`\n\n'
                                    f'{len(recording.steps)} saved frames / '
                                    f'{recording.feature_shape[0]}x{recording.feature_shape[1]} feature grid\n\n'
                                    f'Decoder: {recording.metadata.get("decoder_layer", 0) + 1}\n\n'
                                    f'{reference_note}'
                                    f'Attention routing; not causal attribution. {scale_note}')
            self.error.visible = False
        with self.lock:
            self.last_panels, self.last_maps = panels, maps
            self.last_step, self.last_label = step, recording.label

    def worker(self):
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    if self.playing and time.monotonic() >= self.next_tick:
                        recording = self.by_label[self.model.value]
                        index = recording.nearest_index(self.frame.value)
                        following = index + 1
                        if following >= len(recording.steps):
                            if self.loop.value:
                                following = 0
                            else:
                                self.set_playing(False)
                                following = index
                        gap = abs(int(recording.steps[following]) - int(recording.steps[index]))
                        if following == 0 or gap == 0:
                            gap = self.frame_stride(recording)
                        speed = float(self.speed.value[:-1])
                        self.next_tick = time.monotonic() + gap / recording.metadata.get('fps', 20) / speed
                        self.frame.value = int(recording.steps[following])
                        self.dirty.set()
                if self.dirty.is_set():
                    self.dirty.clear()
                    self.render()
            except Exception as error:
                traceback.print_exc()
                self.set_playing(False)
                self.error.content = f'Visualization error: {error}'
                self.error.visible = True
            self.stop_event.wait(0.02)

    def download_image(self):
        with self.lock:
            if self.last_panels is None:
                return
            canvas = Image.new('RGB', (PANEL_WIDTH * 2, PANEL_HEIGHT * 2))
            for index, image in enumerate(self.last_panels):
                canvas.paste(Image.fromarray(image), ((index % 2) * PANEL_WIDTH, (index // 2) * PANEL_HEIGHT))
            buffer = io.BytesIO()
            canvas.save(buffer, format='PNG')
            file_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', self.last_label).strip('_') or 'recording'
            self.client.send_file_download(f'act_attention_{file_label}_{self.last_step:04d}.png',
                                           buffer.getvalue(), save_immediately=True)

    def download_maps(self):
        with self.lock:
            if self.last_maps is None:
                return
            buffer = io.BytesIO()
            recording = self.by_label[self.last_label]
            np.savez_compressed(buffer, step=self.last_step, model=self.last_label,
                                checkpoint=recording.metadata['checkpoint'],
                                checkpoint_sha256=recording.metadata.get('checkpoint_sha256', ''),
                                **self.last_maps)
            file_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', self.last_label).strip('_') or 'recording'
            self.client.send_file_download(f'act_attention_maps_{file_label}_{self.last_step:04d}.npz',
                                           buffer.getvalue(), save_immediately=True)

    def close(self):
        self.stop_event.set()


def main(cli):
    torch.set_num_threads(2)
    path = cli.results.resolve() if cli.results else latest_recording()
    recordings = load_recordings(path)
    server = viser.ViserServer(host=cli.host, port=cli.port, label='ACT attention')
    server.scene.set_up_direction('+y')
    server.scene.set_background_image(np.full((2, 2, 3), (239, 242, 243), dtype=np.uint8), format='png')
    sessions = {}

    @server.on_client_connect
    def connect(client):
        sessions[client.client_id] = ViewerSession(client, recordings)

    @server.on_client_disconnect
    def disconnect(client):
        session = sessions.pop(client.client_id, None)
        if session is not None:
            session.close()

    print(f'ACT attention Viser: http://{cli.host}:{server.get_port()}', flush=True)
    print(f'Recordings: {path}', flush=True)
    print(', '.join(f'{run.label}: {len(run.steps)} frames' for run in recordings), flush=True)
    try:
        server.sleep_forever()
    finally:
        for session in list(sessions.values()):
            session.close()
        server.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, help='Export directory or one model subdirectory; defaults to latest export')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18767)
    main(parser.parse_args())
