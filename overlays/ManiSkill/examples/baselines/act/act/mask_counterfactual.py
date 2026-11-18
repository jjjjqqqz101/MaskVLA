"""Pure-data metrics and figures for fixed-observation mask interventions.

Action differences measure a policy's response to an input intervention. They
are not measured end-effector displacement or evidence of successful grasping.
Attention maps describe routing; their weights do not establish causal feature
importance. This module does not load policies or instantiate a simulator.
"""

from pathlib import Path
import textwrap

import numpy as np
import torch
import torch.nn.functional as F


def _finite_array(value, name):
    array = np.asarray(value, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite, nonempty numeric data')
    return array


def _difference(left, right, name):
    left = _finite_array(left, name)
    right = _finite_array(right, name)
    if left.shape != right.shape:
        raise ValueError(f'{name} shapes differ: {left.shape} versus {right.shape}')
    return left - right


def _rmse(values):
    return float(np.sqrt(np.mean(np.square(values))))


def compare_predictions(left, right):
    """Compare two conditions from the same checkpoint using JSON scalars.

    Required keys are ``normalized_actions`` and ``actions`` ([Q, 8]),
    ``encoder_attention`` ([L, N, N]), ``decoder_attention`` ([Q, N]),
    ``maps`` (a compute_attention_maps result), and ``masked_rgb`` (uint8
    [3, H, W] or None). Actions must have already been transformed back using
    checkpoint action_mean/action_std. For Panda pd_joint_pos, their first
    seven components are absolute joint target angles in radians. The last
    component is a normalized gripper controller command, not a distance.

    Chunk metrics compare all predicted queries, without temporal aggregation
    or execution. First-query L2 is the norm over seven joint target angles.
    Attention L1 is the sum of absolute *raw* map-weight differences and thus
    retains changes in the total attention mass allocated to that stream.
    """
    normalized = _difference(left['normalized_actions'], right['normalized_actions'], 'normalized_actions')
    actions = _difference(left['actions'], right['actions'], 'actions')
    if normalized.ndim != 2 or normalized.shape[1] != 8 or actions.shape != normalized.shape:
        raise ValueError('Expected matching [Q, 8] normalized and controller actions')
    encoder = _difference(left['encoder_attention'], right['encoder_attention'], 'encoder_attention')
    decoder = _difference(left['decoder_attention'], right['decoder_attention'], 'decoder_attention')
    if encoder.ndim != 3 or encoder.shape[1] != encoder.shape[2]:
        raise ValueError('Expected encoder attention with shape [L, N, N]')
    if decoder.shape != (actions.shape[0], encoder.shape[1]):
        raise ValueError('Expected decoder attention with shape [Q, N]')
    original = _difference(left['maps']['action_original'], right['maps']['action_original'], 'action_original')
    if original.ndim != 2:
        raise ValueError('Expected two-dimensional action attention maps')
    has_masked = ['action_masked' in row['maps'] for row in (left, right)]
    if has_masked[0] != has_masked[1]:
        raise ValueError('Compare conditions with the same visual stream count')
    masked_l1 = None
    if has_masked[0]:
        masked = _difference(left['maps']['action_masked'], right['maps']['action_masked'], 'action_masked')
        if masked.shape != original.shape:
            raise ValueError('Original and masked attention grids must match')
        masked_l1 = float(np.abs(masked).sum())
    left_image, right_image = left['masked_rgb'], right['masked_rgb']
    if left_image is None or right_image is None:
        identical = left_image is None and right_image is None
    else:
        for image in (left_image, right_image):
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3:
                raise ValueError('masked_rgb must be uint8 [3, H, W] or None')
        identical = np.array_equal(left_image, right_image)
    return {
        'normalized_chunk_rmse': _rmse(normalized),
        'arm_chunk_rmse_rad': _rmse(actions[:, :7]),
        'arm_first_query_l2_rad': float(np.linalg.norm(actions[0, :7])),
        'arm_max_abs_rad': float(np.abs(actions[:, :7]).max()),
        'gripper_chunk_rmse_command': _rmse(actions[:, 7]),
        'encoder_mean_abs_diff': float(np.abs(encoder).mean()),
        'decoder_mean_abs_diff': float(np.abs(decoder).mean()),
        'action_original_l1': float(np.abs(original).sum()),
        'action_masked_l1': masked_l1,
        'masked_input_identical': bool(identical),
    }


def region_scores(maps, masks):
    """Score the same three cube regions under every intervention condition.

    ``masks`` is [3, H, W], with one fixed segmentation region per cube. Each
    mask is area-pooled onto the feature grid. A stream's overlap is
    sum(attention * region_coverage) / sum(attention), and lift divides that
    overlap by mean(region_coverage), the uniform-attention reference. Scores
    remain coarse because CNN receptive fields overlap and regions may share
    feature cells. Invisible regions and unavailable streams return None.
    """
    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.shape[0] != 3 or min(masks.shape[1:]) < 1:
        raise ValueError('masks must have shape [3, H, W]')
    if not np.isfinite(masks).all():
        raise ValueError('masks contains non-finite values')
    original = _finite_array(maps['action_original'], 'action_original')
    if original.ndim != 2 or (original < 0).any():
        raise ValueError('action_original must be a nonnegative feature-grid map')
    stream_maps = {'original': original, 'masked': None}
    if 'action_masked' in maps:
        masked = _finite_array(maps['action_masked'], 'action_masked')
        if masked.shape != original.shape or (masked < 0).any():
            raise ValueError('action_masked must match the original feature grid')
        stream_maps['masked'] = masked
    masks = masks > 0
    pooled = F.adaptive_avg_pool2d(torch.from_numpy(masks.astype(np.float32))[:, None], original.shape)
    coverage = pooled[:, 0].numpy()
    scores = {}
    for index in range(3):
        prefix = f'cube{index + 1}'
        pixels = int(masks[index].sum())
        baseline = float(coverage[index].mean())
        scores[f'{prefix}_pixels'] = pixels
        scores[f'{prefix}_area_fraction'] = baseline
        for stream, weights in stream_maps.items():
            overlap = lift = None
            if pixels and weights is not None and weights.sum() > 0:
                overlap = float((weights * coverage[index]).sum() / weights.sum())
                lift = float(overlap / baseline)
            scores[f'{prefix}_{stream}_overlap'] = overlap
            scores[f'{prefix}_{stream}_lift'] = lift
    return scores


def render_comparison(rows, destination, title):
    """Save five-column input/attention rows with a shared scale per column.

    Each row supplies ``condition``, ``rgb`` (uint8 [S, 3, H, W]),
    ``target_mask`` ([H, W]), and ``maps``. Input images are displayed without
    outlines or modification. Empty/full mask controls are labeled by their
    actual coverage. Single-stream rows display the binary reference mask;
    they have no masked-stream action attention. Heat overlays keep the RGB
    background visible and retain a common zero-to-maximum scale for every
    condition in the same column. Returns the output Path.
    """
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from PIL import Image

    if not rows:
        raise ValueError('At least one comparison row is required')
    prepared = []
    for row in rows:
        rgb = np.asarray(row['rgb'])
        mask = np.asarray(row['target_mask'])
        if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[0] not in (1, 2) or rgb.shape[1] != 3:
            raise ValueError('Each rgb input must be uint8 [1 or 2, 3, H, W]')
        if mask.shape != rgb.shape[-2:] or not np.isfinite(mask).all():
            raise ValueError('target_mask must match the input image dimensions')
        images = rgb.transpose(0, 2, 3, 1)
        mask = mask > 0
        dual = len(images) == 2
        maps = row['maps']
        relation_valid = bool(maps['mask_relation_valid']) if dual else True
        heatmaps = [
            maps['action_original'], maps.get('action_masked') if dual else None,
            maps['mask_to_original'] if dual and relation_valid else (None if dual else maps['decoder_original']),
        ]
        for index, heatmap in enumerate(heatmaps):
            if heatmap is not None:
                heatmap = _finite_array(heatmap, 'attention heatmap')
                if heatmap.ndim != 2 or (heatmap < 0).any():
                    raise ValueError('Attention heatmaps must be nonnegative two-dimensional arrays')
                heatmaps[index] = heatmap
        prepared.append((row, images, mask, dual, heatmaps))
    maxima = [max((float(item[4][column].max()) for item in prepared if item[4][column] is not None),
                  default=0.0) for column in range(3)]
    available = [any(item[4][column] is not None for item in prepared) for column in range(3)]
    cmap = colormaps['inferno']
    figure = Figure(figsize=(17.5, 3.5 * len(rows) + 1.1), layout='constrained', facecolor='white')
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(rows), 5, squeeze=False)
    figure.suptitle(title, fontsize=17)

    def heat_image(image, weights, maximum):
        fraction = np.zeros_like(weights) if maximum <= 0 else weights / maximum
        fraction = np.asarray(Image.fromarray(fraction.astype(np.float32)).resize(
            (image.shape[1], image.shape[0]), Image.Resampling.NEAREST))
        fraction = np.clip(fraction, 0, 1)
        alpha = (0.72 * fraction)[..., None]
        return np.round(image * (1 - alpha) + cmap(fraction)[..., :3] * 255 * alpha).astype(np.uint8)

    for row_index, (row, images, mask, dual, heatmaps) in enumerate(prepared):
        pixels = int(mask.sum())
        control = 'Empty mask control' if pixels == 0 else ('Full mask control' if mask.all() else f'{pixels} target pixels')
        condition = textwrap.fill(str(row['condition']), width=24)
        axes[row_index, 0].set_ylabel(f'{condition}\n{control}', fontsize=10, rotation=0,
                                      labelpad=82, va='center', ha='center')
        binary = np.repeat(mask[..., None], 3, axis=-1).astype(np.uint8) * 255
        pictures = [images[0], images[1] if dual else binary]
        names = ['Original RGB\nPolicy input', 'Masked RGB\nPolicy input' if dual else 'Reference mask\nNot a policy input']
        backgrounds = [images[0], images[1] if dual else images[0], images[0]]
        query = row['maps'].get('query_index', 0)
        query_text = 'all queries' if query == -1 else f'query {query}'
        names.extend([
            f'Action -> RGB\nRollout / {query_text}',
            f'Action -> Masked RGB\nRollout / {query_text}' if dual else 'Action -> Masked RGB\nUnavailable',
            f'Mask -> RGB\nEncoder {row["maps"]["selected_encoder_layer"] + 1}' if dual
            else f'Decoder -> RGB\nDirect / {query_text}',
        ])
        for column, (weights, background) in enumerate(zip(heatmaps, backgrounds)):
            if weights is None:
                pictures.append(np.full_like(images[0], 242))
            else:
                pictures.append(heat_image(background, weights, maxima[column]))
        for column, (axis, picture, name) in enumerate(zip(axes[row_index], pictures, names)):
            axis.imshow(picture, interpolation='nearest')
            axis.set_title(name, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if column >= 2 and heatmaps[column - 2] is None:
                message = 'No masked stream' if column == 3 else 'No visible target query cells'
                axis.text(0.5, 0.5, message, ha='center', va='center', fontsize=9,
                          color='#555555', transform=axis.transAxes, wrap=True)
    for column in range(3):
        if not available[column]:
            continue
        maximum = maxima[column]
        scalar = ScalarMappable(norm=Normalize(0, maximum if maximum > 0 else 1), cmap=cmap)
        colorbar = figure.colorbar(scalar, ax=axes[:, column + 2].tolist(), location='bottom',
                                  shrink=0.9, fraction=0.045, pad=0.025, aspect=24)
        colorbar.set_label('Raw weight / feature cell; shared across rows', fontsize=8)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_ticks([0, maximum] if maximum > 0 else [0])
        colorbar.set_ticklabels(['0', f'{maximum:.5f}'] if maximum > 0 else ['0 (all weights zero)'])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130, format='png')
    figure.clear()
    return destination
