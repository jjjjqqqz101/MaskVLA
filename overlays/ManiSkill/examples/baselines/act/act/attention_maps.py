"""Read ACT attention weights without changing the policy's forward pass.

Attention rollout is a visualization heuristic, not a causal explanation of an
action. It omits value projections, feed-forward blocks, normalization, and
channel-dependent effects. Direct attention relates contextualized token
locations; a location is not restricted to information from that image patch.
"""

import numpy as np
import torch
import torch.nn.functional as F


def visual_token_indices(height, width, num_streams):
    """Return [streams, H, W] indices after z/state and width concatenation.

    DETRVAE concatenates images along feature width *before* flattening. For
    two 2x2 feature maps the visual token indices are therefore::

        original = [[2, 3], [6, 7]]
        masked   = [[4, 5], [8, 9]]
    """
    if height < 1 or width < 1 or num_streams not in (1, 2):
        raise ValueError("Expected positive feature dimensions and 1 or 2 streams")
    indices = np.arange(2, 2 + height * width * num_streams)
    return indices.reshape(height, num_streams, width).transpose(1, 0, 2)


def _validate_attention(attention, name):
    if not np.isfinite(attention).all():
        raise ValueError(f"{name} contains non-finite attention weights")
    if (attention < -1e-7).any():
        raise ValueError(f"{name} contains negative attention weights")
    if not np.allclose(attention.sum(axis=-1), 1.0, atol=2e-4, rtol=2e-4):
        raise ValueError(f"{name} rows must sum to 1; run the policy in eval mode")


class AttentionRecorder:
    """Context manager recording one eval forward of a visual DETRVAE.

    Usage::

        with AttentionRecorder(agent.model) as recorder:
            recorder.clear()
            actions = agent.get_action(observation)
            maps = recorder.maps(2, target_mask, query_index=0)

    Hooks only inspect existing outputs. All hooks are removed on context exit,
    including exceptional exits. ``clear`` must precede each new forward.
    ``encoder_attention`` stores one [B, N, N] numpy array per encoder layer;
    ``decoder_attention`` stores [B, Q, N] cross-attention from the decoder
    layer whose output the existing DETRVAE consumes. Weights are head means.
    The present model returns intermediate decoder outputs and selects [0],
    so its action-producing decoder layer is layer 0. This module preserves
    that behavior instead of switching inference to another decoder layer.
    """

    def __init__(self, model):
        self.model = model
        self._handles = []
        decoder = model.transformer.decoder
        self.decoder_layer_index = 0 if decoder.return_intermediate else len(decoder.layers) - 1
        self.clear()

    def clear(self):
        self.encoder_attention = []
        self.decoder_attention = None
        self.feature_shape = None
        self._feature_shapes = []

    def __enter__(self):
        if self._handles:
            raise RuntimeError("AttentionRecorder is already active")
        if self.model.training:
            raise ValueError("Attention visualization requires model.eval()")
        if not hasattr(self.model, "input_proj"):
            raise ValueError("Attention visualization requires a visual ACT model")
        self.clear()
        try:
            for layer in self.model.transformer.encoder.layers:
                self._handles.append(layer.self_attn.register_forward_hook(self._capture_encoder))
            decoder = self.model.transformer.decoder.layers[self.decoder_layer_index]
            self._handles.append(decoder.multihead_attn.register_forward_hook(self._capture_decoder))
            self._handles.append(self.model.input_proj.register_forward_hook(self._capture_grid))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    @staticmethod
    def _weights(output):
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            raise RuntimeError("ACT attention module did not return weights")
        weights = output[1].detach().float()
        if weights.ndim == 4:
            weights = weights.mean(dim=1)
        if weights.ndim != 3:
            raise ValueError(f"Expected batched attention weights, got {tuple(weights.shape)}")
        return weights.cpu().numpy().copy()

    def _capture_encoder(self, module, inputs, output):
        self.encoder_attention.append(self._weights(output))

    def _capture_decoder(self, module, inputs, output):
        if self.decoder_attention is not None:
            raise RuntimeError("Call recorder.clear() before each policy forward")
        self.decoder_attention = self._weights(output)

    def _capture_grid(self, module, inputs, output):
        if output.ndim != 4 or output.shape[0] != 1:
            raise ValueError("Attention visualization requires image features with batch size 1")
        shape = tuple(output.shape[-2:])
        self._feature_shapes.append(shape)
        self.feature_shape = shape

    def maps(self, num_streams, target_mask, query_index=0, encoder_layer=0):
        """Validate the capture and compute maps using ``compute_attention_maps``."""
        if num_streams not in (1, 2):
            raise ValueError("num_streams must be 1 (RGB) or 2 (RGB + masked RGB)")
        if self.model.training:
            raise ValueError("Attention visualization requires model.eval()")
        if self.feature_shape is None or self.decoder_attention is None:
            raise RuntimeError("No complete forward recorded; run the policy inside the context")
        expected_layers = len(self.model.transformer.encoder.layers)
        if len(self.encoder_attention) != expected_layers:
            raise ValueError("Expected one complete encoder forward; clear the recorder before inference")
        if len(self._feature_shapes) != num_streams or any(
            shape != self.feature_shape for shape in self._feature_shapes
        ):
            raise ValueError("Recorded feature maps do not match num_streams and a common feature grid")

        return compute_attention_maps(
            self.encoder_attention, self.decoder_attention, self.feature_shape,
            num_streams, target_mask, query_index, encoder_layer, self.decoder_layer_index,
        )


def compute_attention_maps(
    encoder_attention, decoder_attention, feature_shape, num_streams, target_mask,
    query_index=0, encoder_layer=0, decoder_layer_index=0,
):
    """Compute feature-grid maps and mass summaries from captured numpy data.

    ``encoder_attention`` is [L, N, N] or [L, 1, N, N], including a list of
    [1, N, N] arrays from AttentionRecorder. ``decoder_attention`` is [Q, N]
    or [1, Q, N]. Inputs are head-averaged probabilities for one observation.
    ``feature_shape`` is the (height, width) of one image's feature map.
    This function does not need a model and does not modify its input arrays.

    For encoder weights A_l, residual rollout is
    R_l = row_normalize(A_l + I), R = R_last @ ... @ R_first.
    ``action_*`` is decoder cross-attention @ R for one action query;
    query_index=-1 averages all action queries in the predicted chunk.
    ``decoder_*`` is the direct cross-attention to encoder memory.

    ``mask_to_original`` uses the selected encoder's direct self-attention:
    sum_i target_coverage_i * A_layer[masked_i, original_j] / sum_i coverage_i.
    The default first layer has queries derived from the masked image before
    cross-stream fusion; later layers contain mixed information. Negative
    encoder_layer indices count from the last layer. The mask chooses query
    patches by area coverage; it is not itself painted as a heatmap. With an
    empty target mask this relation is zero and ``mask_relation_valid`` is
    False. Single-stream models do not have a mask-to-original map.

    All maps retain probability mass, without per-map normalization.
    ``target_overlap_original`` is action mass weighted by the fraction of
    each original feature cell occupied by target pixels, divided by total
    original-image mass. It is a coarse spatial statistic, not segmentation
    accuracy or a causal importance score. ``decoder_layer_index`` records
    which layer produced the supplied decoder attention; it selects no data.
    """
    if num_streams not in (1, 2):
        raise ValueError("num_streams must be 1 (RGB) or 2 (RGB + masked RGB)")
    if len(feature_shape) != 2:
        raise ValueError("feature_shape must be (height, width)")
    height, width = feature_shape
    indices = visual_token_indices(height, width, num_streams)
    token_count = 2 + height * width * num_streams
    encoder = np.asarray(encoder_attention)
    decoder = np.asarray(decoder_attention)
    if encoder.ndim == 4 and encoder.shape[1] == 1:
        encoder = encoder[:, 0]
    if decoder.ndim == 3 and decoder.shape[0] == 1:
        decoder = decoder[0]
    if encoder.ndim != 3 or encoder.shape[0] < 1 or encoder.shape[1:] != (token_count, token_count):
        raise ValueError(f"Unexpected encoder attention shape: {encoder.shape}")
    if decoder.ndim != 2 or decoder.shape[0] < 1 or decoder.shape[1] != token_count:
        raise ValueError(f"Unexpected decoder attention shape: {decoder.shape}")
    _validate_attention(encoder, "Encoder")
    _validate_attention(decoder, "Decoder")
    expected_layers = encoder.shape[0]
    if not -expected_layers <= encoder_layer < expected_layers:
        raise ValueError(f"encoder_layer must be in [{-expected_layers}, {expected_layers - 1}]")
    encoder_layer %= expected_layers
    if query_index != -1 and not 0 <= query_index < decoder.shape[0]:
        raise ValueError(f"query_index must be -1 or in [0, {decoder.shape[0] - 1}]")

    target_mask = np.asarray(target_mask)
    if target_mask.ndim != 2 or min(target_mask.shape) < 1:
        raise ValueError("target_mask must be a nonempty HxW array")
    if not np.isfinite(target_mask).all():
        raise ValueError("target_mask contains non-finite values")
    mask_tensor = torch.from_numpy((target_mask > 0).astype(np.float32))[None, None]
    coverage = F.adaptive_avg_pool2d(mask_tensor, (height, width))[0, 0].numpy()

    identity = np.eye(token_count, dtype=np.float32)
    rollout = identity.copy()
    for layer_attention in encoder:
        residual = layer_attention + identity
        residual /= residual.sum(axis=-1, keepdims=True)
        rollout = residual @ rollout
    direct = decoder.mean(axis=0) if query_index == -1 else decoder[query_index]
    action_mass = direct @ rollout
    _validate_attention(rollout, "Encoder rollout")
    _validate_attention(action_mass[None], "Action rollout")

    original = action_mass[indices[0]]
    original_mass = float(original.sum())
    target_overlap = float((original * coverage).sum()) / original_mass if original_mass > 0 else 0.0
    target_baseline = float(coverage.mean())
    maps = {
        "action_original": original,
        "decoder_original": direct[indices[0]],
        "encoder_rollout": rollout,
        "action_token_mass": action_mass,
        "target_coverage": coverage,
        "visual_token_indices": indices,
        "feature_height": height,
        "feature_width": width,
        "decoder_layer_index": decoder_layer_index,
        "selected_encoder_layer": encoder_layer,
        "query_index": query_index,
        "action_mass_latent": float(action_mass[0]),
        "action_mass_state": float(action_mass[1]),
        "action_mass_original": original_mass,
        "action_mass_masked": 0.0,
        "target_overlap_original": target_overlap,
        "target_uniform_overlap": target_baseline,
        "target_attention_lift": target_overlap / target_baseline if target_baseline > 0 else 0.0,
        "target_visible": bool(coverage.sum() > 0),
        "mask_relation_valid": False,
    }
    if num_streams == 2:
        maps["action_masked"] = action_mass[indices[1]]
        maps["decoder_masked"] = direct[indices[1]]
        maps["action_mass_masked"] = float(maps["action_masked"].sum())
        relation = np.zeros((height, width), dtype=np.float32)
        if maps["target_visible"]:
            query_weights = coverage.reshape(-1) / coverage.sum()
            cross_stream = encoder[encoder_layer][np.ix_(indices[1].ravel(), indices[0].ravel())]
            relation = (query_weights @ cross_stream).reshape(height, width)
            maps["mask_relation_valid"] = True
        maps["mask_to_original"] = relation
        maps["mask_to_original_mass"] = float(relation.sum())
    return maps
