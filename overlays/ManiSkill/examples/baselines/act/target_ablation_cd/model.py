"""Isolated ACT models for centroid (C) and binary-mask (D) ablations.

C adds one centroid token to the fusion encoder while retaining the original
28-dimensional state and CVAE posterior. D uses the existing dual-image ACT
path with a three-channel binary image as its second stream. Both deliberately
retain the existing selection of the first intermediate decoder output.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T

from act.detr.backbone import build_backbone
from act.detr.detr_vae import DETRVAE, build_encoder, reparametrize
from act.detr.transformer import build_transformer


@dataclass
class ModelConfig:
    condition: str = 'C'
    backbone: str = 'resnet18'
    hidden_dim: int = 256
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    nheads: int = 8
    dropout: float = 0.1
    num_queries: int = 30
    pre_norm: bool = False
    kl_weight: float = 10.0
    lr_backbone: float = 1e-5
    position_embedding: str = 'sine'
    masks: bool = False
    dilation: bool = False
    include_depth: bool = False

    def __post_init__(self):
        if self.condition not in ('C', 'D'):
            raise ValueError('condition must be C (centroid) or D (binary mask)')
        if self.include_depth:
            raise ValueError('C/D ablations use RGB inputs without depth')
        if self.hidden_dim < 1 or self.nheads < 1 or self.hidden_dim % self.nheads:
            raise ValueError('hidden_dim must be positive and divisible by nheads')
        if min(self.enc_layers, self.dec_layers, self.dim_feedforward, self.num_queries) < 1:
            raise ValueError('Layer counts, feedforward dimension and query count must be positive')
        if not 0 <= self.dropout < 1 or self.kl_weight < 0:
            raise ValueError('Expected dropout in [0, 1) and nonnegative KL weight')


class ConditionalDETRVAE(DETRVAE):
    """Preserve the base ACT parameters and add C's target modules last.

    C fusion ordering is [latent, state, centroid, original image tokens].
    Centroid content is projected from normalized image coordinates; its
    learned position embedding identifies the token's role. An invisible
    target uses a separate learned missing token, independent of coordinates.
    D registers no additional parameters, preserving base DETRVAE state keys.
    """

    def __init__(self, backbones, transformer, encoder, config):
        super().__init__(backbones, transformer, encoder, state_dim=28,
                         action_dim=8, num_queries=config.num_queries)
        self.condition = config.condition
        self.decoder_output_layer = 0
        if self.condition == 'C':
            # Initialize only after every shared ACT parameter has been built.
            self.center_proj = nn.Linear(2, transformer.d_model)
            self.center_pos = nn.Embedding(1, transformer.d_model)
            self.missing_target = nn.Embedding(1, transformer.d_model)

    def forward(self, obs, actions=None):
        if self.condition == 'D':
            return super().forward(obs, actions)

        state = obs['state']
        batch = state.shape[0]
        if actions is not None:
            cls = self.cls_embed.weight.unsqueeze(0).repeat(batch, 1, 1)
            state_embed = self.encoder_state_proj(state).unsqueeze(1)
            action_embed = self.encoder_action_proj(actions)
            posterior_inputs = torch.cat([cls, state_embed, action_embed], dim=1).permute(1, 0, 2)
            padding = torch.full((batch, posterior_inputs.shape[0]), False, device=state.device)
            posterior_pos = self.pos_table.clone().detach().permute(1, 0, 2)
            posterior = self.encoder(posterior_inputs, pos=posterior_pos, src_key_padding_mask=padding)
            latent_info = self.latent_proj(posterior[0])
            mu, logvar = latent_info[:, :self.latent_dim], latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
        else:
            mu = logvar = None
            latent_sample = torch.zeros((batch, self.latent_dim), dtype=torch.float32, device=state.device)
        latent = self.latent_out_proj(latent_sample)
        proprio = self.input_proj_robot_state(state)

        features, positions = self.backbones[0](obs['rgb'][:, 0])
        visual = self.input_proj(features[0]).flatten(2).permute(2, 0, 1)
        visual_pos = positions[0].flatten(2).permute(2, 0, 1).repeat(1, batch, 1)
        projected_center = self.center_proj(obs['target_xy'])
        missing = self.missing_target.weight.expand(batch, -1)
        center = torch.where(obs['target_valid'][:, None], projected_center, missing)
        prefix = torch.stack([latent, proprio, center], dim=0)
        prefix_pos = torch.cat([self.additional_pos_embed.weight, self.center_pos.weight], dim=0)
        prefix_pos = prefix_pos.unsqueeze(1).repeat(1, batch, 1)
        source = torch.cat([prefix, visual], dim=0)
        position = torch.cat([prefix_pos, visual_pos], dim=0)
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, batch, 1)
        memory = self.transformer.encoder(source, src_key_padding_mask=None, pos=position)
        decoded = self.transformer.decoder(torch.zeros_like(queries), memory,
                                           memory_key_padding_mask=None, pos=position, query_pos=queries)
        hidden = decoded.transpose(1, 2)[self.decoder_output_layer]
        return self.action_head(hidden), [mu, logvar]


class ACTPolicy(nn.Module):
    """ACT policy accepting already-normalized state and raw uint8 images.

    ``obs['rgb']`` is [B, S, 3, 224, 224], with S=1 for C or S=2 for D.
    ``obs['state']`` is [B, 28] normalized using training statistics. C also
    requires ``target_xy`` [B, 2] in [-1, 1] and ``target_valid`` [B] bool.
    The D binary stream must be encoded as uint8 0/255 by the data adapter.
    Inputs are never modified in place. Outputs are normalized [B, Q, 8]
    action chunks; action statistics and temporal execution live outside.
    """

    def __init__(self, config):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError('config must be a ModelConfig')
        self.config = config
        self.state_dim = 28
        self.act_dim = 8
        self.kl_weight = config.kl_weight
        self.include_depth = False
        self.num_streams = 1 if config.condition == 'C' else 2
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Match original Agent construction order for identical shared weights.
        backbones = [build_backbone(config)]
        transformer = build_transformer(config)
        encoder = build_encoder(config)
        self.model = ConditionalDETRVAE(backbones, transformer, encoder, config)

    def _inputs(self, obs):
        rgb, state = obs['rgb'], obs['state']
        if rgb.dtype != torch.uint8 or rgb.ndim != 5 or tuple(rgb.shape[1:]) != (self.num_streams, 3, 224, 224):
            raise ValueError(f'Expected uint8 RGB with shape [B, {self.num_streams}, 3, 224, 224]')
        if state.shape != (rgb.shape[0], 28) or not state.is_floating_point():
            raise ValueError('Expected normalized floating-point state with shape [B, 28]')
        if state.device != rgb.device:
            raise ValueError('RGB and state must be on the same device')
        inputs = {'rgb': self.normalize(rgb.float() / 255.0), 'state': state}
        if self.config.condition == 'C':
            center, valid = obs['target_xy'], obs['target_valid']
            if center.shape != (rgb.shape[0], 2) or not center.is_floating_point():
                raise ValueError('Expected target_xy with shape [B, 2]')
            if valid.shape != (rgb.shape[0],) or valid.dtype != torch.bool:
                raise ValueError('Expected boolean target_valid with shape [B]')
            if center.device != rgb.device or valid.device != rgb.device:
                raise ValueError('Target coordinates, validity and RGB must be on the same device')
            if not torch.isfinite(center).all() or (center.abs() > 1).any():
                raise ValueError('target_xy must contain finite coordinates in [-1, 1]')
            inputs.update(target_xy=center, target_valid=valid)
        return inputs

    def get_action(self, obs):
        prediction, _ = self.model(self._inputs(obs))
        return prediction

    def compute_loss(self, obs, normalized_actions):
        inputs = self._inputs(obs)
        expected = (inputs['state'].shape[0], self.config.num_queries, 8)
        if normalized_actions.shape != expected or not normalized_actions.is_floating_point():
            raise ValueError(f'Expected normalized action chunks with shape {expected}')
        prediction, (mu, logvar) = self.model(inputs, normalized_actions)
        l1 = F.l1_loss(prediction, normalized_actions, reduction='mean')
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(dim=1).mean()
        return {'loss': l1 + self.kl_weight * kl, 'l1': l1, 'kl': kl}
