# -*- coding: utf-8 -*-
"""RT-DETR detection head used by FireRGBTNet.

Sec. 2.2.1 ("Head"). FireRGBTNet adopts the RT-DETR set-prediction paradigm, so
no anchors and no NMS post-processing are involved. This is a compact
re-implementation with hidden dim 128, 2 decoder layers and 300 object queries,
plus an IoU-aware quality branch whose score is fused into the classification
confidence at inference time.

The cross-attention uses multi-scale deformable attention (pure PyTorch
``grid_sample`` implementation, no custom CUDA kernels required).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import MishGLU

__all__ = ["RTDETRDecoder"]


def inverse_sigmoid(x, eps=1e-3):
    """Inverse of the sigmoid function with clamping for numerical stability."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return (torch.log(x1 / x2)).clamp(min=-10, max=10)


class MLP(nn.Module):
    """Simple multi-layer perceptron with ReLU on all but the last layer."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# Multi-scale deformable attention
# ---------------------------------------------------------------------------
def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Pure-PyTorch multi-scale deformable attention.

    Args:
        value: ``[bs, sum(H_l*W_l), n_head, C//n_head]``
        value_spatial_shapes: ``[n_levels, 2]`` tensor of ``(H_l, W_l)``
        sampling_locations: ``[bs, Len_q, n_head, n_levels, n_points, 2]`` in ``[0, 1]``
        attention_weights: ``[bs, Len_q, n_head, n_levels, n_points]``

    Returns:
        ``[bs, Len_q, n_head * C//n_head]``
    """
    bs, _, n_head, c = value.shape
    _, len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)

    # [0, 1] -> [-1, 1] for grid_sample
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(bs * n_head, c, h, w)
        sampling_grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

        sampling_value_list.append(
            F.grid_sample(
                value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        )

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * n_head, 1, len_q, n_levels * n_points
    )
    output = (
        torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
    ).sum(-1).view(bs, n_head * c, len_q)

    return output.transpose(1, 2).contiguous()


class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention (pure PyTorch)."""

    def __init__(self, embed_dim=256, num_heads=8, num_levels=3, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]

        value = self.value_proj(value).view(bs, len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, len_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, len_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = (
            F.softmax(attention_weights.float(), -1)
            .type_as(query)
            .view(bs, len_q, self.num_heads, self.num_levels, self.num_points)
        )

        if reference_points.shape[-1] != 2:
            raise ValueError("Reference points should be (x, y).")
        offset_normalizer = torch.stack(
            [value_spatial_shapes[..., 1], value_spatial_shapes[..., 0]], -1
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )

        output = multi_scale_deformable_attn_pytorch(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)


# ---------------------------------------------------------------------------
# Positional encodings and decoder layer
# ---------------------------------------------------------------------------
def gen_sineembed_for_coords(coords, num_pos_feats=128, temperature=10000, scale=2 * math.pi):
    """Sine positional embedding for normalized ``(x, y)`` coordinates."""
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=coords.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    x_embed = coords[:, :, 0] * scale
    y_embed = coords[:, :, 1] * scale

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t

    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    return torch.cat((pos_y, pos_x), dim=2)


class PositionEmbeddingSine(nn.Module):
    """2-D sine positional embedding for multi-scale feature maps."""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x):
        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class RTDETRDecoderLayer(nn.Module):
    """One decoder layer: self-attention -> deformable cross-attention -> FFN."""

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.0, n_levels=3, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = MSDeformableAttention(d_model, nhead, n_levels, n_points)
        # The decoder FFN uses a 3x3 depthwise kernel (the released configuration).
        self.ffn = MishGLU(d_model, dim_feedforward, dw_kernel=3)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, reference_points, spatial_shapes, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt = self.norm1(tgt + self.dropout1(self.self_attn(q, k, value=tgt, need_weights=False)[0]))

        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, query_pos),
            reference_points=reference_points,
            value=memory,
            value_spatial_shapes=spatial_shapes,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # MishGLU operates on [B, C, H, W]; feed the queries as H x W = 1 x Q.
        tgt2 = self.ffn(tgt.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)
        return self.norm3(tgt + self.dropout3(tgt2))


class RTDETRDecoder(nn.Module):
    """RT-DETR decoder with IoU-aware prediction heads.

    Args:
        num_classes: number of foreground classes (no background class here;
            the extra logit slot is the background/no-object slot).
        hidden_dim: decoder hidden dimension (128 in the paper).
        num_queries: number of object queries (300 in the paper).
        nhead: attention heads.
        num_decoder_layers: number of decoder layers (2 in the paper).
        in_channels: channels of the ``(P3, P4, P5)`` neck outputs. All levels must
            already match ``hidden_dim``.
    """

    def __init__(
        self,
        num_classes,
        hidden_dim=128,
        num_queries=300,
        nhead=8,
        num_decoder_layers=2,
        in_channels=(128, 128, 128),
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_levels = len(in_channels)

        # Levels are expected to already be projected to hidden_dim by the neck.
        self.input_proj = nn.ModuleList([nn.Identity() for _ in in_channels])

        self.pos_generator = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True)

        # Encoder prediction head used for query selection.
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes + 1),
        )
        self.pos_trans = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            [
                RTDETRDecoderLayer(
                    hidden_dim,
                    nhead,
                    dim_feedforward=hidden_dim * 2,
                    dropout=0.0,
                    n_levels=self.num_levels,
                    n_points=4,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.iou_head = MLP(hidden_dim, hidden_dim, 1, 3)

        self._reset_parameters()

    def _reset_parameters(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        if hasattr(self.enc_output[-1], "bias"):
            self.enc_output[-1].bias.data = torch.ones(self.num_classes + 1) * bias_value
        self.class_head.bias.data = torch.ones(self.num_classes + 1) * bias_value

        nn.init.constant_(self.bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_head.layers[-1].bias.data, 0)
        nn.init.constant_(self.iou_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.iou_head.layers[-1].bias.data, 0)

    def forward(self, feats):
        """Args:
            feats: list of neck feature maps ``[P3, P4, P5]``.

        Returns:
            Training: dict with ``pred_logits``/``pred_boxes``/``pred_ious`` plus
            ``aux_outputs`` and ``enc_outputs`` for deep supervision.
            Inference: only the last layer's predictions.
        """
        bs = feats[0].shape[0]

        proj_feats = []
        spatial_shapes = []
        for feat, layer in zip(feats, self.input_proj):
            proj_feats.append(layer(feat))
            spatial_shapes.append(feat.shape[-2:])
        spatial_shapes = torch.tensor(
            spatial_shapes, device=proj_feats[0].device, dtype=torch.long
        )

        srcs = [feat.flatten(2).transpose(1, 2) for feat in proj_feats]
        memory = torch.cat(srcs, dim=1)

        # ---- Query selection -------------------------------------------------
        enc_outputs_class = self.enc_output(memory)
        topk_score, topk_inds = torch.topk(
            enc_outputs_class[..., :-1].max(-1)[0], self.num_queries, dim=1
        )

        batch_idx = torch.arange(bs, device=memory.device).unsqueeze(1)
        tgt = memory[batch_idx, topk_inds]

        grid_list = []
        for h, w in spatial_shapes:
            y, x = torch.meshgrid(
                torch.arange(h, device=memory.device),
                torch.arange(w, device=memory.device),
                indexing="ij",
            )
            grid = torch.stack((x, y), -1).float()
            grid = (grid + 0.5) / torch.tensor([w, h], device=memory.device)
            grid_list.append(grid.flatten(0, 1))

        flattened_grid = torch.cat(grid_list, dim=0).unsqueeze(0).repeat(bs, 1, 1)
        topk_coords = flattened_grid[batch_idx, topk_inds]

        query_pos = self.pos_trans(
            gen_sineembed_for_coords(topk_coords, num_pos_feats=self.hidden_dim // 2)
        )

        # ---- Encoder box prediction (initial anchors) -------------------------
        enc_bbox_embed = self.bbox_head(tgt)
        enc_bbox_embed[..., :2] += inverse_sigmoid(topk_coords)
        enc_outputs_coord = enc_bbox_embed.sigmoid()

        ref_points = enc_outputs_coord.detach()
        enc_outputs_class_selected = enc_outputs_class[batch_idx, topk_inds]

        # ---- Decoder loop with iterative box refinement ----------------------
        output = tgt
        outputs_class_list = []
        outputs_coord_list = []
        outputs_iou_list = []

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points[..., :2].unsqueeze(2).repeat(1, 1, self.num_levels, 1)

            output = layer(
                tgt=output,
                memory=memory,
                reference_points=ref_points_input,
                spatial_shapes=spatial_shapes,
                query_pos=query_pos,
            )

            tmp_box_delta = self.bbox_head(output)
            ref_points_inv = inverse_sigmoid(ref_points)
            new_ref_points = (ref_points_inv + tmp_box_delta).sigmoid()
            new_ref_points = torch.clamp(new_ref_points, min=1e-4, max=1.0 - 1e-4)
            outputs_coord_list.append(new_ref_points)
            ref_points = new_ref_points.detach()

            if self.training or i == len(self.layers) - 1:
                outputs_class_list.append(self.class_head(output))
                outputs_iou_list.append(self.iou_head(output))

        if not self.training:
            return {
                "pred_logits": outputs_class_list[-1],
                "pred_boxes": outputs_coord_list[-1],
                "pred_ious": outputs_iou_list[-1],
            }

        outputs_class = torch.stack(outputs_class_list)
        outputs_coord = torch.stack(outputs_coord_list)
        outputs_iou = torch.stack(outputs_iou_list)

        return {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_ious": outputs_iou[-1],
            "aux_outputs": self._set_aux_loss(outputs_class, outputs_coord, outputs_iou),
            "enc_outputs": {
                "pred_logits": enc_outputs_class_selected,
                "pred_boxes": enc_outputs_coord,
            },
        }

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_iou):
        """Per-layer predictions used for deep supervision."""
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_ious": c}
            for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_iou[:-1])
        ]
