"""
FlashAttentionDenoiserFourierRPE: Transformer denoiser with Fourier SE3 Relative Position Encoding.

Extends FlashAttentionDenoiser by enriching raw 6D relative SE3 transforms with
Fourier features (NeRF-style positional encoding) before projecting to per-head
attention bias. This provides the denoiser with high-frequency spatial representations
that help distinguish fine-grained geometric relationships between robot links
and between robot links and object patches.

Encoding: relative_se3 [6D] -> Fourier [6*(2*num_freqs+1)] -> MLP -> per-head bias

Reuses all core components (TransformerDenoiserLayer, FlashSelfAttention, etc.)
from flash_denoiser.py to avoid code duplication.
"""

import torch
import torch.nn as nn

from model.flash_denoiser import (
    NaiveMLP,
    TransformerDenoiserLayer,
)


class FourierSE3Encoding(nn.Module):
    """Sinusoidal positional encoding for SE3 relative transforms (NeRF-style).

    Maps a D-dimensional input to D*(2*num_freqs+1) dimensions using
    exponentially-spaced frequencies: [x, sin(2^0*x), cos(2^0*x), ...,
    sin(2^{L-1}*x), cos(2^{L-1}*x)].
    """

    def __init__(self, in_dim=6, num_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.out_dim = in_dim * (2 * num_freqs + 1)
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        """
        Args:
            x: [..., in_dim] raw SE3 relative transforms

        Returns:
            [..., in_dim * (2 * num_freqs + 1)] Fourier-encoded features
        """
        encoded = [x]
        for freq in self.freqs:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)


class FlashAttentionDenoiserFourierRPE(nn.Module):
    """Transformer denoiser with Fourier SE3 Relative Position Encoding.

    Same architecture as FlashAttentionDenoiser but enriches edge features
    with Fourier positional encoding before projecting to attention bias.
    """

    def __init__(
        self,
        # Initial Config
        M: int = 1000,
        object_patch: int = 25,
        max_link_node: int = 25,
        # Embedding
        t_embed_dim: int = 200,
        # Input Encoder
        V_object_dims: list = [3, 1, 64],
        V_robot_dims: list = [3, 3, 128],
        E_or_dims: list = [3, 3],
        E_rr_dims: list = [3, 3],
        # Backbone
        v_conv_dim: int = 384,
        num_heads: int = 16,
        ffn_dim: int = 1536,
        num_layers: int = 8,
        dropout: float = 0.0,
        # Output
        v_out_hidden_dims: list = [256, 128],
        se3_out_dim: list = [3, 3],
        # Fourier RPE
        fourier_num_freqs: int = 8,
        # Compat (accepted but unused)
        e_conv_dim: int = 384,
        c_atten_head: int = 32,
        e_out_hidden_dims: list = [256, 128],
    ):
        super().__init__()

        self.M = M
        self.object_patch = object_patch
        self.max_link_node = max_link_node
        self.v_conv_dim = v_conv_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # ---- Sinusoidal time embedding ----
        self.t_embed_dim = t_embed_dim
        self.register_buffer("t_embed", self._sinusoidal_embedding(M, t_embed_dim))

        # Time embedding MLP: project to d_model for AdaLN conditioning
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, v_conv_dim),
            nn.SiLU(),
            nn.Linear(v_conv_dim, v_conv_dim),
        )

        # ---- Node input encoders (multi-stream sum) ----
        self.V_object_dims = V_object_dims
        self.V_robot_dims = V_robot_dims
        self.V_object_in_layers = nn.ModuleList(
            [nn.Linear(d, v_conv_dim) for d in V_object_dims]
        )
        self.V_robot_in_layers = nn.ModuleList(
            [nn.Linear(d, v_conv_dim) for d in V_robot_dims]
        )

        # ---- Fourier SE3 encoding + edge bias projections ----
        self.E_or_dims = E_or_dims
        self.E_rr_dims = E_rr_dims
        e_rr_total = sum(E_rr_dims)  # 6
        e_or_total = sum(E_or_dims)  # 6

        self.fourier_encoding = FourierSE3Encoding(e_rr_total, fourier_num_freqs)
        fourier_dim = self.fourier_encoding.out_dim  # 6 * (2*8+1) = 102

        # Also create a separate Fourier encoder for OR edges if dims differ
        if e_or_total != e_rr_total:
            self.fourier_encoding_or = FourierSE3Encoding(e_or_total, fourier_num_freqs)
            fourier_dim_or = self.fourier_encoding_or.out_dim
        else:
            self.fourier_encoding_or = self.fourier_encoding
            fourier_dim_or = fourier_dim

        self.rr_bias_proj = nn.Sequential(
            nn.Linear(fourier_dim, num_heads * 4),
            nn.SiLU(),
            nn.Linear(num_heads * 4, num_heads),
        )
        self.or_bias_proj = nn.Sequential(
            nn.Linear(fourier_dim_or, num_heads * 4),
            nn.SiLU(),
            nn.Linear(num_heads * 4, num_heads),
        )

        # ---- Transformer layers ----
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerDenoiserLayer(
                    d_model=v_conv_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    cond_dim=v_conv_dim,
                )
            )

        # ---- Feature aggregation (wide FC) ----
        self.v_robot_wide_fc = nn.Linear(
            v_conv_dim * (1 + num_layers), v_conv_dim
        )

        # ---- Output MLPs ----
        self.se3_out_dim = se3_out_dim
        self.v_robot_output_module = nn.ModuleList()
        for f_dim in se3_out_dim:
            self.v_robot_output_module.append(
                NaiveMLP(
                    f_dim + v_conv_dim,
                    f_dim,
                    v_out_hidden_dims,
                )
            )

    @staticmethod
    def _sinusoidal_embedding(n, d):
        embedding = torch.zeros(n, d)
        wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
        wk = wk.reshape((1, d))
        t = torch.arange(n).reshape((n, 1))
        embedding[:, ::2] = torch.sin(t * wk[:, ::2])
        embedding[:, 1::2] = torch.cos(t * wk[:, ::2])
        return embedding

    def _encoder_(self, x, in_dims, in_layers):
        """Multi-stream projection fusion."""
        cur = 0
        feat = None
        for i, d in enumerate(in_dims):
            partial = x[..., cur : cur + d]
            out = in_layers[i](partial)
            feat = out if feat is None else feat + out
            cur += d
        return feat

    def forward(
        self,
        V_O,
        noisy_V_R,
        noisy_E_OR,
        noisy_E_RR,
        t,
        skip_or=False,
        return_features=False,
    ):
        """
        Args:
            V_O: [B, P, 68] object patch features
            noisy_V_R: [B, L, 134] noisy robot link features
            noisy_E_OR: [B, L, P, 6] robot-object relative SE(3) edges
            noisy_E_RR: [B, L, L, 6] robot-robot relative SE(3) edges
            t: [B] timestep indices
            skip_or: bool, skip cross-attention (for CFG unconditional pass)
            return_features: bool, also return intermediate features

        Returns:
            v_robot_pred: [B, L, 6] predicted noise/velocity
            (optional) robot_node_f: [B, L, d_model]
            (optional) object_node_f: [B, P, d_model]
        """
        # ---- Time embedding ----
        t_emb = self.t_embed[t]  # [B, t_embed_dim]
        t_cond = self.t_mlp(t_emb)  # [B, d_model]

        # ---- Encode inputs ----
        object_node_f = self._encoder_(V_O, self.V_object_dims, self.V_object_in_layers)
        robot_node_f = self._encoder_(noisy_V_R, self.V_robot_dims, self.V_robot_in_layers)

        # ---- Fourier-encoded edge bias ----
        rr_fourier = self.fourier_encoding(noisy_E_RR)    # [B, L, L, fourier_dim]
        rr_bias = self.rr_bias_proj(rr_fourier)            # [B, L, L, num_heads]
        rr_bias = rr_bias.permute(0, 3, 1, 2)              # [B, num_heads, L, L]

        or_bias = None
        if not skip_or:
            or_fourier = self.fourier_encoding_or(noisy_E_OR)  # [B, L, P, fourier_dim]
            or_bias = self.or_bias_proj(or_fourier)             # [B, L, P, num_heads]
            or_bias = or_bias.permute(0, 3, 1, 2)              # [B, num_heads, L, P]

        # ---- Transformer layers ----
        robot_f_list = [robot_node_f]

        for layer in self.layers:
            robot_node_f = layer(
                robot_node_f,
                object_node_f,
                t_cond,
                rr_bias=rr_bias,
                or_bias=or_bias,
                skip_or=skip_or,
            )
            robot_f_list.append(robot_node_f)

        # ---- Feature aggregation (wide FC) ----
        update_robot_node_f = self.v_robot_wide_fc(
            torch.cat(robot_f_list, dim=-1)
        )

        # ---- Output MLPs ----
        v_robot_pred = []
        cur = 0
        for i, f_dim in enumerate(self.se3_out_dim):
            inp = torch.cat(
                [update_robot_node_f, noisy_V_R[:, :, cur : cur + f_dim]], dim=-1
            )
            v_robot_pred.append(self.v_robot_output_module[i](inp))
            cur += f_dim
        v_robot_pred = torch.cat(v_robot_pred, dim=-1)

        if return_features:
            return v_robot_pred, update_robot_node_f, object_node_f
        return v_robot_pred
