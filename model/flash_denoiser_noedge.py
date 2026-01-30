"""
FlashAttentionDenoiserNoEdge: Transformer denoiser WITHOUT edge (SE3) bias.

Ablation variant of FlashAttentionDenoiser. Removes pairwise relative SE(3)
attention bias; the model learns spatial relationships purely from absolute
poses in node features via Q@K^T.

Differences from FlashAttentionDenoiser:
1. No rr_bias_proj / or_bias_proj parameters
2. forward() does not accept noisy_E_OR / noisy_E_RR
3. Transformer layers receive rr_bias=None, or_bias=None
"""

import torch
import torch.nn as nn

from model.flash_denoiser import NaiveMLP, TransformerDenoiserLayer


class FlashAttentionDenoiserNoEdge(nn.Module):
    """Denoiser without edge features -- clean ablation of FlashAttentionDenoiser."""

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
        # Backbone
        v_conv_dim: int = 384,
        num_heads: int = 16,
        ffn_dim: int = 1536,
        num_layers: int = 6,
        dropout: float = 0.0,
        # Output
        v_out_hidden_dims: list = [256, 128],
        se3_out_dim: list = [3, 3],
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

        # ---- Node input encoders (multi-stream sum, same as FlashAttentionDenoiser) ----
        self.V_object_dims = V_object_dims
        self.V_robot_dims = V_robot_dims
        self.V_object_in_layers = nn.ModuleList(
            [nn.Linear(d, v_conv_dim) for d in V_object_dims]
        )
        self.V_robot_in_layers = nn.ModuleList(
            [nn.Linear(d, v_conv_dim) for d in V_robot_dims]
        )

        # NOTE: No edge bias projections (rr_bias_proj / or_bias_proj) --
        # this is the key ablation difference from FlashAttentionDenoiser.

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
        """Multi-stream projection fusion (same as FlashAttentionDenoiser)."""
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
        t,
        skip_or=False,
        return_features=False,
    ):
        """
        Args:
            V_O: [B, P, 68] object patch features
            noisy_V_R: [B, L, 134] noisy robot link features
            t: [B] timestep indices
            skip_or: bool, skip cross-attention
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

        # NOTE: No edge bias -- pure Q@K^T attention

        # ---- Transformer layers ----
        robot_f_list = [robot_node_f]

        for layer in self.layers:
            robot_node_f = layer(
                robot_node_f,
                object_node_f,
                t_cond,
                rr_bias=None,
                or_bias=None,
                skip_or=skip_or,
            )
            robot_f_list.append(robot_node_f)

        # ---- Feature aggregation (wide FC) ----
        update_robot_node_f = self.v_robot_wide_fc(
            torch.cat(robot_f_list, dim=-1)  # [B, L, d_model * (1 + num_layers)]
        )  # [B, L, d_model]

        # ---- Output MLPs ----
        v_robot_pred = []
        cur = 0
        for i, f_dim in enumerate(self.se3_out_dim):
            inp = torch.cat(
                [update_robot_node_f, noisy_V_R[:, :, cur : cur + f_dim]], dim=-1
            )
            v_robot_pred.append(self.v_robot_output_module[i](inp))
            cur += f_dim
        v_robot_pred = torch.cat(v_robot_pred, dim=-1)  # [B, L, 6]

        if return_features:
            return v_robot_pred, update_robot_node_f, object_node_f
        return v_robot_pred
