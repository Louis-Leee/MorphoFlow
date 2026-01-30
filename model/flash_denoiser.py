"""
FlashAttentionDenoiser: Standard transformer denoiser with Flash Attention.

Drop-in replacement for GraphDenoiser with identical interface.
Uses F.scaled_dot_product_attention for efficient attention computation.
Edge features (SE3 relative transforms) encoded as attention bias.
Time conditioning via AdaLN-Zero (DiT style).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveMLP(nn.Module):
    """MLP with skip connections (same as denoiser.py)."""

    def __init__(self, z_dim, out_dim, hidden_dims):
        super().__init__()
        c_wide_out_cnt = z_dim
        self.layers = nn.ModuleList()
        c_in = z_dim
        for c_out in hidden_dims:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(c_in, c_out),
                    nn.LayerNorm(c_out),
                    nn.LeakyReLU(),
                )
            )
            c_wide_out_cnt += c_out
            c_in = c_out
        self.out_fc0 = nn.Linear(c_wide_out_cnt, out_dim)

    def forward(self, x):
        f_list = [x]
        for layer in self.layers:
            x = layer(x)
            f_list.append(x)
        f = torch.cat(f_list, -1)
        return self.out_fc0(f)


class AdaLNZero(nn.Module):
    """Adaptive LayerNorm with zero-initialized gate (DiT style).

    Given conditioning vector c, produces (scale, shift, gate) to modulate:
        output = gate * sublayer(LN(x) * (1 + scale) + shift)
    """

    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # Produces 3 * d_model: (scale, shift, gate)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * d_model),
        )
        # Zero-init the gate so initial output is near-zero (residual stream dominant)
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, x, cond):
        """
        Args:
            x: [B, N, d_model] input features
            cond: [B, d_model] conditioning vector (time embedding)

        Returns:
            normed: [B, N, d_model] modulated features
            gate: [B, 1, d_model] gate values
        """
        scale, shift, gate = self.proj(cond).chunk(3, dim=-1)  # each [B, d_model]
        scale = scale.unsqueeze(1)  # [B, 1, d_model]
        shift = shift.unsqueeze(1)
        gate = gate.unsqueeze(1)
        normed = self.norm(x) * (1.0 + scale) + shift
        return normed, gate


class FlashSelfAttention(nn.Module):
    """Multi-head self-attention with optional additive bias.

    Uses F.scaled_dot_product_attention for Flash Attention support.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        """
        Args:
            x: [B, N, d_model]
            attn_bias: [B, num_heads, N, N] additive bias for attention logits
            key_padding_mask: [B, N] True for valid positions (False = padding)

        Returns:
            [B, N, d_model]
        """
        B, N, _ = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)  # each [B, heads, N, head_dim]

        # Build attention mask combining bias and padding
        attn_mask = self._build_mask(attn_bias, key_padding_mask, B, N, x.device, x.dtype)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.out_proj(out)

    def _build_mask(self, attn_bias, key_padding_mask, B, N, device, dtype):
        """Combine attn_bias and key_padding_mask into a single mask."""
        if attn_bias is None and key_padding_mask is None:
            return None

        mask = torch.zeros(B, self.num_heads, N, N, device=device, dtype=dtype)

        if attn_bias is not None:
            mask = mask + attn_bias

        if key_padding_mask is not None:
            # key_padding_mask: [B, N], True = valid, False = padding
            # Need to set padding positions in key dimension to -inf
            pad_mask = ~key_padding_mask  # True = padding
            pad_mask = pad_mask[:, None, None, :]  # [B, 1, 1, N]
            mask = mask.masked_fill(pad_mask, float("-inf"))

        return mask


class FlashCrossAttention(nn.Module):
    """Multi-head cross-attention with optional additive bias.

    Query from decoder (robot), Key/Value from encoder (object).
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, context, attn_bias=None):
        """
        Args:
            query: [B, L, d_model] (robot node features)
            context: [B, P, d_model] (object node features)
            attn_bias: [B, num_heads, L, P] additive bias

        Returns:
            [B, L, d_model]
        """
        B, L, _ = query.shape
        P = context.shape[1]

        q = self.q_proj(query).reshape(B, L, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [B, heads, L, head_dim]

        kv = self.kv_proj(context).reshape(B, P, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, heads, P, head_dim]
        k, v = kv.unbind(0)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Standard FFN with GELU activation."""

    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerDenoiserLayer(nn.Module):
    """One transformer block: AdaLN + SelfAttn + AdaLN + CrossAttn + AdaLN + FFN."""

    def __init__(self, d_model, num_heads, ffn_dim, cond_dim):
        super().__init__()
        # Self-attention (RR)
        self.adaln_self = AdaLNZero(d_model, cond_dim)
        self.self_attn = FlashSelfAttention(d_model, num_heads)

        # Cross-attention (OR)
        self.adaln_cross = AdaLNZero(d_model, cond_dim)
        self.cross_attn = FlashCrossAttention(d_model, num_heads)

        # Feed-forward
        self.adaln_ffn = AdaLNZero(d_model, cond_dim)
        self.ffn = FeedForward(d_model, ffn_dim)

    def forward(
        self,
        robot_f,       # [B, L, d_model]
        object_f,      # [B, P, d_model]
        t_cond,        # [B, d_model]
        rr_bias=None,  # [B, heads, L, L]
        or_bias=None,  # [B, heads, L, P]
        key_padding_mask=None,  # [B, L]
        skip_or=False,
    ):
        # Self-attention (robot links attend to each other)
        normed, gate = self.adaln_self(robot_f, t_cond)
        robot_f = robot_f + gate * self.self_attn(
            normed, attn_bias=rr_bias, key_padding_mask=key_padding_mask
        )

        # Cross-attention (robot links attend to object patches)
        if not skip_or:
            normed, gate = self.adaln_cross(robot_f, t_cond)
            robot_f = robot_f + gate * self.cross_attn(
                normed, object_f, attn_bias=or_bias
            )

        # Feed-forward
        normed, gate = self.adaln_ffn(robot_f, t_cond)
        robot_f = robot_f + gate * self.ffn(normed)

        return robot_f


class FlashAttentionDenoiser(nn.Module):
    """Drop-in replacement for GraphDenoiser using standard transformer attention.

    Same input/output interface as GraphDenoiser for compatibility.
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
        num_layers: int = 6,
        dropout: float = 0.0,
        # Output
        v_out_hidden_dims: list = [256, 128],
        se3_out_dim: list = [3, 3],
        # Compat (accepted but unused by flash denoiser)
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

        # ---- Node input encoders (multi-stream sum, same as GraphDenoiser) ----
        self.V_object_dims = V_object_dims
        self.V_robot_dims = V_robot_dims
        self.V_object_in_layers = nn.ModuleList(
            [nn.Linear(d, v_conv_dim) for d in V_object_dims]
        )
        self.V_robot_in_layers = nn.ModuleList(
            [nn.Linear(d, v_conv_dim) for d in V_robot_dims]
        )

        # ---- Edge bias projections ----
        # Project raw SE(3) edge features to per-head attention bias
        self.E_or_dims = E_or_dims
        self.E_rr_dims = E_rr_dims
        e_or_total = sum(E_or_dims)  # 6
        e_rr_total = sum(E_rr_dims)  # 6
        self.rr_bias_proj = nn.Sequential(
            nn.Linear(e_rr_total, num_heads * 2),
            nn.SiLU(),
            nn.Linear(num_heads * 2, num_heads),
        )
        self.or_bias_proj = nn.Sequential(
            nn.Linear(e_or_total, num_heads * 2),
            nn.SiLU(),
            nn.Linear(num_heads * 2, num_heads),
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

        # ---- Feature aggregation (wide FC, same pattern as GraphDenoiser) ----
        self.v_robot_wide_fc = nn.Linear(
            v_conv_dim * (1 + num_layers), v_conv_dim
        )

        # ---- Output MLPs (same as GraphDenoiser) ----
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
        """Multi-stream projection fusion (same as GraphDenoiser)."""
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
            skip_or: bool, skip cross-attention (for Stage 1)
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

        # ---- Edge bias ----
        rr_bias = self.rr_bias_proj(noisy_E_RR)  # [B, L, L, num_heads]
        rr_bias = rr_bias.permute(0, 3, 1, 2)    # [B, num_heads, L, L]

        or_bias = None
        if not skip_or:
            or_bias = self.or_bias_proj(noisy_E_OR)  # [B, L, P, num_heads]
            or_bias = or_bias.permute(0, 3, 1, 2)    # [B, num_heads, L, P]

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
