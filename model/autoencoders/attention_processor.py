from diffusers.models.attention_processor import Attention
import flash_attn
import torch

try:
    import flash_attn_interface

    flash_attn = flash_attn_interface
    print("Using flash attn 3")
except ImportError:
    print("Using flash attn", flash_attn.__version__)

class CragVarlenFlashAttentionProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_k: int | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ):
        assert cu_seqlens_q is not None, "Must provide cu_seqlens_q for query"
        assert max_seqlen_q is not None, "Must provide max_seqlen_q for query"
        assert attention_mask is None, "Attention mask not supported for variable length attention"
        if encoder_hidden_states is not None:
            assert cu_seqlens_k is not None, "Must provide cu_seqlens_k for key/value"
            assert max_seqlen_k is not None, "Must provide max_seqlen_k for key/value"

        residual = hidden_states

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            cu_seqlens_k = cu_seqlens_q
            max_seqlen_k = max_seqlen_q
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(-1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(-1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(-1, attn.heads, head_dim)
        key = key.view(-1, attn.heads, head_dim)
        value = value.view(-1, attn.heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query).to(query.dtype)
        if attn.norm_k is not None:
            key = attn.norm_k(key).to(key.dtype)

        hidden_states = flash_attn.flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
        )

        hidden_states = hidden_states.reshape(-1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor
