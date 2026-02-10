import numpy as np
import torch
from torch_cluster import fps

from .attention_processor import CragVarlenFlashAttentionProcessor
from third_party.tripo_sg.triposg.models.attention_processor import TripoSGAttnProcessor2_0
from third_party.tripo_sg.triposg.models.autoencoders import TripoSGVAEModel
from third_party.tripo_sg.triposg.models.autoencoders.autoencoder_kl_triposg import TripoSGEncoder
from third_party.tripo_sg.triposg.models.autoencoders.vae import DiagonalGaussianDistribution


class TripoSGVarlenEncoder(TripoSGEncoder):
    def set_varlen_attn_processor(self):
        for block in self.blocks:
            for attn in block.modules():
                if hasattr(attn, "set_processor"):
                    attn.set_processor(CragVarlenFlashAttentionProcessor())

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        num_tokens_q: torch.Tensor,
        num_tokens_kv: torch.Tensor,
    ):
        max_seqlen_q = num_tokens_q.max().item()
        max_seqlen_kv = num_tokens_kv.max().item()
        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(num_tokens_q, dim=0), (1, 0), value=0).int()
        cu_seqlens_kv = torch.nn.functional.pad(torch.cumsum(num_tokens_kv, dim=0), (1, 0), value=0).int()

        hidden_states = self.proj_in(x_q)
        encoder_hidden_states = self.proj_in(x_kv)

        for layer, block in enumerate(self.blocks):
            if layer == 0:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    attention_kwargs={
                        "cu_seqlens_q": cu_seqlens_q,
                        "max_seqlen_q": max_seqlen_q,
                        "cu_seqlens_k": cu_seqlens_kv,
                        "max_seqlen_k": max_seqlen_kv,
                    },
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_kwargs={
                        "cu_seqlens_q": cu_seqlens_q,
                        "max_seqlen_q": max_seqlen_q,
                    },
                )

        return self.norm_out(hidden_states), {
            "cu_seqlens_q": cu_seqlens_q,
            "max_seqlen_q": max_seqlen_q,
            "cu_seqlens_kv": cu_seqlens_kv,
            "max_seqlen_kv": max_seqlen_kv,
        }


class TripoSGVarlenVAEModel(TripoSGVAEModel):
    def __init__(
        self,
        *,
        in_channels=3,
        latent_channels=64,
        num_attention_heads=8,
        width_encoder=512,
        width_decoder=1024,
        num_layers_encoder=8,
        num_layers_decoder=16,
        embedding_type="frequency",
        embed_frequency=8,
        embed_include_pi=False,
    ):
        super().__init__(
            in_channels,
            latent_channels,
            num_attention_heads,
            width_encoder,
            width_decoder,
            num_layers_encoder,
            num_layers_decoder,
            embedding_type,
            embed_frequency,
            embed_include_pi,
        )

        self.encoder = TripoSGVarlenEncoder(
            in_channels=in_channels + self.embedder.out_dim,
            dim=width_encoder,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers_encoder,
        )

        self.encoder.set_varlen_attn_processor()
        self.set_varlen_decoder()

    def encode(self, x: torch.Tensor, num_tokens: torch.Tensor, ratio: float = 0.25, *, use_decoded_features: bool = False):
        position_channels = self.config.get("in_channels", 3)
        position, features = x[..., :position_channels], x[..., position_channels:]
        x_kv = torch.cat([self.embedder(position), features], dim=-1)

        sampled_indices = fps(
            src=position.float(),
            batch=torch.arange(
                len(num_tokens),
                device=num_tokens.device,
            ).repeat_interleave(num_tokens),
            ratio=ratio,
        )
        x_q = torch.cat([self.embedder(position[sampled_indices]), features[sampled_indices]], dim=-1)

        x, seqlen = self.encoder(
            x_q,
            x_kv,
            torch.ceil(num_tokens * ratio).int(),
            num_tokens,
        )
        h = self.quant(x)
        posterior = DiagonalGaussianDistribution(h, feature_dim=-1)

        z = posterior.sample()
        if not use_decoded_features:
            return x, z, sampled_indices, (num_tokens * ratio).ceil().int()

        feat = self.post_quant(z)
        for block in self.decoder.blocks[:-1]:
            feat = block(
                feat,
                attention_kwargs={
                    "cu_seqlens_q": seqlen["cu_seqlens_q"],
                    "max_seqlen_q": seqlen["max_seqlen_q"],
                },
            )

        return (feat, z, sampled_indices, (num_tokens * ratio).ceil().int())

    def encode_shape(
        self,
        x: torch.Tensor,
        num_tokens: int,
        batch_size: int,
        seed: int = 42,
        use_decoded_features: bool = False,
        
    ):
        x = x.reshape(batch_size, -1, x.shape[-1])
        rng = np.random.default_rng(seed)
        indices = rng.choice(x.shape[1], num_tokens * 4, replace=num_tokens * 4 > x.shape[1])
        selected_points = x[:, indices].reshape(-1, x.shape[-1])

        position_channels = self.config.get("in_channels", 3)
        position, features = selected_points[..., :position_channels], selected_points[..., position_channels:]
        x_kv = torch.cat([self.embedder(position), features], dim=-1)

        sampled_indices = fps(
            src=position.float(),
            batch=torch.arange(
                batch_size,
                device=position.device,
            ).repeat_interleave(num_tokens * 4),
            ratio=1 / 4.0,
        )
        x_q = torch.cat([self.embedder(position[sampled_indices]), features[sampled_indices]], dim=-1)
        x, seqlen = self.encoder(
            x_q,
            x_kv,
            (torch.ones(batch_size, device=x_q.device) * num_tokens).int(),
            (torch.ones(batch_size, device=x_q.device) * num_tokens * 4).int(),
        )
        h = self.quant(x)
        posterior = DiagonalGaussianDistribution(h, feature_dim=-1)
        if not use_decoded_features:
            return x, posterior
        
        feat = self.post_quant(posterior.sample())
        for block in self.decoder.blocks[:-1]:
            feat = block(
                feat,
                attention_kwargs={
                    "cu_seqlens_q": seqlen["cu_seqlens_q"],
                    "max_seqlen_q": seqlen["max_seqlen_q"],
                },
            )

        return feat, posterior

    def set_varlen_decoder(self):
        for block in self.decoder.blocks[:-1]:
            block.attn1.set_processor(CragVarlenFlashAttentionProcessor())

    def set_tripo_decoder(self):
        for block in self.decoder.blocks[:-1]:
            block.attn1.set_processor(TripoSGAttnProcessor2_0())


from_pretrained = TripoSGVarlenVAEModel.from_pretrained
