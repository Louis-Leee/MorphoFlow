# V4 CE VAE: TripoSG VAE Link Feature Extraction

## Overview

V4 replaces the BPS (Ball Point Set) link embeddings in V3 with frozen TripoSG VAE encoder latents. Each robot link's 512-point cloud (with normals) is encoded to a single 64D latent token, then projected to 128D to match the existing denoiser architecture.

## Architecture Comparison

```
V3 (BPS):
  link_pc [N, 3] → unit_ball_normalize → bps.encode → [1, 124]
  → cat(centroid[3], scale[1]) → [1, 128]
  → MLP(128→128→128) → [1, 128]

V4 (VAE):
  link_pc [512, 3] + normal [512, 3] → cat → [512, 6]
  → TripoSG encode_shape(num_tokens=1, fp16) → posterior.sample() → [1, 64]
  → MLP(64→128→128) → [1, 128]

Both → cat with pose [6D] → robot_nodes [B, L, 134] → denoiser (unchanged)
```

## Key Design Decisions

1. **Pretrained + frozen encoder**: TripoSG VAE from `VAST-AI/TripoSG` (HuggingFace), frozen weights
2. **1 token per link**: `encode_shape(num_tokens=1)` uses 4x oversampling + FPS to 1 query token
3. **64D → 128D projection**: Linear(64, 128) to keep `V_robot_dims=[3, 3, 128]` unchanged
4. **fp16 required**: Flash attention only supports fp16/bf16 — VAE runs in fp16
5. **Pre-computed at init**: All link embeddings computed once, VAE deleted from GPU after

## Files Created

| File | Description |
|------|-------------|
| `model/tro_graph_v4_ce.py` | `RobotGraphV4CE` — replaces `construct_bps()` with `construct_vae()` |
| `train_diff_v4_ce_vae.py` | Training script (identical to V3 except import) |
| `config/train_diff_v4_ce_vae.yaml` | Config with `vae_config` instead of `bps_config` |

## Files NOT Modified

- `model/tro_graph_v3_ce.py` — untouched
- `train_diff_v3_ce.py` — untouched
- `utils/hand_model.py` — untouched
- `model/flash_denoiser_noedge.py` — untouched
- `dataset/CrossEmbodimentDataset.py` — untouched

## Data Requirements

Uses `data/PointCloud_512_uniform/robot/{name}.pt` with keys:
- `'filtered'`: `{link_name: Tensor[512, 3]}` — per-link point clouds
- `'normal'`: `{link_name: Tensor[512, 3]}` — per-point normals

Available robots: allegro, barrett, leaphand, shadowhand, xhand

## Config Changes (vs V3)

```yaml
# V3 config:
model:
  bps_config:
    bps_type: 'random_uniform'
    n_bps_points: 124
    radius: 1.0
    n_dims: 3

# V4 config:
model:
  vae_config:
    pretrained_model: 'VAST-AI/TripoSG'
    subfolder: 'vae'
    latent_channels: 64
    link_embed_dim: 128
```

## Tensor Shapes

| Stage | V3 (BPS) | V4 (VAE) |
|-------|----------|----------|
| Raw link embedding | `[num_links, 128]` | `[num_links, 64]` |
| After token encoder | `[num_links, 128]` | `[num_links, 128]` |
| Robot nodes | `[B, 25, 134]` | `[B, 25, 134]` |
| Denoiser input | `[B, 25, 134]` | `[B, 25, 134]` |
| Denoiser output | `[B, 25, 6]` | `[B, 25, 6]` |

## Training Commands

```bash
# Single GPU
conda run -n rpf python train_diff_v4_ce_vae.py train.gpus=1

# Multi-GPU (8x)
conda run -n rpf python train_diff_v4_ce_vae.py

# Resume
conda run -n rpf python train_diff_v4_ce_vae.py train.resume_from=graph_exp/diff_v4_ce_vae/ckpt/epoch=99.ckpt
```

## Verified

- Model instantiation: all 5 robots load correctly
- Link embedding shapes: `[num_links, 64]` for all robots
- Token encoder output: `[num_links, 128]` matches V_robot_dims
- 1-epoch training: loss ~0.48 (GT), ~0.46 (no-GT) — comparable to V3 epoch 0
- Speed: ~9.96 it/s on single L40S GPU
