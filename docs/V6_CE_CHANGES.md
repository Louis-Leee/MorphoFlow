# V6 CE: Fourier SE3 Relative Position Encoding

## Problem

The original `train.py` (using `RobotGraph` + `GraphDenoiser`) converges better and generalizes more robustly to diverse unseen objects than the V3/V4 CE models (using `FlashAttentionDenoiserNoEdge`). The root cause is that graph-based models explicitly encode spatial relationships via edge features, while V3/V4 must infer spatial relationships from absolute poses alone.

| Aspect | GraphDenoiser (Original) | NoEdge (V3/V4) |
|--------|-------------------------|-----------------|
| Edge features | E_RR [B,L,L,6] + E_OR [B,L,P,6] | None |
| Edges in attention | Part of VALUE computation | Pure Q@K^T |
| Spatial relationships | Explicit pairwise relative SE3 | Must learn from absolute poses |

## Solution: Fourier SE3 Relative Position Encoding

V6 re-introduces relative SE3 edge features but encodes them as **attention bias** (not graph edges), maintaining a standard transformer architecture:

```
relative_se3 [6D] -> Fourier encoding [102D] -> MLP -> per-head bias [16 heads]
```

### Fourier Encoding (NeRF-style)

The `FourierSE3Encoding` module maps 6D relative SE3 transforms to 102D using 8 exponentially-spaced frequencies:

```
x -> [x, sin(2^0*x), cos(2^0*x), ..., sin(2^7*x), cos(2^7*x)]
output_dim = 6 * (2*8 + 1) = 102
```

This provides the attention mechanism with high-frequency spatial representations, enabling it to distinguish fine-grained geometric relationships.

### Bias Projection

```
102D -> Linear(102, 64) -> SiLU -> Linear(64, 16) -> per-head bias
```

Two separate projections: `rr_bias_proj` (robot-robot) and `or_bias_proj` (object-robot).

## Architecture

### FlashAttentionDenoiserFourierRPE

File: `model/flash_denoiser_fourier_rpe.py`

- Extends `FlashAttentionDenoiser` with Fourier SE3 encoding
- Imports reusable components from `flash_denoiser.py` (TransformerDenoiserLayer, etc.)
- **8 layers** (increased from 6)
- Forward: `(V_O, noisy_V_R, noisy_E_OR, noisy_E_RR, t, skip_or) -> pred_noise`

### RobotGraphV6CE

File: `model/tro_graph_v6_ce.py`

Key changes from V4:
1. **Edge computation**: Computes noisy E_RR and E_OR at each forward/inference step via `compute_batch_relative_se3`
2. **BPS/VAE switch**: `link_feature_type` config param selects between BPS (V3-style) and TripoSG VAE (V4-style)
3. **Fourier RPE denoiser**: Uses `FlashAttentionDenoiserFourierRPE` instead of `FlashAttentionDenoiserNoEdge`

### Edge Computation

At each diffusion step (training and inference), edges are recomputed from noisy poses:

```python
noisy_V_R_se3 = vector_to_matrix(noisy_V_R[:, :, :6])  # [B, L, 4, 4]
noisy_E_RR = matrix_to_vector(compute_batch_relative_se3(noisy_V_R_se3, noisy_V_R_se3))  # [B, L, L, 6]
# ... similar for E_OR with object positions
```

## Configuration

### Link Feature Types

**VAE mode** (default):
```yaml
model:
  link_feature_type: 'vae'
  vae_config:
    pretrained_model: 'VAST-AI/TripoSG'
    subfolder: 'vae'
    latent_channels: 64
    link_embed_dim: 128
```

**BPS mode**:
```yaml
model:
  link_feature_type: 'bps'
  bps_config:
    n_bps_points: 124
    radius: 1.0
    n_dims: 3
```

### Denoiser Config (new params)

```yaml
model:
  denoiser_config:
    E_or_dims: [3, 3]        # RE-ADDED from original
    E_rr_dims: [3, 3]        # RE-ADDED from original
    num_layers: 8             # INCREASED from 6
    fourier_num_freqs: 8      # NEW: Fourier encoding frequencies
```

## Files

| File | Description |
|------|-------------|
| `model/flash_denoiser_fourier_rpe.py` | Fourier RPE denoiser |
| `model/tro_graph_v6_ce.py` | V6 model (BPS/VAE + edges) |
| `train_diff_v6_ce_vae.py` | Training script |
| `config/train_diff_v6_ce_vae.yaml` | Training config |
| `test_diff_v6_ce_vae.py` | Test script |
| `config/test_diff_v6_ce_vae.yaml` | Test config |

## Dataset

Uses `CrossEmbodimentDatasetV5` with dual object splits (same as V5).

## Memory Impact

Edge tensors are small: E_RR [B*T, L, L, 6] and E_OR [B*T, L, P, 6] with B=16, T=4, L=25, P=25 use under 2MB total. Fourier encoding (102D per pair) is also negligible.
