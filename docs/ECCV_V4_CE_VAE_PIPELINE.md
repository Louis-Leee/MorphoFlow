# V4 Cross-Embodiment VAE: Complete Training & Inference Pipeline

> **Target**: ECCV paper — exhaustive technical reference for method section writing.
> **Command**: `python train_diff_v4_ce_vae.py --config-name train_diff_v4_ce_vae.yaml model.object_encoder_type=triposg_vae model.object_patch=50`

---

## 0. Notation System

### Spaces & Sets

| Symbol | Definition |
|--------|-----------|
| $\mathcal{E}$ | Set of robot embodiments, e.g. {allegro, barrett, shadowhand, leaphand, ezgripper, robotiq\_3finger, xhand} |
| $SE(3)$ | Special Euclidean group (rigid body transformations) |
| $SO(3)$ | Special orthogonal group (rotations) |
| $\mathfrak{so}(3)$ | Lie algebra of $SO(3)$ (axis-angle vectors in $\mathbb{R}^3$) |

### Index & Dimension Symbols

| Symbol | Value | Definition |
|--------|-------|-----------|
| $B$ | 128 | Batch size |
| $P$ | 50 | Number of object patch tokens (TripoSG VAE output) |
| $L$ | 25 | Maximum number of robot link nodes (padded) |
| $L_e$ | varies | Actual link count for embodiment $e$ (see table below) |
| $N_t$ | 10 | Multi-timestep training copies per sample |
| $B' = B \cdot N_t$ | 1280 | Expanded batch size |
| $M$ | 1000 | Total forward diffusion steps |
| $K$ | 100 | DDIM sampling steps (inference) |
| $d$ | 384 | Transformer hidden dimension ($d_\text{model}$) |
| $H$ | 16 | Number of attention heads |
| $d_h = d/H$ | 24 | Per-head dimension |
| $d_\text{vae}$ | 64 | TripoSG VAE latent dimension |
| $d_\text{link}$ | 128 | Link embedding dimension (after projection) |
| $N_\text{pts}$ | 512 | Object/link point cloud size |

### Robot Embodiment Table

| Embodiment $e$ | Links $L_e$ | DOF ($6+J_e$) | Description |
|----------------|-------------|---------------|-------------|
| allegro | 21 | 27 | 4-finger, 16 joints |
| barrett | 10 | 16 | 3-finger, 8+2 joints |
| shadowhand | 17 | 28 | 5-finger, 20 joints |
| leaphand | 17 | 22 | 4-finger, 16 joints |
| ezgripper | 5 | 11 | 2-finger parallel |
| robotiq\_3finger | 13 | 19 | 3-finger adaptive |
| xhand | 13 | 19 | 5-finger, 13 links |

### Core Variable Symbols

| Symbol | Shape | Definition |
|--------|-------|-----------|
| $\mathbf{x}_O$ | $[B, N_\text{pts}, 3]$ | Raw object point cloud (xyz) |
| $\mathbf{n}_O$ | $[B, N_\text{pts}, 3]$ | Object surface normals |
| $\hat{\mathbf{x}}_O$ | $[B, N_\text{pts}, 3]$ | Normalized object point cloud |
| $\mathbf{c}_O, s_O$ | $[B,1,3], [B,1,1]$ | Object centroid and scale |
| $\mathbf{V}_O$ | $[B, P, 65]$ | Object node features (VAE latent + scale) |
| $\mathbf{V}_R$ | $[B, L, 134]$ | Robot node features (pose + link embedding) |
| $\mathbf{M}_R$ | $[B, L]$ | Robot link validity mask (boolean) |
| $\mathbf{p}_i = [\mathbf{t}_i, \mathbf{r}_i]$ | $\mathbb{R}^6$ | Link $i$ pose: translation $\mathbf{t}_i \in \mathbb{R}^3$ + axis-angle $\mathbf{r}_i \in \mathfrak{so}(3)$ |
| $\mathbf{q}$ | $\mathbb{R}^{6+J_e}$ | Joint configuration (6 virtual root + $J_e$ finger joints) |
| $\boldsymbol{\epsilon}^t, \boldsymbol{\epsilon}^r$ | $[B', L, 3]$ | Ground-truth noise (translation / rotation) |
| $\hat{\boldsymbol{\epsilon}}^t, \hat{\boldsymbol{\epsilon}}^r$ | $[B', L, 3]$ | Predicted noise |
| $\bar{\alpha}_m$ | scalar | Cumulative noise schedule: $\prod_{s=0}^{m} (1-\beta_s)$ |

---

## 1. Data Pipeline

### 1.1 Dataset: `CrossEmbodimentDataset`

**Source**: `dataset/CrossEmbodimentDataset.py`

Each `__getitem__` returns a complete batch of $B=128$ samples from a **single** robot embodiment (homogeneous batch). Robot selection is uniform across all embodiments per epoch.

**GT (ground-truth) batch flow** for robot $e$:

1. Randomly sample $(q_\text{target}, o_\text{name})$ from filtered grasp dataset
2. Object PC: sample $N_\text{pts}=512$ from 65536 pre-sampled mesh points + Gaussian noise ($\sigma=0.002$)
3. Forward kinematics: $q_\text{target} \xrightarrow{\text{FK}} \{T_i \in SE(3)\}_{i=1}^{L_e}$ via `pytorch_kinematics`
4. Convert SE3 to pose vectors: $T_i \mapsto [\mathbf{t}_i, \text{AxisAngle}(\mathbf{R}_i)]$
5. Initial config: perturbed $q_\text{initial}$ with random root rotation ($\leq \pi/6$) and finger retraction

**Output dictionary (GT batch)**:

| Key | Type / Shape | Description |
|-----|-------------|-------------|
| `object_pc` | $[B, 512, 3]$ | Stacked object point clouds |
| `object_pc_normal` | $[B, 512, 3]$ | Stacked surface normals |
| `target_vec` | list of $[L_e, 6]$ | Per-link pose vectors $[\mathbf{t}, \mathbf{r}]$ |
| `robot_name` | list of str | Robot embodiment name |
| `has_gt` | bool | `True` for GT batches |

### 1.2 Object Point Cloud Sampling Modes

| Mode | Method | Training augmentation |
|------|--------|----------------------|
| `random` | Uniform random 512 from 65536 | + Gaussian noise $\mathcal{N}(0, 0.002^2)$ |
| `fixed` | Pre-computed from `data/PointCloud/` | None |
| `partial` | Sort by random direction, take far half | Simulates single-view |

---

## 2. Object Encoding

### 2.1 Normalization ($L^\infty$ centering)

```
c_O = mean(x_O, dim=1)                    [B, 1, 3]
x_hat = x_O - c_O                         [B, 512, 3]
s_O = max(max(|x_hat|, dim=1), dim=2)     [B, 1, 1]
x_hat = x_hat / s_O                       [B, 512, 3]
```

### 2.2 TripoSG VAE Encoding (`object_encoder_type=triposg_vae`)

**Architecture**: Frozen pretrained TripoSG VAE encoder (`VAST-AI/TripoSG`, subfolder `vae`).

**Step-by-step**:

| Step | Operation | Input Shape | Output Shape |
|------|-----------|-------------|--------------|
| 1 | Concatenate: $[\hat{\mathbf{x}}_O, \mathbf{n}_O]$ | $[B,512,3]+[B,512,3]$ | $[B, 512, 6]$ |
| 2 | Flatten + cast fp16 | $[B, 512, 6]$ | $[B \cdot 512, 6]$ |
| 3 | Reshape to batch | $[65536, 6]$ | $[128, 512, 6]$ |
| 4 | Random subset: $4P=200$ points | $[128, 512, 6]$ | $[128, 200, 6] \to [25600, 6]$ |
| 5 | Frequency pos. embed: $\gamma(\mathbf{x})$ | $[25600, 3]$ | $[25600, 51]$ |
| 6 | Concat features: $[\gamma(\mathbf{x}), \mathbf{n}]$ | $[25600, 51]+[25600, 3]$ | $[25600, 54]$ ← KV tokens |
| 7 | FPS (ratio $\frac{1}{4}$): select $P=50$ per sample | $25600$ points | $6400$ points |
| 8 | Query tokens from FPS points | $[6400, 6]$ | $[6400, 54]$ ← Q tokens |
| 9 | Encoder: 1 cross-attn + 8 self-attn DiTBlocks | Q: $[6400, 54]$, KV: $[25600, 54]$ | $[6400, 512]$ |
| 10 | Proj + LayerNorm | $[6400, 512]$ | $[6400, 512]$ |
| 11 | Quant: Linear($512 \to 128$) | $[6400, 512]$ | $[6400, 128]$ |
| 12 | Split $\to$ DiagGaussian: $\mu,\log\sigma^2$ | $[6400, 128]$ | $\mu: [6400, 64]$, $\log\sigma^2: [6400, 64]$ |
| 13 | `posterior.mode()` = $\mu$ | $[6400, 64]$ | $[6400, 64]$ |
| 14 | Reshape + float32 | $[6400, 64]$ | $[128, 50, 64]$ |
| 15 | Concat scale: $[\mathbf{z}_O, s_O]$ | $[B,50,64]+[B,50,1]$ | $[B, 50, 65]$ = $\mathbf{V}_O$ |

**Frequency Positional Embedding** $\gamma: \mathbb{R}^3 \to \mathbb{R}^{51}$:

$$\gamma(\mathbf{x}) = [\mathbf{x},\; \sin(f_0 \mathbf{x}),\ldots,\sin(f_7 \mathbf{x}),\; \cos(f_0 \mathbf{x}),\ldots,\cos(f_7 \mathbf{x})]$$

where $f_k = 2^k$ for $k=0,\ldots,7$ (8 log-spaced frequencies, `include_pi=False`).
Per-coordinate: $1 + 2 \times 8 = 17$ features $\times$ 3 coordinates = 51D.

**TripoSG Encoder Architecture**:

| Layer | Type | Details |
|-------|------|---------|
| `proj_in` | Linear(54, 512) | Shared for Q and KV |
| Block 0 | Cross-Attention DiTBlock | Q=$P$ tokens, KV=$4P$ tokens, 8 heads $\times$ 64 dim/head |
| Blocks 1-8 | Self-Attention DiTBlock ($\times 8$) | $P$ tokens self-attend, 8 heads $\times$ 64 dim/head |
| Each DiTBlock | Norm + Attn + Norm + FFN | FP32LayerNorm, GELU FFN(512$\to$2048$\to$512), no QK-norm, no bias |
| `norm_out` | LayerNorm(512) | Final normalization |

**Attention**: Variable-length flash attention (`flash_attn_varlen_func`) via `CragVarlenFlashAttentionProcessor`. All samples in the batch are flattened into a single sequence with cumulative sequence length indices (`cu_seqlens`), avoiding padding overhead.

### 2.3 VQ-VAE Encoding (Baseline, `object_encoder_type=vqvae`)

For comparison — the default when not using TripoSG:

| Step | Operation | Output Shape |
|------|-----------|-------------|
| 1 | PointNet++ SA1(256, r=0.2) $\to$ SA2(128, r=0.4) $\to$ SA3($P$=25, r=0.8) | $[B, 25, 512]$ |
| 2 | Conv1d(512, 64) | $[B, 25, 64]$ |
| 3 | Reshape to $[B, 100, 16]$ $\to$ VQ nearest-neighbor (codebook: 1024$\times$16) $\to$ reshape back | $\mathbf{z}_q: [B, 25, 64]$ |
| 4 | Concat: $[\text{xyz}_\text{SA3}, s_O, \mathbf{z}_q]$ | $[B, 25, 68]$ |

**Denoiser input dimensions**: `V_object_dims = [3, 1, 64]` (VQ-VAE) vs `[64, 1]` (TripoSG).

### 2.4 Comparison Summary

| Aspect | VQ-VAE | TripoSG VAE |
|--------|--------|-------------|
| Input | 512 pts, xyz | 512 pts, xyz + normals |
| Encoder | PointNet++ (3 SA layers) | Transformer (9 DiTBlocks) |
| Tokenization | SA3 farthest-point ($P$=25) | FPS from $4P$ random subset ($P$=50) |
| Quantization | Discrete VQ (1024 codes, 16D) | Continuous KL ($\mu$ of DiagGaussian, 64D) |
| Object feature | $[B, 25, 68]$ | $[B, 50, 65]$ |
| Precision | fp32 | fp16 (flash\_attn) |
| Pretrained | `ckpt/vqvae.ckpt` | `VAST-AI/TripoSG` (HuggingFace) |

---

## 3. Robot Representation

### 3.1 Link Embedding via TripoSG VAE

**Pre-computed at initialization** (once, then cached as persistent buffers):

For each embodiment $e \in \mathcal{E}$ and each link $l \in \{1, \ldots, L_e\}$:

| Step | Operation | Shape |
|------|-----------|-------|
| 1 | Load canonical link PC + normals | $\mathbf{x}_l \in \mathbb{R}^{512 \times 3}$, $\mathbf{n}_l \in \mathbb{R}^{512 \times 3}$ |
| 2 | Unit-ball normalize: $\hat{\mathbf{x}}_l = (\mathbf{x}_l - \bar{\mathbf{x}}_l) / \max\|\mathbf{x}_l - \bar{\mathbf{x}}_l\|_2$ | $\hat{\mathbf{x}}_l \in \mathbb{R}^{512 \times 3}$ |
| 3 | Concat: $[\hat{\mathbf{x}}_l, \mathbf{n}_l]$ | $[512, 6]$ |
| 4 | Batch all links: stack | $[L_e, 512, 6]$ |
| 5 | Flatten + encode\_shape(num\_tokens=1) | $[L_e, 64]$ |
| 6 | `posterior.mode()` | $\mathbf{z}_l^\text{link} \in \mathbb{R}^{64}$ |

**Note**: Link uses unit-ball ($L^2$) normalization; object uses $L^\infty$ normalization. Different!

**Link token encoder** (learnable MLP):

$$\phi_\text{link}(\mathbf{z}) = W_2 \cdot \text{ReLU}(W_1 \mathbf{z} + b_1) + b_2$$

where $W_1 \in \mathbb{R}^{128 \times 64}$, $W_2 \in \mathbb{R}^{128 \times 128}$.

Output: $\mathbf{e}_l = \phi_\text{link}(\mathbf{z}_l^\text{link}) \in \mathbb{R}^{128}$ per link.

### 3.2 Robot Node Construction

For each sample $b$ in a batch with robot $e$ having $L_e$ links:

1. **Target pose normalization** (object-centric):
$$\tilde{\mathbf{t}}_i = \frac{\mathbf{t}_i - \mathbf{c}_O}{s_O}, \quad \tilde{\mathbf{r}}_i = \mathbf{r}_i \quad \text{(rotation unchanged)}$$

2. **Padded tensor assembly**:
$$\mathbf{V}_R[b, i, :] = \begin{cases} [\tilde{\mathbf{t}}_i, \tilde{\mathbf{r}}_i, \mathbf{e}_i] \in \mathbb{R}^{134} & \text{if } i < L_e \\ \mathbf{0} \in \mathbb{R}^{134} & \text{if } i \geq L_e \end{cases}$$

3. **Link mask**: $\mathbf{M}_R[b, i] = \mathbb{1}[i < L_e]$

**Final shapes**: $\mathbf{V}_R \in \mathbb{R}^{B \times L \times 134}$, $\mathbf{M}_R \in \{0,1\}^{B \times L}$

Decomposition of the 134D robot feature:

| Dims | Content | Symbol |
|------|---------|--------|
| 0-2 | Normalized translation | $\tilde{\mathbf{t}}_i$ |
| 3-5 | Axis-angle rotation | $\tilde{\mathbf{r}}_i$ |
| 6-133 | Link geometry embedding | $\mathbf{e}_i$ |

---

## 4. Diffusion Process

### 4.1 Noise Schedule

**Linear schedule**: $\beta_m = \beta_\min + \frac{m}{M-1}(\beta_\max - \beta_\min)$ for $m = 0, \ldots, M-1$.

$$\alpha_m = 1 - \beta_m, \quad \bar{\alpha}_m = \prod_{s=0}^{m} \alpha_s$$

| Parameter | Value |
|-----------|-------|
| $\beta_\min$ | $10^{-4}$ |
| $\beta_\max$ | $0.02$ |
| $M$ | 1000 |
| Prediction type | $\epsilon$ (noise prediction) |
| Timestep spacing | trailing |

### 4.2 Forward Diffusion (Training)

The diffusion target is the 6D pose vector $\mathbf{p}_i = [\mathbf{t}_i, \mathbf{r}_i]$. Translation and rotation are diffused **independently**:

$$\mathbf{t}_i^{(m)} = \sqrt{\bar{\alpha}_m}\, \tilde{\mathbf{t}}_i + \sqrt{1 - \bar{\alpha}_m}\, \boldsymbol{\epsilon}_i^t, \quad \boldsymbol{\epsilon}_i^t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_3)$$

$$\mathbf{r}_i^{(m)} = \sqrt{\bar{\alpha}_m}\, \tilde{\mathbf{r}}_i + \sqrt{1 - \bar{\alpha}_m}\, \boldsymbol{\epsilon}_i^r, \quad \boldsymbol{\epsilon}_i^r \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_3)$$

**Multi-timestep amortization** ($N_t = 10$):

To amortize the cost of encoding $\mathbf{V}_O$ (TripoSG forward pass), each sample is expanded to $N_t = 10$ copies with independently sampled timesteps. The expansion via `_expand_and_reshape_`:

$$[B, N, D] \xrightarrow{\text{unsqueeze}} [B, 1, N, D] \xrightarrow{\text{expand}} [B, N_t, N, D] \xrightarrow{\text{reshape}} [B \cdot N_t, N, D]$$

Effective batch: $B' = B \cdot N_t = 128 \times 10 = 1280$.

**Noisy robot node** (link embedding is NOT diffused):

$$\text{noisy\_V}_R^{(m)} = [\mathbf{t}^{(m)}_i,\; \mathbf{r}^{(m)}_i,\; \mathbf{e}_i] \in \mathbb{R}^{134}$$

### 4.3 Classifier-Free Guidance (CFG) Dropout

During training, with probability $p_\text{uncond} = 0.1$:

- Object nodes zeroed: $\mathbf{V}_O \leftarrow \mathbf{0}$
- Cross-attention skipped: `skip_or = True`

This trains the model to predict noise both conditionally and unconditionally.

---

## 5. Denoiser Architecture

### 5.1 Overview: `FlashAttentionDenoiserNoEdge`

**Source**: `model/flash_denoiser_noedge.py`

**Signature**: $f_\theta(\mathbf{V}_O, \text{noisy\_V}_R, m) \to \hat{\boldsymbol{\epsilon}} \in \mathbb{R}^{B' \times L \times 6}$

**Key design**: No pairwise SE(3) edge features — spatial relationships learned purely from absolute poses via Q@K^T attention. This is an intentional ablation from the edge-based variant.

### 5.2 Input Encoding (Multi-Stream Sum Fusion)

Each input is split into sub-features, independently projected to $d$, then **summed** (not concatenated):

$$h = \sum_{k=1}^{K} W_k \mathbf{x}_{[d_k]} + b_k$$

**Object streams** (`V_object_dims = [64, 1]`):

| Stream | Slice | Projection | Meaning |
|--------|-------|-----------|---------|
| 1 | $\mathbf{V}_O[\ldots, 0\!:\!64]$ | Linear(64, 384) | VAE latent |
| 2 | $\mathbf{V}_O[\ldots, 64\!:\!65]$ | Linear(1, 384) | Scale |

$$\mathbf{h}_O = W_\text{lat} \mathbf{z}_O + W_\text{scale} s_O \in \mathbb{R}^{B' \times P \times d}$$

**Robot streams** (`V_robot_dims = [3, 3, 128]`):

| Stream | Slice | Projection | Meaning |
|--------|-------|-----------|---------|
| 1 | $\text{noisy\_V}_R[\ldots, 0\!:\!3]$ | Linear(3, 384) | Noisy translation |
| 2 | $\text{noisy\_V}_R[\ldots, 3\!:\!6]$ | Linear(3, 384) | Noisy rotation |
| 3 | $\text{noisy\_V}_R[\ldots, 6\!:\!134]$ | Linear(128, 384) | Link embedding |

$$\mathbf{h}_R = W_t \mathbf{t}^{(m)} + W_r \mathbf{r}^{(m)} + W_e \mathbf{e} \in \mathbb{R}^{B' \times L \times d}$$

### 5.3 Time Conditioning

**Sinusoidal embedding** (fixed buffer, $[M, 200]$):

$$\text{emb}(m, 2j) = \sin\!\Big(\frac{m}{10000^{2j/200}}\Big), \quad \text{emb}(m, 2j+1) = \cos\!\Big(\frac{m}{10000^{2j/200}}\Big)$$

**Time MLP**: $\text{emb}(m) \xrightarrow{\text{Linear}(200, 384)} \xrightarrow{\text{SiLU}} \xrightarrow{\text{Linear}(384, 384)} \mathbf{c}_m \in \mathbb{R}^{d}$

### 5.4 Transformer Layers ($\times 6$)

Each layer is a `TransformerDenoiserLayer` with three AdaLN-Zero-gated sub-layers:

#### AdaLN-Zero (DiT-style)

$$(\gamma, \beta, \alpha) = \text{Linear}(\text{SiLU}(\mathbf{c}_m)) \in \mathbb{R}^{3d}$$

$$\text{AdaLN}(\mathbf{x}, \mathbf{c}_m) = \text{LN}(\mathbf{x}) \odot (1 + \gamma) + \beta$$

$$\text{output} = \mathbf{x} + \alpha \odot \text{SubLayer}(\text{AdaLN}(\mathbf{x}, \mathbf{c}_m))$$

Gate $\alpha$ is **zero-initialized** — at initialization, all sub-layers contribute nothing (residual-dominant).

#### Sub-Layer 1: Self-Attention (Robot-Robot)

$$\mathbf{Q}, \mathbf{K}, \mathbf{V} = \text{split}(W_{qkv}\, \text{AdaLN}(\mathbf{h}_R)) \in \mathbb{R}^{B' \times H \times L \times d_h}$$

$$\text{SelfAttn} = \text{softmax}\!\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}}\Big)\mathbf{V}$$

Uses `F.scaled_dot_product_attention` (Flash Attention kernel). No attention mask — pure Q@K^T.

#### Sub-Layer 2: Cross-Attention (Object $\to$ Robot)

$$\mathbf{Q}_R = W_q\, \text{AdaLN}(\mathbf{h}_R) \in \mathbb{R}^{B' \times H \times L \times d_h}$$

$$\mathbf{K}_O, \mathbf{V}_O = \text{split}(W_{kv}\, \mathbf{h}_O) \in \mathbb{R}^{B' \times H \times P \times d_h}$$

$$\text{CrossAttn} = \text{softmax}\!\Big(\frac{\mathbf{Q}_R \mathbf{K}_O^\top}{\sqrt{d_h}}\Big)\mathbf{V}_O$$

**Skipped entirely** when `skip_or=True` (CFG unconditional pass).

#### Sub-Layer 3: Feed-Forward Network

$$\text{FFN}(\mathbf{x}) = W_2\, \text{GELU}(W_1 \mathbf{x})$$

where $W_1 \in \mathbb{R}^{1536 \times 384}$, $W_2 \in \mathbb{R}^{384 \times 1536}$ (4x expansion).

### 5.5 Feature Aggregation (DenseNet-style Wide FC)

All 7 snapshots (1 input + 6 layer outputs) are concatenated:

$$\mathbf{h}_\text{agg} = W_\text{wide}\, [\mathbf{h}_R^{(0)} \| \mathbf{h}_R^{(1)} \| \cdots \| \mathbf{h}_R^{(6)}]$$

$$[B', L, 7 \times 384] = [B', L, 2688] \xrightarrow{\text{Linear}(2688, 384)} [B', L, 384]$$

### 5.6 Output Heads (NaiveMLP with Skip Connections)

Two separate heads for translation and rotation noise:

**NaiveMLP architecture** (for each head):

$$\text{Input: } \mathbf{x} = [\mathbf{h}_\text{agg}, \text{noisy\_input}_{0:3}] \in \mathbb{R}^{387}$$

| Layer | Operation | Output | Accumulated |
|-------|-----------|--------|-------------|
| 0 | Linear(387, 256) $\to$ LN $\to$ LeakyReLU | $\mathbf{f}_0 \in \mathbb{R}^{256}$ | $[\mathbf{x}, \mathbf{f}_0]$: 643D |
| 1 | Linear(256, 128) $\to$ LN $\to$ LeakyReLU | $\mathbf{f}_1 \in \mathbb{R}^{128}$ | $[\mathbf{x}, \mathbf{f}_0, \mathbf{f}_1]$: 771D |
| Out | Linear(771, 3) | $\hat{\boldsymbol{\epsilon}} \in \mathbb{R}^3$ | Final prediction |

$$\hat{\boldsymbol{\epsilon}} = [\hat{\boldsymbol{\epsilon}}^t \| \hat{\boldsymbol{\epsilon}}^r] \in \mathbb{R}^{B' \times L \times 6}$$

### 5.7 Parameter Count Summary

| Component | Parameters |
|-----------|-----------|
| Time MLP | 225,024 |
| Object encoder (2 streams) | 25,728 |
| Robot encoder (3 streams) | 52,608 |
| Transformer layers ($\times 6$) | 22,169,088 |
| $\quad$ AdaLN-Zero ($3 \times 6$) | 7,983,360 |
| $\quad$ Self-Attention ($\times 6$) | 3,548,160 |
| $\quad$ Cross-Attention ($\times 6$) | 3,548,160 |
| $\quad$ FFN ($\times 6$) | 7,089,408 |
| Feature aggregation (wide FC) | 1,032,576 |
| Output MLPs ($\times 2$ heads) | 270,616 |
| **Denoiser total** | **~23.8M** |

---

## 6. Loss Function

### 6.1 Masked MSE on Noise Prediction

$$\mathcal{L}_\text{trans} = \frac{\sum_{b,i} \mathbf{M}_R[b,i] \cdot \frac{1}{3}\|\boldsymbol{\epsilon}_i^t - \hat{\boldsymbol{\epsilon}}_i^t\|^2}{\sum_{b,i} \mathbf{M}_R[b,i] + \varepsilon}$$

$$\mathcal{L}_\text{rot} = \frac{\sum_{b,i} \mathbf{M}_R[b,i] \cdot \frac{1}{3}\|\boldsymbol{\epsilon}_i^r - \hat{\boldsymbol{\epsilon}}_i^r\|^2}{\sum_{b,i} \mathbf{M}_R[b,i] + \varepsilon}$$

**Total loss** (GT path):

$$\mathcal{L} = w_t \cdot \mathcal{L}_\text{trans} + w_r \cdot \mathcal{L}_\text{rot}$$

with $w_t = w_r = 1.0$.

**No-GT path**: $\mathcal{L}_\text{nogt} = \lambda_\text{nogt} \cdot (w_t \cdot \mathcal{L}_\text{trans} + w_r \cdot \mathcal{L}_\text{rot})$ with $\lambda_\text{nogt} = 1.0$.

### 6.2 Loss Shapes Trace

| Tensor | Shape |
|--------|-------|
| $\boldsymbol{\epsilon}^t, \hat{\boldsymbol{\epsilon}}^t$ | $[1280, 25, 3]$ |
| Error (per-dim) | $[1280, 25, 3]$ |
| Error (mean over xyz) | $[1280, 25]$ |
| Masked error | $[1280, 25]$ |
| $\mathcal{L}_\text{trans}, \mathcal{L}_\text{rot}$ | scalar |
| $\mathcal{L}_\text{total}$ | scalar |

---

## 7. Inference: DDIM Sampling with CFG

### 7.1 Unconditioned Inference

Starting from pure noise: $\mathbf{t}^{(M)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),\; \mathbf{r}^{(M)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

At each DDIM step $i$ with timestep $m_i$ (from $K$ trailing-spaced timesteps):

**Step 1 — CFG noise prediction** (two forward passes when $s \neq 1$):

$$\hat{\boldsymbol{\epsilon}}_\text{uncond} = f_\theta(\mathbf{0}, \text{noisy\_V}_R, m_i, \text{skip\_or=True})$$

$$\hat{\boldsymbol{\epsilon}}_\text{cond} = f_\theta(\mathbf{V}_O, \text{noisy\_V}_R, m_i, \text{skip\_or=False})$$

$$\hat{\boldsymbol{\epsilon}}_\text{guided} = \hat{\boldsymbol{\epsilon}}_\text{uncond} + s \cdot (\hat{\boldsymbol{\epsilon}}_\text{cond} - \hat{\boldsymbol{\epsilon}}_\text{uncond})$$

with guidance scale $s = 1.5$.

**Step 2 — Predict $\mathbf{x}_0$**:

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_m - \sqrt{1-\bar{\alpha}_m}\, \hat{\boldsymbol{\epsilon}}_\text{guided}}{\sqrt{\bar{\alpha}_m}}$$

**Step 3 — DDIM update**:

Let $\bar{\alpha}_\text{prev} = \bar{\alpha}_{m_{i+1}}$ (or $1.0$ at final step).

$$\sigma_m = \eta \sqrt{\frac{1 - \bar{\alpha}_\text{prev}}{1 - \bar{\alpha}_m} \cdot \Big(1 - \frac{\bar{\alpha}_m}{\bar{\alpha}_\text{prev}}\Big)}$$

$$c_\text{ddim} = \sqrt{1 - \bar{\alpha}_\text{prev} - \sigma_m^2}$$

$$\mathbf{x}_{m-1} = \sqrt{\bar{\alpha}_\text{prev}}\, \hat{\mathbf{x}}_0 + c_\text{ddim} \cdot \hat{\boldsymbol{\epsilon}}_\text{guided} + \lambda_\text{noise} \cdot \sigma_m \cdot \mathbf{z}$$

where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ (zero at final step), $\eta = 1.0$, $\lambda_\text{noise} = 0.2$.

**Step 4 — De-normalize and convert**:

$$\hat{\mathbf{t}}_i = \mathbf{t}^{(0)}_i \cdot s_O + \mathbf{c}_O, \quad \hat{T}_i = \text{AxisAngle2Mat}(\hat{\mathbf{r}}^{(0)}_i)$$

### 7.2 Palm-Conditioned Inference

Starts from a partial-noise initialization at timestep $m^\ast$ (not from pure noise):

**Start timestep**: $m^\ast = \arg\min_m \big|1 - \bar{\alpha}_m - (\theta_\text{err}/\mu)^2\big|$

where $\theta_\text{err} = \pi/2$ (90 deg) and $\mu = 1.596$.

**Initialization**:

$$\mathbf{r}^{(m^\ast)} = \sqrt{\bar{\alpha}_{m^\ast}}\, \mathbf{r}_0 + \sqrt{1-\bar{\alpha}_{m^\ast}}\, \boldsymbol{\epsilon}^r$$

$$\mathbf{t}^{(m^\ast)} = \sqrt{1-\bar{\alpha}_{m^\ast}}\, \boldsymbol{\epsilon}^t$$

**Rotation guidance** at each step $i$ (applied after noise prediction, before DDIM update):

1. Compute progress: $\text{prog} = (i+1)/K$
2. Interpolation strength: $s_m = \rho \cdot \sin(0.5 \cdot \text{prog} \cdot \pi)$ with $\rho = 0.5$
3. Palm rotation error:
   $$\mathbf{R}_\text{err} = \mathbf{R}_\text{init} \cdot \mathbf{R}_\text{cur}^{-1}, \quad \mathbf{r}_\text{err} = \text{Log}(\mathbf{R}_\text{err})$$
4. Apply correction to **all links** (global rotation):
   $$\mathbf{R}_\delta = \text{Exp}(s_m \cdot \mathbf{r}_\text{err}), \quad \mathbf{R}_l' = \mathbf{R}_\delta \cdot \mathbf{R}_l \;\;\forall l$$
5. Back-project to noise space:
   $$\hat{\boldsymbol{\epsilon}}^r = \frac{\mathbf{x}_m^r - \sqrt{\bar{\alpha}_m}\, \text{Log}(\mathbf{R}')}{\sqrt{1-\bar{\alpha}_m}}$$

---

## 8. Training Infrastructure

### 8.1 PyTorch Lightning Module

| Component | Configuration |
|-----------|--------------|
| Framework | PyTorch Lightning + Hydra |
| Optimizer | Adam, lr=$10^{-4}$ |
| Scheduler | StepLR(step=20, gamma=0.8) |
| Gradient clipping | max\_norm=1.0 |
| GPUs | 7 (DDP, `find_unused_parameters=True`) |
| Precision | fp32 (denoiser), fp16 (TripoSG VAE) |
| Epochs | 700 |
| Checkpoint | Every 10 epochs |
| Tracking | W&B (project: `trograsp-fm`) |

### 8.2 Data Flow Summary

```
DataLoader(batch_size=1)
  → CrossEmbodimentDataset.__getitem__ returns full B=128 batch
  → ce_custom_collate_fn just unwraps: batch[0]
  → DiffusionV4CEVAEModule.training_step
    → _prepare_input (move to device)
    → model._forward_gt / _forward_nogt
      → _normalize_pc_ + _encode_object_vae (TripoSG, fp16)
      → CFG dropout (p=0.1)
      → Robot node construction (link embeddings from buffers)
      → _expand_and_reshape_ (×N_t=10)
      → Forward diffusion (add_noise)
      → Denoiser forward (transformer, fp32)
      → Masked MSE loss
```

---

## 9. Complete Tensor Shape Trace

```
DATA:
  object_pc:             [128, 512, 3]
  object_pc_normal:      [128, 512, 3]
  target_vec[b]:         [L_e, 6]

OBJECT ENCODING:
  normalize →            hat_x_O: [128, 512, 3], s_O: [128, 1, 1]
  vae_input:             [128, 512, 6]
  flat (fp16):           [65536, 6]
  TripoSG encode_shape:
    selected_pts:        [25600, 6]  (200 pts × 128)
    x_kv:                [25600, 54] (freq_embed: 51 + normals: 3)
    FPS → x_q:           [6400, 54]  (50 pts × 128)
    encoder:             [6400, 512] (9 DiTBlocks)
    quant:               [6400, 128]
    posterior.mode():    [6400, 64]
  reshape:               [128, 50, 64]
  V_O:                   [128, 50, 65] (+ scale)

ROBOT NODES:
  link_embeddings:       [L_e, 64]  (buffer)
  link_token_encoder:    [L_e, 128]
  link_target_poses:     [128, 25, 6]   (padded)
  link_robot_embeds:     [128, 25, 128] (padded)
  M_R:                   [128, 25]      (bool mask)
  V_R:                   [128, 25, 134]

DIFFUSION:
  t:                     [1280]
  V_O (expanded):        [1280, 50, 65]
  V_R (expanded):        [1280, 25, 134]
  epsilon^t, epsilon^r:  [1280, 25, 3]
  noisy_V_R:             [1280, 25, 134]

DENOISER:
  t_emb:                 [1280, 200]
  t_cond:                [1280, 384]
  h_O:                   [1280, 50, 384]
  h_R:                   [1280, 25, 384]
  × 6 TransformerDenoiserLayer:
    Self-Attn Q,K,V:     [1280, 16, 25, 24]
    Cross-Attn Q:        [1280, 16, 25, 24]
    Cross-Attn K,V:      [1280, 16, 50, 24]
    FFN hidden:          [1280, 25, 1536]
  wide_fc:               [1280, 25, 2688] → [1280, 25, 384]
  output MLP ×2:         [1280, 25, 387] → [1280, 25, 3]
  hat_epsilon:           [1280, 25, 6]

LOSS:
  M_R (float):           [1280, 25]
  error_trans/rot:       [1280, 25, 3]
  L_trans, L_rot:        scalar
  L_total:               scalar
```

---

## 10. Key Hyperparameters Reference

| Parameter | Symbol | Value | Section |
|-----------|--------|-------|---------|
| Batch size | $B$ | 128 | 1 |
| Object patches | $P$ | 50 | 2 |
| Max link nodes | $L$ | 25 | 3 |
| Timestep copies | $N_t$ | 10 | 4 |
| Diffusion steps | $M$ | 1000 | 4 |
| DDIM steps | $K$ | 100 | 7 |
| DDIM eta | $\eta$ | 1.0 | 7 |
| Noise lambda | $\lambda_\text{noise}$ | 0.2 | 7 |
| CFG dropout | $p_\text{uncond}$ | 0.1 | 4.3 |
| Guidance scale | $s$ | 1.5 | 7 |
| d\_model | $d$ | 384 | 5 |
| Attention heads | $H$ | 16 | 5 |
| FFN dim | — | 1536 | 5 |
| Num layers | — | 6 | 5 |
| VAE latent dim | $d_\text{vae}$ | 64 | 2, 3 |
| Link embed dim | $d_\text{link}$ | 128 | 3 |
| Loss weights | $w_t, w_r$ | 1.0, 1.0 | 6 |
| Learning rate | — | $10^{-4}$ | 8 |
| LR step / gamma | — | 20 / 0.8 | 8 |
| Gradient clip | — | 1.0 | 8 |
| Epochs | — | 700 | 8 |

---

## 11. Source File Index

| File | Content |
|------|---------|
| `train_diff_v4_ce_vae.py` | Training script (Lightning + Hydra) |
| `config/train_diff_v4_ce_vae.yaml` | Full training configuration |
| `model/tro_graph_v4_ce.py` | `RobotGraphV4CE` — main model |
| `model/flash_denoiser_noedge.py` | `FlashAttentionDenoiserNoEdge` — transformer denoiser |
| `model/flash_denoiser.py` | Shared components: `TransformerDenoiserLayer`, `AdaLNZero`, attention, `NaiveMLP` |
| `model/autoencoders/triposg_varlen_autoencoder.py` | `TripoSGVarlenVAEModel` — varlen flash attn encoder |
| `model/autoencoders/attention_processor.py` | `CragVarlenFlashAttentionProcessor` |
| `model/vqvae/vq_vae.py` | `VQVAE` — baseline object encoder |
| `model/vqvae/pn2.py` | `PN2` — PointNet++ encoder |
| `dataset/CrossEmbodimentDataset.py` | Dataset + collate function |
| `utils/hand_model.py` | `HandModel` — FK, link PC management |
| `utils/rotation.py` | SE3/SO3 conversion utilities |
