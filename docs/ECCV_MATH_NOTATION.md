# Mathematical Notation Quick Reference (ECCV Paper)

> Concise notation guide for writing the method section. See `ECCV_V4_CE_VAE_PIPELINE.md` for exhaustive details.

---

## Core Notation

### Input

| Symbol | Meaning |
|--------|---------|
| $\mathcal{O} = \{\mathbf{x}_i\}_{i=1}^{N}$ | Object point cloud, $\mathbf{x}_i \in \mathbb{R}^3$ |
| $\{\mathbf{n}_i\}_{i=1}^{N}$ | Surface normals |
| $\mathcal{R} = (e, \mathbf{q})$ | Robot: embodiment $e \in \mathcal{E}$, joint config $\mathbf{q} \in \mathbb{R}^{D_q}$ |
| $\{T_l\}_{l=1}^{L_e}$ | Per-link SE(3) poses from FK($\mathbf{q}$), $T_l \in SE(3)$ |
| $\mathbf{p}_l = [\mathbf{t}_l, \mathbf{r}_l] \in \mathbb{R}^6$ | Pose vector: translation + axis-angle |

### Encoding

| Symbol | Meaning |
|--------|---------|
| $\Phi_\text{obj}^\text{VAE}$ | Frozen TripoSG VAE encoder |
| $\Phi_\text{link}^\text{VAE}$ | Same encoder, applied to link geometry |
| $\phi_\text{link}$ | Learnable link projection MLP (64$\to$128) |
| $\gamma(\cdot)$ | Frequency positional embedding ($\mathbb{R}^3 \to \mathbb{R}^{51}$) |
| $\text{FPS}(\cdot, k)$ | Farthest point sampling to $k$ points |

### Graph / Token Representation

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $\mathbf{V}_O$ | $[B, P, d_O]$ | Object tokens ($P$=50 patches, $d_O$=65) |
| $\mathbf{V}_R$ | $[B, L, d_R]$ | Robot tokens ($L$=25 max links, $d_R$=134) |
| $\mathbf{M}$ | $[B, L]$ | Link validity mask |
| $\mathbf{e}_l$ | $\mathbb{R}^{128}$ | Link geometry embedding for link $l$ |

### Diffusion

| Symbol | Meaning |
|--------|---------|
| $\bar{\alpha}_m$ | Cumulative noise schedule at step $m$ |
| $\boldsymbol{\epsilon} = [\boldsymbol{\epsilon}^t, \boldsymbol{\epsilon}^r]$ | Gaussian noise ($\mathbb{R}^6$ per link) |
| $\hat{\boldsymbol{\epsilon}}_\theta$ | Predicted noise by denoiser $f_\theta$ |
| $f_\theta(\mathbf{V}_O, \mathbf{V}_R^{(m)}, m)$ | Denoiser (transformer) |
| $s$ | CFG guidance scale (1.5) |

---

## Method Section Equations

### Object Encoding (TripoSG VAE)

$$\mathbf{z}_O = \mu\big(\Phi_\text{obj}^\text{VAE}(\hat{\mathcal{O}}, \mathbf{n}_\mathcal{O};\, P)\big) \in \mathbb{R}^{P \times d_\text{vae}}$$

$$\mathbf{V}_O = [\mathbf{z}_O \,\|\, s_O \cdot \mathbf{1}_P] \in \mathbb{R}^{P \times (d_\text{vae}+1)}$$

where $\hat{\mathcal{O}}$ is the $L^\infty$-normalized point cloud, and $s_O$ is the normalization scale.

### Link Embedding

$$\mathbf{e}_l = \phi_\text{link}\Big(\mu\big(\Phi_\text{link}^\text{VAE}(\hat{\mathcal{P}}_l, \mathbf{n}_l;\, 1)\big)\Big) \in \mathbb{R}^{d_\text{link}}$$

Pre-computed once; $\hat{\mathcal{P}}_l$ is the unit-ball-normalized link point cloud.

### Robot Token

$$\mathbf{V}_R^{(0)}[l] = \Big[\frac{\mathbf{t}_l - \mathbf{c}_O}{s_O},\; \mathbf{r}_l,\; \mathbf{e}_l\Big] \in \mathbb{R}^{3+3+d_\text{link}}$$

### Forward Diffusion

$$\mathbf{V}_R^{(m)} = \sqrt{\bar{\alpha}_m}\, \mathbf{V}_R^{(0)}_{[0:6]} + \sqrt{1-\bar{\alpha}_m}\, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

(Applied to the 6D pose part only; link embedding $\mathbf{e}_l$ is concatenated unchanged.)

### Denoiser

$$\hat{\boldsymbol{\epsilon}}_\theta = f_\theta(\mathbf{V}_O, [\mathbf{V}_R^{(m)}_{[0:6]} \,\|\, \mathbf{e}], m)$$

where $f_\theta$ is a 6-layer transformer with AdaLN-Zero time conditioning:

- **Self-attention**: Robot tokens attend to each other
- **Cross-attention**: Robot tokens attend to object tokens
- **FFN**: Standard GELU feed-forward

### Training Loss

$$\mathcal{L} = \mathbb{E}_{m, \boldsymbol{\epsilon}} \Bigg[\frac{\sum_{l=1}^{L} \mathbf{M}_l \cdot \|\boldsymbol{\epsilon}_l - \hat{\boldsymbol{\epsilon}}_{\theta,l}\|^2}{\sum_{l=1}^{L} \mathbf{M}_l}\Bigg]$$

With separate tracking: $\mathcal{L} = w_t \mathcal{L}_\text{trans} + w_r \mathcal{L}_\text{rot}$.

### Classifier-Free Guidance (Training)

With probability $p_\text{uncond}$: $\mathbf{V}_O \leftarrow \mathbf{0}$, cross-attention disabled.

### Classifier-Free Guidance (Inference)

$$\hat{\boldsymbol{\epsilon}}_\text{cfg} = (1-s)\, \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{0}, \cdot, m) + s\, \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{V}_O, \cdot, m)$$

### DDIM Sampling

$$\mathbf{x}_{m-1} = \sqrt{\bar{\alpha}_{m-1}}\, \hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{m-1}-\sigma_m^2}\, \hat{\boldsymbol{\epsilon}}_\text{cfg} + \lambda\sigma_m\, \mathbf{z}$$

where $\hat{\mathbf{x}}_0 = (\mathbf{x}_m - \sqrt{1-\bar{\alpha}_m}\, \hat{\boldsymbol{\epsilon}}_\text{cfg}) / \sqrt{\bar{\alpha}_m}$.

---

## Recommended Paper Notation Style

1. Use **boldface lowercase** for vectors: $\mathbf{t}, \mathbf{r}, \mathbf{e}, \boldsymbol{\epsilon}$
2. Use **boldface uppercase** for matrices/tensors: $\mathbf{V}_O, \mathbf{V}_R, \mathbf{M}$
3. Use **calligraphic** for sets: $\mathcal{E}, \mathcal{O}$
4. Use **subscript** for component type: $\boldsymbol{\epsilon}^t$ (translation), $\boldsymbol{\epsilon}^r$ (rotation)
5. Use **superscript in parens** for diffusion step: $\mathbf{x}^{(m)}$
6. Use $\theta$ for learnable params: $f_\theta, \hat{\boldsymbol{\epsilon}}_\theta$
7. Use $\Phi$ for frozen encoders: $\Phi^\text{VAE}$
8. Use $\phi$ for small learnable projections: $\phi_\text{link}$
