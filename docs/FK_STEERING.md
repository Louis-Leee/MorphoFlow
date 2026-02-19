# FK Steering: Feynman-Kac Diffusion Steering for Grasp Inference

## Overview

FK Steering applies particle-based Sequential Monte Carlo (SMC) resampling during DDIM denoising to steer grasp pose generation toward higher-quality grasps at inference time. Based on the Feynman-Kac Diffusion Steering framework, it maintains multiple particles per sample and resamples them based on grasp quality rewards at intermediate denoising steps.

## Files

| File | Role |
|------|------|
| `test_diff_v4_ce_vae_fk.py` | Evaluation script with FK Steering support |
| `config/test_diff_v4_ce_vae_fk.yaml` | Config (model + FK Steering parameters) |
| `utils/reward.py` | Reference reward functions (ERF, SPF, SRF) |
| `third_party/Fk-Diffusion-Steering/` | Reference implementation (text diffusion) |

## Usage

```bash
# Without FK Steering (identical to test_diff_v4_ce_vae.py)
python test_diff_v4_ce_vae_fk.py --config config/test_diff_v4_ce_vae_fk.yaml \
    --hands allegro --ckpt path/to/ckpt --gpu 0

# With FK Steering enabled
python test_diff_v4_ce_vae_fk.py --config config/test_diff_v4_ce_vae_fk.yaml \
    --hands barrett fk_steering.enabled=true fk_steering.num_particles=4 \
    --ckpt path/to/ckpt --gpu 0

# Use JAX on CPU to save GPU memory for IK:
JAX_PLATFORM_NAME=cpu python test_diff_v4_ce_vae_fk.py ...
```

## Backward Compatibility

`fk_steering.enabled=false` (default) makes behavior identical to `test_diff_v4_ce_vae.py`. The DDIM loop, CFG, IK, and validation are all unchanged.

## Algorithm

```
Input: trained diffusion model, object PC, hand model, IK solver

1. Expand S samples → S×k particles (independent noise per particle)
2. For each DDIM step t:
   a. Predict noise ε_θ via CFG two-pass (cond + uncond)
   b. Compute x0_pred = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t
   c. Step to x_{t-1} via DDIM
   d. If t in resample_schedule:
      ┌─────────────────────────────────────────────────┐
      │ IK-Based Reward Computation:                    │
      │   x0_pred → denormalize → SE3 matrices          │
      │   → process_transform → target positions         │
      │   → batch_retarget (JAX IK) → joint angles q    │
      │   → extract_fingertip_joints                     │
      │   → per-sample FK(q):                            │
      │       get_surface_pc → hand mesh points          │
      │       get_keypoints → self-collision points      │
      │       get_dis_keypoints → fingertip points       │
      │   → ERF + SPF + SRF → reward                    │
      └─────────────────────────────────────────────────┘
      e. Compute potentials: w = exp(λ · Δreward)
      f. (Optional) Check ESS; skip if ESS > 0.5k
      g. Multinomial resample within each group of k particles
3. Select best particle per group (highest cumulative reward)
4. Return final poses → standard IK → validation
```

## Reward Functions

| Reward | Function | Measures |
|--------|----------|----------|
| **ERF** (Elastic Repulsion Field) | `ERF_loss_single` | Penetration depth: KNN from hand→object, normal-signed distance |
| **SPF** (Surface Proximity Field) | `SPF_loss_single` | Fingertip-to-object proximity (thres=0.02) |
| **SRF** (Self-Repulsion Field) | `SRF_loss_single` | Pairwise self-collision between keypoints |

All rewards are single-sample (`[1,N,D]` → scalar). Combined as:
```
reward = -(w_ERF · ERF + w_SPF · SPF + w_SRF · SRF)
```

## Why IK-Based Rewards (Not Direct SE3)

The initial implementation computed rewards directly from x0_pred SE3 link poses. This was incorrect because:

- At intermediate DDIM steps, x0_pred represents a "peek" at clean data — the per-link SE3 transforms are **kinematically inconsistent** (links float disconnected in space)
- Computing penetration/contact from disconnected link geometry yields meaningless rewards
- FK Steering had no real effect ("用不用这个FK steering效果都差不多")

The fix runs **IK → FK(q)** at each resampling step to recover physically valid hand configurations before computing rewards:

```
Old (broken):  x0_pred SE3 → transform geometry directly → rewards (meaningless)
New (correct): x0_pred SE3 → IK solve → joint angles q → FK(q) → mesh → rewards
```

## Key Implementation Details

### get_surface_pc(hand, q)

Standalone function (not a HandModel method). Computes FK(q) and concatenates per-link point clouds with FPS downsampling for finger links:

- Palm/base links: full resolution (no downsampling)
- Finger links: FPS to 128 points (64 for shadowhand)
- Barrett: `bh_base_link` kept full; Allegro: `base_link` kept full

### Particle Expansion

```python
# S original samples → S*k particles (contiguous groups)
object_nodes_exp = object_nodes.repeat_interleave(k, dim=0)  # [S*k, P, D]
initial_q_exp = batch["initial_q"].repeat_interleave(k, dim=0)  # [S*k, DOF]
# Independent noise per particle
noisy_V_R_trans = torch.randn([S*k, max_link, 3])
```

### Grouped Resampling

Each group of k particles (belonging to the same original sample) is resampled independently:
```python
rewards_grouped = rewards.view(S, k)           # [S, k]
weights = softmax(lmbda * delta_rewards, dim=1) # [S, k]
indices = multinomial(weights, k, replacement=True)
# Gather resampled particles per group
```

### Potential Types

- **DIFF** (default): `exp(λ · (reward_t - reward_{t-1}))` — scale-invariant, robust

### Adaptive Resampling

When `adaptive_resampling=true`, skips resampling if ESS > 0.5k (particles already well-distributed).

### Fingertip Joint Extraction

Pure-rotation joints (fingertips) have zero positional Jacobian — IK cannot optimize them. After IK, these joint angles are extracted from the diffusion model's predicted SE3 rotations via `extract_fingertip_joints()`. Per-robot configurations are defined in `FINGERTIP_JOINTS` dicts.

## Config Parameters

```yaml
fk_steering:
  enabled: false            # Toggle FK Steering (false = identical to base script)
  num_particles: 4          # k particles per sample
  potential_type: 'diff'    # Potential function type
  lmbda: 5.0                # Temperature for importance weights
  resample_start: 20        # First DDIM step to resample (of 100)
  resample_end: 95          # Last DDIM step to resample
  resample_frequency: 10    # Resample every N steps (10 = ~8 IK calls)
  adaptive_resampling: true # Skip resampling when ESS > 0.5k
  reward_weights:
    ERF: 1.0                # Penetration penalty weight
    SPF: 1.0                # Fingertip contact weight
    SRF: 0.5                # Self-collision weight
```

## Supported Robots

Only robots with keypoint JSON files can use FK Steering rewards:
- `allegro` — `data/data_urdf/robot/allegro/key_points.json` + `dis_key_points.json`
- `barrett` — `data/data_urdf/robot/barrett/key_points.json` + `dis_key_points.json`
- `shadowhand` — `data/data_urdf/robot/shadowhand/key_points.json` + `dis_key_points.json`

HandModel keypoint loading is conditional (try/except with None fallback). Robots without keypoints get SPF=0, SRF=0 (only ERF applies).

## CLI Override Parsing

OmegaConf dotlist overrides (e.g. `fk_steering.enabled=true`) are passed alongside `--hands` arguments. The script filters them by detecting `=` in the argument:
```python
hands = args.hands
dotlist = [h for h in hands if "=" in h]
hands = [h for h in hands if "=" not in h]
if dotlist:
    overrides = OmegaConf.from_dotlist(dotlist)
    config = OmegaConf.merge(config, overrides)
```

## Performance Notes

- Each resampling step runs IK (JAX) + FK + KNN for all S*k particles
- `resample_frequency=10` with 100 DDIM steps = ~8 resampling calls
- Use `JAX_PLATFORM_NAME=cpu` to offload IK to CPU and save GPU memory
- `get_surface_pc` uses FPS downsampling to reduce KNN cost
