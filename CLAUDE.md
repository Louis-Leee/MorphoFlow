# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

T(R,O) Grasp is a diffusion-based framework for generating dexterous grasps across multiple robotic hands. It uses a graph representation (T(R,O) Graph) of robot-object spatial transformations with a graph diffusion model for both unconditioned and palm-conditioned grasp synthesis.

## Commands

### Environment Setup
```bash
conda create -n tro python==3.10 -y
conda activate tro
pip install -r requirements.txt
```

### Training
```bash
python train.py --config config/train.yaml
```
Checkpoints save to `$train.save_dir$/ckpt/` every `save_interval` epochs. W&B tracks experiments.

### Testing / Evaluation
```bash
python test.py --config config/test_palm_unconditioned.yaml
# Use JAX_PLATFORM_NAME=cpu to reduce GPU memory consumption for IK solving
export JAX_PLATFORM_NAME=cpu
```
Per-robot test configs exist: `config/test_palm_unconditioned_{allegro,barrett,shadow,leap,ezgripper,robotiq_3finger}.yaml`. Outputs `res.txt` (metrics) and `vis.pt` (visualization data) to `test.save_dir`.

### Visualization
```bash
python vis.py  # Interactive 3D via Viser; requires vis.pt from test.py
```

### Batch Evaluation
```bash
bash eval.sh  # Runs test.py across multiple embodiments
```

## Architecture

### Data Flow (Training)
1. **CMapDataset** (`dataset/CMapDataset.py`) loads (robot, object, grasp) tuples from `.pt` files in `data/CMapDataset_filtered/`
2. **RobotGraph** (`model/tro_graph.py`) — the core model — builds a heterogeneous graph:
   - **Object nodes**: 25 patches encoded by a frozen VQ-VAE (PointNet++ encoder in `model/vqvae/`) → 68D features (xyz + scale + 64D VQ codes)
   - **Robot nodes**: up to 25 links with 6D poses + 128D BPS embeddings → 70D features
   - **Edges**: relative SE3 transforms between robot-robot (E_RR) and object-robot (E_OR) pairs
3. **Forward diffusion** adds Gaussian noise to target robot link poses
4. **GraphDenoiser** (`model/denoiser.py`) predicts noise via 6 graph attention layers (object→robot cross-attention + robot→robot self-attention)
5. **Loss**: weighted MSE on translation (3D) and rotation (3D) noise predictions

### Data Flow (Inference)
1. DDIM sampling (20–100 steps) denoises robot poses from noise
2. Two modes: `unconditioned` (pure noise start) or `palm_conditioned` (initialized palm rotation + guidance)
3. **Pyroki IK** (`utils/pyroki_ik.py`) — JAX-based inverse kinematics — retargets denoised link positions to joint angles
4. **Isaac Gym validation** (`validation/`) simulates grasp closure and computes success metrics

### Key Modules
- `model/tro_graph.py` — `RobotGraph`: graph construction, diffusion scheduling, forward/inference logic
- `model/denoiser.py` — `GraphDenoiser`: graph neural network with alternating cross/self-attention
- `model/vqvae/` — Frozen PointNet++ VQ-VAE for object point cloud encoding (weights: `ckpt/vqvae.ckpt`)
- `utils/hand_model.py` — `HandModel`: forward kinematics via pytorch-kinematics, link geometry loading from URDFs
- `utils/rotation.py` — SE3/SO3 conversions and batch relative transform computation
- `utils/pyroki_ik.py` — `PyrokiRetarget`: batched JAX IK solver for joint angle recovery
- `dataset/CMapDataset.py` — Dataset with random PC sampling (512 from 65536 points), noise augmentation, and partial observation support

### Configuration System
All configs are OmegaConf YAML files in `config/`. Key sections:
- `dataset`: robot_names, batch_size, object_pc_type (`random`/`fixed`/`partial`)
- `model`: vqvae_cfg, bps_config, diffusion_config (M=1000 steps, ddim_steps), denoiser_config (num_layers=6), inference_config
- `train`/`test`: lr, epochs, checkpoint paths, GPU selection

### Supported Robot Hands
Allegro (21 links), Barrett (10 links), ShadowHand (17 links), LeapHand (17 links), EZGripper, Robotiq 3-Finger. URDF files and metadata live in `data/data_urdf/`.

## Key Dependencies
- **PyTorch** (2.5.1+cu121) — core DL framework
- **JAX** (0.6.2) — used by Pyroki for fast batched IK
- **pytorch-kinematics** — forward kinematics from URDFs
- **pytorch3d** — 3D transform operations
- **theseus** — Lie group (SO3/SE3) operations
- **bps_torch** — Ball Point Set features for link embeddings
- **pyroki** — JAX-based inverse kinematics (installed from git)
- **Isaac Gym** — physics validation (separate conda env, Python 3.8)
- **wandb** — experiment tracking
- **viser** — interactive 3D visualization
