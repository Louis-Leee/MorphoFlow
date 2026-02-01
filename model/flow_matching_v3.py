"""
FlowMatchingV3: Flow matching with FlashAttentionDenoiserNoEdge.

Based on tro_graph_v3.py (RobotGraphV3 / diff_v3) with the generative
framework changed from diffusion to flow matching:
1. Linear interpolation: x_t = (1-σ)·x_target + σ·noise  (both trans & rot)
2. Velocity matching loss: v = noise - x_target
3. Euler ODE integration for inference via FlowMatchEulerDiscreteScheduler

Same network architecture (FlashAttentionDenoiserNoEdge, no edge computation).
This is a clean comparison: the ONLY difference from diff_v3 is diffusion vs
flow matching — everything else (denoiser, BPS, VQVAE) is identical.
"""

import math
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from theseus.geometry.so3 import SO3
from diffusers import FlowMatchEulerDiscreteScheduler

from bps_torch.bps import bps_torch
from model.vqvae.vq_vae import VQVAE
from utils.rotation import (
    vector_to_matrix,
    matrix_to_vector,
)
from model.flash_denoiser_noedge import FlashAttentionDenoiserNoEdge
from utils.hand_model import create_hand_model


class FlowMatchingV3(nn.Module):

    def __init__(
        self,
        vqvae_cfg,
        vqvae_pretrain,
        object_patch,
        max_link_node,
        robot_links,
        inference_config,
        bps_config,
        N_t_training,
        flow_matching_config,
        denoiser_config,
        embodiment,
        loss_config,
        mode="train",
    ):
        super(FlowMatchingV3, self).__init__()

        # ---- VQ-VAE encoder (frozen) ----
        if isinstance(vqvae_cfg, dict):
            vqvae_cfg = OmegaConf.create(vqvae_cfg)
        self.vqvae = VQVAE(vqvae_cfg)
        if vqvae_pretrain is not None:
            state_dict = torch.load(vqvae_pretrain, map_location="cpu")
            self.vqvae.load_state_dict(state_dict)
            print(f"Loaded pretrained VQVAE from {vqvae_pretrain}.")
        for param in self.vqvae.parameters():
            param.requires_grad = False

        # ---- Meta ----
        self.embodiment = embodiment
        self.hand_dict = {}
        for hand_name in self.embodiment:
            self.hand_dict[hand_name] = create_hand_model(hand_name)

        self.object_patch = object_patch
        self.max_link_node = max_link_node

        # ---- Link embedding (BPS) ----
        self.robot_links = robot_links
        if isinstance(bps_config, dict):
            bps_config = OmegaConf.create(bps_config)
        self.link_embed_dim = bps_config.n_bps_points + 4
        self.bps = bps_torch(**bps_config)
        self.link_token_encoder = nn.Sequential(
            nn.Linear(self.link_embed_dim, self.link_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.link_embed_dim, self.link_embed_dim),
        )

        # ---- Flow matching schedule ----
        self.N_t_training = N_t_training
        if isinstance(flow_matching_config, dict):
            flow_matching_config = OmegaConf.create(flow_matching_config)
        self.init_flow_matching(flow_matching_config)

        # ---- Flash Attention Denoiser (No Edge) ----
        denoiser_dict = (
            OmegaConf.to_container(denoiser_config, resolve=True)
            if not isinstance(denoiser_config, dict)
            else denoiser_config
        )
        self.denoiser = FlashAttentionDenoiserNoEdge(
            M=flow_matching_config["M"],
            object_patch=self.object_patch,
            max_link_node=self.max_link_node,
            **denoiser_dict,
        )

        self.mode = mode
        self.link_embeddings = self.construct_bps()

        if self.mode == "train":
            self.loss_config = loss_config
        elif self.mode == "test":
            if not isinstance(inference_config, dict):
                inference_config = OmegaConf.to_container(
                    inference_config, resolve=True
                )
            self.inference_mode = inference_config["inference_mode"]
            assert self.inference_mode in ["unconditioned", "palm_conditioned"]
            if self.inference_mode == "palm_conditioned":
                self.palm_names = inference_config["palm_names"]
                self.t_start = inference_config.get("t_start", 0.15)
                self.interpolation_rate = inference_config["interpolation_rate"]
                self.interpolation_clip = (
                    inference_config["interpolation_clip"] * math.pi / 180
                )

    # ------------------------------------------------------------------
    # Flow matching init
    # ------------------------------------------------------------------
    def init_flow_matching(self, cfg):
        self.M = cfg["M"]
        self.ode_steps = cfg.get("ode_steps", 100)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.M,
            shift=1.0,
        )

    # ------------------------------------------------------------------
    # Helper methods (identical to tro_graph_v3.py)
    # ------------------------------------------------------------------
    def construct_bps(self):
        link_embedding_dict = {}
        for embodiment, hand_model in self.hand_dict.items():
            links_pc = hand_model.links_pc
            embodiment_bps = []
            for link_name, link_pc in links_pc.items():
                centroid, scale = self._unit_ball_(link_pc)
                link_pc = (link_pc - centroid) / scale
                link_bps = self.bps.encode(
                    link_pc,
                    feature_type=["dists"],
                    x_features=None,
                    custom_basis=None,
                )["dists"]
                link_bps = torch.cat([link_bps, centroid, scale.view(1, 1)], dim=-1)
                embodiment_bps.append(link_bps)
            link_embedding_dict[embodiment] = torch.cat(embodiment_bps, dim=0)
        return link_embedding_dict

    def _unit_ball_(self, pc):
        centroid = torch.mean(pc, dim=0, keepdim=True)
        pc = pc - centroid
        max_radius = pc.norm(dim=-1).max()
        return centroid, max_radius

    def _normalize_pc_(self, pc):
        B, N, _ = pc.shape
        centroids = torch.mean(pc, dim=1, keepdim=True)
        pc = pc - centroids
        scale, _ = torch.max(torch.abs(pc), dim=1, keepdim=True)
        scale, _ = torch.max(scale, dim=2, keepdim=True)
        pc = pc / scale
        return pc, centroids, scale

    def _expand_and_reshape_(self, x, name):
        shape = x.shape
        B = x.shape[0]
        if len(shape) == 3:  # Node [B, N, D]
            return (
                x[:, None, :, :]
                .expand(-1, self.N_t_training, -1, -1)
                .reshape(B * self.N_t_training, shape[1], shape[2])
            )
        elif len(shape) == 4:  # Edge [B, N, M, D]
            return (
                x[:, None, :, :, :]
                .expand(-1, self.N_t_training, -1, -1, -1)
                .reshape(B * self.N_t_training, shape[1], shape[2], shape[3])
            )
        elif len(shape) == 2:  # Mask [B, N]
            return (
                x[:, None, :]
                .expand(-1, self.N_t_training, -1)
                .reshape(B * self.N_t_training, shape[1])
            )
        else:
            raise ValueError(f"Unsupported shape for {name}: {shape}")

    # ------------------------------------------------------------------
    # Forward: Flow matching velocity training
    # ------------------------------------------------------------------
    def forward(self, batch, eps=1e-8):

        object_pc = batch["object_pc"]
        B = object_pc.shape[0]
        device = object_pc.device
        dtype = object_pc.dtype

        # ---- Object nodes ----
        with torch.no_grad():
            normal_pc, centroids, scale = self._normalize_pc_(object_pc)
            object_tokens = self.vqvae.encode(normal_pc)

        object_nodes = torch.cat(
            [
                object_tokens["xyz"],
                scale.expand(-1, self.object_patch, -1),
                object_tokens["z_q"],
            ],
            dim=-1,
        )  # [B, P, 68]

        # ---- Target link nodes ----
        target_vec = batch["target_vec"]
        link_target_poses = torch.zeros(
            [B, self.max_link_node, 6], device=device, dtype=dtype
        )
        link_robot_embeds = torch.zeros(
            [B, self.max_link_node, self.link_embed_dim], device=device, dtype=dtype
        )
        link_node_masks = torch.zeros(
            [B, self.max_link_node], device=device, dtype=torch.bool
        )

        for b in range(B):
            robot_name = batch["robot_name"][b]
            num_link = self.robot_links[robot_name]
            target_pose_vec = target_vec[b]
            target_trans = target_pose_vec[:, :3]
            target_rot = target_pose_vec[:, 3:]

            object_center, object_scale = centroids[b], scale[b]
            target_trans = (target_trans - object_center) / object_scale
            target_pose_vec = torch.cat([target_trans, target_rot], dim=-1)
            link_target_poses[b, :num_link, :] = target_pose_vec

            link_bps = self.link_embeddings[robot_name].to(device)
            link_embed = self.link_token_encoder(link_bps)
            link_robot_embeds[b, :num_link, :] = link_embed
            link_node_masks[b, :num_link] = True

        robot_nodes = torch.cat([link_target_poses, link_robot_embeds], dim=-1)

        # ---- Flow matching interpolation ----
        t_idx = torch.randint(0, self.M, (B * self.N_t_training,), device=device)
        sigma = t_idx.float() / self.M  # σ ∈ [0, 1)

        V_O = self._expand_and_reshape_(object_nodes, "V_O")
        V_R = self._expand_and_reshape_(robot_nodes, "V_R")
        V_R_trans, V_R_rot, V_R_embed = V_R[:, :, :3], V_R[:, :, 3:6], V_R[:, :, 6:]

        # Sample noise
        noise_trans = torch.randn_like(V_R_trans)
        noise_rot = torch.randn_like(V_R_rot)

        # Direct linear interpolation for both trans and rot
        sigma_bc = sigma[:, None, None]  # [BT, 1, 1]
        noisy_trans = (1 - sigma_bc) * V_R_trans + sigma_bc * noise_trans
        noisy_rot = (1 - sigma_bc) * V_R_rot + sigma_bc * noise_rot
        noisy_V_R = torch.cat([noisy_trans, noisy_rot, V_R_embed], dim=-1)

        # Velocity targets: v = noise - target (points from clean → noisy)
        v_target_trans = noise_trans - V_R_trans
        v_target_rot = noise_rot - V_R_rot

        # ---- Predict velocity (no edge computation) ----
        pred_velocity = self.denoiser(V_O, noisy_V_R, t_idx)

        # ---- Velocity matching loss ----
        M_V_R = self._expand_and_reshape_(link_node_masks, "M_V_R").float()
        pred_v_trans = pred_velocity[:, :, :3]
        pred_v_rot = pred_velocity[:, :, 3:]

        error_trans = (v_target_trans - pred_v_trans) ** 2
        error_trans = error_trans.mean(dim=-1)
        loss_trans = (error_trans * M_V_R).sum() / (M_V_R.sum() + eps)

        error_rot = (v_target_rot - pred_v_rot) ** 2
        error_rot = error_rot.mean(dim=-1)
        loss_rot = (error_rot * M_V_R).sum() / (M_V_R.sum() + eps)

        total_loss = (
            self.loss_config["trans_weight"] * loss_trans
            + self.loss_config["rot_weight"] * loss_rot
        )
        loss_dict = {
            "loss_rot": loss_rot,
            "loss_trans": loss_trans,
            "loss_total": total_loss,
        }
        return loss_dict

    # ------------------------------------------------------------------
    # Inference: Euler ODE integration (no edge computation)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference(self, batch, eps=1e-8):

        object_pc = batch["object_pc"]
        B = object_pc.shape[0]
        device = object_pc.device
        dtype = object_pc.dtype

        robot_name = batch["robot_name"]
        link_names = list(batch["robot_links_pc"][0].keys())
        valid_links = self.robot_links[robot_name]

        # Object nodes
        with torch.no_grad():
            normal_pc, centroids, scale = self._normalize_pc_(object_pc)
            object_tokens = self.vqvae.encode(normal_pc)

        object_nodes = torch.cat(
            [
                object_tokens["xyz"],
                scale.expand(-1, self.object_patch, -1),
                object_tokens["z_q"],
            ],
            dim=-1,
        )

        if self.inference_mode == "unconditioned":
            return self._inference_unconditioned(
                batch, object_nodes, centroids, scale,
                B, device, dtype, link_names, valid_links,
            )
        elif self.inference_mode == "palm_conditioned":
            return self._inference_palm_conditioned(
                batch, object_nodes, centroids, scale,
                B, device, dtype, link_names, valid_links, eps,
            )

    def _prepare_link_embeds(self, batch, B, device, dtype):
        """Prepare link embeddings for inference."""
        link_robot_embeds = torch.zeros(
            [B, self.max_link_node, self.link_embed_dim], device=device, dtype=dtype
        )
        robot_name = batch["robot_name"]
        num_link = self.robot_links[robot_name]
        link_bps = self.link_embeddings[robot_name].to(device)
        link_embed = self.link_token_encoder(link_bps)
        for b in range(B):
            link_robot_embeds[b, :num_link, :] = link_embed
        return link_robot_embeds

    def _inference_unconditioned(
        self, batch, object_nodes, centroids, scale,
        B, device, dtype, link_names, valid_links,
    ):
        self.noise_scheduler.set_timesteps(self.ode_steps, device=device)
        timesteps = self.noise_scheduler.timesteps
        all_step_poses_dict = {}

        # Start from pure noise (σ=1)
        V_R_trans = torch.randn(
            [B, self.max_link_node, 3], device=device, dtype=dtype
        )
        V_R_rot = torch.randn(
            [B, self.max_link_node, 3], device=device, dtype=dtype
        )
        link_robot_embeds = self._prepare_link_embeds(batch, B, device, dtype)

        for i, t in enumerate(timesteps):
            t_idx = t.round().long().clamp(0, self.M - 1)
            t_batch = t_idx.expand(B)

            noisy_V_R = torch.cat(
                [V_R_trans, V_R_rot, link_robot_embeds], dim=-1
            )
            pred_velocity = self.denoiser(object_nodes, noisy_V_R, t_batch)

            # Euler step via scheduler (concatenate [trans, rot] for single call)
            pose = torch.cat([V_R_trans, V_R_rot], dim=-1)  # [B, L, 6]
            result = self.noise_scheduler.step(pred_velocity, t, pose)
            new_pose = result.prev_sample
            V_R_trans = new_pose[:, :, :3]
            V_R_rot = new_pose[:, :, 3:]

            # Save snapshot at every step
            pred_trans = V_R_trans * scale + centroids
            pred_pose = torch.cat([pred_trans, V_R_rot], dim=-1)

            predict_link_pose_dict = {}
            step_key = self.ode_steps - 1 - i
            predict_link_pose = vector_to_matrix(pred_pose[:, :valid_links])
            for link_id, link_name in enumerate(link_names):
                predict_link_pose_dict[link_name] = predict_link_pose[:, link_id]
            all_step_poses_dict[step_key] = predict_link_pose_dict

        return all_step_poses_dict

    def _inference_palm_conditioned(
        self, batch, object_nodes, centroids, scale,
        B, device, dtype, link_names, valid_links, eps,
    ):
        robot_name = batch["robot_name"]
        palm_name = self.palm_names[robot_name]
        palm_index = link_names.index(palm_name)
        palm_R_target = batch["initial_se3"][:, palm_index, :3, :3]
        initial_pose = matrix_to_vector(batch["initial_se3"])

        self.noise_scheduler.set_timesteps(self.ode_steps, device=device)
        timesteps = self.noise_scheduler.timesteps

        # sigma_start in timestep space
        sigma_start = self.t_start
        sigma_start_t = torch.tensor(sigma_start * self.M, device=device)
        start_idx = int(torch.argmin(torch.abs(timesteps - sigma_start_t)).item())

        # Initialize rotation from initial pose with direct linear interpolation
        link_robot_rots = torch.zeros(
            [B, self.max_link_node, 3], device=device, dtype=dtype
        )
        link_robot_embeds = self._prepare_link_embeds(batch, B, device, dtype)
        num_link = self.robot_links[robot_name]
        for b in range(B):
            link_robot_rots[b, :num_link] = initial_pose[b, :num_link, 3:]

        # Direct linear interpolation to sigma_start
        noise_rot = torch.randn_like(link_robot_rots)
        noise_trans = torch.randn_like(link_robot_rots)
        V_R_rot = (1 - sigma_start) * link_robot_rots + sigma_start * noise_rot
        V_R_trans = sigma_start * noise_trans  # target_trans unknown → 0

        all_step_poses_dict = {}
        for i in range(start_idx, len(timesteps)):
            t = timesteps[i]
            t_idx = t.round().long().clamp(0, self.M - 1)
            t_batch = t_idx.expand(B)

            noisy_V_R = torch.cat(
                [V_R_trans, V_R_rot, link_robot_embeds], dim=-1
            )
            pred_velocity = self.denoiser(object_nodes, noisy_V_R, t_batch)

            # Euler step via scheduler
            pose = torch.cat([V_R_trans, V_R_rot], dim=-1)
            result = self.noise_scheduler.step(pred_velocity, t, pose)
            new_pose = result.prev_sample
            V_R_trans = new_pose[:, :, :3]
            V_R_rot = new_pose[:, :, 3:]

            # Palm SO(3) rotation correction (guidance, not interpolation)
            progress = (i - start_idx + 1) / (len(timesteps) - start_idx)
            s_t = self.interpolation_rate * math.sin(0.5 * progress * math.pi)

            R_all = SO3.exp_map(
                V_R_rot.reshape(-1, 3)
            ).to_matrix().reshape(B, self.max_link_node, 3, 3)

            R_palm_current = R_all[:, palm_index]
            R_err = palm_R_target @ R_palm_current.transpose(-2, -1)
            r_err = SO3(tensor=R_err).log_map()

            theta = r_err.norm(dim=-1, keepdim=True).clamp_min(eps)
            torch.clamp(theta, max=self.interpolation_clip)

            r_delta = s_t * r_err
            R_delta = SO3.exp_map(r_delta).to_matrix()[:, None, :, :]
            R_corrected = torch.matmul(R_delta, R_all)
            V_R_rot = SO3(
                tensor=R_corrected.reshape(-1, 3, 3)
            ).log_map().reshape(B, self.max_link_node, 3)

            # Save snapshot
            pred_trans = V_R_trans * scale + centroids
            pred_pose = torch.cat([pred_trans, V_R_rot], dim=-1)

            predict_link_pose_dict = {}
            step_key = len(timesteps) - 1 - i
            predict_link_pose = vector_to_matrix(pred_pose[:, :valid_links])
            for link_id, link_name in enumerate(link_names):
                predict_link_pose_dict[link_name] = predict_link_pose[:, link_id]
            all_step_poses_dict[step_key] = predict_link_pose_dict

        return all_step_poses_dict
