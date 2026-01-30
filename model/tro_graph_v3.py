"""
RobotGraphV3: Diffusion model using FlashAttentionDenoiserNoEdge.

Based on tro_graph_v2.py (RobotGraphV2) with one change:
1. FlashAttentionDenoiserNoEdge replaces FlashAttentionDenoiser
2. All explicit edge computation (_build_noisy_edges) removed
3. All diffusion logic (schedule, forward, DDIM sampling, palm conditioning) preserved

This is a clean ablation: the ONLY difference from diff_v2 is removal of
explicit SE(3) edge features. The model must learn spatial relationships
purely from absolute poses in node features.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from theseus.geometry.so3 import SO3

from bps_torch.bps import bps_torch
from model.vqvae.vq_vae import VQVAE
from utils.rotation import (
    vector_to_matrix,
    matrix_to_vector,
)
from model.flash_denoiser_noedge import FlashAttentionDenoiserNoEdge
from utils.hand_model import create_hand_model


class RobotGraphV3(nn.Module):

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
        diffusion_config,
        denoiser_config,
        embodiment,
        loss_config,
        mode="train",
    ):
        super(RobotGraphV3, self).__init__()

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

        # ---- Diffusion schedule ----
        self.N_t_training = N_t_training
        if isinstance(diffusion_config, dict):
            diffusion_config = OmegaConf.create(diffusion_config)
        self.init_diffusion(diffusion_config)

        # ---- Flash Attention Denoiser (No Edge) ----
        denoiser_dict = (
            OmegaConf.to_container(denoiser_config, resolve=True)
            if not isinstance(denoiser_config, dict)
            else denoiser_config
        )
        self.denoiser = FlashAttentionDenoiserNoEdge(
            M=diffusion_config["M"],
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
                self.interpolation_clip = (
                    inference_config["interpolation_clip"] * math.pi / 180
                )
                self.interpolation_rate = inference_config["interpolation_rate"]
                self.rotation_error = (
                    inference_config["rotation_error"] * math.pi / 180
                )
                self.t_star = self.get_start_timestamp()

    # ------------------------------------------------------------------
    # Diffusion schedule (identical to tro_graph.py)
    # ------------------------------------------------------------------
    def init_diffusion(self, cfg):
        self.M = cfg["M"]
        self.scheduling = cfg["scheduling"]
        if self.scheduling == "linear":
            self.beta_min, self.beta_max = cfg["beta_min"], cfg["beta_max"]
            betas = torch.linspace(self.beta_min, self.beta_max, self.M)
        else:
            raise NotImplementedError()
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer(
            "alpha_bars",
            torch.tensor(
                [torch.prod(self.alphas[: i + 1]) for i in range(len(self.alphas))]
            ),
        )
        self.ddim_steps = cfg["ddim_steps"]
        self.eta = cfg["ddim_eta"]
        self.noise_lambda = cfg["lambda"]

    # ------------------------------------------------------------------
    # Helper methods (identical to tro_graph.py)
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

    # NOTE: No _build_noisy_edges() method -- edges are not computed in v3.

    # ------------------------------------------------------------------
    # Forward: Diffusion training (identical logic, no edge computation)
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

        # ---- Forward diffusion ----
        t = np.random.randint(0, self.M, (B * self.N_t_training))
        V_O = self._expand_and_reshape_(object_nodes, "V_O")
        V_R = self._expand_and_reshape_(robot_nodes, "V_R")
        V_R_trans, V_R_rot, V_R_embed = V_R[:, :, :3], V_R[:, :, 3:6], V_R[:, :, 6:]

        eta_V_R_trans = torch.randn_like(V_R_trans)
        eta_V_R_rot = torch.randn_like(V_R_rot)
        a_bar = self.alpha_bars[t][:, None, None]

        noisy_trans = a_bar.sqrt() * V_R_trans + (1 - a_bar).sqrt() * eta_V_R_trans
        noisy_rot = a_bar.sqrt() * V_R_rot + (1 - a_bar).sqrt() * eta_V_R_rot
        noisy_V_R = torch.cat([noisy_trans, noisy_rot, V_R_embed], dim=-1)

        # ---- Predict noise (no edge computation) ----
        pred_link_noise = self.denoiser(V_O, noisy_V_R, t)

        # ---- Loss ----
        M_V_R = self._expand_and_reshape_(link_node_masks, "M_V_R").float()
        pred_trans_noise = pred_link_noise[:, :, :3]
        pred_rot_noise = pred_link_noise[:, :, 3:]

        error_trans_noise = (eta_V_R_trans - pred_trans_noise) ** 2
        error_trans_noise = error_trans_noise.mean(dim=-1)
        loss_trans_noise = (error_trans_noise * M_V_R).sum() / (M_V_R.sum() + eps)

        error_rot_noise = (eta_V_R_rot - pred_rot_noise) ** 2
        error_rot_noise = error_rot_noise.mean(dim=-1)
        loss_rot_noise = (error_rot_noise * M_V_R).sum() / (M_V_R.sum() + eps)

        total_loss = (
            self.loss_config["trans_weight"] * loss_trans_noise
            + self.loss_config["rot_weight"] * loss_rot_noise
        )
        loss_dict = {
            "loss_rot": loss_rot_noise,
            "loss_trans": loss_trans_noise,
            "loss_total": total_loss,
        }
        return loss_dict

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def get_start_timestamp(self, mu=1.596, eps=1e-8):
        target = torch.tensor(self.rotation_error / mu).clamp_min(eps) ** 2
        idx = torch.argmin(torch.abs((1.0 - self.alpha_bars) - target)).item()
        return int(idx)

    # ------------------------------------------------------------------
    # Inference: DDIM sampling (no edge computation)
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
        step = self.M // self.ddim_steps
        ddim_t = torch.arange(self.M - 1, -1, -step, device=device, dtype=torch.long)
        all_diffuse_step_poses_dict = {}

        # Start from pure noise
        noisy_V_R_trans = torch.randn(
            [B, self.max_link_node, 3], device=device, dtype=dtype
        )
        noisy_V_R_rot = torch.randn(
            [B, self.max_link_node, 3], device=device, dtype=dtype
        )
        link_robot_embeds = self._prepare_link_embeds(batch, B, device, dtype)
        noisy_V_R = torch.cat(
            [noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1
        )

        for i, diffuse_step in enumerate(ddim_t):
            diffuse_step = diffuse_step.item()

            # Predict noise (no edge computation)
            pred_link_pose_noise = self.denoiser(
                object_nodes,
                noisy_V_R,
                t=torch.full(
                    (object_nodes.shape[0],),
                    diffuse_step,
                    dtype=torch.long,
                    device=device,
                ),
            )
            pred_link_trans_noise = pred_link_pose_noise[:, :, :3]
            pred_link_rot_noise = pred_link_pose_noise[:, :, 3:]

            # Predict x_0
            a_bar_t = self.alpha_bars[diffuse_step]
            if i == len(ddim_t) - 1:
                a_bar_prev = torch.tensor(1.0, device=device, dtype=dtype)
            else:
                a_bar_prev = self.alpha_bars[ddim_t[i + 1]]

            x_t_trans = noisy_V_R_trans
            x_t_rot = noisy_V_R_rot
            x_0_trans = (
                x_t_trans - (1 - a_bar_t).sqrt() * pred_link_trans_noise
            ) / a_bar_t.sqrt()
            x_0_rot = (
                x_t_rot - (1 - a_bar_t).sqrt() * pred_link_rot_noise
            ) / a_bar_t.sqrt()

            # DDIM step
            sigma_t = self.eta * torch.sqrt(
                ((1 - a_bar_prev) / (1 - a_bar_t)) * (1 - a_bar_t / a_bar_prev)
            )
            ddim_coeffient = torch.sqrt(1 - a_bar_prev - sigma_t**2)

            if i == len(ddim_t) - 1:
                z_trans = torch.zeros_like(x_0_trans)
                z_rot = torch.zeros_like(x_0_rot)
            else:
                z_trans = torch.randn_like(x_0_trans)
                z_rot = torch.randn_like(x_0_rot)

            x_prev_trans = (
                a_bar_prev.sqrt() * x_0_trans
                + ddim_coeffient * pred_link_trans_noise
                + sigma_t * z_trans * self.noise_lambda
            )
            x_prev_rot = (
                a_bar_prev.sqrt() * x_0_rot
                + ddim_coeffient * pred_link_rot_noise
                + sigma_t * z_rot * self.noise_lambda
            )

            noisy_V_R_trans = x_prev_trans
            noisy_V_R_rot = x_prev_rot

            # Update node features (no edge rebuild needed)
            noisy_V_R = torch.cat(
                [noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1
            )

            # Save snapshot
            pred_trans = noisy_V_R_trans * scale + centroids
            pred_rot = noisy_V_R_rot
            pred_pose = torch.cat([pred_trans, pred_rot], dim=-1)

            predict_link_pose_dict = {}
            denoised_step = ddim_t[i + 1].item() if i < len(ddim_t) - 1 else 0
            predict_link_pose = vector_to_matrix(pred_pose[:, :valid_links])
            for link_id, link_name in enumerate(link_names):
                predict_link_pose_dict[link_name] = predict_link_pose[:, link_id]
            all_diffuse_step_poses_dict[denoised_step] = predict_link_pose_dict

        return all_diffuse_step_poses_dict

    def _inference_palm_conditioned(
        self, batch, object_nodes, centroids, scale,
        B, device, dtype, link_names, valid_links, eps,
    ):
        robot_name = batch["robot_name"]
        palm_name = self.palm_names[robot_name]
        palm_index = link_names.index(palm_name)
        palm_r3 = batch["initial_se3"][:, palm_index][:, :3, :3]
        initial_pose = matrix_to_vector(batch["initial_se3"])

        step = self.M // self.ddim_steps
        ddim_t = torch.arange(self.M - 1, -1, -step, device=device, dtype=torch.long)
        start_idx = int(torch.argmin(torch.abs(ddim_t - self.t_star)).item())
        t_start = int(ddim_t[start_idx].item())
        a_bar_s = self.alpha_bars[t_start]

        # Initial links
        link_robot_rots = torch.zeros(
            [B, self.max_link_node, 3], device=device, dtype=dtype
        )
        link_robot_embeds = self._prepare_link_embeds(batch, B, device, dtype)
        num_link = self.robot_links[robot_name]
        for b in range(B):
            link_robot_rots[b, :num_link] = initial_pose[b, :, 3:]

        noisy_V_R_rot = (
            a_bar_s.sqrt() * link_robot_rots
            + (1.0 - a_bar_s).sqrt() * torch.randn_like(link_robot_rots)
        )
        noisy_V_R_trans = (1.0 - a_bar_s).sqrt() * torch.randn_like(link_robot_rots)
        noisy_V_R = torch.cat(
            [noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1
        )

        all_diffuse_step_poses_dict = {}
        for i in range(start_idx, len(ddim_t)):
            diffuse_step = int(ddim_t[i].item())

            # Predict noise (no edge computation)
            pred_link_pose_noise = self.denoiser(
                object_nodes,
                noisy_V_R,
                t=torch.full(
                    (object_nodes.shape[0],),
                    diffuse_step,
                    dtype=torch.long,
                    device=device,
                ),
            )
            pred_link_trans_noise = pred_link_pose_noise[:, :, :3]
            pred_link_rot_noise = pred_link_pose_noise[:, :, 3:]

            # Predict x_0
            a_bar_t = self.alpha_bars[diffuse_step]
            x_t_trans = noisy_V_R_trans
            x_t_rot = noisy_V_R_rot
            x_0_trans = (
                x_t_trans - (1 - a_bar_t).sqrt() * pred_link_trans_noise
            ) / a_bar_t.sqrt()
            x_0_rot = (
                x_t_rot - (1 - a_bar_t).sqrt() * pred_link_rot_noise
            ) / a_bar_t.sqrt()

            # ---- Palm rotation guidance ----
            progress = (i + 1) / self.ddim_steps
            s_t = self.interpolation_rate * math.sin(0.5 * progress * math.pi)

            with torch.enable_grad():
                pred_link_rot_noise_with_grad = (
                    pred_link_rot_noise.detach().clone().requires_grad_(True)
                )
                x_0_rot_with_grad = (
                    x_t_rot
                    - (1 - a_bar_t).sqrt() * pred_link_rot_noise_with_grad
                ) / a_bar_t.sqrt()

                BL = x_0_rot_with_grad.shape[0]
                L = x_0_rot_with_grad.shape[1]
                R_cur_all = SO3.exp_map(
                    x_0_rot_with_grad.reshape(-1, 3)
                ).to_matrix().reshape(BL, L, 3, 3)

                r_cur = x_0_rot_with_grad[:, palm_index]
                R_cur = SO3.exp_map(r_cur).to_matrix()
                R_init = palm_r3.detach().expand(BL, 3, 3)

                # Interpolation
                R_err = R_init @ torch.linalg.inv(R_cur)
                r_err = SO3(tensor=R_err).log_map()

                theta = r_err.norm(dim=-1, keepdim=True).clamp_min(eps)
                torch.clamp(theta, max=self.interpolation_clip)
                delta_rate = s_t

                r_delta = delta_rate * r_err
                R_delta = SO3.exp_map(r_delta).to_matrix()[:, None, :, :]
                R_intered = torch.matmul(R_delta, R_cur_all)
                r_intered = (
                    SO3(tensor=R_intered.reshape(-1, 3, 3))
                    .log_map()
                    .reshape(BL, L, 3)
                )
                pred_link_rot_noise = (
                    x_t_rot - a_bar_t.sqrt() * r_intered
                ) / (1 - a_bar_t).sqrt()

                x_0_rot = (
                    x_t_rot - (1 - a_bar_t).sqrt() * pred_link_rot_noise
                ) / a_bar_t.sqrt()

            # DDIM step
            if i == len(ddim_t) - 1:
                a_bar_prev = torch.tensor(1.0, device=device, dtype=dtype)
            else:
                a_bar_prev = self.alpha_bars[ddim_t[i + 1]]

            sigma_t = self.eta * torch.sqrt(
                ((1 - a_bar_prev) / (1 - a_bar_t)) * (1 - a_bar_t / a_bar_prev)
            )
            ddim_coeffient = torch.sqrt(1 - a_bar_prev - sigma_t**2)

            if i == len(ddim_t) - 1:
                z_trans = torch.zeros_like(x_0_trans)
                z_rot = torch.zeros_like(x_0_rot)
            else:
                z_trans = torch.randn_like(x_0_trans)
                z_rot = torch.randn_like(x_0_rot)

            x_prev_trans = (
                a_bar_prev.sqrt() * x_0_trans
                + ddim_coeffient * pred_link_trans_noise
                + sigma_t * z_trans * self.noise_lambda
            )
            x_prev_rot = (
                a_bar_prev.sqrt() * x_0_rot
                + ddim_coeffient * pred_link_rot_noise
                + sigma_t * z_rot * self.noise_lambda
            )

            noisy_V_R_trans = x_prev_trans
            noisy_V_R_rot = x_prev_rot

            # Update node features (no edge rebuild needed)
            noisy_V_R = torch.cat(
                [noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1
            )

            # Save snapshot
            pred_trans = noisy_V_R_trans * scale + centroids
            pred_rot = noisy_V_R_rot
            pred_pose = torch.cat([pred_trans, pred_rot], dim=-1)

            predict_link_pose_dict = {}
            denoised_step = ddim_t[i + 1].item() if i < len(ddim_t) - 1 else 0
            predict_link_pose = vector_to_matrix(pred_pose[:, :valid_links])
            for link_id, link_name in enumerate(link_names):
                predict_link_pose_dict[link_name] = predict_link_pose[:, link_id]
            all_diffuse_step_poses_dict[denoised_step] = predict_link_pose_dict

        return all_diffuse_step_poses_dict
