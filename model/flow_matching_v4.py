"""
FlowMatchingV4: Flow matching with proper SE(3) geodesic interpolation.

Based on FlowMatchingV3, with key improvement:
- Uses SE3FlowScheduler for proper SO(3) geodesic interpolation
- Rotation interpolation via log/exp map on SO(3) manifold
- Euler integration on SE(3) for inference

Same network architecture (FlashAttentionDenoiserNoEdge, no edge computation).
"""

import math
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from theseus.geometry.so3 import SO3

from bps_torch.bps import bps_torch
from model.vqvae.vq_vae import VQVAE
from utils.rotation import (
    vector_to_matrix,
    matrix_to_vector,
)
from model.flash_denoiser_noedge import FlashAttentionDenoiserNoEdge
from model.se3_flow_scheduler import SE3FlowScheduler
from utils.hand_model import create_hand_model


class FlowMatchingV4(nn.Module):

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
        super(FlowMatchingV4, self).__init__()

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

        # ---- SE3 Flow matching schedule ----
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
            M=self.M,
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
    # SE3 Flow matching init
    # ------------------------------------------------------------------
    def init_flow_matching(self, cfg):
        self.M = cfg.get("M", 1000)
        self.ode_steps = cfg.get("ode_steps", 50)

        self.scheduler = SE3FlowScheduler(
            sigma_min=cfg.get("sigma_min", 1e-5),
            t_schedule=cfg.get("t_schedule", "linear"),
            t_schedule_kwargs=cfg.get("t_schedule_kwargs", {}),
            shift_t=cfg.get("shift_t", 1.0),
        )

    # ------------------------------------------------------------------
    # Helper methods (identical to flow_matching_v3.py)
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
    # Forward: SE3 Flow matching velocity training
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

        # ---- SE3 Flow matching interpolation ----
        # Sample t ∈ [0, 1]
        t = self.scheduler.sample_t(B * self.N_t_training, device=device)

        # Convert timestep to discrete index for denoiser
        t_idx = (t * self.M).long().clamp(0, self.M - 1)

        V_O = self._expand_and_reshape_(object_nodes, "V_O")
        V_R = self._expand_and_reshape_(robot_nodes, "V_R")
        V_R_pose = V_R[:, :, :6]  # [BT, N, 6] (trans + rot_vec)
        V_R_embed = V_R[:, :, 6:]

        # SE3 interpolation and velocity target
        noisy_pose, noise = self.scheduler.scale_noise_se3(V_R_pose, t)
        v_target = self.scheduler.get_v_se3(V_R_pose, noise, t)

        noisy_V_R = torch.cat([noisy_pose, V_R_embed], dim=-1)

        # ---- Predict velocity (no edge computation) ----
        pred_velocity = self.denoiser(V_O, noisy_V_R, t_idx)

        # ---- Velocity matching loss ----
        M_V_R = self._expand_and_reshape_(link_node_masks, "M_V_R").float()
        pred_v_trans = pred_velocity[:, :, :3]
        pred_v_rot = pred_velocity[:, :, 3:]
        v_target_trans = v_target[:, :, :3]
        v_target_rot = v_target[:, :, 3:]

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
    # Inference: Euler ODE integration on SE(3)
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
        # Get timesteps with shifting
        timesteps = self.scheduler.get_timesteps(self.ode_steps, device=device)
        all_step_poses_dict = {}

        # Start from pure noise (t=1)
        # For SE3, initialize with random rotations and Gaussian translations
        import pytorch3d.transforms as p3dt
        V_R_trans = torch.randn(
            [B, self.max_link_node, 3], device=device, dtype=dtype
        )
        noise_rot_quat = p3dt.random_quaternions(
            B * self.max_link_node, device=device, dtype=dtype
        ).reshape(B, self.max_link_node, 4)
        V_R_rot = p3dt.so3_log_map(
            p3dt.quaternion_to_matrix(noise_rot_quat).reshape(-1, 3, 3)
        ).reshape(B, self.max_link_node, 3)

        link_robot_embeds = self._prepare_link_embeds(batch, B, device, dtype)

        for i in range(len(timesteps) - 1):
            t = timesteps[i].item()
            t_prev = timesteps[i + 1].item()

            # Discrete timestep for denoiser
            t_idx = int(t * self.M)
            t_idx = min(max(t_idx, 0), self.M - 1)
            t_batch = torch.tensor([t_idx] * B, device=device, dtype=torch.long)

            noisy_V_R = torch.cat(
                [V_R_trans, V_R_rot, link_robot_embeds], dim=-1
            )
            pred_velocity = self.denoiser(object_nodes, noisy_V_R, t_batch)

            # SE3 Euler step
            pose = torch.cat([V_R_trans, V_R_rot], dim=-1)  # [B, L, 6]
            new_pose = self.scheduler.step_se3(pred_velocity, t, t_prev, pose)
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
        import pytorch3d.transforms as p3dt

        robot_name = batch["robot_name"]
        palm_name = self.palm_names[robot_name]
        palm_index = link_names.index(palm_name)
        palm_R_target = batch["initial_se3"][:, palm_index, :3, :3]
        initial_pose = matrix_to_vector(batch["initial_se3"])

        timesteps = self.scheduler.get_timesteps(self.ode_steps, device=device)

        # sigma_start in timestep space
        sigma_start = self.t_start
        start_idx = int(sigma_start * len(timesteps))
        start_idx = min(max(start_idx, 0), len(timesteps) - 2)

        # Initialize rotation from initial pose
        link_robot_rots = torch.zeros(
            [B, self.max_link_node, 3], device=device, dtype=dtype
        )
        link_robot_embeds = self._prepare_link_embeds(batch, B, device, dtype)
        num_link = self.robot_links[robot_name]
        for b in range(B):
            link_robot_rots[b, :num_link] = initial_pose[b, :num_link, 3:]

        # SE3 interpolation to sigma_start
        noise_trans = torch.randn_like(link_robot_rots)
        noise_rot_quat = p3dt.random_quaternions(
            B * self.max_link_node, device=device, dtype=dtype
        ).reshape(B, self.max_link_node, 4)
        noise_rot = p3dt.so3_log_map(
            p3dt.quaternion_to_matrix(noise_rot_quat).reshape(-1, 3, 3)
        ).reshape(B, self.max_link_node, 3)

        # Interpolate using SE3 scheduler
        initial_rot_pose = link_robot_rots
        t_start_tensor = torch.tensor([sigma_start] * B, device=device)

        # For translation, start from noise (target unknown)
        V_R_trans = sigma_start * noise_trans

        # For rotation, interpolate from initial pose to noise
        init_rot_mat = p3dt.so3_exp_map(initial_rot_pose.reshape(-1, 3)).reshape(B, self.max_link_node, 3, 3)
        noise_rot_mat = p3dt.so3_exp_map(noise_rot.reshape(-1, 3)).reshape(B, self.max_link_node, 3, 3)
        rot_rel = torch.matmul(noise_rot_mat, init_rot_mat.transpose(-2, -1))
        rot_rel_log = p3dt.so3_log_map(rot_rel.reshape(-1, 3, 3)).reshape(B, self.max_link_node, 3)
        sigma_coef = self.scheduler.sigma_min + (1 - self.scheduler.sigma_min) * sigma_start
        rot_rel_log_scaled = sigma_coef * rot_rel_log
        scaled_rel_rot = p3dt.so3_exp_map(rot_rel_log_scaled.reshape(-1, 3)).reshape(B, self.max_link_node, 3, 3)
        V_R_rot_mat = torch.matmul(scaled_rel_rot, init_rot_mat)
        V_R_rot = p3dt.so3_log_map(V_R_rot_mat.reshape(-1, 3, 3)).reshape(B, self.max_link_node, 3)

        all_step_poses_dict = {}
        for i in range(start_idx, len(timesteps) - 1):
            t = timesteps[i].item()
            t_prev = timesteps[i + 1].item()

            t_idx = int(t * self.M)
            t_idx = min(max(t_idx, 0), self.M - 1)
            t_batch = torch.tensor([t_idx] * B, device=device, dtype=torch.long)

            noisy_V_R = torch.cat(
                [V_R_trans, V_R_rot, link_robot_embeds], dim=-1
            )
            pred_velocity = self.denoiser(object_nodes, noisy_V_R, t_batch)

            # SE3 Euler step
            pose = torch.cat([V_R_trans, V_R_rot], dim=-1)
            new_pose = self.scheduler.step_se3(pred_velocity, t, t_prev, pose)
            V_R_trans = new_pose[:, :, :3]
            V_R_rot = new_pose[:, :, 3:]

            # Palm SO(3) rotation correction (guidance)
            progress = (i - start_idx + 1) / (len(timesteps) - 1 - start_idx)
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
            step_key = len(timesteps) - 2 - i
            predict_link_pose = vector_to_matrix(pred_pose[:, :valid_links])
            for link_id, link_name in enumerate(link_names):
                predict_link_pose_dict[link_name] = predict_link_pose[:, link_id]
            all_step_poses_dict[step_key] = predict_link_pose_dict

        return all_step_poses_dict
