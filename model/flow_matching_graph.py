"""
FlowMatchingRobotGraph: SE(3) flow matching model for cross-embodiment dexterous grasping.

Uses proper geometric handling:
- R³ linear velocity for translation
- so(3) Lie algebra velocity for rotation (geodesic interpolation on SO(3))

Reuses GraphDenoiser and VQVAE from the existing diffusion codebase without modification.
"""

import math
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from theseus.geometry.so3 import SO3
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix

from bps_torch.bps import bps_torch
from model.vqvae.vq_vae import VQVAE
from utils.rotation import (
    vector_to_matrix,
    matrix_to_vector,
    compute_relative_se3,
    compute_batch_relative_se3,
)
from model.denoiser import GraphDenoiser
from utils.hand_model import create_hand_model


class FlowMatchingRobotGraph(nn.Module):

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
        super(FlowMatchingRobotGraph, self).__init__()

        # ---- VQ-VAE encoder (frozen) ----
        # Convert plain dicts back to OmegaConf (PN2/VQVAE use attribute access)
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
        # Ensure dict-like access for bps_config
        if isinstance(bps_config, dict):
            bps_config = OmegaConf.create(bps_config)
        self.link_embed_dim = bps_config.n_bps_points + 4  # BPS dists + centroid(3) + scale(1)
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

        # ---- Denoiser (reused directly, predicts velocity instead of noise) ----
        denoiser_dict = (
            OmegaConf.to_container(denoiser_config, resolve=True)
            if not isinstance(denoiser_config, dict)
            else denoiser_config
        )
        self.denoiser = GraphDenoiser(
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
                inference_config = OmegaConf.to_container(inference_config, resolve=True)
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
    # Flow matching init (replaces init_diffusion)
    # ------------------------------------------------------------------
    def init_flow_matching(self, cfg):
        self.M = cfg["M"]  # sinusoidal embedding table size (e.g. 1000)
        self.ode_steps = cfg["ode_steps"]  # number of ODE integration steps at inference
        self.solver = cfg.get("solver", "euler")  # "euler" or "midpoint"

    # ------------------------------------------------------------------
    # Helper methods (identical to RobotGraph)
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
    # Graph construction helpers
    # ------------------------------------------------------------------
    def _build_object_nodes(self, object_pc):
        """Encode object point cloud via frozen VQ-VAE."""
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
        )  # [B, P, 3+1+64]
        return object_nodes, centroids, scale

    def _build_robot_nodes(self, batch, B, centroids, scale, device, dtype):
        """Build target robot link poses + BPS embeddings."""
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
        return robot_nodes, link_node_masks

    def _build_edges(self, robot_nodes, object_nodes, batch, B, device, dtype):
        """Build robot-robot and robot-object relative SE(3) edges."""
        norm_robot_vec = robot_nodes[:, :, :6]

        # Robot-Robot edges
        link_rel_poses = torch.zeros(
            [B, self.max_link_node, self.max_link_node, 6], device=device, dtype=dtype
        )
        for b in range(B):
            robot_name = batch["robot_name"][b]
            num_link = self.robot_links[robot_name]
            T = vector_to_matrix(norm_robot_vec[b])[:num_link]
            rel_pose = compute_relative_se3(T, T)
            link_rel_poses[b, :num_link, :num_link, :] = matrix_to_vector(rel_pose)

        # Robot-Object edges
        link_object_rel_poses = torch.zeros(
            [B, self.max_link_node, self.object_patch, 6], device=device, dtype=dtype
        )
        object_positions = object_nodes[:, :, :3]
        object_se3 = (
            torch.eye(4, device=device, dtype=dtype)
            .expand(B, self.object_patch, -1, -1)
            .clone()
        )
        object_se3[:, :, :3, 3] = object_positions
        for b in range(B):
            robot_name = batch["robot_name"][b]
            num_link = self.robot_links[robot_name]
            T_r = vector_to_matrix(norm_robot_vec[b])[:num_link]
            T_o = object_se3[b]
            T_rel = compute_relative_se3(T_r, T_o)
            link_object_rel_poses[b, :num_link, : self.object_patch] = matrix_to_vector(
                T_rel
            )

        return link_rel_poses, link_object_rel_poses

    def _build_noisy_edges(self, noisy_V_R, V_O, device, dtype):
        """Build graph edges from noisy/interpolated robot poses."""
        noisy_V_R_se3 = vector_to_matrix(noisy_V_R[:, :, :6])
        noisy_E_RR = matrix_to_vector(
            compute_batch_relative_se3(noisy_V_R_se3, noisy_V_R_se3)
        )
        object_positions = V_O[:, :, :3]
        BT, P, _ = object_positions.shape
        object_se3 = (
            torch.eye(4, device=device, dtype=dtype).expand(BT, P, -1, -1).clone()
        )
        object_se3[:, :, :3, 3] = object_positions
        noisy_E_OR = matrix_to_vector(
            compute_batch_relative_se3(noisy_V_R_se3, object_se3)
        )
        return noisy_E_OR, noisy_E_RR

    # ------------------------------------------------------------------
    # Forward: SE(3) flow matching training
    # ------------------------------------------------------------------
    def forward(self, batch, eps=1e-8):

        object_pc = batch["object_pc"]
        B = object_pc.shape[0]
        device = object_pc.device
        dtype = object_pc.dtype

        # ---- Graph construction (identical to RobotGraph) ----
        object_nodes, centroids, scale = self._build_object_nodes(object_pc)
        robot_nodes, link_node_masks = self._build_robot_nodes(
            batch, B, centroids, scale, device, dtype
        )

        # ---- Expand for N_t_training timesteps ----
        BT = B * self.N_t_training
        V_O = self._expand_and_reshape_(object_nodes, "V_O")  # [BT, P, 68]
        V_R = self._expand_and_reshape_(robot_nodes, "V_R")  # [BT, L, 6+128]
        V_R_trans = V_R[:, :, :3]  # target translation (x₁)
        V_R_rot = V_R[:, :, 3:6]  # target rotation axis-angle (x₁)
        V_R_embed = V_R[:, :, 6:]  # link embeddings (unchanged)

        # ---- Sample continuous time t ~ U(0,1) ----
        t_continuous = torch.rand(BT, device=device, dtype=dtype)
        t_idx = (t_continuous * (self.M - 1)).long()  # discretize for denoiser

        # ---- Translation: R³ linear interpolation ----
        x0_trans = torch.randn_like(V_R_trans)  # noise
        t_bc = t_continuous[:, None, None]  # [BT, 1, 1]
        xt_trans = (1.0 - t_bc) * x0_trans + t_bc * V_R_trans
        v_target_trans = V_R_trans - x0_trans  # velocity target

        # ---- Rotation: SO(3) geodesic interpolation ----
        eps_rot = torch.randn_like(V_R_rot)  # noise in so(3)
        flat_eps = eps_rot.reshape(-1, 3)  # [BT*L, 3]
        R0_flat = SO3.exp_map(flat_eps).to_matrix()  # [BT*L, 3, 3]

        flat_x1_rot = V_R_rot.reshape(-1, 3)  # [BT*L, 3]
        R1_flat = axis_angle_to_matrix(flat_x1_rot)  # [BT*L, 3, 3]

        # Relative rotation: R₀ᵀ · R₁
        R0T_R1 = R0_flat.transpose(-2, -1) @ R1_flat  # [BT*L, 3, 3]

        # Log map: velocity target = log(R₀ᵀ · R₁) ∈ so(3)
        log_R0T_R1 = SO3(tensor=R0T_R1).log_map()  # [BT*L, 3]

        # Geodesic interpolation: Rₜ = R₀ · exp(t · log(R₀ᵀ · R₁))
        t_rot_bc = (
            t_continuous[:, None]
            .expand(-1, self.max_link_node)
            .reshape(-1, 1)
        )  # [BT*L, 1]
        scaled_log = t_rot_bc * log_R0T_R1  # [BT*L, 3]
        Rt_delta = SO3.exp_map(scaled_log).to_matrix()  # [BT*L, 3, 3]
        Rt_flat = R0_flat @ Rt_delta  # [BT*L, 3, 3]

        # Convert Rₜ to axis-angle for graph features
        xt_rot = matrix_to_axis_angle(Rt_flat).reshape(
            BT, self.max_link_node, 3
        )

        # Velocity target in so(3)
        v_target_rot = log_R0T_R1.reshape(BT, self.max_link_node, 3)

        # ---- Assemble interpolated node features ----
        noisy_V_R = torch.cat([xt_trans, xt_rot, V_R_embed], dim=-1)

        # ---- Build graph edges from interpolated poses ----
        noisy_E_OR, noisy_E_RR = self._build_noisy_edges(
            noisy_V_R, V_O, device, dtype
        )

        # ---- Predict velocity ----
        pred_velocity = self.denoiser(
            V_O, noisy_V_R, noisy_E_OR, noisy_E_RR, t_idx
        )  # [BT, L, 6]

        # ---- Loss computation ----
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
    # Inference: ODE integration on SE(3)
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

        # Object encoding
        object_nodes, centroids, scale = self._build_object_nodes(object_pc)

        if self.inference_mode == "unconditioned":
            return self._inference_unconditioned(
                batch, object_nodes, centroids, scale,
                B, device, dtype, link_names, valid_links, eps,
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

    def _ode_step_euler(self, V_R_trans, V_R_rot, pred_v_trans, pred_v_rot, dt):
        """Single Euler step on SE(3)."""
        # R³ translation
        V_R_trans = V_R_trans + dt * pred_v_trans

        # SO(3) rotation via exponential map
        R_current = axis_angle_to_matrix(V_R_rot.reshape(-1, 3))
        delta_R = SO3.exp_map(dt * pred_v_rot.reshape(-1, 3)).to_matrix()
        R_new = R_current @ delta_R
        V_R_rot = matrix_to_axis_angle(R_new).reshape(
            V_R_trans.shape[0], self.max_link_node, 3
        )
        return V_R_trans, V_R_rot

    def _inference_unconditioned(
        self, batch, object_nodes, centroids, scale,
        B, device, dtype, link_names, valid_links, eps,
    ):
        N_steps = self.ode_steps
        dt = 1.0 / N_steps
        all_step_poses_dict = {}

        # Initialize at t=0: noise prior
        V_R_trans = torch.randn([B, self.max_link_node, 3], device=device, dtype=dtype)
        eps_rot = torch.randn([B, self.max_link_node, 3], device=device, dtype=dtype)
        R0 = SO3.exp_map(eps_rot.reshape(-1, 3)).to_matrix()
        V_R_rot = matrix_to_axis_angle(R0).reshape(B, self.max_link_node, 3)

        link_robot_embeds = self._prepare_link_embeds(batch, B, device, dtype)

        for step in range(N_steps):
            t_current = step * dt
            t_idx = int(round(t_current * (self.M - 1)))
            t_idx = min(t_idx, self.M - 1)
            t_tensor = torch.full((B,), t_idx, dtype=torch.long, device=device)

            # Build graph
            noisy_V_R = torch.cat(
                [V_R_trans, V_R_rot, link_robot_embeds], dim=-1
            )
            noisy_E_OR, noisy_E_RR = self._build_noisy_edges(
                noisy_V_R, object_nodes, device, dtype
            )

            # Predict velocity
            pred_velocity = self.denoiser(
                object_nodes, noisy_V_R, noisy_E_OR, noisy_E_RR, t_tensor
            )
            pred_v_trans = pred_velocity[:, :, :3]
            pred_v_rot = pred_velocity[:, :, 3:]

            if self.solver == "euler":
                V_R_trans, V_R_rot = self._ode_step_euler(
                    V_R_trans, V_R_rot, pred_v_trans, pred_v_rot, dt
                )
            elif self.solver == "midpoint":
                V_R_trans, V_R_rot = self._ode_step_midpoint(
                    V_R_trans, V_R_rot, pred_v_trans, pred_v_rot, dt,
                    t_current, link_robot_embeds, object_nodes, B, device, dtype,
                )

        # Output: denormalize and format
        pred_trans = V_R_trans * scale + centroids
        pred_rot = V_R_rot
        pred_pose = torch.cat([pred_trans, pred_rot], dim=-1)

        predict_link_pose = vector_to_matrix(pred_pose[:, :valid_links])
        predict_link_pose_dict = {}
        for link_id, link_name in enumerate(link_names):
            predict_link_pose_dict[link_name] = predict_link_pose[:, link_id]
        all_step_poses_dict[0] = predict_link_pose_dict

        return all_step_poses_dict

    def _ode_step_midpoint(
        self, V_R_trans, V_R_rot, pred_v_trans, pred_v_rot, dt,
        t_current, link_robot_embeds, object_nodes, B, device, dtype,
    ):
        """Midpoint method on SE(3) for better accuracy."""
        # Half step
        mid_trans = V_R_trans + 0.5 * dt * pred_v_trans
        R_mid = axis_angle_to_matrix(V_R_rot.reshape(-1, 3))
        delta_R_half = SO3.exp_map(
            0.5 * dt * pred_v_rot.reshape(-1, 3)
        ).to_matrix()
        R_mid = R_mid @ delta_R_half
        mid_rot = matrix_to_axis_angle(R_mid).reshape(B, self.max_link_node, 3)

        # Re-evaluate velocity at midpoint
        t_mid_idx = int(round((t_current + 0.5 * dt) * (self.M - 1)))
        t_mid_idx = min(t_mid_idx, self.M - 1)
        t_mid_tensor = torch.full((B,), t_mid_idx, dtype=torch.long, device=device)

        noisy_V_R_mid = torch.cat([mid_trans, mid_rot, link_robot_embeds], dim=-1)
        noisy_E_OR_mid, noisy_E_RR_mid = self._build_noisy_edges(
            noisy_V_R_mid, object_nodes, device, dtype
        )
        pred_vel_mid = self.denoiser(
            object_nodes, noisy_V_R_mid, noisy_E_OR_mid, noisy_E_RR_mid, t_mid_tensor
        )
        v_trans_mid = pred_vel_mid[:, :, :3]
        v_rot_mid = pred_vel_mid[:, :, 3:]

        # Full step using midpoint velocity
        V_R_trans = V_R_trans + dt * v_trans_mid
        R_current = axis_angle_to_matrix(V_R_rot.reshape(-1, 3))
        delta_R = SO3.exp_map(dt * v_rot_mid.reshape(-1, 3)).to_matrix()
        R_new = R_current @ delta_R
        V_R_rot = matrix_to_axis_angle(R_new).reshape(B, self.max_link_node, 3)

        return V_R_trans, V_R_rot

    def _inference_palm_conditioned(
        self, batch, object_nodes, centroids, scale,
        B, device, dtype, link_names, valid_links, eps,
    ):
        robot_name = batch["robot_name"]
        palm_name = self.palm_names[robot_name]
        palm_index = link_names.index(palm_name)
        palm_R_target = batch["initial_se3"][:, palm_index, :3, :3]  # [B, 3, 3]

        initial_pose = matrix_to_vector(batch["initial_se3"])  # [B, L, 6]

        N_steps = self.ode_steps
        t_start = self.t_start
        dt = (1.0 - t_start) / N_steps
        all_step_poses_dict = {}

        # Link embeddings
        link_robot_embeds = self._prepare_link_embeds(batch, B, device, dtype)

        # Initialize rotation from initial pose (partially interpolated)
        link_robot_rots = torch.zeros(
            [B, self.max_link_node, 3], device=device, dtype=dtype
        )
        num_link = self.robot_links[robot_name]
        for b in range(B):
            link_robot_rots[b, :num_link] = initial_pose[b, :num_link, 3:]

        # Geodesic interpolation to t_start from noise
        eps_rot = torch.randn_like(link_robot_rots)
        R0_flat = SO3.exp_map(eps_rot.reshape(-1, 3)).to_matrix()
        R1_flat = axis_angle_to_matrix(link_robot_rots.reshape(-1, 3))
        R0T_R1 = R0_flat.transpose(-2, -1) @ R1_flat
        log_R0T_R1 = SO3(tensor=R0T_R1).log_map()
        t_start_bc = torch.full(
            (R0_flat.shape[0], 1), t_start, device=device, dtype=dtype
        )
        Rt_init = R0_flat @ SO3.exp_map(t_start_bc * log_R0T_R1).to_matrix()
        V_R_rot = matrix_to_axis_angle(Rt_init).reshape(B, self.max_link_node, 3)

        # Translation: partial interpolation from noise
        x0_trans = torch.randn([B, self.max_link_node, 3], device=device, dtype=dtype)
        V_R_trans = (1.0 - t_start) * x0_trans  # at t_start

        # ODE integration from t_start to 1
        for step in range(N_steps):
            t_current = t_start + step * dt
            t_idx = int(round(t_current * (self.M - 1)))
            t_idx = min(t_idx, self.M - 1)
            t_tensor = torch.full((B,), t_idx, dtype=torch.long, device=device)

            # Build graph
            noisy_V_R = torch.cat(
                [V_R_trans, V_R_rot, link_robot_embeds], dim=-1
            )
            noisy_E_OR, noisy_E_RR = self._build_noisy_edges(
                noisy_V_R, object_nodes, device, dtype
            )

            # Predict velocity
            pred_velocity = self.denoiser(
                object_nodes, noisy_V_R, noisy_E_OR, noisy_E_RR, t_tensor
            )
            pred_v_trans = pred_velocity[:, :, :3]
            pred_v_rot = pred_velocity[:, :, 3:]

            # Euler step
            V_R_trans, V_R_rot = self._ode_step_euler(
                V_R_trans, V_R_rot, pred_v_trans, pred_v_rot, dt
            )

            # Palm rotation correction (Lie algebra interpolation)
            progress = (step + 1) / N_steps
            s_t = self.interpolation_rate * math.sin(0.5 * progress * math.pi)

            R_all = axis_angle_to_matrix(V_R_rot.reshape(-1, 3)).reshape(
                B, self.max_link_node, 3, 3
            )
            R_palm_current = R_all[:, palm_index]  # [B, 3, 3]
            R_err = palm_R_target @ R_palm_current.transpose(-2, -1)  # [B, 3, 3]
            r_err = SO3(tensor=R_err).log_map()  # [B, 3]

            theta = r_err.norm(dim=-1, keepdim=True).clamp_min(eps)
            torch.clamp(theta, max=self.interpolation_clip)
            delta_rate = s_t

            r_delta = delta_rate * r_err  # [B, 3]
            R_delta = SO3.exp_map(r_delta).to_matrix()[:, None, :, :]  # [B, 1, 3, 3]
            R_corrected = torch.matmul(R_delta, R_all)  # [B, L, 3, 3]
            V_R_rot = matrix_to_axis_angle(
                R_corrected.reshape(-1, 3, 3)
            ).reshape(B, self.max_link_node, 3)

        # Output
        pred_trans = V_R_trans * scale + centroids
        pred_rot = V_R_rot
        pred_pose = torch.cat([pred_trans, pred_rot], dim=-1)

        predict_link_pose = vector_to_matrix(pred_pose[:, :valid_links])
        predict_link_pose_dict = {}
        for link_id, link_name in enumerate(link_names):
            predict_link_pose_dict[link_name] = predict_link_pose[:, link_id]
        all_step_poses_dict[0] = predict_link_pose_dict

        return all_step_poses_dict
